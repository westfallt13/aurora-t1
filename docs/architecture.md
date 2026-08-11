# Architecture

## The model: `SimpleLLM`

Aurora-T1 is a decoder-only transformer, written from scratch in plain PyTorch (`model.py`, duplicated identically — bar one default — in `scripts/` and `cloud/`) rather than built on top of Hugging Face's `GPT2LMHeadModel`. It is deliberately architecturally equivalent to GPT-2:

- Causal (autoregressive) self-attention
- Learned absolute position embeddings (not RoPE or ALiBi)
- Pre-norm transformer blocks (LayerNorm before attention/FFN, not after)
- GELU-activated MLP
- Tied input/output embeddings (the output projection reuses the token embedding matrix — standard since GPT-2, and saves `vocab_size * hidden_size` parameters)

That equivalence is a deliberate design choice, not an accident: it's what makes `cloud/export_hf_gpt2.py` able to losslessly repack a trained `SimpleLLM` checkpoint into a real `GPT2LMHeadModel`, which is what unlocks GGUF export and Ollama testing (see [Export pipeline](#export-pipeline-checkpoint--gguf) below).

### Component breakdown (`model.py`)

- **`MultiHeadAttention`** — batched: one `Linear(hidden_size, hidden_size)` each for Q/K/V across *all* heads at once (`q_proj`/`k_proj`/`v_proj`), reshaped to `(batch, heads, seq, head_dim)` for a single `F.scaled_dot_product_attention(..., is_causal=True)` call, then `output_linear` back to `hidden_size`. This wasn't the original design — see the `local-110m` migration note below for what it replaced and why, and `migrate_fused_attention.py` for the one-time checkpoint migration this required. On this project's Turing GPU (RTX 2060, compute capability 7.5) the true flash-attention backend isn't available (`SDPBackend.FLASH_ATTENTION` fails with "no available kernel" — confirmed directly against the hardware), so SDPA lands on the memory-efficient attention backend instead; still a real fused kernel, just not literally Flash Attention.
  - **Migration history**: originally each head had its own separate small `q`/`k`/`v` `Linear(hidden_size, head_dim)` run in a Python loop (`AttentionHead`, one instance per head). A first attempt only swapped that loop's manual matmul+mask+softmax+matmul for `F.scaled_dot_product_attention` *without* removing the loop — mathematically identical (verified, max abs diff ~5e-7) but barely moved throughput (~2700-2790 tok/s before vs. ~2770-2800 tok/s after) because the loop itself, not the attention math, was the bottleneck: GPU power draw peaked at 136W of a 170W budget and utilization sat around 40-45%, meaning the GPU was idling between many small kernel launches rather than being compute-bound. Fusing Q/K/V into batched projections (this version) fixed that directly — GPU utilization went to ~98-99%, power draw to ~150-167W, and real end-to-end throughput on `local-110m` went from ~2700-2800 tok/s to a measured, consistent **~16,400 tok/s** (nearly 6x, and faster than the original 23M-param model's ~11,100 tok/s despite this model being 4.8x bigger). Verified numerically before touching the live checkpoint (max abs diff ~4e-6 between old and new architectures on identical weights) and checked end-to-end (migrated checkpoint reproduces the exact same logits as the original).
  - **A real mistake happened during this migration**, worth remembering: the first migration attempt only repacked `model_state_dict`, forgetting that AdamW's `optimizer_state_dict` is keyed by parameter *position*, not name — loading the old per-head optimizer state into an optimizer built from the new fused-parameter model crashed with `ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group` immediately after the checkpoint finished downloading on resume. Fixed by having `migrate_fused_attention.py` set `optimizer_state_dict` to `None` for a migrated checkpoint, and `try_load_checkpoint` in `train.py` skip loading it (starting a fresh optimizer) when it sees `None` instead of crashing — the trained weights are unaffected, only Adam's momentum/variance history resets and rebuilds over the next several steps.
- **`FeedForward`** — the standard two-linear-layer MLP with GELU in between and a final dropout.
- **`TransformerBlock`** — pre-norm residual block: `x = x + Attention(LayerNorm(x))`, then `x = x + FFN(LayerNorm(x))`.
- **`GPTConfig`** — a plain config object (vocab size, hidden size, layer/head counts, intermediate size, dropout, max position embeddings). No `transformers` dependency.
- **`SimpleLLM`** — token embeddings + position embeddings → N `TransformerBlock`s → final LayerNorm → tied output projection to vocab logits.

### Two configs, two purposes

| | `scripts/` (tutorial) | `cloud/` (real pretraining, current: `local-110m`) |
|---|---|---|
| Dataset | WikiText-2 (small, in-memory) | FineWeb-Edu (streamed, ~10BT sample) |
| `vocab_size` | 30,000 | 32,000 |
| `hidden_size` | 256 | 768 |
| `num_hidden_layers` | 4 | 12 |
| `num_attention_heads` | 4 | 12 |
| `intermediate_size` | 512 | 3072 |
| `max_position_embeddings` | 128 | 512 |
| Params | ~a few million | 110,025,216 |

The `scripts/` version exists to validate the architecture and training loop cheaply, end-to-end, before committing to a real pretraining run. It is not part of the Aurora-T1 pretraining lineage — its checkpoints (`checkpoints/model_checkpoint_epoch_*.pt`, `simple_llm_final.pt`, `simple_llm_sentiment.pt`) are tutorial artifacts, not usable base weights for a fork.

The `cloud/` config isn't fixed — it's whatever shape `start_training.bat` currently passes. The prior `local-main` lineage (23,132,160 params: `hidden_size=384, num_layers=6, num_heads=6, intermediate_size=1536`) concluded at ~200M tokens and is now historical (see `research_docs/samples/23M Params/205M/`); `local-110m` is the current, architecturally-incompatible successor described in the table above.

## Tokenizers

There are **two separate ByteLevel-BPE tokenizers** in this repo, and they are not interchangeable:

- `tokenizer/` — 30K vocab, trained on WikiText-2 by `scripts/02_train_tokenizer.py`. Used only by `scripts/`.
- `cloud/tokenizer/` — 32K vocab, trained on the first 300,000 documents of the FineWeb-Edu stream by `cloud/train_tokenizer.py` (streamed rather than downloading the full ~48GB sample, since fitting a stable BPE vocab doesn't need the whole corpus). Used by `cloud/` and by every downstream export/GGUF step.

Both use the same special-token scheme: `<s>` (bos), `<pad>`, `</s>` (eos), `<unk>`, `<mask>`.

## Data pipeline: FineWeb-Edu streaming (`cloud/train.py`)

Real pretraining never downloads the dataset to disk. `open_fineweb_stream()` opens `HuggingFaceFW/fineweb-edu` (`sample-10BT` config) in streaming mode, filters to documents with `int_score >= --min-quality-score` (default 3), and shuffles with a 10,000-document buffer. `int_score` is FineWeb-Edu's own educational-quality classifier score; the paper behind the dataset found that filtering to the higher-scoring subset beats training on more unfiltered tokens at equal compute, so this is a deliberate quality-over-quantity choice, not a subsampling shortcut.

`pack_documents()` then tokenizes documents on the fly and packs the token stream into fixed-length `seq_length + 1` chunks (append an `</s>` between documents, no padding). Every position in every training batch is therefore a real target — unlike the `scripts/` tutorial path, which pads each WikiText-2 line to `max_length` individually.

The first `EVAL_RESERVED_DOCS = 3000` documents of the (seeded, shuffled) stream are permanently reserved for held-out evaluation (`build_eval_batch`) and never seen in training; the training stream (`PackedFineWebStream`) explicitly skips them. This split is stable across restarts/resumes because it depends only on the fixed seed, not on run state.

For multi-worker `DataLoader`s, `PackedFineWebStream` can't use HF's `dataset.shard()` (FineWeb-Edu's stream only exposes one physical shard, so sharding past the first worker raises `IndexError`). Instead each worker walks the *same* stream and keeps only every Nth item via `itertools.islice` — correct, at the cost of every worker independently fetching and discarding the items it skips.

## Training loop (`cloud/train.py`)

- **Optimizer**: AdamW, cosine LR schedule with linear warmup (`get_cosine_schedule_with_warmup`), gradient clipping at norm 1.0.
- **Mixed precision**: `torch.amp.autocast` + `GradScaler`, active only on CUDA.
- **Gradient accumulation**: configurable via `--grad-accum-steps`; the optimizer only steps once every N micro-batches.
- **Checkpointing**: no local checkpoint directory — the checkpoint (model, optimizer, scheduler state, step count, cumulative tokens seen) is written to a temp file and uploaded to `s3://<bucket>/checkpoints/<run-name>/latest.pt`, single-part/single-threaded (`TransferConfig(multipart_threshold=512MB, use_threads=False)`) to avoid a memory-hungry multipart upload for what's normally a ~265MB file at 23M params (~1.3GB at 110M). On start, `try_load_checkpoint` restores from that same S3 key if present, so model/optimizer/scheduler state is resumable across restarts, including on a different machine.
  - **Checkpoint saving is stop-only**, not periodic: `save_checkpoint` (synchronous — the training loop blocks until the upload completes or exhausts `retries`) runs exactly once, right before the process exits, and nowhere else in the training loop. This is a deliberate tradeoff, not an oversight: at 110M params a checkpoint upload is ~1.3GB, and periodic uploads (previously every `--checkpoint-every` steps) were blocking the loop often enough to meaningfully slow training. The cost is that only a *clean* exit (`Ctrl+C`, `--stop-file`, or hitting `--max-hours`/`--max-steps`) saves anything — a crash, forced reboot, or power loss loses all progress back to the previous clean stop, not just "recent" progress. (History: an earlier version made periodic saves non-blocking via a background upload thread instead of removing them, but the resulting sustained, non-pulsing GPU load excited a mechanical rattle on the local training rig, so that approach was reverted in favor of going stop-only instead.)
  - **Downloading** a checkpoint on resume uses the same `use_threads=False` config, which makes it single-connection and slow at 1.3GB — measured 206s and 680s in back-to-back tests (network-dependent, not a fixed cost). With no progress output, this is indistinguishable from a hang; `try_load_checkpoint` now calls `head_object` first to get the total size and passes a `Callback` to `download_file` that prints `[resume] downloading checkpoint: ...` at ~10% increments so a resume doesn't look stuck. `use_threads=False` was deliberately left alone here rather than re-enabled for speed — it was added to fix a real upload-side MemoryError and downloads weren't confirmed safe to parallelize, so this is a visibility fix, not a speed fix. Revisit if resume time becomes a real problem (e.g. the AWS CLI's default multipart download was dramatically faster for the same object in a side-by-side test).
- **`tokens_seen` vs `tokens_seen_this_session`**: `tokens_seen` is the all-time cumulative count carried in the checkpoint across every resume; `tokens_seen_this_session` resets to 0 each run and is only used for the live tokens/sec readout. `metrics.csv` logs the all-time figure.
- **Data stream and resume**: `PackedFineWebStream` reads a single deterministic shuffle (`seed=42`, fixed regardless of `--seed` not being exposed) of the `--fineweb-config` split (default `sample-10BT`; measured to have ~10.3B usable tokens post-filter — see below). `EVAL_RESERVED_DOCS` (3000) reserves the *first* documents of that shuffle as a fixed, reproducible held-out eval set, permanently skipped by training. What checkpoints do **not** capture is stream position — the dataset itself carries no resumable cursor. Left alone, every process restart would re-open the same seed-42 shuffle and start reading right after the eval reservation again, silently re-training on the same early slice of the corpus every time the process is stopped and resumed (the project's normal operating pattern). `resume_skip_docs` (estimated on startup as `tokens_seen / AVG_TOKENS_PER_DOC`, an empirically-measured constant) skips further into the stream to compensate. This is an approximation, not an exact resume — packing buffers make the true document count unrecoverable from `tokens_seen` alone — but it avoids the much worse failure mode of guaranteed full-prefix replay on every restart. Changing `--fineweb-config` or `--min-quality-score` for an existing checkpoint lineage isn't supported (it changes the underlying document stream and invalidates both the resume estimate and the eval-quarantine split) — pair a change here with a new `--run-name`, same as a model-shape change.
  - `--min-quality-score` (default 3) filters on FineWeb-Edu's own educational-quality classifier score, but `HuggingFaceFW/fineweb-edu` (unlike plain `fineweb`) is already pre-filtered to score≥3 at the dataset level — verified empirically (100% of a 30K-doc sample from `sample-10BT` passed). At the default, this filter is a no-op; it only starts actually excluding documents if raised to 4 or 5.
- **Graceful stop**: three independent ways to stop cleanly (always followed by a checkpoint save): `Ctrl+C`, hitting `--max-hours`/`--max-steps`, or dropping a file named by `--stop-file` (default `STOP`) into the working directory — `stop_training.bat` does exactly this.
- **Pause-for-process**: `--pause-for-processes` takes a comma-separated list of process names (e.g. a game); while any of them is running, the training loop polls and does no GPU work at all, resuming automatically once they exit. This is what lets pretraining run in the background on a machine also used interactively. `--background-priority` additionally lowers the process's OS scheduling priority (Windows-specific, no-op elsewhere).
- **Logging**: `metrics.csv` is an append-only `(timestamp, step, tokens_seen, split, loss)` log, written on every `--log-every` train step and every `--eval-every` eval — this is what survives across restarts and is what `training_progress.png` is plotted from (console output does not survive closing the window).

## Export pipeline: checkpoint → GGUF

Three stages, all driven by `refresh_gguf.bat`:

1. **`cloud/export_hf_gpt2.py`** — loads a `SimpleLLM` checkpoint and repacks its state dict into a real `transformers.GPT2LMHeadModel`. This is a pure tensor-layout translation (not a re-training or approximation), reconciling two real differences between the two implementations:
   - GPT-2 fuses all heads' Q/K/V into one `c_attn` matrix per block; `SimpleLLM` keeps them as separate per-head `nn.Linear` layers. Concatenating the per-head weight/bias tensors in head order along the output dimension produces the identical computation as GPT-2's one fused matrix.
   - GPT-2's HF implementation uses `Conv1D`, which stores weights as `(in_features, out_features)` and computes `x @ W + b` — the transpose of `nn.Linear`'s `(out_features, in_features)` / `x @ W.T + b`. Every weight crossing that boundary gets `.T`.

   It also explicitly sets `bos_token_id`/`eos_token_id` from the tokenizer and adds a legacy `n_ctx` config key (equal to `max_position_embeddings`) — llama.cpp's GPT-2 converter looks for that older key name, which modern `transformers` no longer serializes by default.

2. **`tools/llama.cpp/convert_hf_to_gguf.py`** — the standard llama.cpp HF→GGUF converter, run against the HF export directory with `--outtype f16`.

3. **`ollama create <name> -f Modelfile`** — registers the resulting `.gguf` file with Ollama for interactive testing. `cloud/Modelfile`: `FROM ./aurora-t1.gguf`, stop token `</s>`, temperature 0.8, `repeat_penalty 1.3`. That last parameter was added after a real finding: the original `local-110m` qualitative test transcripts (`research_docs/samples/110M/tests/`) showed prompts collapsing into dash/symbol-loop repetition within a few dozen tokens, which looked like a base-model defect. Re-running the same prompts through this pipeline with an explicit repetition penalty set (Ollama/llama.cpp's undocumented default is a much weaker 1.1, and the Modelfile previously set none at all) eliminated the collapse entirely — confirming it was a decoding-settings artifact of this pipeline, not a pretraining problem. Worth remembering before diagnosing future "the model degenerates" reports as a model issue without checking sampling settings first.

The full command sequence, plus how to run it, is in [Guide.md](Guide.md#converting-a-checkpoint-to-gguf).

## Fine-tuning path (`scripts/07_finetune.py`)

The tutorial path also includes a worked example of task fine-tuning: `SimpleLLMForClassification` wraps the pretrained backbone with a linear classification head that pools the hidden state at each sequence's last real (non-pad) token — the only position that, under a causal mask, has attended to the entire input — and fine-tunes on IMDB sentiment at a much lower LR (2e-5 vs. 3e-4 for pretraining) to avoid catastrophic forgetting of the pretrained backbone. This predates, and is architecturally unrelated to, the real SFT pipeline below (`scripts/model.py`'s tiny tutorial config vs. `cloud/model.py`'s real one) — kept as-is as a from-scratch worked example, not superseded by it.

## SFT pipeline (`cloud/sft_train.py`)

Post-pretraining instruction tuning on the fused-attention `SimpleLLM`, starting from a pretraining checkpoint's `model_state_dict` with a fresh AdamW optimizer (much lower LR than pretraining — `5e-5` vs. `3e-4` — since the backbone already holds 2.2B tokens of learned structure).

**No chat special tokens.** The tokenizer (`cloud/tokenizer/tokenizer.json`) only defines `<s>`/`<pad>`/`</s>`/`<unk>`/`<mask>` — adding real `<|user|>`/`<|assistant|>`-style tokens would mean resizing the embedding table before an existing checkpoint could use them. Instead, turns are marked with plain-text tags (`### System:\n`, `### User:\n`, `### Assistant:\n`) that tokenize as ordinary subwords.

**Loss masking.** `encode_multiturn` tokenizes a full conversation once and returns `(ids, mask)`, where `mask[i] == True` marks a position to exclude from the loss (`ignore_index=-100` in `cross_entropy`). System and user turns are masked; assistant turn *content* is not — the model is trained to answer, not to role-play asking the question or reproduce the system prompt. System messages are **kept**, not dropped — an earlier version dropped them as generic persona boilerplate (true for `openhermes-100k`'s "You are an AI assistant...") but that silently breaks tool-calling data (`apigen-80k`), where the system message *is* the tool schema the assistant's response depends on.

**Batched training via right-padding**, not one example per forward pass. Padding tokens are placed strictly after each sequence's real content, and their label positions are masked out (`-100`) the same way prompt tokens are. Under causal attention, a real token's prediction only ever depends on earlier positions — which are always real content, never padding that comes after it — so this is provably exact, not an approximation, despite the model itself having no explicit padding-mask machinery. Verified directly before trusting it: a batched, padded forward pass over two different-length examples produced a loss bit-for-bit identical to the weighted average of computing each example's loss individually.

**Dataset sources**, selected via repeated `--dataset name[:config][:limit]` flags (e.g. `--dataset "HuggingFaceTB/smoltalk:openhermes-100k"`, `--dataset "rajpurkar/squad_v2::35000"` to shuffle-and-cap a source to ~35K usable examples):

- **`messages`-schema sources** (`HuggingFaceH4/no_robots`, `HuggingFaceTB/smoltalk`'s various configs) — a list of `{role, content}` turns, handled generically by `encode_multiturn`.
- **`rajpurkar/squad_v2`** (special-cased, different schema: `context`/`question`/`answers`) — teaches grounded answering: respond from the given passage when it supports an answer, or hedge with one of several varied phrasings (`HEDGE_PHRASES`, sampled per-example to avoid the model learning one memorized magic string) when it doesn't. ~33% of squad_v2's questions are deliberately unanswerable from the given passage — added by the original SQuAD 2.0 authors specifically to stop models from guessing. Chosen for this project because it doubles as practice for the tool-use case below: retrieved search results often won't actually answer the question either, and the behavior needed is the same.
- **`smoltalk:apigen-80k`** (`messages`-schema, but the system message is a JSON tool schema and every assistant turn is a `<tool_call>[...]</tool_call>` block, never a direct answer) — teaches the tool-calling *format*. Heavily filtered by the 512-token limit (~78% of rows dropped — the tool-schema system messages are long) but still yields ~18K usable examples.

**Why squad_v2 and apigen-80k compose**, not just coexist: apigen-80k's assistant turns are *only* ever a tool call — there's no "here's the function result, now answer" turn in that dataset, so a model trained on it alone would learn to call tools but not to use what they return. squad_v2's grounded-answer-or-hedge task is exactly that missing skill (given retrieved-looking text, answer or say you can't) — see `agent_loop.py` below for how the two are chained at inference time.

## Web search tool (`cloud/web_search.py` + `tools/searxng/`)

`web_search.py` is a stdlib-only client (`urllib`, no new dependency) for a self-hosted [SearXNG](https://github.com/searxng/searxng) instance's JSON API — chosen over a hosted search API specifically because the project's requirement was free, self-owned, Docker/Proxmox-deployable, and private (no query data leaving the user's own infrastructure). `tools/searxng/` has the deployment files: `docker-compose.yml` (single container, no redis — the public-instance rate-limiter/bot-detection SearXNG normally needs doesn't apply to a private, single-consumer instance) and `settings.yml` (`formats: [html, json]` explicitly enabled — disabled by default on public instances to prevent scraping abuse, safe here). **Currently deployed locally on the training machine** (2026-08-07, `docker compose up -d` in `tools/searxng/`, reachable at `http://localhost:8888`, verified returning real results) — the plan is a real Docker/Proxmox host eventually, but local-for-now was the explicit choice to unblock testing the tool-use pipeline immediately. `web_search.py` only needs network access to wherever it ends up, so moving it later is a `--base-url` change, not a code change.

## Tool-use agent loop (`cloud/agent_loop.py`)

Two-phase generation, not one continuous pass, because of how the training data above is shaped:

1. **Phase 1**: prompt the model with `TOOL_SYSTEM_PROMPT` (deliberately copied in the same shape `apigen-80k` trains on — tool schema as JSON, `<tool_call>[...]</tool_call>` as the required output format — a from-scratch phrasing here would be off-distribution for what the SFT checkpoint actually learned) plus the user's question. `parse_tool_call` looks for a well-formed `<tool_call>` block; a 110M model's output won't always have one, and that's treated as the expected failure mode, not an error — malformed JSON, a missing block, or an explicit empty list all fall through to treating the raw generation as a direct answer.
2. **Phase 2** (only if a tool call was found): execute it against `web_search.py` (or return canned `MOCK_RESULTS` with `--mock`, for testing without a live SearXNG instance), then prompt again using `sft_train.py`'s exact `SQUAD_INSTRUCTION` template with the search results concatenated in as the "passage" — reusing the grounded-answer-or-hedge behavior squad_v2 trained, now applied to retrieved text instead of a SQuAD paragraph.

**Stated plainly, not oversold**: a 110M model reliably doing reason → decide → call → use-result → answer is a real stretch — this checkpoint can't yet reliably produce a syntactically valid Python function, let alone well-formed JSON tool calls under all conditions. Expect partial, inconsistent tool use from this pipeline, not dependable self-correction.
