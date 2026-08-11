# Operator's Guide

Step-by-step instructions for the things you'll actually do in this repo. For *why* things work this way, see [architecture.md](architecture.md).

## Running the tutorial pipeline (`scripts/`)

Only needed if you're validating a change to the architecture itself before touching the real pretraining run. Run in order from the repo root, with `scripts/` as the working directory:

```
cd scripts
..\.venv\Scripts\python.exe 02_train_tokenizer.py   # builds tokenizer/ (30K vocab, WikiText-2)
..\.venv\Scripts\python.exe 03_process_dataset.py    # sanity-checks tokenization
..\.venv\Scripts\python.exe 04_train.py              # trains SimpleLLM, 5 epochs, saves to checkpoints/
..\.venv\Scripts\python.exe 05_evaluate.py           # reports held-out perplexity
..\.venv\Scripts\python.exe 06_generate.py           # samples text at a few temperatures
..\.venv\Scripts\python.exe 07_finetune.py           # fine-tunes checkpoints/simple_llm_final.pt on IMDB sentiment
```

Each script hardcodes its own model config to match `checkpoints/simple_llm_final.pt` (hidden_size=256, 4 layers, 4 heads) — if you change `GPTConfig` in `scripts/model.py`, retrain from `04_train.py` before running the later scripts, or they'll fail to load the checkpoint.

## Running real pretraining (`cloud/`)

**Start (or resume):**

```
start_training.bat
```

This calls `cloud/train.py` with the project's standard flags (`--s3-bucket aurora-llm-checkpoints-752988091124 --run-name local-110m ...`, `--hidden-size 768 --num-layers 12 --num-heads 12 --intermediate-size 3072`, `--batch-size 4 --grad-accum-steps 2` (effective batch size still 8; micro-batch of 4 fits comfortably now that attention is batched/fused — see [architecture.md](architecture.md#the-model-simplellm) — it needed 2 before the fusion), `--max-steps 540000` (~2.2B tokens, the Chinchilla-optimal ratio for 110M params), `--max-hours 999`, `--num-workers 2`). On startup it downloads `s3://.../checkpoints/local-110m/latest.pt` if it exists and resumes from there — running the script again after a stop always continues where it left off (model/scheduler state exactly; optimizer state too, *unless* the checkpoint just went through an architecture migration, in which case it intentionally starts fresh — see architecture.md; the data stream position approximately — see `--fineweb-config` below).

`--background-priority` was dropped: with training now GPU-bound at ~99% utilization instead of leaving headroom, yielding CPU priority to foreground apps isn't a meaningful UX benefit anymore, and this project's current priority is throughput. VRAM headroom is tighter now too (~5.75GB used of 6GB, stable but with less margin than before) — if you add other GPU-using apps while training runs, watch for OOM and drop back to `--batch-size 2 --grad-accum-steps 4` if it happens.

The prior `local-main` lineage (23.1M params: `hidden_size=384, num_layers=6, num_heads=6, intermediate_size=1536`) concluded at its `--max-steps 50000` ceiling (~200M tokens) — see `research_docs/samples/23M Params/205M/`. `local-110m` is a new, architecturally incompatible checkpoint lineage (a 23M-shaped checkpoint cannot be loaded into the 110M model and vice versa), not a continuation of it.

**Dataset size for future, bigger milestones**: `--fineweb-config` (default `sample-10BT`) selects which FineWeb-Edu size bucket to stream from. `sample-10BT` has ~10.3B usable tokens (measured directly, not estimated from its name) — comfortably enough for `local-110m`'s 2.2B-token target, but short of the ~20B tokens a future 1B-param run would want at the same Chinchilla-optimal ratio. When that milestone comes, switch to a larger config (`sample-100BT` or bigger) *and* start a new `--run-name` — reusing a run-name across a config change silently breaks both the resume-skip estimate and the eval-quarantine split, since they're built on that specific stream's shuffle order.

**Stop cleanly:**

```
stop_training.bat
```

Drops a `STOP` file next to `cloud/train.py`; the training loop checks for it every step, deletes it, saves a checkpoint, and exits. This is the *safe* way to stop — closing the training window with the X button can kill the process mid-checkpoint-upload and lose recent progress. `Ctrl+C` in the training window works too and is handled the same way.

**Pausing for other GPU use (e.g. gaming):** already configured via `--pause-for-processes` in `start_training.bat`'s invocation if you add process names there (comma-separated, e.g. `League of Legends.exe,LeagueClientUx.exe`) — training fully idles (no GPU work) while any of them is running, and resumes automatically once they close. Not required for a normal foreground training session.

**Monitoring progress:**

- Live console output while `start_training.bat`'s window is open (`step=... loss=... lr=... tokens=... tok/s=... elapsed=...`).
- `cloud/metrics.csv` — persistent, append-only log of every train/eval loss point, survives restarts.
- `training_progress.png` — regenerated automatically (via `cloud/plot_progress.py`) every time a training session ends (including on a `stop_training.bat`-triggered stop). Regenerate manually any time with:
  ```
  cd cloud
  ..\.venv\Scripts\python.exe plot_progress.py
  ```

## Converting a checkpoint to GGUF

```
refresh_gguf.bat
```

This does **not** require training to be paused — it downloads a fresh snapshot of the current S3 checkpoint without touching the training process. It:

1. Downloads `s3://aurora-llm-checkpoints-752988091124/checkpoints/local-110m/latest.pt` → `cloud/latest_checkpoint.pt`.
2. Runs `cloud/export_hf_gpt2.py` to repack it into a Hugging Face `GPT2LMHeadModel` at `cloud/hf_export/` (model config flags in the script must match the checkpoint's training config — currently hidden-size 768 / 12 layers / 12 heads / intermediate 3072 / seq-length 512 / vocab 32000, matching `start_training.bat`'s flags).
3. Runs `tools/llama.cpp/convert_hf_to_gguf.py` to produce `cloud/aurora-t1.gguf` (f16).
4. Runs `ollama create aurora-t1-scratch -f cloud/Modelfile` to register it with Ollama.

Then test it interactively:

```
ollama run aurora-t1-scratch
```

If you change the model config (layers, hidden size, etc.) for a new pretraining run, update both `start_training.bat`'s flags and the `export_hf_gpt2.py` flags inside `refresh_gguf.bat` together — they must describe the same architecture or the state-dict repacking in step 2 will fail or silently produce garbage.

### Doing it manually (equivalent to `refresh_gguf.bat`)

Useful if you want to convert a specific checkpoint rather than always "latest", or you're on a machine without the AWS CLI configured:

```
cd cloud
..\.venv\Scripts\python.exe export_hf_gpt2.py --checkpoint <path-to-.pt> --tokenizer-path tokenizer/tokenizer.json --output-dir hf_export --hidden-size 768 --num-layers 12 --num-heads 12 --intermediate-size 3072 --seq-length 512 --vocab-size 32000
..\.venv\Scripts\python.exe ..\tools\llama.cpp\convert_hf_to_gguf.py hf_export --outfile aurora-t1.gguf --outtype f16
ollama create aurora-t1-scratch -f Modelfile
```

## Running SFT (`cloud/sft_train.py`)

```
cd cloud
..\.venv\Scripts\python.exe sft_train.py ^
  --dataset "HuggingFaceTB/smoltalk:openhermes-100k" ^
  --dataset "HuggingFaceTB/smoltalk:everyday-conversations" ^
  --dataset "rajpurkar/squad_v2::35000" ^
  --dataset "HuggingFaceTB/smoltalk:apigen-80k" ^
  --epochs 1
```

`--dataset` is repeatable, format `name[:config][:limit]` — omit `config` for datasets like `rajpurkar/squad_v2` that don't have one (`name::limit`, empty middle segment), omit `limit` to use the full filtered dataset. With no `--dataset` flags at all, it defaults to just `openhermes-100k` + `everyday-conversations`. Saves a checkpoint after every epoch (`<output>_epoch<N>.pt`) plus a final one (`--output-checkpoint`, default `sft_checkpoint.pt`) — **rename or move the previous `sft_checkpoint.pt` first** if you want to keep it, the script overwrites that filename by default and doesn't check for an existing one.

Before committing to a long run: use `--max-examples N --epochs 1` to dry-run a tiny slice first and catch bugs in seconds/minutes instead of hours — dataset loaders (especially anything hitting a Hub dataset for the first time) should be schema-checked directly (`datasets.load_dataset(...)[0]`) rather than assumed, the same way `cloud/train.py`'s FineWeb-Edu pipeline was.

Throughput is meaningfully worse than pretraining's ~16,400 tok/s — this script trains on short, independent instruction examples rather than packed long sequences, and GPU utilization sits near 100% but power draw stays low (~72W of 170W), meaning it's kernel-launch/Python-overhead-bound, not compute-bound. Measured ~2.08s/step at `--batch-size 8 --grad-accum-steps 2` (16 examples/step) — budget accordingly; a ~123K-example, 1-epoch run took ~4.5 hours on the project's RTX 2060.

## Testing a checkpoint's generations (`cloud/generate.py`, `cloud/agent_loop.py`)

Direct PyTorch sampling, no GGUF/Ollama round-trip needed:

```
cd cloud
..\.venv\Scripts\python.exe generate.py --checkpoint sft_checkpoint.pt --prompt-format chat "Why is the sky blue?"
```

`--prompt-format chat` wraps the prompt in `### User:\n...\n\n### Assistant:\n` (what current SFT checkpoints are trained on); `--prompt-format alpaca` uses the older `### Instruction:`/`### Response:` format the first (`no_robots`-only) SFT checkpoint used. `--raw` skips templating entirely. Tune `--repetition-penalty`/`--temperature`/`--top-k`/`--top-p` here the same way `cloud/Modelfile` does for Ollama — a missing repetition penalty is what made the base model's raw test transcripts look like they were collapsing into token loops when they mostly weren't (see [architecture.md](architecture.md#export-pipeline-checkpoint--gguf)).

For tool-use testing:

```
..\.venv\Scripts\python.exe agent_loop.py --checkpoint sft_checkpoint.pt --mock "What's the capital of France?"
```

`--mock` uses canned search results instead of a live SearXNG instance. **A live instance is running locally as of 2026-08-07** (`http://localhost:8888`, the default `--search-base-url`), so dropping `--mock` works today — no need to wait on a separate deployment.

## The search backend (`tools/searxng/`)

Currently deployed locally (`docker compose up -d` in `tools/searxng/`, running on this training machine at `http://localhost:8888`) — a deliberate "unblock testing now" choice, not the final home. Moving it to a dedicated Docker/Proxmox host later:

```
cd tools/searxng
openssl rand -hex 32          # paste the output into settings.yml's server.secret_key (already done for the local instance)
docker compose up -d
curl "http://<new-host>:8888/search?q=test&format=json"   # should return JSON, not an error
```

Then update `--search-base-url` (`agent_loop.py`) / `--base-url` (`web_search.py`) to point at the new host — nothing else changes.

## The milestone rhythm

The project moves in fixed-size pretraining increments, each one converted, tested, and written up before moving to the next:

1. Pretrain a batch of tokens (`start_training.bat`, stop at the target token count with `stop_training.bat`).
2. Convert the resulting checkpoint to GGUF and load it into Ollama (`refresh_gguf.bat`).
3. Test it by hand against the fixed prompt set in `research_docs/samples/reusable/tests/`, and write up the results plus free-form notes in a new dated/token-count folder under `research_docs/samples/` (e.g. `research_docs/samples/200M/`), following the pattern already established by `research_docs/samples/100M/`.
4. Log the milestone in `research_docs/Aurora-T1_Benchmark_Spreadsheet.ods`.
5. Move to the next increment.

Current status: the `local-main` (23.1M param) lineage concluded at ~200M tokens; `local-110m` (110M param, GPT-2-small scale) concluded 2026-08-06 at ~2.2B tokens (its Chinchilla-optimal target). Pretraining itself is done for this lineage — active work has moved to SFT and tool-use (see below). `docs/` (this folder) doesn't need updating for each milestone — it only needs to change when the *repository itself* changes (new scripts, changed architecture, new workflow). Milestone-specific results belong in `research_docs/`, not here.

## Current direction: unified SFT + tool-use, not per-variant forks

An earlier version of this plan was to copy the repo per capability (a coding Aurora, a conversational Aurora, etc.) once base pretraining finished. That's not what happened in practice: SFT so far has instead layered multiple capabilities (general instruction-following, grounded answer-or-hedge, tool-call format) into the *same* checkpoint lineage via one combined dataset mix, on the reasoning that a small model doing "reason/hedge, then decide whether to call a tool" needs those skills to compose in a single model, not live in separate forks. See [architecture.md](architecture.md#sft-pipeline-cloudsft_trainpy) for the full reasoning and [architecture.md](architecture.md#tool-use-agent-loop-cloudagent_looppy) for how it's used at inference time. `scripts/07_finetune.py`'s classification-head fine-tune remains a valid template if a genuinely separate-head task ever comes up (e.g. a real classifier, not causal-LM instruction tuning), but isn't the active path for this project.
