# Aurora-T1

Aurora-T1 is a from-scratch GPT-2-style language model, built and trained end-to-end by hand (no pre-built model classes) as both a learning project and the seed for a family of fine-tuned "Aurora" variants.

The long-term plan: pretrain one solid base model, then copy this repository per variant and fine-tune each copy toward a different capability (e.g. a coding-focused Aurora, a conversational Aurora, a creative-writing Aurora) without having to repeat the expensive pretraining step each time.

## Project status

As of 2026-08-04, the first pretraining lineage (`local-main`, 23.1M params: `hidden_size=384, num_layers=6, num_heads=6, intermediate_size=1536`) concluded at its `--max-steps 50000` ceiling, ~200M tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Results are written up in `research_docs/samples/23M Params/205M/` — output was grammatical at the phrase level but still degenerating into repeated-token loops on some prompts, not yet coherent prose, which is expected for a model this small.

Pretraining moved to a **`local-110m`** lineage: a GPT-2-small-scale model (110M params: `hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072`), an architecturally incompatible fresh start rather than a continuation. **As of 2026-08-06, `local-110m` pretraining concluded**: 536,625/540,000 steps, ~2.2B tokens (the Chinchilla-optimal ratio for 110M params), final eval loss 3.26 (down from ~10.7 at initialization). Results in `research_docs/samples/110M/tests/`.

**Since then, the project has moved into SFT and tool-use** rather than forking per capability. `cloud/sft_train.py` fine-tunes the pretrained checkpoint on instruction/QA data with loss masking (only response tokens count toward the loss). Three experiments so far, each producing its own checkpoint lineage (not overwritten — see `cloud/sft_checkpoint_*.pt`):

1. A small probe on `HuggingFaceH4/no_robots` (~7K examples) — confirmed SFT teaches topical relevance and response format that the base model lacked, but not new capability (math/code answers still wrong, just more confidently formatted).
2. A bigger run on `HuggingFaceTB/smoltalk`'s `openhermes-100k` + `everyday-conversations` (~71K examples) — more fluent and structurally sophisticated, but *not* more accurate; confirmed the "sounds right" vs. "is right" gap doesn't close with more/better instruction data alone.
3. A combined run adding `rajpurkar/squad_v2` (grounded answer-or-hedge behavior, ~35K subsampled) and `smoltalk`'s `apigen-80k` (tool-call format, ~18K) on top of the above (~123K examples total) — aimed at the model reasoning/hedging *before* deciding whether to invoke a tool, rather than answering from parametric memory alone.

Full write-up and reasoning: `research_docs/samples/110M/sft-comparisons/` and [architecture.md](architecture.md#sft-pipeline-cloudsft_trainpy).

**Tool use**: `cloud/web_search.py` queries a self-hosted [SearXNG](https://github.com/searxng/searxng) instance (deployment files in `tools/searxng/`) for web search results, with no external API dependency. `cloud/agent_loop.py` wires this into a two-phase generation loop — see [architecture.md](architecture.md#tool-use-agent-loop-cloudagent_looppy) for why it's two phases, not one.

**Known limitation, stated plainly**: a 110M model reliably doing reason → decide → call a tool → use the result → answer is a real stretch. Expect partial, inconsistent tool use, not dependable self-correction — this is being built and tested incrementally, not sold as solved.

## Repository map

| Path | What it is |
|---|---|
| `scripts/` | The original tutorial-scale implementation: a tiny SimpleLLM trained on WikiText-2, used to build and validate the architecture before scaling up. See [architecture.md](architecture.md#two-tracks-tutorial-vs-real-pretraining). |
| `cloud/` | The real Aurora-T1 pretraining pipeline: streams FineWeb-Edu, trains a larger model, checkpoints to S3, and can run locally or on a cloud GPU instance. This is where all current pretraining happens. |
| `cloud/sft_train.py` | Post-pretraining instruction tuning: loss-masked SFT over `messages`-schema datasets (multi-turn, system messages kept), plus special-cased loaders for `rajpurkar/squad_v2` (grounded answer-or-hedge) and tool-call data. Batched with correctness-verified right-padding. |
| `cloud/generate.py` | Direct PyTorch sampling from a checkpoint (temperature/top-k/top-p/repetition-penalty) — used for base-vs-SFT comparisons without going through the GGUF/Ollama pipeline. |
| `cloud/web_search.py` | stdlib-only client for a self-hosted SearXNG instance's JSON API. |
| `cloud/agent_loop.py` | Two-phase tool-use orchestration: generate with a tool schema in context, parse a `<tool_call>` block if the model produces one, execute it, then generate a final answer grounded in the result. `--mock` runs it without a live search backend. |
| `tools/searxng/` | Deployment files (`docker-compose.yml` + `settings.yml`) for a self-hosted, private SearXNG instance — the search backend `web_search.py` talks to. **Currently running locally** on the training machine (`http://localhost:8888`) — moving to a dedicated Docker/Proxmox host later is a `--base-url` change, not a re-deploy. |
| `checkpoints/` | Saved weights from the `scripts/` tutorial run (WikiText-2 epochs, IMDB sentiment fine-tune). Not part of the FineWeb-Edu pretraining lineage. |
| `tokenizer/` | 30K-vocab BPE tokenizer trained on WikiText-2, used by `scripts/`. |
| `cloud/tokenizer/` | 32K-vocab BPE tokenizer trained on a FineWeb-Edu sample, used by `cloud/`. These two tokenizers are **not interchangeable**. |
| `tools/llama.cpp` | Vendored [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout, used for its `convert_hf_to_gguf.py` conversion script; its bundled server also has native tool-calling/MCP support, not yet explored as an alternative to `agent_loop.py`. |
| `gguf-Results/` | Saved transcripts from manually testing GGUF exports in Ollama (from the earlier, since-abandoned 23M lineage). |
| `research_docs/` | Per-milestone qualitative research journal (fixed test-prompt categories + free-form journal + a benchmark spreadsheet), plus `samples/110M/sft-comparisons/` for base-vs-SFT generation comparisons. Maintained by hand alongside each milestone — see below. |
| `docs/` | This folder: standing documentation of the repo itself (not milestone-specific). |
| `start_training.bat` / `stop_training.bat` / `refresh_gguf.bat` | Windows entry points for the cloud training loop and the checkpoint → GGUF → Ollama pipeline. |
| `training_progress.png` | Auto-generated train/eval loss plot from the (now concluded) `local-110m` pretraining run. |

Full technical detail on the model and pipelines: [architecture.md](architecture.md). Step-by-step operational instructions: [Guide.md](Guide.md).

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`torch` is intentionally left out of `requirements.txt` — install it separately with the CUDA-specific index URL for your GPU, e.g.:

```
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
```

AWS CLI (configured with credentials for the `aurora-llm-checkpoints-*` S3 bucket) and [Ollama](https://ollama.com) are required for the GGUF export/test workflow but not for training itself.

## Research documentation

`research_docs/` is where qualitative evaluation lives: a `samples/reusable/` folder holds a fixed set of prompts across seven categories (coding, conversation, creative writing, following instructions, grammar, knowledge, math) plus a training-journal template. At each milestone (e.g. `samples/100M/`), that reusable template gets copied and filled in with the model's actual outputs and journal notes for that checkpoint, alongside a running benchmark spreadsheet. That folder is maintained by hand, separately from this `docs/` folder — this `docs/` folder instead documents the repository itself: what exists, how it fits together, and how to operate it.
