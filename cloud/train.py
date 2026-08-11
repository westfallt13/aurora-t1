import argparse
import csv
import itertools
import os
import tempfile
import time

import boto3
import psutil
import torch
import torch.nn.functional as F
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import get_cosine_schedule_with_warmup

from model import GPTConfig, SimpleLLM
from plot_progress import generate_plot

# Checkpoint size scales with model size (fp32 model + AdamW's two fp32
# moment tensors per param) -- ~265MB for the 23M-param local-main model,
# ~1.3GB for the 110M-param local-110m model. Comfortably under S3's 5GB
# single-PUT limit either way. boto3 defaults to multipart + multiple
# upload threads for anything over
# 8MB, which buffers several chunks concurrently -- more memory-hungry than
# it needs to be for a file this size, and implicated in a MemoryError that
# crashed a training run mid-checkpoint. Force a plain single-part,
# single-threaded transfer instead.
S3_TRANSFER_CONFIG = TransferConfig(multipart_threshold=512 * 1024 * 1024, use_threads=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--intermediate-size", type=int, default=1536)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    # Smoke-test-scale defaults. For the real run, pass explicit --max-steps
    # (calculated from measured tokens/sec) so the cosine LR schedule decays
    # to near-zero right around when the run actually ends, not partway
    # through a schedule sized for a much longer run.
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--max-hours", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-examples", type=int, default=200)
    # FineWeb-Edu's own classifier score (0-5ish) for how useful a page is
    # as educational content. The FineWeb-Edu paper found that filtering to
    # the higher-scoring subset beats training on more unfiltered tokens at
    # equal compute -- essentially a free quality win. Note: the
    # "HuggingFaceFW/fineweb-edu" dataset (as opposed to plain "fineweb") is
    # itself already pre-filtered to score>=3, so at the default of 3 this
    # is a no-op (verified: 100% of a 30K-doc sample passed) -- it only
    # starts actually filtering if raised to 4 or 5.
    p.add_argument("--min-quality-score", type=int, default=3)
    # Which FineWeb-Edu size bucket to stream from (see the dataset's
    # config list on the HF Hub: sample-10BT, sample-100BT, sample-350BT,
    # or the full ~1.3T-token default). sample-10BT has ~10.3B usable
    # tokens (measured) -- comfortably enough for the current 110M-param
    # run's ~2.2B-token target, but well short of the ~20B tokens a future
    # 1B-param run would want at the Chinchilla-optimal ratio. Changing
    # this for an existing checkpoint lineage isn't supported (it changes
    # the underlying document stream entirely, invalidating both the
    # resume-skip estimate and the eval-quarantine split) -- pair a config
    # change with a new --run-name, same as a model-shape change.
    p.add_argument("--fineweb-config", type=str, default="sample-10BT")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--s3-bucket", type=str, required=True)
    p.add_argument("--s3-prefix", type=str, default="checkpoints")
    p.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    p.add_argument("--run-name", type=str, default="run1")
    # For running on a machine you're also using for other things at the
    # same time (browsing, gaming): pause training entirely whenever any of
    # these process names are running, and resume automatically once they
    # exit. Comma-separated, e.g. "League of Legends.exe,LeagueClientUx.exe".
    p.add_argument("--pause-for-processes", type=str, default="")
    p.add_argument("--pause-poll-seconds", type=float, default=5.0)
    # Sleep this long after every optimizer step, even outside a pause --
    # keeps the GPU from being saturated 100% of the time, leaving more
    # consistent headroom for foreground use (browsing, light gaming) at
    # the cost of overall training throughput.
    p.add_argument("--step-delay", type=float, default=0.0)
    # Lower this process's OS scheduling priority so foreground interactive
    # apps get preferential CPU access under contention. Windows-specific;
    # harmless no-op elsewhere.
    p.add_argument("--background-priority", action="store_true")
    # If this file exists, pause exactly like a watched process is running
    # (and delete it once acted on). Lets a session be paused without
    # needing to be at the keyboard -- from a script, or by me creating the
    # file directly when asked to pause on your behalf.
    p.add_argument("--stop-file", type=str, default="STOP")
    # Local, append-only log of (timestamp, step, tokens, split, loss) rows,
    # one per log/eval event, that persists across every start/stop/resume
    # cycle -- what a loss curve gets plotted from later, since console
    # output itself doesn't survive closing the window.
    p.add_argument("--metrics-path", type=str, default="metrics.csv")
    return p.parse_args()


def lower_process_priority():
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print("[priority] process priority lowered to below-normal", flush=True)
    except Exception as e:
        print(f"[priority] could not lower process priority: {e}", flush=True)


def log_metric(metrics_path, step, tokens_seen, split, loss):
    is_new = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "step", "tokens_seen", "split", "loss"])
        writer.writerow([time.time(), step, tokens_seen, split, f"{loss:.6f}"])


def check_stop_file(stop_file):
    if os.path.exists(stop_file):
        os.remove(stop_file)
        return True
    return False


def wait_while_processes_running(process_names, poll_seconds):
    if not process_names:
        return
    announced = False
    while True:
        running = {
            p.info["name"]
            for p in psutil.process_iter(["name"])
            if p.info["name"] in process_names
        }
        if not running:
            if announced:
                print("[resume] watched process(es) closed, resuming training", flush=True)
            return
        if not announced:
            print(f"[pause] detected running: {sorted(running)} -- pausing training", flush=True)
            announced = True
        time.sleep(poll_seconds)


# Number of (shuffled) documents permanently reserved for validation and
# never seen during training. Held fixed regardless of --seed so eval stays
# comparable across resumed/restarted runs.
EVAL_RESERVED_DOCS = 3000

# Measured empirically (avg tokens/doc under our tokenizer, sampled from the
# shuffled stream): used only to estimate how many documents to skip past on
# resume, so a restart continues roughly where the data stream left off
# instead of re-reading the same prefix every time. Approximate by design --
# packing buffers make an exact document count unrecoverable from tokens_seen
# alone -- but close enough to avoid grossly re-training on the same early
# slice of the corpus every time the process restarts.
AVG_TOKENS_PER_DOC = 1050


def open_fineweb_stream(seed, min_quality_score, fineweb_config="sample-10BT"):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", fineweb_config, split="train", streaming=True
    )
    dataset = dataset.filter(lambda x: x["int_score"] >= min_quality_score)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    return dataset


def pack_documents(dataset, tokenizer, seq_length, limit=None):
    eos_id = tokenizer.token_to_id("</s>")
    buffer = []
    count = 0
    for example in dataset:
        ids = tokenizer.encode(example["text"]).ids
        buffer.extend(ids)
        buffer.append(eos_id)

        while len(buffer) >= seq_length + 1:
            chunk = buffer[: seq_length + 1]
            buffer = buffer[seq_length + 1 :]
            yield torch.tensor(chunk, dtype=torch.long)
            count += 1
            if limit is not None and count >= limit:
                return


def build_eval_batch(tokenizer_path, seq_length, min_quality_score, num_examples, fineweb_config="sample-10BT", seed=42):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    dataset = open_fineweb_stream(seed, min_quality_score, fineweb_config).take(EVAL_RESERVED_DOCS)
    examples = list(pack_documents(dataset, tokenizer, seq_length, limit=num_examples))
    return torch.stack(examples)


@torch.no_grad()
def evaluate(model, eval_batch, device, eval_batch_size=16):
    # Chunk through the held-out set in small mini-batches rather than one
    # giant forward pass -- a (200, seq_len) batch materializes a
    # (200, seq_len, vocab_size) logits tensor, which is large enough to
    # OOM even generously-sized GPUs, and doesn't shrink just because the
    # eval set itself is small in example count.
    model.eval()
    total_loss, total_batches = 0.0, 0
    for start in range(0, eval_batch.size(0), eval_batch_size):
        chunk = eval_batch[start : start + eval_batch_size].to(device)
        input_ids = chunk[:, :-1]
        labels = chunk[:, 1:]
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / total_batches


class PackedFineWebStream(IterableDataset):
    """Streams FineWeb-Edu, tokenizes on the fly, and packs the token stream
    into fixed-length chunks with no padding -- every position in every
    batch is a real training target, unlike the per-line padded approach
    used for the small WikiText tutorial run. Skips the first
    EVAL_RESERVED_DOCS documents of the (seeded, shuffled) stream, which are
    permanently held out for validation by build_eval_batch, plus
    resume_skip_docs more on top of that (see AVG_TOKENS_PER_DOC)."""

    def __init__(
        self,
        tokenizer_path,
        seq_length,
        min_quality_score,
        fineweb_config="sample-10BT",
        seed=42,
        resume_skip_docs=0,
    ):
        self.tokenizer_path = tokenizer_path
        self.seq_length = seq_length
        self.min_quality_score = min_quality_score
        self.fineweb_config = fineweb_config
        self.seed = seed
        self.resume_skip_docs = resume_skip_docs

    def __iter__(self):
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        dataset = open_fineweb_stream(self.seed, self.min_quality_score, self.fineweb_config)
        dataset = dataset.skip(EVAL_RESERVED_DOCS + self.resume_skip_docs)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            # dataset.shard() needs enough underlying physical file-shards
            # to split across workers -- FineWeb-Edu's stream only exposes
            # one, so that call crashes with an IndexError for any worker
            # past the first. Split at the iteration level instead: each
            # worker walks the same stream but keeps only every Nth item.
            # Correct regardless of physical shard count, at the cost of
            # every worker independently fetching (and discarding) the
            # items it skips.
            dataset = itertools.islice(dataset, worker_info.id, None, worker_info.num_workers)

        yield from pack_documents(dataset, tokenizer, self.seq_length)


def checkpoint_key(args):
    return f"{args.s3_prefix}/{args.run_name}/latest.pt"


def _write_checkpoint_file(step, tokens_seen, model, optimizer, scheduler, args):
    # Save to a real temp file rather than an in-memory BytesIO buffer:
    # torch.save can miscalculate the zip end-of-file position when writing
    # directly into a BytesIO ("unexpected pos" RuntimeError), a known quirk
    # with non-file buffers. A temp file sidesteps it and is the more usual
    # pattern for checkpoints of any real size anyway.
    #
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        },
        tmp_path,
    )
    return tmp_path


def _upload_checkpoint_file(tmp_path, step, tokens_seen, args, retries):
    try:
        key = checkpoint_key(args)
        # A transient upload failure (network hiccup, momentary memory
        # pressure) shouldn't be able to kill an unattended multi-day
        # training run. Retry a couple times, and if it still fails, log
        # and let training continue rather than crashing -- the next
        # periodic checkpoint will simply try again.
        last_error = None
        for attempt in range(retries + 1):
            try:
                boto3.client("s3").upload_file(
                    tmp_path, args.s3_bucket, key, Config=S3_TRANSFER_CONFIG
                )
                print(
                    f"[checkpoint] step={step} tokens={tokens_seen:,} -> s3://{args.s3_bucket}/{key}",
                    flush=True,
                )
                return True
            except Exception as e:
                last_error = e
                if attempt < retries:
                    print(f"[checkpoint] upload attempt {attempt + 1} failed ({e}), retrying...", flush=True)
                    time.sleep(5)
        print(f"[checkpoint] FAILED to save step={step} after {retries + 1} attempts: {last_error}", flush=True)
        return False
    finally:
        os.remove(tmp_path)


def save_checkpoint(step, tokens_seen, model, optimizer, scheduler, args, retries=1):
    """Writes and uploads before returning. Blocks the training loop for the
    duration of the S3 upload."""
    tmp_path = _write_checkpoint_file(step, tokens_seen, model, optimizer, scheduler, args)
    return _upload_checkpoint_file(tmp_path, step, tokens_seen, args, retries)


def _checkpoint_download_progress(total_bytes):
    # use_threads=False makes a large checkpoint download take minutes over
    # a single connection (observed: ~1.3GB in ~3.5 minutes) with zero
    # output from boto3 in the meantime -- indistinguishable from a hang
    # without this. Reports at ~10% increments rather than every chunk.
    state = {"downloaded": 0, "last_pct": -10}

    def callback(bytes_amount):
        state["downloaded"] += bytes_amount
        pct = int(state["downloaded"] * 100 / total_bytes) if total_bytes else 0
        if pct >= state["last_pct"] + 10:
            state["last_pct"] = pct
            print(
                f"[resume] downloading checkpoint: {state['downloaded'] / 1e6:.0f}MB / "
                f"{total_bytes / 1e6:.0f}MB ({pct}%)",
                flush=True,
            )

    return callback


def try_load_checkpoint(model, optimizer, scheduler, args, device):
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        total_bytes = s3.head_object(Bucket=args.s3_bucket, Key=checkpoint_key(args))["ContentLength"]
        print(f"[resume] downloading checkpoint ({total_bytes / 1e6:.0f}MB)...", flush=True)
        s3.download_file(
            args.s3_bucket,
            checkpoint_key(args),
            tmp_path,
            Config=S3_TRANSFER_CONFIG,
            Callback=_checkpoint_download_progress(total_bytes),
        )
    except ClientError as e:
        os.remove(tmp_path)
        # A confirmed "this object doesn't exist" is the only legitimate
        # reason to start fresh. Any other S3 error (permissions, etc.)
        # must not be treated the same way -- silently proceeding could
        # overwrite real progress with a step-0 model.
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return 0, 0
        raise
    except Exception:
        os.remove(tmp_path)
        # Network/memory/etc failure while a checkpoint might genuinely
        # exist -- NOT the same as "no checkpoint exists yet". Fail loudly
        # rather than silently starting over.
        raise

    checkpoint = torch.load(tmp_path, map_location=device)
    os.remove(tmp_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    # None marks a checkpoint whose optimizer state isn't compatible with
    # the current model (e.g. right after an architecture migration --
    # AdamW's state is keyed by parameter position, not name, so it can't
    # carry over across a shape change). Keep the freshly-initialized
    # optimizer instead of crashing; only the momentum/variance history is
    # lost, not the trained weights.
    if checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("[resume] optimizer state not compatible (post-migration checkpoint) -- starting fresh optimizer state", flush=True)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(
        f"[resume] loaded step={checkpoint['step']} tokens_seen={checkpoint['tokens_seen']:,}",
        flush=True,
    )
    return checkpoint["step"], checkpoint["tokens_seen"]


def main():
    args = parse_args()
    pause_names = {n.strip() for n in args.pause_for_processes.split(",") if n.strip()}

    if args.background_priority:
        lower_process_priority()

    config = GPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_length,
    )
    model = SimpleLLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    step, tokens_seen = try_load_checkpoint(model, optimizer, scheduler, args, device)

    print("Building held-out validation set...", flush=True)
    eval_batch = build_eval_batch(
        args.tokenizer_path, args.seq_length, args.min_quality_score, args.eval_examples, args.fineweb_config
    )
    print(f"Validation set: {eval_batch.shape[0]} examples of {eval_batch.shape[1]} tokens", flush=True)

    resume_skip_docs = round(tokens_seen / AVG_TOKENS_PER_DOC) if tokens_seen else 0
    if resume_skip_docs:
        print(
            f"[resume] skipping ~{resume_skip_docs:,} already-seen documents in the training stream",
            flush=True,
        )
    dataset = PackedFineWebStream(
        args.tokenizer_path,
        args.seq_length,
        args.min_quality_score,
        fineweb_config=args.fineweb_config,
        resume_skip_docs=resume_skip_docs,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    model.train()
    start_time = time.time()
    deadline = start_time + args.max_hours * 3600
    running_loss, running_count = 0.0, 0
    tokens_seen_this_session = 0  # separate from tokens_seen (all-time, incl. resumed history)
    optimizer.zero_grad()

    try:
        for batch_idx, batch in enumerate(dataloader):
            if time.time() >= deadline:
                print("[stop] time limit reached", flush=True)
                break
            if step >= args.max_steps:
                print("[stop] max steps reached", flush=True)
                break
            if check_stop_file(args.stop_file):
                print(f"[stop] {args.stop_file} found, saving checkpoint and exiting", flush=True)
                break

            # Fully pause -- no GPU work at all -- while a watched process
            # (e.g. a game) is running, so it never has to compete for the
            # GPU. Checked every step; the poll itself is cheap relative to
            # a training step and only spins while actually paused.
            wait_while_processes_running(pause_names, args.pause_poll_seconds)

            batch = batch.to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum_steps
            running_count += 1
            tokens_seen += batch.numel()
            tokens_seen_this_session += batch.numel()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = tokens_seen_this_session / elapsed if elapsed > 0 else 0
                    train_loss = running_loss / running_count
                    print(
                        f"step={step} loss={train_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} tokens={tokens_seen:,} "
                        f"tok/s={tok_per_sec:,.0f} elapsed={elapsed / 60:.1f}min",
                        flush=True,
                    )
                    log_metric(args.metrics_path, step, tokens_seen, "train", train_loss)
                    running_loss, running_count = 0.0, 0

                if step % args.eval_every == 0:
                    eval_loss = evaluate(model, eval_batch, device)
                    print(
                        f"step={step} eval_loss={eval_loss:.4f} eval_ppl={torch.exp(torch.tensor(eval_loss)):.1f}",
                        flush=True,
                    )
                    log_metric(args.metrics_path, step, tokens_seen, "eval", eval_loss)

                if args.step_delay > 0:
                    time.sleep(args.step_delay)
    except KeyboardInterrupt:
        # Ctrl+C is the manual equivalent of --pause-for-processes: stop
        # cleanly with a checkpoint instead of losing progress since the
        # last periodic save.
        print("\n[interrupt] Ctrl+C received, saving checkpoint before exiting", flush=True)
    except Exception as e:
        # Anything else unexpected (not just checkpoint upload failures,
        # which save_checkpoint already handles internally without
        # raising) -- still make a best-effort attempt to preserve
        # progress rather than silently losing everything back to the
        # last periodic save.
        print(f"\n[error] unexpected exception: {e!r}", flush=True)
        print("[error] attempting to save checkpoint before exiting", flush=True)

    save_checkpoint(step, tokens_seen, model, optimizer, scheduler, args, retries=3)

    try:
        n_train, n_eval = generate_plot()
        print(f"[plot] saved training_progress.png ({n_train} train points, {n_eval} eval points)", flush=True)
    except Exception as e:
        # Never let a plotting failure mask a clean training exit.
        print(f"[plot] failed to generate plot: {e!r}", flush=True)

    print("[done] training loop exited cleanly", flush=True)


if __name__ == "__main__":
    main()
