import argparse
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from model import GPTConfig, SimpleLLM

# Plain-text turn tags instead of chat special tokens: the tokenizer
# (tokenizer/tokenizer.json) only defines <s>/<pad>/</s>/<unk>/<mask>, no
# <|user|>/<|assistant|>-style tokens, and adding real special tokens would
# mean resizing the embedding table before this checkpoint could use them.
# These tokenize as ordinary subwords, same as anything else the base model
# saw during pretraining.
SYSTEM_TAG = "### System:\n"
USER_TAG = "### User:\n"
ASSISTANT_TAG = "### Assistant:\n"

# Grounded-answer-or-hedge target for squad_v2's ~33% deliberately
# unanswerable questions. Several varied phrasings, sampled per-example, so
# the model learns the *behavior* (hedge when the passage doesn't support an
# answer) rather than memorizing one magic string. Also reused verbatim by
# agent_loop.py's phase 2 -- retrieved search snippets often don't answer the
# question either, and that's the same skill applied to a different passage.
HEDGE_PHRASES = [
    "I don't have enough information here to answer that accurately.",
    "The passage doesn't say -- I can't answer that from what's given here.",
    "I'm not able to find that in the information provided.",
    "That isn't covered by the text I have, so I can't say for sure.",
    "I don't know based on what's given here.",
]

SQUAD_INSTRUCTION = (
    "Answer the question using only the information in the passage below. "
    "If the passage does not contain the answer, say so honestly instead of guessing.\n\n"
    "Passage:\n{context}\n\n"
    "Question: {question}"
)


def encode_multiturn(tokenizer, messages, eos_id, max_length):
    """Tokenizes a full multi-turn conversation and returns (ids, mask) where
    mask[i]==True means ids[i] should be ignored as a loss target. System and
    user turns are masked; assistant *content* is not -- the model is trained
    to answer, not to role-play asking the question or reproduce the system
    prompt. System messages are kept (not dropped): generic persona text
    (openhermes-100k's "You are an AI assistant...") is low value, but for
    tool-calling data (apigen-80k) the system message *is* the tool schema
    the assistant's response depends on -- dropping it there would silently
    break that data. Returns None if the conversation doesn't fit or has no
    assistant content at all."""
    ids: list[int] = []
    masked: list[bool] = []

    def add(text, mask_it):
        piece = tokenizer.encode(text).ids
        ids.extend(piece)
        masked.extend([mask_it] * len(piece))

    saw_assistant = False
    for msg in messages:
        if msg["role"] == "system":
            add(SYSTEM_TAG + msg["content"] + "\n\n", True)
        elif msg["role"] == "user":
            add(USER_TAG + msg["content"] + "\n\n", True)
        elif msg["role"] == "assistant":
            add(ASSISTANT_TAG, True)
            add(msg["content"] + "\n\n", False)
            saw_assistant = True

    if not saw_assistant:
        return None
    ids.append(eos_id)
    masked.append(False)

    if len(ids) > max_length:
        return None
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masked, dtype=torch.bool)


def load_squad_v2(split, limit, seed):
    """Special-cased: squad_v2's schema (context/question/answers) isn't the
    messages list encode_multiturn expects, so build a synthetic 2-turn
    conversation and hand it to the same encoder -- keeps loss masking
    identical to every other source instead of a parallel code path."""
    dataset = load_dataset("rajpurkar/squad_v2", split=split)
    if limit:
        dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    rng = random.Random(seed)
    rows = []
    for row in dataset:
        answers = row["answers"]["text"]
        answer = answers[0] if answers else rng.choice(HEDGE_PHRASES)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": SQUAD_INSTRUCTION.format(context=row["context"], question=row["question"])},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return rows


def build_examples(tokenizer, sources, split, max_length, seed):
    """sources: list of (dataset_name, config_or_None, limit_or_None)
    triples. Most sources expose a `messages` column of {role, content}
    dicts (no_robots, and smoltalk's various configs, all share this shape)
    -- rajpurkar/squad_v2 is special-cased (different schema, no `test`
    split, so `validation` is substituted for eval)."""
    eos_id = tokenizer.token_to_id("</s>")
    examples = []
    skipped_shape, skipped_length = 0, 0

    for name, config, limit in sources:
        if name == "rajpurkar/squad_v2":
            squad_split = "validation" if split == "test" else split
            rows = load_squad_v2(squad_split, limit, seed)
        else:
            dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
            if limit:
                dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
            rows = dataset

        for row in rows:
            result = encode_multiturn(tokenizer, row["messages"], eos_id, max_length)
            if result is None:
                messages = row["messages"]
                has_assistant = any(m["role"] == "assistant" for m in messages)
                if not has_assistant:
                    skipped_shape += 1
                else:
                    skipped_length += 1
                continue
            examples.append(result)

    print(
        f"[data] {split}: {len(examples)} usable examples from {[n for n, _, _ in sources]} "
        f"(skipped {skipped_shape} malformed, {skipped_length} too long)",
        flush=True,
    )
    return examples


def collate(batch, pad_id):
    """Right-pads a batch to its longest sequence. Padding is always placed
    after the real content, so under causal attention no real token's
    prediction is ever affected by what's in the padding -- a real position
    can only attend to earlier positions, which are all real. Loss on padded
    target positions is masked out the same way prompt tokens are, so this
    is exact, not an approximation, despite the model having no explicit
    padding-mask support."""
    max_len = max(ids.size(0) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len - 1), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)
    for i, (ids, masked) in enumerate(batch):
        length = ids.size(0)
        input_ids[i, : length - 1] = ids[:-1]
        seq_labels = ids.clone()
        seq_labels[masked] = -100
        labels[i, : length - 1] = seq_labels[1:]
    return input_ids, labels


@torch.no_grad()
def evaluate(model, examples, device, pad_id, batch_size=8):
    model.eval()
    total_loss, total_count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        input_ids, labels = collate(chunk, pad_id)
        input_ids, labels = input_ids.to(device), labels.to(device)
        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        total_loss += loss.item()
        total_count += 1
    model.train()
    return total_loss / total_count


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-checkpoint", type=str, default="latest_checkpoint.pt")
    p.add_argument("--output-checkpoint", type=str, default="sft_checkpoint.pt")
    p.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--intermediate-size", type=int, default=3072)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    # Much lower than pretraining's 3e-4 -- the backbone already holds
    # useful structure from 2.2B tokens of pretraining; a large LR here
    # would wreck it before it learns the instruction/response behavior.
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-examples", type=int, default=None, help="debug: cap dataset size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            'name[:config[:limit]], e.g. "HuggingFaceTB/smoltalk:openhermes-100k" or '
            '"rajpurkar/squad_v2::35000" (empty config, shuffle-and-cap to 35K). Repeatable.'
        ),
    )
    return p.parse_args()


# Default recipe: general instructions + conversational tone + grounded
# answer-or-hedge behavior + tool-call format. squad_v2 capped at 35K so it
# doesn't dominate the mix by sheer row count; apigen-80k's long tool-schema
# system messages already get heavily filtered by the 512-token limit.
DEFAULT_SOURCES = [
    ("HuggingFaceTB/smoltalk", "openhermes-100k", None),
    ("HuggingFaceTB/smoltalk", "everyday-conversations", None),
    ("rajpurkar/squad_v2", None, 35000),
    ("HuggingFaceTB/smoltalk", "apigen-80k", None),
]


def parse_sources(dataset_args):
    if not dataset_args:
        return DEFAULT_SOURCES
    sources = []
    for spec in dataset_args:
        parts = spec.split(":", 2)
        name = parts[0]
        config = parts[1] if len(parts) > 1 and parts[1] else None
        limit = int(parts[2]) if len(parts) > 2 and parts[2] else None
        sources.append((name, config, limit))
    return sources


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")
    sources = parse_sources(args.dataset)

    train_examples = build_examples(tokenizer, sources, "train", args.seq_length, args.seed)
    eval_examples = build_examples(tokenizer, sources, "test", args.seq_length, args.seed)
    if args.max_examples:
        train_examples = train_examples[: args.max_examples]
        eval_examples = eval_examples[: min(len(eval_examples), 100)]

    config = GPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_length,
    )
    model = SimpleLLM(config)
    checkpoint = torch.load(args.init_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"[init] loaded pretrained weights from {args.init_checkpoint} "
        f"(step={checkpoint.get('step', '?')}, tokens_seen={checkpoint.get('tokens_seen', '?'):,})",
        flush=True,
    )
    model.to(device)
    model.train()

    print("[eval] pre-SFT loss on held-out set:", flush=True)
    pre_loss = evaluate(model, eval_examples, device, pad_id)
    print(f"  loss={pre_loss:.4f} ppl={torch.exp(torch.tensor(pre_loss)):.1f}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_examples) // (args.batch_size * args.grad_accum_steps))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    step = 0
    running_loss, running_count = 0.0, 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        order = list(range(len(train_examples)))
        random.shuffle(order)
        batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]

        for i, batch_idx in enumerate(batches):
            batch = [train_examples[j] for j in batch_idx]
            input_ids, labels = collate(batch, pad_id)
            input_ids, labels = input_ids.to(device), labels.to(device)

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum_steps
            running_count += 1

            if (i + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    print(
                        f"epoch={epoch + 1} step={step}/{total_steps} "
                        f"loss={running_loss / running_count:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                    running_loss, running_count = 0.0, 0

        epoch_eval_loss = evaluate(model, eval_examples, device, pad_id)
        print(
            f"[eval] after epoch {epoch + 1}: loss={epoch_eval_loss:.4f} "
            f"ppl={torch.exp(torch.tensor(epoch_eval_loss)):.1f}",
            flush=True,
        )
        epoch_path = args.output_checkpoint.replace(".pt", f"_epoch{epoch + 1}.pt")
        torch.save(
            {"model_state_dict": model.state_dict(), "args": vars(args), "base_checkpoint": args.init_checkpoint},
            epoch_path,
        )
        print(f"[checkpoint] saved {epoch_path}", flush=True)

    torch.save(
        {"model_state_dict": model.state_dict(), "args": vars(args), "base_checkpoint": args.init_checkpoint},
        args.output_checkpoint,
    )
    print(f"[done] saved final SFT checkpoint to {args.output_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
