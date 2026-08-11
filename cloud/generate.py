import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPTConfig, SimpleLLM

PROMPT_TEMPLATES = {
    "alpaca": "### Instruction:\n{instruction}\n\n### Response:\n",
    "chat": "### User:\n{instruction}\n\n### Assistant:\n",
}


def load_model(checkpoint_path, args, device):
    config = GPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_length,
    )
    model = SimpleLLM(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, eos_id, seq_length, device):
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = ids[-seq_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)[0, -1, :]

        # Penalize tokens already generated (in the response so far), same
        # idea as HF's repetition_penalty: divide the logit of a token
        # that's pushing the distribution positive, multiply if negative --
        # either way it makes an already-used token less attractive next.
        if repetition_penalty != 1.0:
            for tok in set(ids):
                logits[tok] = logits[tok] / repetition_penalty if logits[tok] > 0 else logits[tok] * repetition_penalty

        logits = logits / max(temperature, 1e-5)

        if top_k > 0:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[-1]] = float("-inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_logits[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        if next_id == eos_id:
            break

    return ids[len(prompt_ids) :]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json")
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--intermediate-size", type=int, default=3072)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw", action="store_true", help="skip the instruction/response template, feed the prompt as raw text")
    p.add_argument("--prompt-format", choices=list(PROMPT_TEMPLATES), default="chat")
    p.add_argument("prompt", type=str)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_id = tokenizer.token_to_id("</s>")
    model = load_model(args.checkpoint, args, device)

    text = args.prompt if args.raw else PROMPT_TEMPLATES[args.prompt_format].format(instruction=args.prompt)
    prompt_ids = tokenizer.encode(text).ids

    output_ids = generate(
        model, tokenizer, prompt_ids, args.max_new_tokens, args.temperature,
        args.top_k, args.top_p, args.repetition_penalty, eos_id, args.seq_length, device,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
