"""One-time migration: repacks a SimpleLLM checkpoint's per-head
AttentionHead weights (attention.heads.{h}.{q,k,v}.{weight,bias}) into the
fused MultiHeadAttention layout (attention.{q,k,v}_proj.{weight,bias}).

Purely a weight-layout change, not an approximation: each head's own
Linear computes `x @ W_h.T + b_h`; concatenating the per-head weight rows
in head order produces one bigger Linear whose output, sliced back into
per-head chunks, is identical to running each head independently (every
output row only depends on its own weight row and the shared input `x`).
Verified numerically before use on this project's real checkpoint -- see
the training-status memory for the comparison.

This exists as a real script rather than a throwaway one-off because it's
a real transformation this project's checkpoint lineage underwent and
should be reproducible/inspectable later, same spirit as
export_hf_gpt2.py.
"""

import argparse

import torch

from model import GPTConfig


def migrate_state_dict(old_state, config: GPTConfig):
    new_state = dict(old_state)
    for i in range(config.num_hidden_layers):
        prefix = f"transformer_blocks.{i}.attention."
        for proj in ("q", "k", "v"):
            weights, biases = [], []
            for h in range(config.num_attention_heads):
                head_prefix = f"{prefix}heads.{h}.{proj}."
                weights.append(old_state[head_prefix + "weight"])
                biases.append(old_state[head_prefix + "bias"])
                del new_state[head_prefix + "weight"]
                del new_state[head_prefix + "bias"]
            new_state[f"{prefix}{proj}_proj.weight"] = torch.cat(weights, dim=0)
            new_state[f"{prefix}{proj}_proj.bias"] = torch.cat(biases, dim=0)
    return new_state


def migrate_checkpoint(checkpoint_path, config: GPTConfig, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state_dict"] = migrate_state_dict(checkpoint["model_state_dict"], config)
    # AdamW's saved state is keyed by parameter *position* (via
    # param_groups), not by name -- it has no way to know that the old
    # per-head heads.{h}.q/k/v tensors and the new fused q_proj/k_proj/v_proj
    # tensors are "the same" parameters in a different shape. Loading the
    # old optimizer_state_dict into an optimizer built from the new model
    # would fail on a parameter-count/shape mismatch (this is what actually
    # crashed the first migration attempt). Dropping it just resets Adam's
    # momentum/variance history -- the trained weights themselves are fully
    # preserved by model_state_dict above; only the per-parameter adaptive
    # learning-rate history is lost, which the optimizer rebuilds over the
    # next handful of steps.
    checkpoint["optimizer_state_dict"] = None
    torch.save(checkpoint, output_path)
    return checkpoint


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--intermediate-size", type=int, default=3072)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    args = p.parse_args()

    config = GPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_length,
    )
    migrate_checkpoint(args.checkpoint, config, args.output)
    print(f"Migrated checkpoint written to {args.output}")
