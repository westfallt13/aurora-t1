"""Convert a SimpleLLM checkpoint into a standard Hugging Face GPT2LMHeadModel.

Why this works at all: SimpleLLM was deliberately designed to be
architecturally equivalent to GPT-2 (causal self-attention, learned
absolute position embeddings, pre-norm blocks, GELU MLP, tied
input/output embeddings) specifically so it could be bridged to GPT-2's
exact tensor layout for GGUF export, without changing how the model
itself is written or trained.

Two real differences in *layout* (not math) need to be reconciled:

1. GPT-2 fuses all heads' Q, K, V into one matrix per block (`c_attn`);
   SimpleLLM keeps them as separate per-head nn.Linear layers. Since each
   head independently computes `head_i.q(x) = x @ Wq_i.T + bq_i`, and
   GPT-2 reshapes its one big Q vector into per-head chunks of head_dim
   contiguous elements, concatenating our per-head weight/bias tensors in
   head order along the output dimension produces the identical
   computation as one fused matrix -- this isn't an approximation, it's
   the same linear map written two different ways.

2. GPT-2 (in HF's implementation) uses `Conv1D`, which stores its weight
   as (in_features, out_features) and computes `x @ W + b` -- the
   transpose of nn.Linear's (out_features, in_features) / `x @ W.T + b`.
   Every weight crossing that boundary needs `.T`.
"""

import argparse

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from model import GPTConfig, SimpleLLM


def build_gpt2_state_dict(simple_llm_state, config: GPTConfig):
    head_dim = config.hidden_size // config.num_attention_heads
    sd = {}

    sd["transformer.wte.weight"] = simple_llm_state["token_embeddings.weight"]
    sd["transformer.wpe.weight"] = simple_llm_state["position_embeddings.weight"]

    for i in range(config.num_hidden_layers):
        prefix = f"transformer_blocks.{i}."
        out_prefix = f"transformer.h.{i}."

        sd[out_prefix + "ln_1.weight"] = simple_llm_state[prefix + "layer_norm1.weight"]
        sd[out_prefix + "ln_1.bias"] = simple_llm_state[prefix + "layer_norm1.bias"]
        sd[out_prefix + "ln_2.weight"] = simple_llm_state[prefix + "layer_norm2.weight"]
        sd[out_prefix + "ln_2.bias"] = simple_llm_state[prefix + "layer_norm2.bias"]

        # Fuse Q, K, V into one (3*hidden, hidden) matrix. SimpleLLM's
        # q_proj/k_proj/v_proj are already each-all-heads-fused (batched
        # multi-head attention, not per-head Linears -- see model.py), and
        # already use the same head-major row ordering GPT-2's c_attn
        # expects after its own per-head reshape, so this is just a
        # concatenation, no reordering needed.
        qkv_weight = torch.cat(
            [
                simple_llm_state[prefix + "attention.q_proj.weight"],
                simple_llm_state[prefix + "attention.k_proj.weight"],
                simple_llm_state[prefix + "attention.v_proj.weight"],
            ],
            dim=0,
        )  # (3*hidden, hidden), nn.Linear layout
        qkv_bias = torch.cat(
            [
                simple_llm_state[prefix + "attention.q_proj.bias"],
                simple_llm_state[prefix + "attention.k_proj.bias"],
                simple_llm_state[prefix + "attention.v_proj.bias"],
            ],
            dim=0,
        )  # (3*hidden,)

        sd[out_prefix + "attn.c_attn.weight"] = qkv_weight.T.contiguous()  # -> Conv1D layout
        sd[out_prefix + "attn.c_attn.bias"] = qkv_bias

        sd[out_prefix + "attn.c_proj.weight"] = (
            simple_llm_state[prefix + "attention.output_linear.weight"].T.contiguous()
        )
        sd[out_prefix + "attn.c_proj.bias"] = simple_llm_state[
            prefix + "attention.output_linear.bias"
        ]

        sd[out_prefix + "mlp.c_fc.weight"] = (
            simple_llm_state[prefix + "feed_forward.linear1.weight"].T.contiguous()
        )
        sd[out_prefix + "mlp.c_fc.bias"] = simple_llm_state[prefix + "feed_forward.linear1.bias"]
        sd[out_prefix + "mlp.c_proj.weight"] = (
            simple_llm_state[prefix + "feed_forward.linear2.weight"].T.contiguous()
        )
        sd[out_prefix + "mlp.c_proj.bias"] = simple_llm_state[prefix + "feed_forward.linear2.bias"]

    sd["transformer.ln_f.weight"] = simple_llm_state["layer_norm.weight"]
    sd["transformer.ln_f.bias"] = simple_llm_state["layer_norm.bias"]
    # Tied embeddings: same tensor under both names, so load_state_dict
    # doesn't complain about a "missing" lm_head.weight.
    sd["lm_head.weight"] = simple_llm_state["token_embeddings.weight"]

    assert head_dim * config.num_attention_heads == config.hidden_size
    return sd


def convert(checkpoint_path, tokenizer_path, config: GPTConfig, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    simple_llm_state = checkpoint["model_state_dict"]

    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_position_embeddings,
        n_embd=config.hidden_size,
        n_layer=config.num_hidden_layers,
        n_head=config.num_attention_heads,
        n_inner=config.intermediate_size,
        # nn.GELU() defaults to the exact erf-based GELU; GPT2Config's own
        # default ("gelu_new") is the tanh approximation. Match ours exactly
        # rather than silently swapping in a different activation.
        activation_function="gelu",
        resid_pdrop=config.hidden_dropout_prob,
        embd_pdrop=config.hidden_dropout_prob,
        attn_pdrop=config.hidden_dropout_prob,
    )
    gpt2_model = GPT2LMHeadModel(gpt2_config)

    new_state = build_gpt2_state_dict(simple_llm_state, config)
    missing, unexpected = gpt2_model.load_state_dict(new_state, strict=False)
    # tie_weights() re-establishes lm_head/wte sharing on the model's own
    # terms; only real gaps matter, not that overlap.
    real_missing = [k for k in missing if k not in ("lm_head.weight",)]
    if real_missing or unexpected:
        raise RuntimeError(f"Conversion left gaps -- missing={real_missing} unexpected={unexpected}")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token = "<s>"
    tokenizer.unk_token = "<unk>"
    gpt2_config.bos_token_id = tokenizer.bos_token_id
    gpt2_config.eos_token_id = tokenizer.eos_token_id
    # llama.cpp's GPT-2 converter looks for the legacy `n_ctx` key; modern
    # transformers only serializes `n_positions` (same value, different
    # name from an older GPT-2 config convention). Add it explicitly so
    # convert_hf_to_gguf.py doesn't KeyError looking for it.
    gpt2_config.n_ctx = config.max_position_embeddings

    gpt2_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved HF-format GPT2LMHeadModel to {output_dir}")
    return gpt2_model, tokenizer


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-path", default="tokenizer/tokenizer.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hidden-size", type=int, default=384)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--intermediate-size", type=int, default=1536)
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
    convert(args.checkpoint, args.tokenizer_path, config, args.output_dir)
