import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Batched multi-head attention: one Linear per q/k/v across all heads
    at once (not one small Linear per head in a Python loop), reshaped
    into (batch, heads, seq, head_dim) for a single fused
    scaled_dot_product_attention call. Mathematically identical to running
    each head independently and concatenating -- a head's slice of a
    concatenated (out_features=hidden_size) projection only depends on its
    own weight rows and the shared input, same as an independent per-head
    Linear would produce -- but as a handful of large GPU ops instead of
    num_heads*3 small ones. (Checkpoints from the earlier per-head layout
    need `migrate_fused_attention.py` to load into this shape.)

    On this project's Turing GPU there's no real flash-attention kernel
    available -- PyTorch's scaled_dot_product_attention falls back to its
    memory-efficient attention backend instead (verified directly:
    SDPBackend.FLASH_ATTENTION errors with "no available kernel" on
    compute capability 7.5, EFFICIENT_ATTENTION works) -- still a real
    fused kernel, just not literally Flash Attention.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape

        def split_heads(x):
            return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.q_proj(hidden_states))
        key = split_heads(self.k_proj(hidden_states))
        value = split_heads(self.v_proj(hidden_states))

        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        return self.output_linear(attn_output)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        # Self-attention with residual connection and layer norm
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual

        # Feed-forward with residual connection and layer norm
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual

        return x


class GPTConfig:
    def __init__(
        self,
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings


class SimpleLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Output head
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying: reuse the input embedding matrix as the output
        # projection instead of learning a separate one. Standard practice
        # since GPT-2 -- the two roles are related (mapping a token to a
        # vector, and mapping a vector back to token scores), tying them
        # saves vocab_size * hidden_size parameters (~7.7M here), and it's
        # also required for our checkpoints to convert cleanly into GPT-2's
        # architecture later (GPT-2 ties these by construction).
        self.output.weight = self.token_embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_hidden_states(self, input_ids):
        batch_size, seq_length = input_ids.size()

        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Create position IDs and embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        x = token_embeds + position_embeds

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final layer norm
        return self.layer_norm(x)

    def forward(self, input_ids):
        hidden_states = self.get_hidden_states(input_ids)
        logits = self.output(hidden_states)
        return logits


if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)
    # Stand-in for GPTConfig, used only to smoke-test the individual pieces above.
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_dropout_prob=0.1,
    )
    batch_size, seq_length = 2, 5

    head = AttentionHead(config.hidden_size, config.hidden_size // config.num_attention_heads)
    dummy_input = torch.randn(batch_size, seq_length, config.hidden_size)
    output = head(dummy_input)
    print("AttentionHead    input shape:", dummy_input.shape, " output shape:", output.shape)

    mha = MultiHeadAttention(config)
    mha_output = mha(dummy_input)
    print("MultiHeadAttn    input shape:", dummy_input.shape, " output shape:", mha_output.shape)

    ff = FeedForward(config)
    ff_output = ff(mha_output)
    print("FeedForward      input shape:", mha_output.shape, " output shape:", ff_output.shape)

    block = TransformerBlock(config)
    block_output = block(dummy_input)
    print("TransformerBlock input shape:", dummy_input.shape, " output shape:", block_output.shape)

    # Now the real thing: a full SimpleLLM with GPTConfig, fed actual token IDs.
    gpt_config = GPTConfig(
        vocab_size=30000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
    )
    model = SimpleLLM(gpt_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nSimpleLLM parameters: {num_params:,}")

    dummy_ids = torch.randint(0, gpt_config.vocab_size, (batch_size, 20))
    logits = model(dummy_ids)
    print("SimpleLLM input_ids shape:", dummy_ids.shape, " logits shape:", logits.shape)
