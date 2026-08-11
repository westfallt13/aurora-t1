import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from model import GPTConfig, SimpleLLM

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

max_length = 128  # must match what the model was trained with

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=max_length,
)
model = SimpleLLM(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("checkpoints/simple_llm_final.pt", map_location=device))
model.to(device)


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # No learned position embedding exists past this length -- stop
            # before we'd index past the end of that lookup table.
            if input_ids.size(1) >= config.max_position_embeddings:
                break

            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    prompt = "Artificial intelligence is"
    for temperature in [0.7, 1.0, 1.3]:
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=60, temperature=temperature)
        print(f"[temperature={temperature}]")
        print(generated)
        print()
