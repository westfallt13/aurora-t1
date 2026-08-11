import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from model import GPTConfig, SimpleLLM

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

max_length = 128


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format("torch")

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

eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=16)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)

            outputs = model(input_ids)

            # Same next-token shift as training.
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="sum",
            )

            non_pad_mask = shift_labels.reshape(-1).ne(tokenizer.pad_token_id)
            num_tokens = non_pad_mask.sum().item()

            total_loss += loss.item()
            total_tokens += num_tokens

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()


if __name__ == "__main__":
    perplexity = evaluate(model, eval_dataloader)
    print(f"Validation perplexity: {perplexity:.2f}")
