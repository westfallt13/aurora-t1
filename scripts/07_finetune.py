import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from model import GPTConfig, SimpleLLM

# stanfordnlp/imdb also ships a 50,000-row "unsupervised" split (unlabeled
# reviews, meant for LM pretraining research) that we don't need here --
# drop it before mapping so we don't waste time tokenizing it.
task_dataset = load_dataset("stanfordnlp/imdb")
task_dataset = DatasetDict({"train": task_dataset["train"], "test": task_dataset["test"]})

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
tokenizer.pad_token = "<pad>"
tokenizer.eos_token = "</s>"
tokenizer.bos_token = "<s>"
tokenizer.unk_token = "<unk>"

max_length = 128


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


processed_datasets = task_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
processed_datasets = processed_datasets.rename_column("label", "labels")
processed_datasets.set_format("torch")


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, attention_mask):
        # Sequences are right-padded, so the last *real* token sits at
        # index (real_token_count - 1), not always at index -1. Under a
        # causal mask it's also the only position that has attended to
        # every earlier token, making it a reasonable single-vector
        # summary of the whole sequence for classification.
        last_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_positions]
        return self.linear(pooled)


class SimpleLLMForClassification(nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.backbone.get_hidden_states(input_ids)
        return self.head(hidden_states, attention_mask)


config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=max_length,
)
backbone = SimpleLLM(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone.load_state_dict(torch.load("checkpoints/simple_llm_final.pt", map_location=device))

model = SimpleLLMForClassification(backbone, num_labels=2)
model.to(device)

train_dataloader = DataLoader(processed_datasets["train"], batch_size=16, shuffle=True)
test_dataloader = DataLoader(processed_datasets["test"], batch_size=16)

# Much smaller than the 3e-4 used for pretraining: the backbone already
# holds useful learned structure, and a large LR here would wreck it
# ("catastrophic forgetting") before the new head has learned anything.
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating accuracy"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    print("Accuracy before fine-tuning (should be near chance, ~0.50):")
    print(f"  {evaluate_accuracy(model, test_dataloader):.4f}")

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Fine-tune epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Fine-tune epoch {epoch + 1} avg loss: {epoch_loss / len(train_dataloader):.4f}")

    print("Accuracy after fine-tuning:")
    print(f"  {evaluate_accuracy(model, test_dataloader):.4f}")

    torch.save(model.state_dict(), "checkpoints/simple_llm_sentiment.pt")
