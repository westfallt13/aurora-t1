import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from model import GPTConfig, SimpleLLM

# --- Data + tokenizer (same as sections 1-3) ---
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

# --- Model ---
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    max_position_embeddings=max_length,
)
model = SimpleLLM(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Training setup ---
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)

optimizer = AdamW(model.parameters(), lr=3e-4)

num_epochs = 5
accumulation_steps = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on {device}")

scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

os.makedirs("checkpoints", exist_ok=True)
batch_losses = []

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            outputs = model(input_ids)

            # Predict token i+1 from position i, not token i from itself.
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        real_loss = loss.item() * accumulation_steps
        epoch_loss += real_loss
        batch_losses.append(real_loss)
        progress_bar.set_postfix({"loss": f"{real_loss:.4f}"})

    if (step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_epoch_loss,
        },
        f"checkpoints/model_checkpoint_epoch_{epoch + 1}.pt",
    )

torch.save(model.state_dict(), "checkpoints/simple_llm_final.pt")

plt.figure()
plt.plot(batch_losses)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.savefig("training_loss.png")
print("Saved loss curve to training_loss.png")
