from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

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


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

tokenized_datasets.set_format("torch")

if __name__ == "__main__":
    print(tokenized_datasets)
    sample = tokenized_datasets["train"][3]
    print("input_ids shape:", sample["input_ids"].shape)
    print("input_ids:", sample["input_ids"][:20])
    print("decoded:", tokenizer.decode(sample["input_ids"]))
