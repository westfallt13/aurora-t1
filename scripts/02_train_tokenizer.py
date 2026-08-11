import os

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

tokenizer = ByteLevelBPETokenizer()


def get_training_corpus():
    for i in range(0, len(dataset["train"])):
        yield dataset["train"][i]["text"]


tokenizer.train_from_iterator(
    get_training_corpus(),
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

os.makedirs("tokenizer", exist_ok=True)
tokenizer.save_model("tokenizer")
tokenizer.save("tokenizer/tokenizer.json")
