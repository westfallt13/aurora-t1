import itertools
import os

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Stream instead of downloading the full 48GB sample -- we only need a few
# hundred thousand documents' worth of text to fit a stable BPE vocabulary,
# not the entire corpus.
NUM_DOCS_FOR_TOKENIZER = 300_000

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
)


def get_training_corpus():
    for example in itertools.islice(dataset, NUM_DOCS_FOR_TOKENIZER):
        yield example["text"]


tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(
    get_training_corpus(),
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

os.makedirs("cloud/tokenizer", exist_ok=True)
tokenizer.save_model("cloud/tokenizer")
tokenizer.save("cloud/tokenizer/tokenizer.json")

print("Saved tokenizer to cloud/tokenizer/")
