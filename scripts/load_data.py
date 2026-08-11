from datasets import load_dataset

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

print(f"Train set size: {len(dataset['train'])}")
print(f"Sample text: {dataset['train'][0]['text'][:20000]}")
