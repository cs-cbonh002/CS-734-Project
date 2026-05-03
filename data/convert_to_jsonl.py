import json
import os

# Create the directory Pyserini will look at
os.makedirs('collection_jsonl', exist_ok=True)

print("Starting conversion... this may take a minute or two.")

# Read the TSV and write out JSONL
with open('collection.tsv', 'r', encoding='utf-8') as f_in, \
     open('collection_jsonl/docs.jsonl', 'w', encoding='utf-8') as f_out:
    for line in f_in:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            f_out.write(json.dumps({"id": parts[0], "contents": parts[1]}) + '\n')

print("Conversion complete! Your data is ready for indexing.")
