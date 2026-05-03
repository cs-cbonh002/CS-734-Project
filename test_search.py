from pyserini.search.lucene import LuceneSearcher
import json

# Point the searcher to the index directory you just built
searcher = LuceneSearcher('indexes/msmarco-passage-index')

# Run a test query
query = "what is the capital of france"
print(f"Searching for: '{query}'...\n")

# Retrieve top 3 results
hits = searcher.search(query, k=3)

for i in range(len(hits)):
    doc_id = hits[i].docid
    score = hits[i].score

    # Retrieve the raw JSON string and parse it to get the passage text
    raw_doc = searcher.doc(doc_id).raw()
    content = json.loads(raw_doc).get("contents", "No content found")

    print(f"Rank {i + 1} | Score: {score:.4f} | ID: {doc_id}")
    print(f"Passage: {content}\n")
    print("-" * 50)
