import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

from haystack import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import ComponentDevice

print("Loading Pyserini Index and MS MARCO Dev Queries...")
searcher = LuceneSearcher('indexes/msmarco-passage-index')
topics = get_topics('msmarco-passage-dev-subset')

# Load official MS MARCO relevance judgments (qrels)
# Structure: { qid (int): { doc_id (int): relevance_score (str), ... }, ... }
print("Loading MS MARCO Qrels...")
qrels = get_qrels('msmarco-passage-dev-subset')

print("Initializing the Cross-Encoder on Apple Silicon (MPS)...")
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
    device=ComponentDevice.from_str("mps")
)
ranker.warm_up()  # Loads the model into unified memory

output_file = "runs/run.msmarco-passage.reranked.txt"

print("Starting Neural Reranking. This will take roughly 30-45 minutes on the M1 Max...")

# Accumulators for metrics
mrr_scores = []
p10_scores = []
queries_ranked = 0

with open(output_file, 'w') as f:
    for qid, topic in tqdm(topics.items(), total=len(topics)):
        query = topic['title']

        # Step 1: BM25 Recall (Grab top 100 candidates)
        hits = searcher.search(query, k=100)

        documents = []
        for hit in hits:
            raw_doc = json.loads(searcher.doc(hit.docid).raw())
            content = raw_doc.get("contents", "")
            documents.append(
                Document(content=content, meta={"doc_id": hit.docid}))

        # Step 2: Neural Precision Reranking
        result = ranker.run(query=query, documents=documents)
        reranked_docs = result["documents"]

        # Step 3: Write to official MS MARCO format (query_id, doc_id, rank)
        for rank, doc in enumerate(reranked_docs):
            f.write(f"{qid}\t{doc.meta['doc_id']}\t{rank + 1}\n")

        # Step 4: Compute MRR@10 and P@10 for this query
        # Note: qrels keys are int, but doc_id stored in meta is a string — cast to int to match
        relevant_docs = qrels.get(qid, {})

        # MRR@10: reciprocal rank of the first relevant doc in top 10
        rr = 0.0
        for rank, doc in enumerate(reranked_docs[:10]):
            if int(doc.meta['doc_id']) in relevant_docs and int(relevant_docs[int(doc.meta['doc_id'])]) > 0:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)

        # P@10: fraction of top 10 results that are relevant
        relevant_in_top10 = sum(
            1 for doc in reranked_docs[:10]
            if int(doc.meta['doc_id']) in relevant_docs and int(relevant_docs[int(doc.meta['doc_id'])]) > 0
        )
        p10_scores.append(relevant_in_top10 / 10.0)

        queries_ranked += 1

# Final metric summary
mrr_at_10 = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
p_at_10   = sum(p10_scores) / len(p10_scores) if p10_scores else 0.0

print(f"\nReranking complete! Results saved to {output_file}")
print("\n" + "#" * 40)
print("Results:")
print("#" * 40)
print(f"MRR@10:         {mrr_at_10:.16f}")
print(f"P@10:           {p_at_10:.16f}")
print(f"QueriesRanked:  {queries_ranked}")
print("#" * 40)
