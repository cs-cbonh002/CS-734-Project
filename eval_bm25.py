import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

print("Loading Pyserini Index and MS MARCO Dev Queries...")
searcher = LuceneSearcher('indexes/msmarco-passage-index')
topics = get_topics('msmarco-passage-dev-subset')
qrels = get_qrels('msmarco-passage-dev-subset')

mrr_scores = []
p10_scores = []

for qid, topic in tqdm(topics.items(), total=len(topics)):
    query = topic['title']

    # BM25 only — top 10 directly, no reranking
    hits = searcher.search(query, k=10)

    relevant_docs = qrels.get(qid, {})

    # MRR@10
    rr = 0.0
    for rank, hit in enumerate(hits[:10]):
        if int(hit.docid) in relevant_docs and int(relevant_docs[int(hit.docid)]) > 0:
            rr = 1.0 / (rank + 1)
            break
    mrr_scores.append(rr)

    # P@10
    relevant_in_top10 = sum(
        1 for hit in hits[:10]
        if int(hit.docid) in relevant_docs and int(relevant_docs[int(hit.docid)]) > 0
    )
    p10_scores.append(relevant_in_top10 / 10.0)

mrr_at_10 = sum(mrr_scores) / len(mrr_scores)
p_at_10   = sum(p10_scores) / len(p10_scores)

print("\n" + "#" * 40)
print("BM25 Baseline Results:")
print("#" * 40)
print(f"MRR@10:        {mrr_at_10:.16f}")
print(f"P@10:          {p_at_10:.16f}")
print(f"QueriesRanked: {len(mrr_scores)}")
print("#" * 40)
