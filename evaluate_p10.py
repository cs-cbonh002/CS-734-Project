"""
CS 734 - RAG System Evaluation
P@10 Evaluation Script for 10 Queries
  - 5 randomly selected MS MARCO dev queries
  - 5 student-authored queries

Outputs results to terminal and saves to evaluation_results.txt and evaluation_results.csv
"""

import json
import csv
import random
from datetime import datetime
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from haystack import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import ComponentDevice

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INDEX_PATH       = "indexes/msmarco-passage-index"
TOPICS_KEY       = "msmarco-passage-dev-subset"
QRELS_KEY        = "msmarco-passage-dev-subset"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_BM25       = 100   # BM25 candidate pool
TOP_K_RERANK     = 10    # Final passages to return per query
RANDOM_SEED      = 42
OUTPUT_TXT       = "evaluation_results.txt"
OUTPUT_CSV       = "evaluation_results.csv"

# ─────────────────────────────────────────────
# Student-authored queries
# ─────────────────────────────────────────────
STUDENT_QUERIES = [
    "What causes inflation in an economy?",
    "How does the immune system fight viruses?",
    "What are the main causes of the American Civil War?",
    "How do black holes form?",
    "What is the difference between machine learning and deep learning?",
]


def load_components():
    print("Loading Pyserini index...")
    searcher = LuceneSearcher(INDEX_PATH)

    print("Loading MS MARCO dev topics and qrels...")
    topics = get_topics(TOPICS_KEY)
    qrels  = get_qrels(QRELS_KEY)

    print("Initializing Cross-Encoder reranker on MPS...")
    ranker = SentenceTransformersSimilarityRanker(
        model=RERANKER_MODEL,
        top_k=TOP_K_RERANK,
        device=ComponentDevice.from_str("mps")
    )
    ranker.warm_up()

    return searcher, topics, qrels, ranker


def select_marco_queries(topics, qrels, n=5, seed=RANDOM_SEED):
    """Pick n MS MARCO dev queries that have at least one qrel judgment."""
    random.seed(seed)
    judged = [qid for qid in topics if qid in qrels and len(qrels[qid]) > 0]
    selected_ids = random.sample(judged, n)
    return {qid: topics[qid]["title"] for qid in selected_ids}


def retrieve_and_rerank(query, searcher, ranker):
    """BM25 retrieval → Cross-Encoder reranking. Returns top-10 docs."""
    hits = searcher.search(query, k=TOP_K_BM25)

    documents = []
    for hit in hits:
        raw = json.loads(searcher.doc(hit.docid).raw())
        content = raw.get("contents", "")
        documents.append(Document(content=content, meta={"doc_id": hit.docid}))

    result = ranker.run(query=query, documents=documents)
    return result["documents"]  # already top-k=10


def calculate_p_at_10(reranked_docs, qrels_for_query):
    """
    P@10 = (number of relevant docs in top 10) / 10
    For student queries with no qrels, returns None.
    """
    if not qrels_for_query:
        return None
    relevant = sum(
        1 for doc in reranked_docs[:10]
        if doc.meta["doc_id"] in qrels_for_query and qrels_for_query[doc.meta["doc_id"]] > 0
    )
    return relevant / 10.0


def format_passages(docs):
    lines = []
    for i, doc in enumerate(docs[:10], 1):
        snippet = doc.content[:200].replace("\n", " ").strip()
        lines.append(f"  [{i:2d}] doc_id={doc.meta['doc_id']}  \"{snippet}...\"")
    return "\n".join(lines)


def run_evaluation():
    searcher, topics, qrels, ranker = load_components()

    # Build query list
    marco_queries = select_marco_queries(topics, qrels)
    all_queries = []

    print("\nSelected MS MARCO queries:")
    for qid, text in marco_queries.items():
        print(f"  [{qid}] {text}")
        all_queries.append({"source": "MS MARCO", "qid": str(qid), "text": text})

    print("\nStudent-authored queries:")
    for i, text in enumerate(STUDENT_QUERIES, start=1):
        qid = f"student_{i}"
        print(f"  [{qid}] {text}")
        all_queries.append({"source": "Student", "qid": qid, "text": text})

    # Run evaluation
    results = []
    output_lines = []
    header = f"\n{'='*70}\nCS 734 P@10 Evaluation  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}"
    print(header)
    output_lines.append(header)

    p10_scores = []

    for q in all_queries:
        print(f"\nQuery [{q['qid']}] ({q['source']}): {q['text']}")
        docs = retrieve_and_rerank(q["text"], searcher, ranker)

        # MS MARCO qrels use integer keys; student queries have no qrels
        if q["source"] == "MS MARCO":
            qrels_for_query = qrels.get(int(q["qid"]), qrels.get(q["qid"], {}))
        else:
            qrels_for_query = {}
        p10 = calculate_p_at_10(docs, qrels_for_query)

        p10_str = f"{p10:.4f}" if p10 is not None else "N/A (no qrels)"
        if p10 is not None:
            p10_scores.append(p10)

        block = (
            f"\n{'─'*70}\n"
            f"Query  : {q['text']}\n"
            f"Source : {q['source']}  |  QID: {q['qid']}\n"
            f"P@10   : {p10_str}\n"
            f"Top 10 Passages:\n{format_passages(docs)}"
        )
        print(block)
        output_lines.append(block)

        results.append({
            "source":    q["source"],
            "qid":       q["qid"],
            "query":     q["text"],
            "p_at_10":   p10_str,
            **{f"rank_{i+1}_doc_id": docs[i].meta["doc_id"] if i < len(docs) else "" for i in range(10)},
            **{f"rank_{i+1}_snippet": (docs[i].content[:150].replace("\n", " ") if i < len(docs) else "") for i in range(10)},
        })

    # Summary
    mean_p10 = sum(p10_scores) / len(p10_scores) if p10_scores else 0
    summary = (
        f"\n{'='*70}\n"
        f"SUMMARY\n"
        f"  Queries evaluated : {len(all_queries)}\n"
        f"  MS MARCO queries  : 5  (with qrel judgments)\n"
        f"  Student queries   : 5  (no qrels — passages shown for manual review)\n"
        f"  Mean P@10 (MS MARCO queries): {mean_p10:.4f}\n"
        f"{'='*70}\n"
    )
    print(summary)
    output_lines.append(summary)

    # Save TXT
    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Results saved to {OUTPUT_TXT}")

    # Save CSV
    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_evaluation()
