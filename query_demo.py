import json
import sys
import requests
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics
from haystack import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import ComponentDevice

# ==========================================
# Query Sets
# ==========================================

# 3 confirmed valid MS MARCO dev subset IDs — 2 more selected dynamically below
CONFIRMED_QUERY_IDS = [1102330, 160885, 1091115]

# 5 custom queries written by student
CUSTOM_QUERIES = [
    "what is agentic RAG",
    "how does machine learning work",
    "best programming languages for beginners",
    "symptoms of vitamin D deficiency",
    "how do black holes form",
]

# ==========================================
# Setup — tee output to screen AND file
# ==========================================
OUTPUT_FILE = "runs/query_demo_results.txt"

class Tee:
    """Writes to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def isatty(self):
        return False

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

tee = Tee(OUTPUT_FILE)
sys.stdout = tee

print("Loading BM25 Index...")
searcher = LuceneSearcher('indexes/msmarco-passage-index')

print("Loading MS MARCO Dev Topics...")
topics = get_topics('msmarco-passage-dev-subset')

# Dynamically pick 2 more valid IDs from the topics dict
extra_ids = [qid for qid in topics.keys() if qid not in CONFIRMED_QUERY_IDS][:2]
MS_MARCO_QUERY_IDS = CONFIRMED_QUERY_IDS + extra_ids
print(f"MS MARCO query IDs selected: {MS_MARCO_QUERY_IDS}")

print("Initializing Cross-Encoder Reranker on Apple Silicon (MPS)...")
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
    device=ComponentDevice.from_str("mps")
)
ranker.warm_up()


# ==========================================
# LLM answer synthesis via Ollama
# ==========================================
def generate_answer(query: str, top_passages: list) -> str:
    """Send the query + top 5 passages to Ollama (Llama3) and return a synthesized answer."""
    passages_text = "\n\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(top_passages))
    prompt = f"""Given the following retrieved passages, answer the query concisely and factually in 2-3 sentences. Base your answer only on the passages provided.

Passages:
{passages_text}

Query: {query}
Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[Answer generation unavailable — is Ollama running? Start it with: ollama serve]"
    except Exception as e:
        return f"[Answer generation error: {str(e)}]"


# ==========================================
# Search, rerank, and generate answer
# ==========================================
def search_and_rerank(query: str, label: str):
    """Run BM25 retrieval + neural reranking + LLM answer synthesis."""
    print("\n" + "=" * 70)
    print(f"  [{label}]")
    print(f"  Query: {query}")
    print("=" * 70)

    # BM25 retrieval — top 100 candidates
    hits = searcher.search(query, k=100)
    documents = []
    for hit in hits:
        raw_doc = json.loads(searcher.doc(hit.docid).raw())
        content = raw_doc.get("contents", "")
        documents.append(
            Document(content=content, meta={"doc_id": hit.docid, "bm25_score": hit.score})
        )

    # Neural reranking
    result = ranker.run(query=query, documents=documents)
    reranked_docs = result["documents"]

    # LLM answer synthesis from top 5 passages
    top_passages = [doc.content.strip() for doc in reranked_docs[:5]]
    answer = generate_answer(query, top_passages)

    print(f"\n  >> SYNTHESIZED ANSWER (Llama3):")
    print(f"  {answer}")
    print(f"  {'- ' * 33}")

    # Display top 10 ranked passages
    print(f"\n  TOP 10 RETRIEVED PASSAGES:")
    for rank, doc in enumerate(reranked_docs[:10], start=1):
        doc_id = doc.meta['doc_id']
        passage = doc.content.strip()
        if len(passage) > 300:
            passage = passage[:300] + "..."
        print(f"\n  Rank {rank:2d} | Doc ID: {doc_id}")
        print(f"  {passage}")
        print(f"  {'-' * 66}")


# ==========================================
# Run MS MARCO Queries
# ==========================================
print("\n\n" + "#" * 70)
print("  SECTION 1: MS MARCO Dev Set Queries")
print("#" * 70)

for qid in MS_MARCO_QUERY_IDS:
    query_text = topics[qid]['title']
    search_and_rerank(query_text, f"MS MARCO QID: {qid}")

# ==========================================
# Run Custom Queries
# ==========================================
print("\n\n" + "#" * 70)
print("  SECTION 2: Custom Student Queries")
print("#" * 70)

for i, query_text in enumerate(CUSTOM_QUERIES, start=1):
    search_and_rerank(query_text, f"Custom Query {i}")

print("\n\nDemo complete!")
print(f"Results saved to: {OUTPUT_FILE}")

tee.close()
sys.stdout = tee.terminal
