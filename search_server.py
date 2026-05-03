"""
RAG Search UI — Flask Backend
Run with: python search_server.py
Then open http://localhost:5000 in your browser.
"""

import json
import requests
from flask import Flask, request, jsonify, send_from_directory
from pyserini.search.lucene import LuceneSearcher
from haystack import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import ComponentDevice

app = Flask(__name__, static_folder="static")

# ==========================================
# Load models once at startup
# ==========================================
print("Loading BM25 Index...")
searcher = LuceneSearcher('indexes/msmarco-passage-index')

print("Initializing Cross-Encoder Reranker on Apple Silicon (MPS)...")
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
    device=ComponentDevice.from_str("mps")
)
ranker.warm_up()
print("Ready! Visit http://localhost:5000")


# ==========================================
# LLM answer synthesis via Ollama
# ==========================================
def generate_answer(query: str, top_passages: list[str]) -> str:
    """Send the query + top passages to Ollama (Llama3) and return a synthesized answer."""
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
# Search endpoint
# ==========================================
@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # BM25 retrieval
    hits = searcher.search(query, k=100)
    documents = []
    for hit in hits:
        raw_doc = json.loads(searcher.doc(hit.docid).raw())
        content = raw_doc.get("contents", "")
        documents.append(
            Document(content=content, meta={
                "doc_id": hit.docid,
                "bm25_score": round(hit.score, 4)
            })
        )

    # Neural reranking
    result = ranker.run(query=query, documents=documents)
    reranked_docs = result["documents"]

    # Generate answer from top 5 passages
    top_passages = [doc.content.strip() for doc in reranked_docs[:5]]
    answer = generate_answer(query, top_passages)

    # Format results
    results = []
    for rank, doc in enumerate(reranked_docs[:10], start=1):
        results.append({
            "rank": rank,
            "doc_id": doc.meta["doc_id"],
            "bm25_score": doc.meta["bm25_score"],
            "passage": doc.content.strip(),
        })

    return jsonify({"query": query, "answer": answer, "results": results})


# ==========================================
# Serve the frontend
# ==========================================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
