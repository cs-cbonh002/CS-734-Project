from pyserini.search.lucene import LuceneSearcher
from haystack import Document, Pipeline
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.utils import ComponentDevice
import json

# ==========================================
# 1. Pyserini Retrieval (Recall Step)
# ==========================================
print("Loading BM25 Index...")
searcher = LuceneSearcher('indexes/msmarco-passage-index')
query = "what is the capital of france"

# Retrieve top 100 candidates to give the Ranker plenty of options
hits = searcher.search(query, k=100)

# Convert Pyserini hits into Haystack Document objects
documents = []
for hit in hits:
    raw_doc = json.loads(searcher.doc(hit.docid).raw())
    content = raw_doc.get("contents", "")
    documents.append(Document(content=content, meta={"doc_id": hit.docid, "bm25_score": hit.score}))

print(f"Retrieved {len(documents)} candidate passages via BM25.")

# ==========================================
# 2. Haystack Pipeline Setup (Precision & Generation)
# ==========================================
# Ranker Node: MS MARCO specific Cross-Encoder, leveraging M1 GPU (MPS)
ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=5,
    device=ComponentDevice.from_str("mps")
)

# PromptBuilder Node: Jinja2 template to stitch passages and query
template = """
Given the following retrieved passages, answer the query concisely and factually.

Passages:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

Query: {{query}}
Answer:
"""
prompt_builder = PromptBuilder(template=template)

# Generate Node: Points to your local Ollama instance
generator = OllamaGenerator(model="llama3")

# Build and connect the modular pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("ranker", ranker)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", generator)

rag_pipeline.connect("ranker.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# ==========================================
# 3. Execute the Pipeline
# ==========================================
print(f"\nExecuting Neural Reranking and LLM Generation for: '{query}'...")

result = rag_pipeline.run(
    {
        "ranker": {"query": query, "documents": documents},
        "prompt_builder": {"query": query}
    }
)

print("\n" + "="*50)
print("FINAL SYNTHESIZED ANSWER:")
print("="*50)
print(result["llm"]["replies"][0])
