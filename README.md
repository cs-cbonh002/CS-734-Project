# Scalable Retrieval-Augmented Generation over MS MARCO

A three-stage retrieve, rerank, and generate pipeline over the full MS MARCO passage collection — 8.8 million passages drawn from roughly 3.2 million web documents. Sparse retrieval supplies recall, a neural cross-encoder supplies precision, and a locally served language model synthesizes an answer grounded in the retrieved text.

The entire system runs on a single laptop. No cloud compute, no API keys, no hosted inference.

Built for CS 734, Introduction to Information Retrieval, Old Dominion University.

## Results

Evaluated on all 6,980 queries in the MS MARCO passage development subset against the official relevance judgments.

| System | MRR@10 | P@10 |
|---|---|---|
| BM25 baseline (Pyserini) | 0.1840 | 0.0394 |
| BM25 + cross-encoder reranker | **0.3637** | **0.0608** |
| Relative gain | **+97.7%** | **+54.3%** |

The BM25 baseline matches the published Pyserini benchmark for this configuration, which confirms the index and evaluation harness are correct before any claim is made about the reranker's contribution.

**Reading P@10 correctly.** MS MARCO uses sparse binary labels — most queries have exactly one judged relevant passage in the entire 8.8 million. The theoretical maximum P@10 is therefore 0.1, not 1.0. The reranked system reaches about 61% of that ceiling, against 39% for BM25 alone. MRR@10 is the benchmark's primary metric for this reason.

## Architecture

**Stage 1 — Sparse retrieval (recall).** Pyserini's BM25 implementation over a Lucene index of the full corpus returns the top 100 candidates per query, using the standard MS MARCO parameters `k1=0.9`, `b=0.4`.

**Stage 2 — Neural reranking (precision).** Those 100 candidates go to `cross-encoder/ms-marco-MiniLM-L-6-v2`, a six-layer distilled transformer fine-tuned on MS MARCO. Unlike a bi-encoder, a cross-encoder jointly encodes each query-passage pair, which costs more at inference time and scores relevance far more finely. The top 10 are returned.

**Stage 3 — Answer synthesis.** The top 5 reranked passages are injected into a prompt template and passed to Llama 3, served locally through Ollama under 4-bit quantization. The prompt constrains the model to a short answer grounded solely in the supplied passages, which is what keeps the output tied to retrieved evidence rather than to model priors.

Retrieval and reranking are wired together with Haystack's pipeline components. A Flask application serves a browser interface over the same pipeline.

## Repository contents

| File | Purpose |
|---|---|
| `rag_pipeline.py` | The full three-stage pipeline: Pyserini retrieval, cross-encoder reranking, Ollama generation. |
| `search_server.py` | Flask backend serving the search UI on `localhost:5000`. |
| `static/index.html` | Front end for the search interface. |
| `query_demo.py` | Runs the ten-query demonstration — five MS MARCO dev queries and five original queries. |
| `eval_bm25.py` | Evaluates the BM25 baseline across the full dev subset, producing MRR@10 and P@10. |
| `evaluate_reranker.py` | Same evaluation with the cross-encoder stage added. |
| `evaluate_p10.py` | Focused P@10 evaluation over the ten demonstration queries, writing results to `.txt` and `.csv`. |
| `evaluation_results.txt` / `.csv` | Recorded output from the demonstration run. |
| `runs/` | TREC-format run files for the BM25 and reranked systems. |
| `data/convert_to_jsonl.py` | Converts the raw collection TSV into the JSONL format Pyserini's indexer expects. |

## Running it

Requires Python 3, Java (for Pyserini's Lucene backend), and [Ollama](https://ollama.com) with a Llama 3 model pulled locally.

```bash
pip install pyserini haystack-ai ollama-haystack sentence-transformers flask tqdm
ollama pull llama3
```

The passage collection is not in this repository — see the note below. Download it from the [MS MARCO passage ranking task](https://microsoft.github.io/msmarco/Datasets) into `data/`, then build the index:

```bash
python data/convert_to_jsonl.py          # collection.tsv -> collection_jsonl/
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/collection_jsonl \
  --index indexes/msmarco-passage-index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 --storePositions --storeDocvectors --storeRaw
```

Then:

```bash
python eval_bm25.py           # baseline numbers
python evaluate_reranker.py   # reranked numbers
python query_demo.py          # ten-query demonstration
python search_server.py       # web UI at http://localhost:5000
```

Queries and relevance judgments are pulled automatically by Pyserini via `get_topics` and `get_qrels` on `msmarco-passage-dev-subset`, so those need no manual download.

**On the missing collection file.** `collection.tsv` is roughly 3 GB and exceeds GitHub's file size limits, so it is not committed here. Every script, the pipeline, and the evaluation harness are present; only the raw corpus has to be fetched separately using the link above.

## Environment

Developed and evaluated on an Apple M1 Max MacBook Pro with 64 GB unified memory. Cross-encoder and language model inference use Apple's Metal Performance Shaders backend for GPU acceleration. Nothing in the pipeline requires a cloud GPU.

## References

1. Nguyen et al. *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*, 2016.
2. Robertson et al. *Okapi at TREC-3*, 1994 — BM25.
3. Lin et al. *Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations*, 2021.
4. Nogueira & Cho. *Passage Re-ranking with BERT*, 2019.
5. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020.
