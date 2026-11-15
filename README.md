# RAG Architectures Playbook

A hands-on, benchmarked repository demonstrating different **Retrieval-Augmented Generation (RAG)** architectures on a shared **TechOps & Product Support** dataset.

## Architectures Implemented

1. **`01_naive_rag`** – Baseline dense vector retrieval with chunking and direct context injection.
2. **`02_hybrid_rerank_rag`** – Hybrid search (dense embeddings + sparse BM25) combined with FlashRank reranking.
3. **`03_hyde_rag`** – Hypothetical Document Embeddings (HyDE) for query expansion before vector search.
4. **`04_corrective_rag`** – Corrective RAG (CRAG) evaluating retrieval relevance and falling back to query rewriting.
5. **`05_agentic_rag`** – Multi-turn tool-calling agent using LangGraph to query docs and error tickets iteratively.
6. **`06_graph_rag`** – Knowledge Graph RAG with NetworkX modeling service dependencies and incident paths.

## Benchmark Results

Evaluated on the curated goldset questions (`goldset/questions.json`):

| Architecture | Avg Accuracy (0-1) | Avg Latency (ms) |
| :--- | :---: | :---: |
| `01_naive_rag` | 0.70 | 1173 |
| `02_hybrid_rerank` | 0.71 | 1197 |
| `03_hyde_rag` | 0.70 | 3848 |
| `04_corrective_rag` | 0.74 | 1939 |
| `05_agentic_rag` | 0.95 | 2747 |
| `06_graph_rag` | 0.47 | 1128 |

## Tech Stack
- **LLM:** Anthropic Claude Haiku (`ANTHROPIC_API_KEY`)
- **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
- **Vector & Hybrid Search:** Qdrant (Dense + Sparse BM25)
- **Reranking:** FlashRank (local CPU)
- **Orchestration:** LangGraph
- **Graph RAG:** NetworkX

## Quick Start
```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 3. (Optional) Start Qdrant via Docker
docker compose up -d
```
