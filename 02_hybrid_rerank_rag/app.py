import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient, models

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.ingest import load_documents
from shared.llm import embed_texts, generate_text

load_dotenv()

COLLECTION_NAME = "hybrid_rerank_rag"


def get_clients() -> tuple[QdrantClient, SparseTextEmbedding, Ranker]:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return qdrant, sparse_model, reranker


def build_hybrid_index(
    qdrant: QdrantClient,
    sparse_model: SparseTextEmbedding,
    docs: list[dict],
) -> None:
    texts = [d["text"] for d in docs]
    dense_vecs = embed_texts(texts)
    sparse_vecs = list(sparse_model.embed(texts))
    dim = len(dense_vecs[0])

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams()},
    )

    points = [
        models.PointStruct(
            id=i,
            vector={
                "dense": dense_vecs[i],
                "bm25": models.SparseVector(
                    indices=sparse_vecs[i].indices.tolist(),
                    values=sparse_vecs[i].values.tolist(),
                ),
            },
            payload=docs[i],
        )
        for i in range(len(docs))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def query_hybrid_rerank_rag(
    qdrant: QdrantClient,
    sparse_model: SparseTextEmbedding,
    reranker: Ranker,
    question: str,
    prefetch_k: int = 6,
    top_k: int = 3,
) -> str:
    q_dense = embed_texts([question])[0]
    q_sparse = next(sparse_model.embed([question]))

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=q_dense, using="dense", limit=prefetch_k),
            models.Prefetch(
                query=models.SparseVector(
                    indices=q_sparse.indices.tolist(),
                    values=q_sparse.values.tolist(),
                ),
                using="bm25",
                limit=prefetch_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_k,
    ).points

    passages = [{"id": h.id, "text": h.payload["text"]} for h in hits]
    reranked = reranker.rerank(RerankRequest(query=question, passages=passages))[:top_k]
    context = "\n\n---\n\n".join(r["text"] for r in reranked)

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return generate_text(prompt)


if __name__ == "__main__":
    gemini_client, qdrant_client, bm25, ranker = get_clients()
    documents = load_documents(Path(__file__).resolve().parent.parent / "data")
    build_hybrid_index(gemini_client, qdrant_client, bm25, documents)

    for q in [
        "How long do AcmeCloud session tokens last by default?",
        "How do I resolve ERR-504B?",
    ]:
        print(f"\nQ: {q}\nA: {query_hybrid_rerank_rag(gemini_client, qdrant_client, bm25, ranker, q)}")
