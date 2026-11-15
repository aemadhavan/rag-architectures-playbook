import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.ingest import load_documents
from shared.llm import embed_texts, generate_text

load_dotenv()

COLLECTION_NAME = "hyde_rag"


def get_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    return QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")


def build_index(qdrant: QdrantClient, docs: list[dict]) -> None:
    vectors = embed_texts([d["text"] for d in docs])
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(id=i, vector=vec, payload=doc)
        for i, (doc, vec) in enumerate(zip(docs, vectors))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def query_hyde_rag(qdrant: QdrantClient, question: str, top_k: int = 3) -> str:
    # Step 1: Generate a hypothetical document/answer
    hyde_prompt = (
        "Write a concise, technical documentation snippet that would directly answer "
        f"this question:\n{question}"
    )
    hypothetical_doc = generate_text(hyde_prompt)

    # Step 2: Embed the hypothetical document instead of the raw query
    q_vec = embed_texts([hypothetical_doc])[0]
    hits = qdrant.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=top_k).points
    context = "\n\n---\n\n".join(h.payload["text"] for h in hits)

    # Step 3: Generate the final grounded answer
    final_prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return generate_text(final_prompt)


if __name__ == "__main__":
    gemini_client, qdrant_client = get_client()
    documents = load_documents(Path(__file__).resolve().parent.parent / "data")
    build_index(gemini_client, qdrant_client, documents)

    q = "How long do AcmeCloud session tokens last by default?"
    print(f"Q: {q}\nA: {query_hyde_rag(gemini_client, qdrant_client, q)}")
