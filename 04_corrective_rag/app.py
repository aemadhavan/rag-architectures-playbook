import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient, models

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.ingest import load_documents
from shared.llm import embed_texts, generate_text

load_dotenv()

COLLECTION_NAME = "corrective_rag"

qdrant_url = os.getenv("QDRANT_URL")
qdrant = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(":memory:")


class CRAGState(TypedDict):
    question: str
    query: str
    docs: list[str]
    is_relevant: bool
    retries: int
    answer: str


def build_index(docs: list[dict]) -> None:
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


def retrieve(state: CRAGState) -> dict:
    q_vec = embed_texts([state["query"]])[0]
    hits = qdrant.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=3).points
    return {"docs": [h.payload["text"] for h in hits]}


def grade_docs(state: CRAGState) -> dict:
    context = "\n\n".join(state["docs"])
    prompt = (
        "Does the following context contain enough information to answer the question? "
        "Reply with ONLY 'YES' or 'NO'.\n\n"
        f"Question: {state['question']}\n\nContext:\n{context}"
    )
    verdict = generate_text(prompt).strip().upper()
    return {"is_relevant": "YES" in verdict}


def rewrite_query(state: CRAGState) -> dict:
    prompt = f"Rewrite this search query to be clearer and more specific:\n{state['query']}"
    new_q = generate_text(prompt).strip()
    return {"query": new_q, "retries": state["retries"] + 1}


def generate(state: CRAGState) -> dict:
    if not state["is_relevant"]:
        return {"answer": "I do not have enough information in the provided context to answer this question."}
    context = "\n\n---\n\n".join(state["docs"])
    prompt = (
        "Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"answer": generate_text(prompt)}


def route_after_grade(state: CRAGState) -> str:
    if state["is_relevant"] or state["retries"] >= 1:
        return "generate"
    return "rewrite_query"


def build_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_docs", grade_docs)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", route_after_grade)
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


if __name__ == "__main__":
    documents = load_documents(Path(__file__).resolve().parent.parent / "data")
    build_index(documents)
    app = build_graph()

    for q in [
        "How long do AcmeCloud session tokens last by default?",
        "What is the pricing for the AcmeCloud Enterprise plan?",
    ]:
        out = app.invoke({"question": q, "query": q, "docs": [], "is_relevant": False, "retries": 0, "answer": ""})
        print(f"\nQ: {q}\nA: {out['answer']}")
