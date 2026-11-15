import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from rich.console import Console
from rich.table import Table

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from shared.ingest import load_documents
from shared.llm import generate_text

console = Console()


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def judge_answer(question: str, expected: str, actual: str) -> float:
    prompt = (
        "Score the actual answer against the expected answer from 0.0 to 1.0 "
        "(1.0 = completely accurate and grounded, 0.0 = wrong or hallucinated). "
        "Reply with ONLY a single float number.\n\n"
        f"Question: {question}\nExpected: {expected}\nActual: {actual}"
    )
    try:
        score_text = generate_text(prompt).strip()
        return max(0.0, min(1.0, float(score_text)))
    except Exception:
        return 0.0


def run_benchmarks():
    docs = load_documents(ROOT_DIR / "data")
    goldset = json.loads((ROOT_DIR / "goldset" / "questions.json").read_text(encoding="utf-8"))

    # Load RAG modules
    naive_mod = load_module("naive", "01_naive_rag/app.py")
    hybrid_mod = load_module("hybrid", "02_hybrid_rerank_rag/app.py")
    hyde_mod = load_module("hyde", "03_hyde_rag/app.py")
    crag_mod = load_module("crag", "04_corrective_rag/app.py")
    agentic_mod = load_module("agentic", "05_agentic_rag/app.py")
    graph_mod = load_module("graph", "06_graph_rag/app.py")

    # Initialize indexes
    q1 = naive_mod.get_client()
    naive_mod.build_index(q1, docs)

    q2, bm25, ranker = hybrid_mod.get_clients()
    hybrid_mod.build_hybrid_index(q2, bm25, docs)

    q3 = hyde_mod.get_client()
    hyde_mod.build_index(q3, docs)

    crag_mod.build_index(docs)
    crag_app = crag_mod.build_graph()
    agentic_app = agentic_mod.build_agent()

    runners = {
        "01_naive_rag": lambda q: naive_mod.query_naive_rag(q1, q),
        "02_hybrid_rerank": lambda q: hybrid_mod.query_hybrid_rerank_rag(q2, bm25, ranker, q),
        "03_hyde_rag": lambda q: hyde_mod.query_hyde_rag(q3, q),
        "04_corrective_rag": lambda q: crag_app.invoke(
            {"question": q, "query": q, "docs": [], "is_relevant": False, "retries": 0, "answer": ""}
        )["answer"],
        "05_agentic_rag": lambda q: agentic_app.invoke({"messages": [("user", q)]})["messages"][-1].content,
        "06_graph_rag": lambda q: graph_mod.query_graph_rag(q, "auth-service"),
    }

    results = []
    for name, fn in runners.items():
        scores, latencies = [], []
        for item in goldset:
            t0 = time.perf_counter()
            ans = fn(item["question"])
            latencies.append((time.perf_counter() - t0) * 1000)
            scores.append(judge_answer(item["question"], item["expected_answer"], ans))

        avg_score = sum(scores) / len(scores)
        avg_lat = sum(latencies) / len(latencies)
        results.append((name, avg_score, avg_lat))

    # Print Rich table
    table = Table(title="RAG Architectures Benchmark (Goldset)")
    table.add_column("Architecture", style="cyan")
    table.add_column("Avg Accuracy (0-1)", justify="right", style="green")
    table.add_column("Avg Latency (ms)", justify="right", style="yellow")

    md_lines = [
        "# RAG Architectures Benchmark Results\n",
        "| Architecture | Avg Accuracy (0-1) | Avg Latency (ms) |",
        "| :--- | :---: | :---: |",
    ]
    for name, score, lat in results:
        table.add_row(name, f"{score:.2f}", f"{lat:.0f}")
        md_lines.append(f"| `{name}` | {score:.2f} | {lat:.0f} |")

    console.print(table)
    out_dir = ROOT_DIR / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    console.print(f"\nSaved markdown report to [bold]{out_dir / 'results.md'}[/bold]")


if __name__ == "__main__":
    run_benchmarks()
