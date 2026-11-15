import json
import sys
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.llm import generate_text

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_dependency_graph() -> nx.DiGraph:
    data = json.loads((DATA_DIR / "service_dependencies.json").read_text(encoding="utf-8"))
    G = nx.DiGraph()
    for svc in data["services"]:
        G.add_node(svc["name"], owner_team=svc["owner_team"])
        for dep in svc["depends_on"]:
            # Edge: dep -> svc (if dep goes down, svc is impacted)
            G.add_edge(dep, svc["name"])
    return G


def get_downstream_impact(G: nx.DiGraph, service_name: str) -> str:
    if service_name not in G:
        return f"Service '{service_name}' not found in dependency graph."
    impacted = nx.descendants(G, service_name)
    lines = [f"Downstream services impacted if {service_name} goes down:"]
    for node in sorted(impacted):
        owner = G.nodes[node].get("owner_team", "Unknown Team")
        lines.append(f"- {node} (Owner: {owner})")
    return "\n".join(lines)


def query_graph_rag(question: str, target_service: str = "auth-service") -> str:
    G = build_dependency_graph()
    graph_context = get_downstream_impact(G, target_service)

    prompt = (
        "Answer the question using ONLY the knowledge graph context below.\n\n"
        f"Graph Context:\n{graph_context}\n\nQuestion: {question}"
    )
    return generate_text(prompt)


if __name__ == "__main__":
    q = "If auth-service goes down, which teams' services are impacted downstream?"
    print(f"Q: {q}\nA: {query_graph_rag(q, 'auth-service')}")
