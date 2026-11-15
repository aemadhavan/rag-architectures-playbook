import json
from pathlib import Path
from typing import Any


def load_documents(data_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Load and chunk all TechOps dataset files into a unified list of documents."""
    data_path = Path(data_dir)
    docs: list[dict[str, Any]] = []

    # 1. Markdown product docs (split by H2 sections)
    md_file = data_path / "product_docs.md"
    if md_file.exists():
        sections = md_file.read_text(encoding="utf-8").split("\n## ")
        for idx, section in enumerate(sections):
            text = section.strip()
            if not text:
                continue
            if idx > 0:
                text = "## " + text
            docs.append({
                "id": f"doc_{idx}",
                "text": text,
                "source": "product_docs.md",
                "category": "documentation",
            })

    # 2. Error tickets JSON
    tickets_file = data_path / "error_tickets.json"
    if tickets_file.exists():
        tickets = json.loads(tickets_file.read_text(encoding="utf-8"))
        for idx, t in enumerate(tickets):
            text = (
                f"Ticket {t['ticket_id']} | Error Code: {t['error_code']} | "
                f"Service: {t['service']}\nSummary: {t['summary']}\n"
                f"Resolution: {t['resolution']}"
            )
            docs.append({
                "id": f"ticket_{idx}",
                "text": text,
                "source": "error_tickets.json",
                "category": "incident_ticket",
            })

    # 3. Service dependencies JSON
    deps_file = data_path / "service_dependencies.json"
    if deps_file.exists():
        deps = json.loads(deps_file.read_text(encoding="utf-8"))
        for idx, s in enumerate(deps.get("services", [])):
            text = (
                f"Service: {s['name']} | Owner Team: {s['owner_team']} | "
                f"Depends On: {', '.join(s['depends_on'])}"
            )
            docs.append({
                "id": f"service_{idx}",
                "text": text,
                "source": "service_dependencies.json",
                "category": "architecture",
            })

    return docs
