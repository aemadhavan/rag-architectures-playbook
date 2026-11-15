import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.ingest import load_documents

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS = load_documents(DATA_DIR)


@tool
def search_docs(query: str) -> str:
    """Search product documentation and FAQs for general platform questions."""
    matches = [d["text"] for d in DOCS if d["category"] == "documentation" and any(w.lower() in d["text"].lower() for w in query.split())]
    return "\n\n".join(matches) if matches else "No matching documentation found."


@tool
def lookup_error_ticket(error_code: str) -> str:
    """Look up an exact error code (e.g., ERR-504B) in incident tickets."""
    tickets = json.loads((DATA_DIR / "error_tickets.json").read_text(encoding="utf-8"))
    for t in tickets:
        if error_code.upper() in t["error_code"].upper():
            return json.dumps(t, indent=2)
    return f"No ticket found for {error_code}."


@tool
def check_service_dependencies(service_name: str) -> str:
    """Check service ownership and upstream/downstream dependencies."""
    deps = json.loads((DATA_DIR / "service_dependencies.json").read_text(encoding="utf-8"))
    return json.dumps(deps, indent=2)


def build_agent():
    llm = ChatAnthropic(model="claude-haiku-4-5", api_key=os.environ["ANTHROPIC_API_KEY"])
    tools = [search_docs, lookup_error_ticket, check_service_dependencies]
    return create_react_agent(llm, tools)


if __name__ == "__main__":
    agent = build_agent()
    q = "If ERR-504B happens on payment-gateway, how do we fix it and which team owns it?"
    result = agent.invoke({"messages": [("user", q)]})
    print(f"Q: {q}\nA: {result['messages'][-1].content}")
