import os
from anthropic import Anthropic
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

ANTHROPIC_MODEL = "claude-haiku-4-5"
_client: Anthropic | None = None
_embedder: TextEmbedding | None = None


def generate_text(prompt: str) -> str:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = _client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def embed_texts(texts: list[str]) -> list[list[float]]:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return [vec.tolist() for vec in _embedder.embed(texts)]
