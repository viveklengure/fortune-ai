"""
Fortune AI - RAG Pipeline
LangChain + ChromaDB retriever + Claude for Q&A over financial data.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = BASE_DIR / "db" / "chroma"
COLLECTION_NAME = "fortune500_financials"

SYSTEM_PROMPT = (
    "You are Fortune AI, a financial intelligence assistant. "
    "You have access to detailed financial data for Fortune 500 technology companies. "
    "Answer questions accurately using the provided context. "
    "Always cite which companies you are referencing. "
    "Format numbers clearly: use $B for billions, $M for millions, % for percentages. "
    "If you don't have data for something, say so clearly."
)

_model: SentenceTransformer | None = None
_collection = None
_llm: ChatAnthropic | None = None


def _get_resources():
    global _model, _collection, _llm
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    if _llm is None:
        _llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=1024,
        )
    return _model, _collection, _llm


def ask(question: str) -> dict:
    """
    Query the RAG pipeline.
    Returns: {"answer": str, "sources": list[str]}
    """
    model, collection, llm = _get_resources()

    query_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"],
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = [m["ticker"] for m in metadatas]

    context = "\n\n".join(
        f"[{m['ticker']} - {m['company_name']}]\n{doc}"
        for doc, m in zip(docs, metadatas)
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content, "sources": sources}


if __name__ == "__main__":
    result = ask("Which companies have the highest revenue growth?")
    print(result["answer"])
    print("Sources:", result["sources"])
