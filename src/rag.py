"""
Fortune AI - RAG Pipeline
ConversationalRetrievalChain with memory: LangChain routes each question through
ChromaDB retrieval and maintains conversation history so follow-up questions work.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings

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
    "If you don't have data for something, say so clearly.\n\n"
    "Context:\n{context}"
)

_chain: ConversationalRetrievalChain | None = None
_memory: ConversationBufferMemory | None = None


def _get_chain() -> tuple[ConversationalRetrievalChain, ConversationBufferMemory]:
    global _chain, _memory
    if _chain is not None:
        return _chain, _memory

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=1024,
    )

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    _memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    combine_docs_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    _chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=_memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": combine_docs_prompt},
    )
    return _chain, _memory


def ask(question: str) -> dict:
    """
    Query the conversational RAG pipeline.
    Returns: {"answer": str, "sources": list[str]}
    """
    chain, _ = _get_chain()
    result = chain.invoke({"question": question})

    sources = list({
        doc.metadata.get("ticker", "")
        for doc in result.get("source_documents", [])
        if doc.metadata.get("ticker")
    })

    return {"answer": result["answer"], "sources": sources}


def clear_memory() -> None:
    """Reset conversation history."""
    global _memory
    if _memory is not None:
        _memory.clear()


if __name__ == "__main__":
    result = ask("Which companies have the highest revenue growth?")
    print(result["answer"])
    print("Sources:", result["sources"])
    result2 = ask("How do their margins compare?")
    print(result2["answer"])
    print("Sources:", result2["sources"])
