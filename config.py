"""Shared configuration helpers for Day 22 lab."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_env() -> None:
    """Load environment variables and normalize LangSmith settings."""
    load_dotenv(override=False)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if not os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    if not os.getenv("LANGCHAIN_PROJECT") and os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


def get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance using env settings."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAIEmbeddings instance using env settings."""
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return the default text splitter for the lab."""
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def load_knowledge_base() -> str:
    """Load the knowledge base text used for RAG."""
    return Path("data/knowledge_base.txt").read_text(encoding="utf-8")
