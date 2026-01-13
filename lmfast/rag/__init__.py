"""
LMFast RAG Module - Retrieval-Augmented Generation for Small Language Models.

Provides lightweight RAG capabilities optimized for Colab T4 and edge devices.

Example:
    >>> from lmfast.rag import LightweightRAG, create_rag
    >>>
    >>> # Quick setup
    >>> rag = create_rag("./my_model", documents=["Doc 1", "Doc 2"])
    >>> answer = rag.query("What is in Doc 1?")
"""

from lmfast.rag.lightweight import (
    LightweightRAG,
    SimpleEmbedder,
    SimpleIndex,
    create_rag,
)

__all__ = [
    "LightweightRAG",
    "SimpleEmbedder",
    "SimpleIndex",
    "create_rag",
]
