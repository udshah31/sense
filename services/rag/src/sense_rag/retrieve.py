"""Public retrieval entry point used by the RAG baseline experiment script."""

from sense_rag.index import PassageIndex


def retrieve(index: PassageIndex, query: str, k: int = 3) -> list[str]:
    return index.search(query, k)
