"""In-memory FAISS index over a committed passages file, embedded with
all-MiniLM-L6-v2. No server process, no network calls at query time — matches
CLAUDE.md's "in-memory vector store" decision for the RAG comparison baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class PassageIndex:
    def __init__(self, passages: list[dict], embeddings: np.ndarray, model: SentenceTransformer):
        self.passages = passages
        self._model = model
        self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self._faiss_index.add(embeddings)

    @classmethod
    def from_passages(cls, passages: list[dict]) -> "PassageIndex":
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        texts = [p["text"] for p in passages]
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return cls(passages, embeddings.astype(np.float32), model)

    @classmethod
    def from_file(cls, path: Path) -> "PassageIndex":
        passages = json.loads(Path(path).read_text())
        return cls.from_passages(passages)

    def search(self, query: str, k: int) -> list[str]:
        query_embedding = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        _, indices = self._faiss_index.search(query_embedding.astype(np.float32), k)
        return [self.passages[i]["text"] for i in indices[0] if i != -1]
