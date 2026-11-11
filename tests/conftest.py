from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from rag_assistant.config import Settings
from rag_assistant.service import RAGService
from rag_assistant.store import VectorStore


class HashEmbedder:
    """Deterministic, dependency-free embedding double for unit tests."""

    dimension = 128

    def encode(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w-]+", text.casefold())
            for token in tokens:
                index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension
                result[row, index] += 1.0
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return result / norms


class OverlapReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(re.findall(r"[\w-]+", query.casefold()))
        return [
            float(len(terms.intersection(re.findall(r"[\w-]+", text.casefold()))))
            for text in texts
        ]


class FakeLLM:
    def __init__(self, response: str = "FAISS performs vector similarity search [S1]."):
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


@pytest.fixture
def fake_components():
    return HashEmbedder(), OverlapReranker(), FakeLLM()


@pytest.fixture
def service_factory(tmp_path, fake_components):
    embedder, reranker, llm = fake_components

    def factory(**overrides):
        settings = Settings(
            data_dir=tmp_path / "data",
            chunk_size=220,
            chunk_overlap=30,
            min_vector_similarity=0.0,
            retrieval_candidates=20,
            final_context_chunks=5,
        )
        for key, value in overrides.items():
            settings = Settings(**{**settings.__dict__, key: value})
        return RAGService(
            settings,
            embedder=embedder,
            reranker=reranker,
            llm=llm,
            store=VectorStore(settings.index_dir),
        )

    return factory
