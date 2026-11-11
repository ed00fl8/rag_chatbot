"""Application service coordinating ingestion, retrieval, and generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Settings
from .documents import load_and_chunk
from .generation import OllamaClient, generate_grounded_answer
from .models import CrossEncoderReranker, SentenceTransformerEmbedder
from .retrieval import HybridRetriever
from .store import VectorStore


class RAGService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder=None,
        reranker=None,
        llm=None,
        store: VectorStore | None = None,
    ):
        self.settings = settings
        self.store = store or VectorStore(
            settings.index_dir, embedding_model=settings.embedding_model
        )
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._retriever: HybridRetriever | None = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                self.settings.embedding_model
            )
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self.settings.reranker_model
            )
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            self._llm = OllamaClient(
                self.settings.ollama_url,
                self.settings.ollama_model,
                self.settings.ollama_timeout_seconds,
            )
        return self._llm

    def ingest(self, paths: list[str | Path]) -> dict:
        chunks = load_and_chunk(
            paths,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No text chunks were extracted from the input documents.")
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedding model returned an invalid matrix.")
        summary = self.store.upsert(chunks, np.asarray(vectors, dtype=np.float32))
        self._retriever = None
        return {
            **summary,
            "sources": sorted({chunk.source for chunk in chunks}),
            "total_indexed_chunks": self.store.count,
        }

    def retrieve(self, question: str, top_k: int | None = None):
        if self.store.count == 0:
            return []
        if self._retriever is None:
            self._retriever = HybridRetriever(
                self.store,
                self.embedder,
                self.reranker,
            )
        return self._retriever.retrieve(
            question,
            top_k=top_k or self.settings.final_context_chunks,
            candidates=self.settings.retrieval_candidates,
        )

    def ask(self, question: str, top_k: int | None = None) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        result = generate_grounded_answer(
            question=question,
            hits=hits,
            llm=self.llm,
            min_vector_similarity=self.settings.min_vector_similarity,
        )
        return {
            "answer": result.answer,
            "abstained": result.abstained,
            "abstention_reason": result.reason,
            "sources": [
                {
                    "citation": source.citation_id,
                    "source": source.source,
                    "page": source.page,
                    "chunk_index": source.chunk_index,
                    "version": source.version,
                    "excerpt": source.excerpt,
                }
                for source in result.sources
            ],
            "retrieved_chunks": len(hits),
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "indexed_chunks": self.store.count,
            "index_ready": self.store.count > 0,
            "llm_model": self.settings.ollama_model,
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model,
        }
