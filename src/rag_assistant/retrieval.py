"""Hybrid lexical/vector retrieval with reciprocal-rank fusion and reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .documents import DocumentChunk
from .store import VectorStore


TOKEN_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


class BM25Index:
    """Small BM25 Okapi implementation to keep lexical retrieval self-contained."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.lengths = [len(document) for document in documents]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        frequencies = Counter(
            token for document in documents for token in set(document)
        )
        count = len(documents)
        self.idf = {
            token: math.log(
                1.0 + (count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in frequencies.items()
        }

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float64)
        for index, document in enumerate(self.documents):
            frequencies = Counter(document)
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.lengths[index] / max(1.0, self.avg_length)
            )
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if frequency:
                    token_idf = self.idf.get(token, 0.0)
                    scores[index] += token_idf * (
                        frequency * (self.k1 + 1.0)
                    ) / (frequency + length_norm)
        return scores


@dataclass(frozen=True)
class RetrievalHit:
    chunk: DocumentChunk
    vector_score: float
    bm25_score: float
    fusion_score: float
    rerank_score: float


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder, reranker):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.chunks = store.chunks
        self.bm25 = (
            BM25Index([tokenize(chunk.text) or ["_"] for chunk in self.chunks])
            if self.chunks
            else None
        )

    def retrieve(self, query: str, top_k: int = 5, candidates: int = 30) -> list[RetrievalHit]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k < 1 or candidates < 1:
            raise ValueError("top_k and candidates must be positive.")

        chunks = self.chunks
        if not chunks or self.bm25 is None or not tokenize(query):
            return []

        candidate_count = min(candidates, len(chunks))
        query_vector = self.embedder.encode([query])[0]
        vector_results = self.store.vector_search(query_vector, candidate_count)
        vector_rank = {idx: rank for rank, (idx, _) in enumerate(vector_results, 1)}
        vector_scores = {idx: score for idx, score in vector_results}

        lexical_scores = np.asarray(
            self.bm25.get_scores(tokenize(query)), dtype=float
        )
        lexical_order = np.argsort(-lexical_scores, kind="stable")[:candidate_count]
        lexical_rank = {
            int(idx): rank
            for rank, idx in enumerate(lexical_order, 1)
            if lexical_scores[int(idx)] > 0
        }
        lexical_values = {
            int(idx): float(lexical_scores[int(idx)])
            for idx in lexical_order
        }

        candidate_indices = set(vector_rank) | set(lexical_rank)
        if not candidate_indices:
            return []

        rrf_constant = 60
        fusion: dict[int, float] = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in vector_rank:
                score += 1.0 / (rrf_constant + vector_rank[idx])
            if idx in lexical_rank:
                score += 1.0 / (rrf_constant + lexical_rank[idx])
            fusion[idx] = score

        ordered = sorted(candidate_indices, key=lambda idx: (-fusion[idx], idx))
        ordered = ordered[:candidate_count]
        texts = [chunks[idx].text for idx in ordered]
        rerank_scores = self.reranker.score(query, texts)
        if len(rerank_scores) != len(ordered):
            raise ValueError("Reranker returned an unexpected number of scores.")

        hits = [
            RetrievalHit(
                chunk=chunks[idx],
                vector_score=float(vector_scores.get(idx, -1.0)),
                bm25_score=float(lexical_values.get(idx, 0.0)),
                fusion_score=float(fusion[idx]),
                rerank_score=float(rerank_scores[position]),
            )
            for position, idx in enumerate(ordered)
        ]
        hits.sort(key=lambda hit: hit.rerank_score, reverse=True)
        return hits[:top_k]
