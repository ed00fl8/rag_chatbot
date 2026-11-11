from __future__ import annotations

from rag_assistant.retrieval import HybridRetriever
from rag_assistant.store import VectorStore


def test_hybrid_retrieval_returns_relevant_document(service_factory, tmp_path):
    service = service_factory()
    path = tmp_path / "faiss-guide.txt"
    path.write_text(
        "FAISS is a library for efficient similarity search over dense vectors. "
        "It supports nearest-neighbor search in vector collections.",
        encoding="utf-8",
    )
    service.ingest([path])

    hits = service.retrieve("What does FAISS do?", top_k=3)

    assert hits
    assert hits[0].chunk.source == "faiss-guide.txt"
    assert hits[0].fusion_score > 0
    assert hits[0].rerank_score >= 0


def test_empty_index_returns_no_hits(service_factory):
    service = service_factory()
    assert service.retrieve("Any question") == []


def test_query_without_searchable_tokens_returns_no_hits(service_factory, tmp_path):
    service = service_factory()
    path = tmp_path / "guide.txt"
    path.write_text("FAISS supports vector search.", encoding="utf-8")
    service.ingest([path])

    assert service.retrieve("?! ...") == []
