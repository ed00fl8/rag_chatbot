from __future__ import annotations

import numpy as np

from rag_assistant.documents import DocumentChunk
from rag_assistant.store import VectorStore


def chunk(source: str, version: str, text: str, index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{source}-{version}-{index}",
        document_id=source,
        source=source,
        version=version,
        chunk_index=index,
        text=text,
    )


def test_store_persists_vectors_chunks_and_versions(tmp_path):
    store = VectorStore(tmp_path / "index")
    chunks = [chunk("manual.txt", "v1", "alpha beta")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    result = store.upsert(chunks, vectors)

    assert result["indexed_chunks"] == 1
    assert store.count == 1
    assert (tmp_path / "index" / "chunks.json").is_file()
    assert (tmp_path / "index" / "embeddings.npy").is_file()
    assert (tmp_path / "index" / "versions.json").is_file()

    loaded = VectorStore(tmp_path / "index")
    assert loaded.chunks[0].text == "alpha beta"
    assert loaded.vector_search(np.array([1.0, 0.0]), 1)[0][0] == 0


def test_upsert_replaces_current_document_and_tracks_versions(tmp_path):
    store = VectorStore(tmp_path / "index")
    store.upsert(
        [chunk("manual.txt", "v1", "old content")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    store.upsert(
        [chunk("manual.txt", "v2", "new content")],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )

    assert store.count == 1
    assert store.chunks[0].text == "new content"
    history = (tmp_path / "index" / "versions.json").read_text()
    assert "v1" in history and "v2" in history


def test_rejects_mismatched_or_zero_embeddings(tmp_path):
    store = VectorStore(tmp_path / "index")
    chunks = [chunk("manual.txt", "v1", "content")]

    try:
        store.upsert(chunks, np.array([[0.0, 0.0]], dtype=np.float32))
    except ValueError as exc:
        assert "zero-length" in str(exc)
    else:
        raise AssertionError("Expected zero-vector validation to fail")


def test_store_rejects_index_created_with_another_embedding_model(tmp_path):
    directory = tmp_path / "index"
    original = VectorStore(directory, embedding_model="model-a")
    original.upsert(
        [chunk("manual.txt", "v1", "content")],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    changed = VectorStore(directory, embedding_model="model-b")
    try:
        changed.load()
    except ValueError as exc:
        assert "different embedding model" in str(exc)
    else:
        raise AssertionError("Expected model mismatch to be rejected")
