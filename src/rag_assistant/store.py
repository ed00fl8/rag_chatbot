"""Persistent vector and document store with FAISS and NumPy fallback."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .documents import DocumentChunk


class VectorStore:
    def __init__(self, directory: str | Path, embedding_model: str | None = None):
        self.directory = Path(directory)
        self.embedding_model = embedding_model
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.directory / "chunks.json"
        self.embeddings_path = self.directory / "embeddings.npy"
        self.faiss_path = self.directory / "vectors.faiss"
        self.versions_path = self.directory / "versions.json"
        self.metadata_path = self.directory / "index_metadata.json"
        self._chunks: list[DocumentChunk] = []
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._index = None
        self._loaded = False

    @property
    def count(self) -> int:
        self.load()
        return len(self._chunks)

    @property
    def chunks(self) -> list[DocumentChunk]:
        self.load()
        return list(self._chunks)

    def load(self) -> None:
        if self._loaded:
            return
        if self.chunks_path.exists() and self.embeddings_path.exists():
            try:
                raw = json.loads(self.chunks_path.read_text(encoding="utf-8"))
                self._chunks = [DocumentChunk.from_dict(item) for item in raw]
                self._embeddings = np.load(
                    self.embeddings_path, allow_pickle=False
                ).astype(np.float32, copy=False)
                if self.metadata_path.exists():
                    metadata = json.loads(
                        self.metadata_path.read_text(encoding="utf-8")
                    )
                    stored_model = metadata.get("embedding_model")
                    if (
                        self.embedding_model is not None
                        and stored_model is not None
                        and stored_model != self.embedding_model
                    ):
                        raise ValueError(
                            "The stored index was built with a different embedding model. "
                            "Remove data/index and re-ingest documents after changing "
                            "EMBEDDING_MODEL."
                        )
                if (
                    self._embeddings.ndim != 2
                    or len(self._chunks) != self._embeddings.shape[0]
                ):
                    raise ValueError("Stored chunks and embeddings do not match.")
                self._build_faiss_index()
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self._chunks = []
                self._embeddings = np.empty((0, 0), dtype=np.float32)
                self._index = None
                raise
        else:
            self._chunks = []
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            self._index = None
        self._loaded = True

    def _build_faiss_index(self) -> None:
        self._index = None
        if self._embeddings.size == 0:
            self.faiss_path.unlink(missing_ok=True)
            return
        try:
            import faiss  # type: ignore[import-not-found]

            dimension = int(self._embeddings.shape[1])
            if self.faiss_path.is_file():
                try:
                    cached = faiss.read_index(str(self.faiss_path))
                    if (
                        cached.d == dimension
                        and cached.ntotal == len(self._chunks)
                    ):
                        self._index = cached
                        return
                except Exception:
                    self.faiss_path.unlink(missing_ok=True)

            index = faiss.IndexFlatIP(dimension)
            index.add(np.ascontiguousarray(self._embeddings, dtype=np.float32))
            self._index = index
            faiss.write_index(index, str(self.faiss_path))
        except ImportError:
            # NumPy exact search keeps unit tests and minimal environments usable.
            self.faiss_path.unlink(missing_ok=True)

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> dict:
        self.load()
        if not chunks:
            return {"indexed_chunks": 0, "documents": 0, "versions": []}

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embeddings must have one row per chunk.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain non-finite values.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embeddings contain zero-length vectors.")
        vectors = vectors / norms

        if self._embeddings.size and self._embeddings.shape[1] != vectors.shape[1]:
            raise ValueError(
                "Embedding dimension changed. Clear the index before switching models."
            )

        source_keys = {chunk.source.casefold() for chunk in chunks}
        keep_indices = [
            i for i, old in enumerate(self._chunks)
            if old.source.casefold() not in source_keys
        ]
        kept_chunks = [self._chunks[i] for i in keep_indices]
        kept_vectors = (
            self._embeddings[keep_indices]
            if keep_indices
            else np.empty((0, vectors.shape[1]), dtype=np.float32)
        )

        self._chunks = kept_chunks + list(chunks)
        self._embeddings = np.vstack([kept_vectors, vectors]).astype(
            np.float32, copy=False
        )
        self._persist()
        self._loaded = True

        versions = self._update_version_history(chunks)
        return {
            "indexed_chunks": len(chunks),
            "documents": len(source_keys),
            "versions": versions,
        }

    def _persist(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.chunks_path,
            json.dumps(
                [chunk.to_dict() for chunk in self._chunks],
                ensure_ascii=False,
                indent=2,
            ),
        )
        _atomic_write_numpy(self.embeddings_path, self._embeddings)
        # Force a rebuild after an upsert; a same-sized old FAISS index may be stale.
        self.faiss_path.unlink(missing_ok=True)
        self._build_faiss_index()
        _atomic_write_text(
            self.metadata_path,
            json.dumps(
                {
                    "embedding_model": self.embedding_model,
                    "dimension": int(self._embeddings.shape[1]),
                    "chunks": len(self._chunks),
                },
                indent=2,
            ),
        )

    def _update_version_history(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[dict]:
        history: dict[str, list[str]] = {}
        if self.versions_path.exists():
            try:
                history = json.loads(
                    self.versions_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                history = {}

        documents: dict[str, DocumentChunk] = {}
        for chunk in chunks:
            documents[chunk.source] = chunk

        records = []
        for source, chunk in documents.items():
            versions = history.setdefault(source, [])
            if chunk.version not in versions:
                versions.append(chunk.version)
            records.append(
                {
                    "source": source,
                    "version": chunk.version,
                    "version_count": len(versions),
                }
            )

        _atomic_write_text(
            self.versions_path,
            json.dumps(history, indent=2, ensure_ascii=False),
        )
        return records

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[int, float]]:
        self.load()
        if not self._chunks or top_k < 1:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._embeddings.shape[1]:
            raise ValueError("Query and index embedding dimensions do not match.")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("Query embedding has zero length.")
        query /= norm
        count = min(top_k, len(self._chunks))

        if self._index is not None:
            scores, indices = self._index.search(query, count)
            return [
                (int(idx), float(score))
                for idx, score in zip(indices[0], scores[0], strict=True)
                if idx >= 0
            ]

        scores = (self._embeddings @ query[0]).astype(float)
        ordered = np.argsort(-scores, kind="stable")[:count]
        return [(int(i), float(scores[i])) for i in ordered]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _atomic_write_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)
