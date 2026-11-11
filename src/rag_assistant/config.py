"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_candidates: int = 30
    final_context_chunks: int = 5
    min_vector_similarity: float = 0.30
    max_upload_mb: int = 20
    ollama_timeout_seconds: int = 120

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "Settings":
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        settings = cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")).expanduser(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", cls.embedding_model
            ),
            reranker_model=os.getenv(
                "RERANKER_MODEL", cls.reranker_model
            ),
            ollama_url=os.getenv(
                "OLLAMA_URL", cls.ollama_url
            ).rstrip("/"),
            ollama_model=os.getenv(
                "OLLAMA_MODEL", cls.ollama_model
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(
                os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)
            ),
            retrieval_candidates=int(
                os.getenv(
                    "RETRIEVAL_CANDIDATES", cls.retrieval_candidates
                )
            ),
            final_context_chunks=int(
                os.getenv(
                    "FINAL_CONTEXT_CHUNKS", cls.final_context_chunks
                )
            ),
            min_vector_similarity=float(
                os.getenv(
                    "MIN_VECTOR_SIMILARITY", cls.min_vector_similarity
                )
            ),
            max_upload_mb=int(
                os.getenv("MAX_UPLOAD_MB", cls.max_upload_mb)
            ),
            ollama_timeout_seconds=int(
                os.getenv(
                    "OLLAMA_TIMEOUT_SECONDS",
                    cls.ollama_timeout_seconds,
                )
            ),
        )
        settings.validate()
        return settings

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE must be at least 100 characters.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
            )
        if self.retrieval_candidates < 1:
            raise ValueError("RETRIEVAL_CANDIDATES must be positive.")
        if self.final_context_chunks < 1:
            raise ValueError("FINAL_CONTEXT_CHUNKS must be positive.")
        if not -1 <= self.min_vector_similarity <= 1:
            raise ValueError(
                "MIN_VECTOR_SIMILARITY must be between -1 and 1."
            )
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be positive.")
