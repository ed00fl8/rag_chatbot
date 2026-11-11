"""FastAPI application exposing document ingestion and grounded Q&A."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .documents import SUPPORTED_EXTENSIONS
from .service import RAGService


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=10)


def create_app(service: RAGService | None = None) -> FastAPI:
    settings = service.settings if service else Settings.from_env()
    rag_service = service or RAGService(settings)
    app = FastAPI(
        title="RAG Knowledge Assistant",
        description=(
            "Local document assistant with hybrid retrieval, reranking, "
            "source citations, and evidence-based abstention."
        ),
        version="1.0.0",
    )
    app.state.rag_service = rag_service

    @app.get("/health")
    def health() -> dict:
        return rag_service.health()

    @app.post("/documents", status_code=201)
    def ingest_documents(files: list[UploadFile] = File(...)) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required.")
        saved_paths: list[Path] = []
        try:
            settings.documents_dir.mkdir(parents=True, exist_ok=True)
            for upload in files:
                original_name = Path(upload.filename or "").name
                suffix = Path(original_name).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            f"Unsupported file type '{suffix}'. Supported: "
                            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
                        ),
                    )
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_name)
                if not safe_name or safe_name.startswith("."):
                    raise HTTPException(status_code=400, detail="Invalid filename.")
                content = upload.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
                if len(content) > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"'{safe_name}' exceeds {settings.max_upload_mb} MB.",
                    )
                destination = settings.documents_dir / safe_name
                destination.write_bytes(content)
                saved_paths.append(destination)

            result = rag_service.ingest(saved_paths)
            return {"status": "indexed", **result}
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Document ingestion failed: {exc}",
            ) from exc

    @app.post("/query")
    def query(request: QueryRequest) -> dict:
        try:
            return rag_service.ask(request.question, request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
