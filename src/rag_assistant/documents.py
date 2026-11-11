"""Document extraction, content versioning, and deterministic chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    source: str
    version: str
    chunk_index: int
    text: str
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "DocumentChunk":
        return cls(**value)


@dataclass(frozen=True)
class ExtractedDocument:
    source: str
    version: str
    pages: tuple[str, ...]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_name(path: Path) -> str:
    # Persist only a safe display name, never an absolute local path.
    return path.name.replace("\\", "_").replace("/", "_")


def extract_document(path: str | Path) -> ExtractedDocument:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    content = file_path.read_bytes()
    version = sha256_bytes(content)
    source = _source_name(file_path)

    if suffix == ".pdf":
        try:
            reader = PdfReader(file_path)
            pages = tuple(
                (page.extract_text() or "").strip()
                for page in reader.pages
            )
        except Exception as exc:
            raise ValueError(f"Could not extract PDF '{source}': {exc}") from exc
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(content, "html.parser")
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        pages = (text,)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Document '{source}' is not valid UTF-8 text."
            ) from exc
        pages = (text.strip(),)

    if not any(page.strip() for page in pages):
        raise ValueError(f"Document '{source}' contains no extractable text.")

    return ExtractedDocument(source=source, version=version, pages=pages)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows near word boundaries."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk_size/overlap configuration.")

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        next_start = max(start + 1, end - overlap)
        # Avoid a leading partial word in the next chunk.
        whitespace = normalized.find(" ", next_start)
        start = whitespace + 1 if whitespace != -1 and whitespace < end else next_start

    return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    doc_id = hashlib.sha256(
        document.source.casefold().encode("utf-8")
    ).hexdigest()[:16]
    result: list[DocumentChunk] = []
    global_index = 0

    for page_number, text in enumerate(document.pages, start=1):
        for piece in split_text(text, chunk_size, overlap):
            identity = (
                f"{document.source}\0{document.version}\0"
                f"{page_number}\0{global_index}\0{piece}"
            )
            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:20]
            result.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    source=document.source,
                    version=document.version,
                    chunk_index=global_index,
                    text=piece,
                    page=page_number if document.source.lower().endswith(".pdf") else None,
                )
            )
            global_index += 1

    return result


def load_and_chunk(
    paths: Iterable[str | Path],
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    seen_sources: set[str] = set()
    for path in paths:
        document = extract_document(path)
        key = document.source.casefold()
        if key in seen_sources:
            raise ValueError(
                f"Multiple input files have the same filename: {document.source}."
            )
        seen_sources.add(key)
        chunks.extend(chunk_document(document, chunk_size, overlap))
    return chunks
