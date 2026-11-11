"""Download a small, fixed corpus from official technical documentation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class CorpusSource:
    filename: str
    url: str
    allowed_hosts: tuple[str, ...]
    topic: str


# Canonical public documentation pages. The downloaded bytes are content-hashed
# and recorded in manifest.json; upstream pages may change over time.
OFFICIAL_SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        filename="faiss-overview.md",
        url="https://raw.githubusercontent.com/facebookresearch/faiss/main/README.md",
        allowed_hosts=("raw.githubusercontent.com",),
        topic="FAISS overview and similarity search",
    ),
    CorpusSource(
        filename="sentence-transformers-semantic-similarity.html",
        url=(
            "https://www.sbert.net/docs/sentence_transformer/usage/"
            "semantic_textual_similarity.html"
        ),
        allowed_hosts=("www.sbert.net", "sbert.net"),
        topic="Sentence Transformers semantic textual similarity",
    ),
    CorpusSource(
        filename="fastapi-first-steps.html",
        url="https://fastapi.tiangolo.com/tutorial/first-steps/",
        allowed_hosts=("fastapi.tiangolo.com",),
        topic="FastAPI first steps and path operations",
    ),
    CorpusSource(
        filename="ollama-chat-api.html",
        url="https://docs.ollama.com/api/chat",
        allowed_hosts=("docs.ollama.com",),
        topic="Ollama chat API",
    ),
)

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 45.0
USER_AGENT = "rag-knowledge-assistant/1.0 (official-docs corpus downloader)"
MANIFEST_NAME = "manifest.json"


class CorpusDownloadError(RuntimeError):
    """Raised when an official corpus source cannot be downloaded safely."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_document(source: CorpusSource, content: bytes) -> None:
    if not content:
        raise CorpusDownloadError(f"{source.filename}: downloaded document is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise CorpusDownloadError(
            f"{source.filename}: document exceeds the "
            f"{MAX_DOCUMENT_BYTES // (1024 * 1024)} MB limit."
        )
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CorpusDownloadError(
            f"{source.filename}: response is not valid UTF-8 text."
        ) from exc

    if source.filename.endswith(".html"):
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        extracted = soup.get_text(separator=" ", strip=True)
    else:
        extracted = text.strip()

    if len(extracted) < 80:
        raise CorpusDownloadError(
            f"{source.filename}: response contains too little readable text; "
            "the upstream URL may have changed or returned an error page."
        )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "documents": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusDownloadError(f"Could not read corpus manifest: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("documents"), dict):
        raise CorpusDownloadError("Corpus manifest has an invalid structure.")
    return value


def download_official_corpus(
    destination: str | Path,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    fetcher: Callable[..., httpx.Response] | None = None,
) -> dict:
    """Download all configured sources and record URL, timestamp and SHA-256.

    Existing files are replaced only after every source has downloaded and
    passed basic format/content checks. The manifest retains hash history,
    but historical document bytes are not archived.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive.")

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    old_documents = manifest.get("documents", {})

    get = fetcher or httpx.get
    downloaded: list[tuple[CorpusSource, bytes, dict]] = []

    for source in OFFICIAL_SOURCES:
        try:
            response = get(
                source.url,
                follow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise CorpusDownloadError(
                f"Could not download {source.url}: {exc}"
            ) from exc

        final_url = str(response.url)
        final_host = (urlparse(final_url).hostname or "").lower()
        if final_host not in source.allowed_hosts:
            raise CorpusDownloadError(
                f"Unexpected redirect host for {source.filename}: {final_host}"
            )

        content = bytes(response.content)
        _validate_document(source, content)
        fetched_at = _utc_now()
        digest = _sha256(content)
        headers = response.headers
        record = {
            "url": source.url,
            "final_url": final_url,
            "topic": source.topic,
            "sha256": digest,
            "size_bytes": len(content),
            "content_type": headers.get("content-type", ""),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "fetched_at_utc": fetched_at,
        }
        downloaded.append((source, content, record))

    documents: dict[str, dict] = {}
    for source, content, record in downloaded:
        previous = old_documents.get(source.filename, {})
        history = list(previous.get("versions", []))
        if not history or history[-1].get("sha256") != record["sha256"]:
            history.append(
                {
                    "sha256": record["sha256"],
                    "fetched_at_utc": record["fetched_at_utc"],
                }
            )
        record["versions"] = history
        documents[source.filename] = record

    # Download and validate every source before replacing any local copy.
    paths: list[str] = []
    for source, content, _ in downloaded:
        path = destination / source.filename
        _atomic_write(path, content)
        paths.append(str(path.resolve()))

    new_manifest = {
        "schema_version": 1,
        "corpus_id": "official-rag-technical-docs",
        "updated_at_utc": _utc_now(),
        "documents": documents,
    }
    _atomic_write(
        manifest_path,
        json.dumps(new_manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )

    return {
        "corpus_id": new_manifest["corpus_id"],
        "documents_downloaded": len(paths),
        "manifest": str(manifest_path.resolve()),
        "files": paths,
        "documents": documents,
        "note": (
            "SHA-256 identifies the downloaded snapshot. Upstream documentation "
            "can change; historical hashes are recorded, but old bytes are not archived."
        ),
    }
