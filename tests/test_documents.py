from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.documents import (
    chunk_document,
    extract_document,
    sha256_bytes,
    split_text,
)


def test_extracts_text_and_calculates_content_version(tmp_path):
    path = tmp_path / "guide.txt"
    content = "A technical guide about vector databases.\n"
    path.write_text(content, encoding="utf-8")

    document = extract_document(path)

    assert document.source == "guide.txt"
    assert document.version == sha256_bytes(content.encode("utf-8"))
    assert document.pages == (content.strip(),)


def test_html_extraction_discards_script_and_style(tmp_path):
    path = tmp_path / "guide.html"
    path.write_text(
        "<html><body><h1>FAISS</h1><p>Vector search</p>"
        "<script>secret()</script><style>.x{}</style></body></html>",
        encoding="utf-8",
    )

    document = extract_document(path)

    assert "FAISS" in document.pages[0]
    assert "Vector search" in document.pages[0]
    assert "secret" not in document.pages[0]
    assert ".x" not in document.pages[0]


def test_rejects_unsupported_and_empty_documents(tmp_path):
    unsupported = tmp_path / "image.png"
    unsupported.write_bytes(b"image")
    with pytest.raises(ValueError, match="Unsupported"):
        extract_document(unsupported)

    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable text"):
        extract_document(empty)


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"token{i}" for i in range(100))
    chunks = split_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_ids_are_deterministic_and_keep_version_metadata(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("A short reproducible document.", encoding="utf-8")
    document = extract_document(path)

    first = chunk_document(document, chunk_size=200, overlap=20)
    second = chunk_document(document, chunk_size=200, overlap=20)

    assert first == second
    assert first[0].source == "guide.md"
    assert first[0].version == document.version
    assert first[0].page is None
