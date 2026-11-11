from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest

from rag_assistant.corpus import (
    OFFICIAL_SOURCES,
    CorpusDownloadError,
    download_official_corpus,
)


class FakeResponse:
    def __init__(self, url: str, content: bytes, headers=None, status_code=200):
        self.url = httpx.URL(url)
        self.content = content
        self.headers = headers or {"content-type": "text/plain; charset=utf-8"}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "HTTP error",
                request=httpx.Request("GET", str(self.url)),
                response=httpx.Response(self.status_code),
            )


@pytest.fixture
def fake_fetcher():
    responses = {}
    for source in OFFICIAL_SOURCES:
        if source.filename.endswith(".html"):
            content = (
                "<html><head><title>Official docs</title><script>bad()</script>"
                "</head><body><h1>Official documentation</h1>"
                "<p>This official documentation describes the API and its behavior. "
                "It contains enough text to validate a non-empty source snapshot.</p>"
                "</body></html>"
            ).encode()
            content_type = "text/html; charset=utf-8"
        else:
            content = (
                "# Official documentation\n\n"
                "This official documentation describes the library and its behavior. "
                "It contains enough text to validate a non-empty source snapshot.\n"
            ).encode()
            content_type = "text/plain; charset=utf-8"
        responses[source.url] = FakeResponse(
            source.url,
            content,
            {"content-type": content_type, "etag": '"test-etag"'},
        )

    def fetch(url, **kwargs):
        return responses[url]

    return fetch


def test_downloads_all_official_sources_and_writes_manifest(
    tmp_path,
    fake_fetcher,
):
    result = download_official_corpus(tmp_path / "official", fetcher=fake_fetcher)

    assert result["documents_downloaded"] == len(OFFICIAL_SOURCES)
    assert len(result["files"]) == len(OFFICIAL_SOURCES)

    manifest = json.loads(Path(result["manifest"]).read_text())
    assert set(manifest["documents"]) == {
        source.filename for source in OFFICIAL_SOURCES
    }

    for source in OFFICIAL_SOURCES:
        path = tmp_path / "official" / source.filename
        assert path.is_file()
        assert manifest["documents"][source.filename]["sha256"]
        assert len(manifest["documents"][source.filename]["sha256"]) == 64


def test_manifest_retains_hash_history_when_source_changes(
    tmp_path,
    fake_fetcher,
):
    destination = tmp_path / "official"
    download_official_corpus(destination, fetcher=fake_fetcher)

    changed = {}
    for source in OFFICIAL_SOURCES:
        if source.filename.endswith(".html"):
            body = (
                "<html><body><h1>Updated official docs</h1>"
                "<p>The upstream documentation was updated with more content "
                "for the corpus downloader regression test.</p></body></html>"
            ).encode()
            content_type = "text/html"
        else:
            body = (
                "# Updated official docs\n\nThe upstream documentation was updated "
                "with more content for the corpus downloader regression test.\n"
            ).encode()
            content_type = "text/plain"
        changed[source.url] = FakeResponse(
            source.url, body, {"content-type": content_type}
        )

    download_official_corpus(
        destination,
        fetcher=lambda url, **kwargs: changed[url],
    )

    manifest = json.loads(
        (destination / "manifest.json").read_text()
    )
    for source in OFFICIAL_SOURCES:
        assert len(manifest["documents"][source.filename]["versions"]) == 2


def test_rejects_unexpected_redirect_host(tmp_path, fake_fetcher):
    source = OFFICIAL_SOURCES[0]

    def fetch(url, **kwargs):
        return FakeResponse(
            "https://attacker.example/payload.md",
            b"# Fake page\n" + b"safe-looking text " * 20,
        )

    with pytest.raises(CorpusDownloadError, match="Unexpected redirect host"):
        download_official_corpus(tmp_path / "official", fetcher=fetch)


def test_rejects_empty_or_tiny_response(tmp_path):
    source = OFFICIAL_SOURCES[0]

    def fetch(url, **kwargs):
        return FakeResponse(url, b"Not found")

    with pytest.raises(CorpusDownloadError, match="too little readable text"):
        download_official_corpus(tmp_path / "official", fetcher=fetch)


def test_does_not_write_manifest_when_a_source_fails(tmp_path):
    calls = 0

    def fetch(url, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise httpx.ConnectError("offline")
        source = OFFICIAL_SOURCES[calls - 1]
        content = (
            "<html><body><p>Enough official document content to pass validation "
            "for this mocked HTTP response used by the corpus tests.</p></body></html>"
            if source.filename.endswith(".html")
            else "# Docs\n\nEnough official document content to pass validation "
                 "for this mocked HTTP response used by the corpus tests."
        ).encode()
        return FakeResponse(url, content)

    destination = tmp_path / "official"
    with pytest.raises(CorpusDownloadError):
        download_official_corpus(destination, fetcher=fetch)

    assert not (destination / "manifest.json").exists()
