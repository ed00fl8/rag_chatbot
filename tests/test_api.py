from __future__ import annotations

from fastapi.testclient import TestClient

from rag_assistant.api import create_app


def test_health_and_document_upload(service_factory):
    service = service_factory()
    client = TestClient(create_app(service))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["index_ready"] is False

    response = client.post(
        "/documents",
        files=[
            (
                "files",
                (
                    "guide.txt",
                    b"FAISS is used for vector similarity search.",
                    "text/plain",
                ),
            )
        ],
    )

    assert response.status_code == 201
    assert response.json()["indexed_chunks"] == 1
    assert client.get("/health").json()["index_ready"] is True


def test_upload_rejects_unsupported_extension(service_factory):
    client = TestClient(create_app(service_factory()))
    response = client.post(
        "/documents",
        files=[("files", ("malware.exe", b"no", "application/octet-stream"))],
    )
    assert response.status_code == 415
