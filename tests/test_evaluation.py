from __future__ import annotations

import json

from rag_assistant.evaluation import evaluate_retrieval, load_benchmark


def test_evaluator_reports_recall_mrr_and_latency(service_factory, tmp_path):
    service = service_factory()
    document = tmp_path / "faiss-guide.txt"
    document.write_text(
        "FAISS is a library for vector similarity search.",
        encoding="utf-8",
    )
    service.ingest([document])

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "question": "What is FAISS?",
                "relevant_sources": ["faiss-guide.txt"],
            }
        ) + "\n",
        encoding="utf-8",
    )

    result = evaluate_retrieval(service, benchmark, k=5)

    assert result["queries"] == 1
    assert result["recall_at_5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["mean_retrieval_latency_ms"] >= 0


def test_benchmark_requires_non_empty_relevant_sources(tmp_path):
    benchmark = tmp_path / "bad.jsonl"
    benchmark.write_text(
        json.dumps({"question": "What?", "relevant_sources": []}) + "\n",
        encoding="utf-8",
    )

    try:
        load_benchmark(benchmark)
    except ValueError as exc:
        assert "relevant_sources" in str(exc)
    else:
        raise AssertionError("Expected invalid benchmark to be rejected")
