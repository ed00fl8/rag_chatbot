"""Offline retrieval evaluation for a manually labeled JSONL benchmark."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path


def load_benchmark(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on benchmark line {line_number}: {exc}"
                ) from exc
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("question"), str)
                or not item["question"].strip()
                or not isinstance(item.get("relevant_sources"), list)
                or not item["relevant_sources"]
            ):
                raise ValueError(
                    f"Benchmark line {line_number} must contain a non-empty "
                    "question and a non-empty relevant_sources list."
                )
            records.append(item)
    if not records:
        raise ValueError("The benchmark contains no queries.")
    return records


def evaluate_retrieval(service, benchmark_path: str | Path, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k must be positive.")
    records = load_benchmark(benchmark_path)
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    # Exclude one-time model loading and initialization from per-query latency.
    service.retrieve(records[0]["question"], top_k=k)

    for item in records:
        started = time.perf_counter()
        hits = service.retrieve(item["question"], top_k=k)
        latencies.append((time.perf_counter() - started) * 1000)

        expected = {str(value).casefold() for value in item["relevant_sources"]}
        retrieved = [hit.chunk.source.casefold() for hit in hits]
        relevant_retrieved = sum(source in expected for source in set(retrieved))
        recalls.append(relevant_retrieved / len(expected))

        first_rank = next(
            (
                rank
                for rank, source in enumerate(retrieved, start=1)
                if source in expected
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

    return {
        "queries": len(records),
        f"recall_at_{k}": round(statistics.mean(recalls), 6),
        "mrr": round(statistics.mean(reciprocal_ranks), 6),
        "mean_retrieval_latency_ms": round(statistics.mean(latencies), 3),
        "latency_scope": "retrieval only; warm models; excludes LLM generation",
        "benchmark": str(benchmark_path),
    }
