# RAG Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) assistant for technical documentation. Supports PDF, HTML, Markdown and TXT files, hybrid retrieval, reranking, source citations and evidence-based abstention.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Internet connection for initial downloads

## Installation

Install uv:

Windows PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS, Linux and WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Install dependencies:

```bash
uv python install 3.13
uv sync --locked --extra dev
```

Install and start Ollama, then download the language model:

```bash
ollama pull llama3.2
```

Embedding and reranking models are downloaded automatically on first use.

## Download the documentation corpus

```bash
uv run rag-assistant corpus
```

Downloads and indexes official technical documentation from:

- FAISS
- Sentence Transformers
- FastAPI
- Ollama

Source URLs, download metadata and SHA-256 hashes are recorded in a local manifest.

## Run the API

```bash
uv run uvicorn rag_assistant.api:app --host 127.0.0.1 --port 8000
```

Open:

http://127.0.0.1:8000/docs

Upload documents through `POST /documents` and ask questions through `POST /query`.

Example query:

```json
{
  "question": "What does the documentation say about vector search?",
  "top_k": 5
}
```

Responses include the generated answer, abstention status and source references with filenames, page numbers, chunk indices and document hashes.

## Command-line usage

Index documents:

```bash
uv run rag-assistant ingest path/to/document.pdf
```

Ask a question:

```bash
uv run rag-assistant ask "What does the documentation say about vector search?"
```

## Evaluation

Run the initial 60-query retrieval benchmark:

```bash
uv run rag-assistant evaluate eval/benchmark.official.seed.jsonl --k 5
```

The included benchmark is an initial seed dataset. Its source annotations require independent review before its scores can be reported as definitive performance results.

The evaluator measures Recall@k, Mean Reciprocal Rank (MRR) and retrieval latency. It does not independently measure answer faithfulness, citation entailment or end-to-end generation latency.

## Tests

```bash
uv run pytest -v
```

## Data and versioning

Downloaded documents, embeddings, FAISS indexes and metadata are stored locally under `data/`.

Document versions are identified by SHA-256. Re-ingesting a document replaces its current indexed version while retaining its version hashes.

The `data/` directory is excluded from Git.

## Limitations

- Scanned PDFs requiring OCR are not supported.
- Ollama must be running locally for answer generation.
- Citation validation checks source references but does not independently prove semantic entailment of every claim.
- Prompt injection defenses are mitigations, not a guarantee of complete security.
- Reranking thresholds are initial heuristics requiring further calibration.
- Historical document contents are not archived; only version hashes are retained.

## Technologies

Python · FastAPI · Sentence Transformers · FAISS · BM25 · Reciprocal Rank Fusion · Cross-Encoder · Ollama · Llama 3.2 · Pytest