"""Command-line entry points for indexing, querying, serving, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import app
from .config import Settings
from .corpus import download_official_corpus
from .evaluation import evaluate_retrieval
from .service import RAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-assistant",
        description="Local document-grounded RAG assistant.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest", help="Index PDF, HTML, TXT, or Markdown files."
    )
    ingest.add_argument("files", nargs="+", help="Input document paths.")

    ask = commands.add_parser(
        "ask", help="Ask a question about indexed documents."
    )
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)

    serve = commands.add_parser("serve", help="Start the FastAPI server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate retrieval on a labeled JSONL file."
    )
    evaluate.add_argument("benchmark", help="Path to a labeled JSONL benchmark.")
    evaluate.add_argument("--k", type=int, default=5)

    corpus = commands.add_parser(
        "corpus",
        help="Download and index the configured official technical-docs corpus.",
    )
    corpus.add_argument(
        "--download-only",
        action="store_true",
        help="Download documents without indexing them.",
    )
    corpus.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="HTTP timeout per source in seconds.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return

    settings = Settings.from_env()
    service = RAGService(settings)

    if args.command == "ingest":
        paths = [Path(value).expanduser().resolve() for value in args.files]
        result = service.ingest(paths)
    elif args.command == "ask":
        result = service.ask(args.question, args.top_k)
    elif args.command == "evaluate":
        result = evaluate_retrieval(service, args.benchmark, args.k)
    elif args.command == "corpus":
        corpus_dir = settings.data_dir / "corpus" / "official"
        downloaded = download_official_corpus(
            corpus_dir,
            timeout=args.timeout,
        )
        indexed = None
        if not args.download_only:
            indexed = service.ingest(
                [Path(value) for value in downloaded["files"]]
            )
        result = {
            "download": {
                "corpus_id": downloaded["corpus_id"],
                "documents_downloaded": downloaded["documents_downloaded"],
                "manifest": downloaded["manifest"],
                "files": downloaded["files"],
                "note": downloaded["note"],
            },
            "index": indexed,
        }
    else:  # pragma: no cover - argparse enforces the subcommand
        raise RuntimeError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
