"""Batch ingest local documents into the configured ChromaDB collection.

The script scans ``data/ingestion_docs`` for supported documents and runs the
same production ingestion flow used by the app:

    Scanner -> Parser -> Chunker -> Embedder -> Storer

It is safe to run repeatedly because the vector store uses deterministic chunk
IDs. Pass ``--force-reprocess`` to delete existing chunks for each source before
storing freshly processed chunks.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from src.config.settings import Settings, load_settings
from src.ingestion.embedder import Embedder
from src.ingestion.processor import IngestionProcessor
from src.ingestion.storer import VectorStorer


DEFAULT_DOCS_DIR = Path("data/ingestion_docs")


@dataclass
class FileIngestionResult:
    """Summary for one processed document."""

    file_path: Path
    chunks_created: int = 0
    chunks_with_embeddings: int = 0
    duration_seconds: float = 0.0
    status: str = "pending"
    error: Optional[str] = None


def main() -> int:
    """Run batch ingestion and return a process exit code."""
    args = _parse_args()
    _load_environment()

    settings = load_settings()
    _configure_logging(settings)

    docs_dir = args.docs_dir
    supported_types = settings.ingestion.supported_file_types
    files = _scan_documents(docs_dir, supported_types)

    _print_header(settings=settings, docs_dir=docs_dir, files=files, force=args.force_reprocess)

    if not files:
        print("No supported documents found. Nothing to ingest.")
        return 0

    if not settings.llm.api_key:
        print("ERROR: OPENAI_API_KEY is not configured. Embeddings cannot be created.")
        return 1

    embedder = Embedder(
        model_name=settings.llm.embedding_model,
        api_key=settings.llm.api_key,
        dimensions=settings.llm.embedding_dimensions,
        max_retries=settings.llm.max_retries,
    )
    storer = VectorStorer(
        collection_name=settings.retrieval.collection_name,
        persist_directory=settings.retrieval.persist_directory,
        embedding_dimension=settings.retrieval.embedding_dimension,
    )
    processor = IngestionProcessor(embedder=embedder, storer=storer)

    results = []
    started_at = time.perf_counter()
    for file_index, file_path in enumerate(files, start=1):
        result = _ingest_file(
            file_path=file_path,
            processor=processor,
            storer=storer,
            settings=settings,
            force_reprocess=args.force_reprocess,
            file_index=file_index,
            total_files=len(files),
        )
        results.append(result)

    elapsed = time.perf_counter() - started_at
    _print_summary(results, elapsed, storer)
    return 1 if any(result.status == "failed" for result in results) else 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest local documents into ChromaDB.")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help="Folder containing PDF/TXT/MD documents to ingest.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Delete existing chunks for each source before ingesting it again.",
    )
    return parser.parse_args()


def _load_environment() -> None:
    """Load local environment files without printing secret values."""
    load_dotenv()
    load_dotenv(".env.txt", override=False)


def _configure_logging(settings: Settings) -> None:
    """Configure logging from the settings layer."""
    logging.basicConfig(level=getattr(logging, settings.app.log_level, logging.WARNING))


def _scan_documents(docs_dir: Path, supported_types: tuple[str, ...]) -> list[Path]:
    """Return supported document files in deterministic order."""
    if not docs_dir.exists():
        return []

    supported = {file_type.lower() for file_type in supported_types}
    return sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )


def _print_header(
    settings: Settings,
    docs_dir: Path,
    files: list[Path],
    force: bool,
) -> None:
    """Print startup configuration without exposing secrets."""
    print("Tata Nexon Document Ingestion")
    print("=" * 30)
    print(f"Documents directory : {docs_dir}")
    print(f"Supported types     : {', '.join(settings.ingestion.supported_file_types)}")
    print(f"Files discovered    : {len(files)}")
    print(f"Chunk size/overlap  : {settings.ingestion.chunk_size}/{settings.ingestion.overlap}")
    print(f"Embedding model     : {settings.llm.embedding_model}")
    print(f"Vector collection   : {settings.retrieval.collection_name}")
    print(f"Chroma directory    : {settings.retrieval.persist_directory}")
    print(f"Force reprocess     : {'yes' if force else 'no'}")
    print()

    for index, file_path in enumerate(files, start=1):
        print(f"{index}. {file_path.name} ({_format_bytes(file_path.stat().st_size)})")
    if files:
        print()


def _ingest_file(
    file_path: Path,
    processor: IngestionProcessor,
    storer: VectorStorer,
    settings: Settings,
    force_reprocess: bool,
    file_index: int,
    total_files: int,
) -> FileIngestionResult:
    """Ingest a single file and return a structured result."""
    result = FileIngestionResult(file_path=file_path)
    started_at = time.perf_counter()

    print(f"[{file_index}/{total_files}] Processing {file_path.name}")
    try:
        if force_reprocess:
            deleted = _delete_existing_source_chunks(storer, source=file_path.name)
            print(f"    Removed existing chunks : {deleted}")

        chunks = processor.process(
            file_path=str(file_path),
            chunk_size=settings.ingestion.chunk_size,
            overlap=settings.ingestion.overlap,
        )

        result.chunks_created = len(chunks)
        result.chunks_with_embeddings = sum(1 for chunk in chunks if chunk.get("embedding"))
        result.status = "completed"
        print(f"    Chunks created          : {result.chunks_created}")
        print(f"    Chunks with embeddings  : {result.chunks_with_embeddings}")
        print("    Status                  : completed")
    except Exception as error:
        result.status = "failed"
        result.error = str(error)
        print(f"    Status                  : failed")
        print(f"    Error                   : {error}")
    finally:
        result.duration_seconds = time.perf_counter() - started_at
        print(f"    Duration                : {result.duration_seconds:.2f}s")
        print()

    return result


def _delete_existing_source_chunks(storer: VectorStorer, source: str) -> int:
    """Delete existing Chroma records for a source and return deleted count."""
    collection = storer.collection
    existing = collection.get(where={"source": source}, include=[])
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _print_summary(
    results: list[FileIngestionResult],
    elapsed_seconds: float,
    storer: VectorStorer,
) -> None:
    """Print final ingestion statistics."""
    completed = [result for result in results if result.status == "completed"]
    failed = [result for result in results if result.status == "failed"]
    total_chunks = sum(result.chunks_created for result in completed)
    total_embeddings = sum(result.chunks_with_embeddings for result in completed)
    collection_count = _safe_collection_count(storer)

    print("Ingestion Summary")
    print("=" * 17)
    print(f"Files processed       : {len(results)}")
    print(f"Files completed       : {len(completed)}")
    print(f"Files failed          : {len(failed)}")
    print(f"Total chunks created  : {total_chunks}")
    print(f"Chunks embedded       : {total_embeddings}")
    print(f"Collection count      : {collection_count}")
    print(f"Total duration        : {elapsed_seconds:.2f}s")

    if failed:
        print()
        print("Failures")
        print("-" * 8)
        for result in failed:
            print(f"- {result.file_path.name}: {result.error}")


def _safe_collection_count(storer: VectorStorer) -> str:
    """Return the collection count with a diagnostic fallback."""
    try:
        return str(storer.collection.count())
    except Exception as error:
        return f"unavailable ({error})"


def _format_bytes(size: int) -> str:
    """Format file sizes for progress output."""
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


if __name__ == "__main__":
    sys.exit(main())
