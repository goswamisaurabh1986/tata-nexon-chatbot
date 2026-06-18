"""Find stored ChromaDB chunks that mention airbags.

The script connects to the configured ChromaDB collection, prints the total
chunk count, and scans stored documents for "airbag" or "airbags". Matching
chunks are printed with their full text and metadata for ingestion debugging.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv

from src.config.settings import load_settings


SEARCH_TERMS = ("airbag", "airbags")


def main() -> None:
    """Run the airbag chunk diagnostic."""
    _configure_stdout()
    load_dotenv()
    load_dotenv(".env.txt", override=False)

    settings = load_settings()
    persist_directory = settings.retrieval.persist_directory
    collection_name = settings.retrieval.collection_name

    print("ChromaDB Airbag Chunk Diagnostic")
    print("=" * 35)
    print(f"Persist directory : {persist_directory}")
    print(f"Directory exists  : {_yes_no(Path(persist_directory).exists())}")
    print(f"Collection        : {collection_name}")
    print(f"Search terms      : {', '.join(SEARCH_TERMS)}")
    print()

    client = chromadb.PersistentClient(path=persist_directory)
    collection_names = _collection_names(client)

    if collection_name not in collection_names:
        print(f"Collection '{collection_name}' was not found.")
        print(f"Available collections: {', '.join(collection_names) or 'none'}")
        return

    collection = client.get_collection(collection_name)
    total_chunks = collection.count()

    print("Database Summary")
    print("-" * 16)
    print(f"Total chunks: {total_chunks}")
    print()

    if total_chunks == 0:
        print("No chunks are stored in the collection.")
        return

    matches = _matching_chunks(collection, total_chunks)

    print("Matching Chunks")
    print("-" * 15)
    print(f"Matches found: {len(matches)}")
    print()

    if not matches:
        print('No chunks containing "airbag" or "airbags" were found.')
        return

    for index, match in enumerate(matches, start=1):
        print(f"Match {index}")
        print("=" * 7)
        print(f"ID: {match['id']}")
        print("Metadata:")
        print(json.dumps(match["metadata"], indent=2, ensure_ascii=False, sort_keys=True))
        print("Full text:")
        print(match["text"])
        print()


def _collection_names(client: Any) -> list[str]:
    """Return Chroma collection names across ChromaDB versions."""
    names: list[str] = []
    for collection in client.list_collections():
        names.append(collection if isinstance(collection, str) else collection.name)
    return sorted(names)


def _matching_chunks(collection: Any, total_chunks: int) -> list[dict[str, Any]]:
    """Return chunks whose stored text contains any configured search term."""
    results = collection.get(limit=total_chunks, include=["documents", "metadatas"])
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    matches: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        text = document or ""
        if not _contains_search_term(text):
            continue

        matches.append(
            {
                "id": ids[index] if index < len(ids) else "unknown",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "text": text,
            }
        )

    return matches


def _contains_search_term(text: str) -> bool:
    """Return whether text contains an airbag search term."""
    lower_text = text.lower()
    return any(term in lower_text for term in SEARCH_TERMS)


def _yes_no(value: bool) -> str:
    """Format a boolean as yes/no."""
    return "yes" if value else "no"


def _configure_stdout() -> None:
    """Use UTF-8 output for brochure symbols on Windows consoles."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
