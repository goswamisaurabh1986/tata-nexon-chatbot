"""Inspect the configured ChromaDB vector store.

This script is intentionally diagnostic-only: it reads the configured ChromaDB
persist directory and prints collection/chunk information without exposing any
secrets from the environment or settings layer.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import shorten
from typing import Any, Optional

import chromadb
from dotenv import load_dotenv

from src.config.settings import load_settings


SAMPLE_LIMIT = 5
TEXT_PREVIEW_WIDTH = 280


def main() -> None:
    """Run the ChromaDB diagnostic report."""
    load_dotenv()
    load_dotenv(".env.txt", override=False)

    settings = load_settings()
    persist_directory = settings.retrieval.persist_directory
    main_collection_name = settings.retrieval.collection_name

    print("ChromaDB Vector Store Diagnostic")
    print("=" * 34)
    print(f"Persist directory : {persist_directory}")
    print(f"Directory exists  : {_yes_no(Path(persist_directory).exists())}")
    print(f"Main collection   : {main_collection_name}")
    print()

    client = chromadb.PersistentClient(path=persist_directory)
    collections = _list_collections(client)

    _print_collections(collections)
    print()

    if main_collection_name not in {collection["name"] for collection in collections}:
        print("Main Collection")
        print("-" * 15)
        print(f"Collection '{main_collection_name}' was not found.")
        return

    collection = client.get_collection(main_collection_name)
    count = collection.count()

    print("Main Collection")
    print("-" * 15)
    print(f"Name         : {main_collection_name}")
    print(f"Chunk count  : {count}")
    print()

    if count == 0:
        print("No chunks are stored in the main collection yet.")
        return

    _print_sample_chunks(collection, limit=min(SAMPLE_LIMIT, count))


def _list_collections(client: Any) -> list[dict[str, Any]]:
    """Return collection names and counts in a version-tolerant shape."""
    collection_items = client.list_collections()
    collections: list[dict[str, Any]] = []

    for item in collection_items:
        name = item if isinstance(item, str) else item.name
        try:
            count = client.get_collection(name).count()
        except Exception as error:  # pragma: no cover - defensive diagnostic path
            count = f"unavailable ({error})"
        collections.append({"name": name, "count": count})

    return sorted(collections, key=lambda collection: collection["name"])


def _print_collections(collections: list[dict[str, Any]]) -> None:
    """Print all Chroma collections with document counts."""
    print("Collections")
    print("-" * 11)

    if not collections:
        print("No collections found.")
        return

    for index, collection in enumerate(collections, start=1):
        print(f"{index}. {collection['name']} ({collection['count']} chunks)")


def _print_sample_chunks(collection: Any, limit: int) -> None:
    """Print a few stored chunks with useful metadata and text previews."""
    results = collection.get(limit=limit, include=["documents", "metadatas"])
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    print(f"Sample Chunks (showing {len(documents)} of {collection.count()})")
    print("-" * 44)

    for index, document in enumerate(documents, start=1):
        metadata = metadatas[index - 1] if index - 1 < len(metadatas) else {}
        chunk_id = ids[index - 1] if index - 1 < len(ids) else "unknown"
        print(f"[{index}] id            : {chunk_id}")
        print(f"    source        : {_metadata_value(metadata, 'source')}")
        print(f"    section_title : {_metadata_value(metadata, 'section_title')}")
        print(f"    chunk_index   : {_metadata_value(metadata, 'chunk_index')}")
        print(f"    total_chunks  : {_metadata_value(metadata, 'total_chunks')}")
        print(f"    page_number   : {_metadata_value(metadata, 'page_number')}")
        print(f"    page_count    : {_metadata_value(metadata, 'page_count')}")
        print(f"    text          : {_preview_text(document)}")
        print()


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    """Return a printable metadata value with a readable fallback."""
    value = metadata.get(key)
    if value in (None, ""):
        return "n/a"
    return str(value)


def _preview_text(text: Optional[str]) -> str:
    """Return a single-line chunk text preview."""
    normalized = " ".join((text or "").split())
    return shorten(normalized, width=TEXT_PREVIEW_WIDTH, placeholder="...")


def _yes_no(value: bool) -> str:
    """Format a boolean as yes/no."""
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
