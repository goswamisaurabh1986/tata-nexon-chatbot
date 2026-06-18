from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import Embedder
from src.ingestion.parsers.text_parser import TextParser
from src.ingestion.scanner import DocumentScanner
from src.ingestion.storer import VectorStorer


class IngestionProcessor:
    def __init__(
        self,
        scanner: DocumentScanner | None = None,
        parser: TextParser | None = None,
        embedder=None,
        storer=None,
    ) -> None:
        self.scanner = scanner or DocumentScanner()
        self.parser = parser or TextParser()
        self.embedder = embedder
        self.storer = storer

    def process(
        self,
        document: str | None = None,
        file_path: str | None = None,
        source_filename: str | None = None,
        chunk_size: int = DocumentChunker.DEFAULT_CHUNK_SIZE,
        overlap: int = DocumentChunker.DEFAULT_OVERLAP,
        embedder=None,
        storer=None,
        metadata_overrides: dict | None = None,
    ) -> list:
        scanned_document = (
            self.scanner.load_document(file_path)
            if file_path is not None
            else self.scanner.load(document or "", source_filename)
        )
        text = scanned_document["text"]
        document_metadata = scanned_document.get("metadata", {})
        source = document_metadata.get("source") or scanned_document["source_filename"]

        if not text.strip():
            return []

        parsed_document = self.parser.parse(text)
        cleaned_text = parsed_document["cleaned_text"]
        sections = [
            (section["title"], section["content"])
            for section in parsed_document["sections"]
        ]

        if not cleaned_text.strip():
            return []

        chunks = DocumentChunker(chunk_size, overlap).chunk(
            cleaned_text,
            chunk_size=chunk_size,
            overlap=overlap,
            source_filename=source,
            sections=sections,
        )
        if file_path is not None:
            self._add_file_metadata(chunks, document_metadata)
        self._add_metadata_overrides(chunks, metadata_overrides)
        chunks = self._embed_chunks(chunks, embedder)
        return self._store_chunks(chunks, source, cleaned_text, storer)

    def _embed_chunks(self, chunks: list[dict], embedder=None) -> list[dict]:
        active_embedder = embedder if embedder is not None else self.embedder
        if active_embedder is None:
            return chunks

        if isinstance(active_embedder, Embedder):
            return active_embedder.embed_chunks(chunks)

        return Embedder(embedding_model=active_embedder).embed_chunks(chunks)

    def _add_file_metadata(self, chunks: list[dict], document_metadata: dict) -> None:
        metadata_keys = ("source", "file_type", "page_count", "total_characters")
        for chunk in chunks:
            chunk["metadata"].update(
                {
                    key: document_metadata[key]
                    for key in metadata_keys
                    if key in document_metadata
                }
            )

    def _add_metadata_overrides(
        self,
        chunks: list[dict],
        metadata_overrides: dict | None,
    ) -> None:
        if not metadata_overrides:
            return

        for chunk in chunks:
            chunk.setdefault("metadata", {}).update(metadata_overrides)

    def _store_chunks(
        self,
        chunks: list[dict],
        source: str | None,
        cleaned_text: str,
        storer=None,
    ) -> list[dict]:
        active_storer = storer if storer is not None else self.storer
        if active_storer is None:
            return chunks

        if isinstance(active_storer, VectorStorer):
            return active_storer.store(chunks, source, cleaned_text)

        return VectorStorer(vector_store=active_storer).store(chunks, source, cleaned_text)


def process_document(
    document: str,
    source_filename: str | None = None,
    chunk_size: int = DocumentChunker.DEFAULT_CHUNK_SIZE,
    overlap: int = DocumentChunker.DEFAULT_OVERLAP,
    embedder=None,
    storer=None,
    metadata_overrides: dict | None = None,
) -> list:
    return IngestionProcessor().process(
        document,
        source_filename=source_filename,
        chunk_size=chunk_size,
        overlap=overlap,
        embedder=embedder,
        storer=storer,
        metadata_overrides=metadata_overrides,
    )


def process_file(
    file_path: str,
    chunk_size: int = DocumentChunker.DEFAULT_CHUNK_SIZE,
    overlap: int = DocumentChunker.DEFAULT_OVERLAP,
    embedder=None,
    storer=None,
    metadata_overrides: dict | None = None,
) -> list:
    return IngestionProcessor().process(
        file_path=file_path,
        chunk_size=chunk_size,
        overlap=overlap,
        embedder=embedder,
        storer=storer,
        metadata_overrides=metadata_overrides,
    )
