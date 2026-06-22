import logging
from typing import Any, Optional, Protocol, TypedDict


Embedding = list[float]


class Chunk(TypedDict):
    citation_id: str
    text: str
    metadata: dict[str, Any]
    score: float


class EmbedderProtocol(Protocol):
    """Interface for embedding user queries."""

    def embed(self, text: str) -> Embedding:
        """Convert query text into an embedding vector."""


class VectorStoreProtocol(Protocol):
    """Interface for vector similarity search."""

    def similarity_search(self, query_embedding: Embedding, top_k: int = 5) -> list[dict]:
        """Return the most similar chunks for the query embedding."""


class Retriever:
    """Main retrieval component for the LangGraph agent with citation support."""

    DEFAULT_TOP_K = 5
    DEFAULT_SIMILARITY_THRESHOLD = 0.0

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        vector_store: Optional[VectorStoreProtocol] = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[Chunk]:
        """Retrieve relevant chunks with citations for the given query."""
        if not query or not query.strip():
            self.logger.debug("Empty query received, returning empty list.")
            return []

        try:
            safe_top_k = self._safe_top_k(top_k)
            self.logger.info(
                "Processing retrieval for query: '%s' (top_k=%d)",
                query[:100],
                safe_top_k,
            )

            query_embedding = self._embed_query(query)
            raw_results = self._perform_search(query_embedding, safe_top_k)
            filtered_results = self._filter_by_threshold(
                raw_results,
                similarity_threshold,
            )
            sorted_results = self._sort_by_score(filtered_results)
            final_results = self._enrich_with_citations(sorted_results[:safe_top_k])

            self.logger.info("Retrieved %d chunks for query.", len(final_results))
            return final_results
        except Exception as error:
            self.logger.error(
                "Error during retrieval for query '%s': %s",
                query[:100],
                error,
                exc_info=True,
            )
            return []

    def _embed_query(self, query: str) -> Embedding:
        """Embed the query text, or return an empty embedding if unavailable."""
        if not self.embedder:
            self.logger.warning("No embedder provided, using empty embedding.")
            return []

        return self.embedder.embed(query)

    def _perform_search(self, query_embedding: Embedding, top_k: int) -> list[dict]:
        """Search the vector store, falling back to dummy data during early development."""
        if self.vector_store:
            return self.vector_store.similarity_search(query_embedding, top_k=top_k)

        self.logger.warning("No vector store provided, returning dummy results.")
        return self._search_chunks(query_embedding)

    def _filter_by_threshold(self, chunks: list[dict], threshold: float) -> list[dict]:
        """Keep only chunks whose score meets the similarity threshold."""
        return [
            chunk
            for chunk in chunks
            if self._score(chunk) >= threshold
        ]

    def _sort_by_score(self, chunks: list[dict]) -> list[dict]:
        """Sort chunks from most relevant to least relevant."""
        return sorted(chunks, key=self._score, reverse=True)

    def _enrich_with_citations(self, chunks: list[dict]) -> list[Chunk]:
        """Normalize retrieved chunks and add stable citation IDs."""
        return [
            self._chunk_with_citation(chunk, fallback_index)
            for fallback_index, chunk in enumerate(chunks)
        ]

    def _chunk_with_citation(self, chunk: dict, fallback_index: int) -> Chunk:
        metadata = dict(chunk.get("metadata") or {})
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", fallback_index)
        metadata.setdefault("source", source)
        metadata.setdefault("section_title", None)
        metadata.setdefault("chunk_index", chunk_index)

        return {
            "citation_id": f"{source}:{chunk_index}",
            "text": str(chunk.get("text", "")),
            "metadata": metadata,
            "score": self._score(chunk),
        }

    def _search_chunks(self, query_embedding: Embedding) -> list[dict]:
        """Compatibility hook for unit tests and dummy retrieval."""
        return self._get_dummy_results()

    def _get_dummy_results(self) -> list[dict]:
        """Return a tiny sample result when no vector store is configured."""
        return [
            {
                "text": "Tata Nexon includes advanced safety features.",
                "metadata": {
                    "source": "Tata_Nexon_Brochure.pdf",
                    "section_title": "Safety",
                    "chunk_index": 0,
                    "page_number": 5,
                },
                "score": 0.85,
            }
        ]

    def _score(self, chunk: dict) -> float:
        return float(chunk.get("score", 0.0))

    def _safe_top_k(self, top_k: int) -> int:
        return max(0, int(top_k))
