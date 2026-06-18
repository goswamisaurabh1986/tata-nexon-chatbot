import hashlib
import os
from pathlib import Path

import chromadb


class VectorStorer:
    DEFAULT_COLLECTION_NAME = "tata_nexon_chunks"
    DEFAULT_PERSIST_DIRECTORY = ".chroma"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        embedding_dimension: int | None = None,
        client=None,
        vector_store=None,
    ) -> None:
        if not isinstance(collection_name, str):
            vector_store = collection_name
            collection_name = self.DEFAULT_COLLECTION_NAME

        self.collection_name = collection_name
        self.persist_directory = persist_directory or self.DEFAULT_PERSIST_DIRECTORY
        self.embedding_dimension = embedding_dimension
        self.vector_store = vector_store
        self.client = client
        self.collection = None if vector_store is not None else self._collection()

    @classmethod
    def from_env(cls) -> "VectorStorer":
        dimension = os.getenv("CHROMA_EMBEDDING_DIMENSION")
        return cls(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", cls.DEFAULT_COLLECTION_NAME),
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", cls.DEFAULT_PERSIST_DIRECTORY),
            embedding_dimension=int(dimension) if dimension else None,
        )

    @classmethod
    def from_config(cls, config: dict) -> "VectorStorer":
        return cls(
            collection_name=config.get("collection_name", cls.DEFAULT_COLLECTION_NAME),
            persist_directory=config.get("persist_directory"),
            embedding_dimension=config.get("embedding_dimension"),
            client=config.get("client"),
            vector_store=config.get("vector_store"),
        )

    def get_collection_name(self) -> str:
        return self.collection_name

    def store(
        self,
        chunks: list[dict],
        source_filename: str | None = None,
        document: str | None = None,
    ) -> list[dict]:
        if self.vector_store is not None:
            return self._store_with_delegate(chunks, source_filename, document)

        records = self._records(chunks, source_filename)
        if records:
            self.collection.upsert(
                ids=[record["id"] for record in records],
                documents=[record["document"] for record in records],
                embeddings=[record["embedding"] for record in records],
                metadatas=[record["metadata"] for record in records],
            )
        return chunks

    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        if not query_embedding:
            return []

        if self.vector_store is not None:
            search = getattr(self.vector_store, "similarity_search", None)
            if callable(search):
                return search(query_embedding, top_k=top_k)

        results = self.collection.query(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self._chunks_from_query_results(results)

    def _collection(self):
        client = self.client
        if client is None:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_directory)
            self.client = client

        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _store_with_delegate(
        self,
        chunks: list[dict],
        source_filename: str | None,
        document: str | None,
    ) -> list[dict]:
        processed_sources = getattr(self.vector_store, "_processed_sources", None)
        if not isinstance(processed_sources, set):
            processed_sources = set()
            self.vector_store._processed_sources = processed_sources

        source_key = source_filename if source_filename is not None else document
        if source_key not in processed_sources:
            self.vector_store.store(chunks)
            processed_sources.add(source_key)

        return chunks

    def _records(self, chunks: list[dict], source_filename: str | None) -> list[dict]:
        records = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding:
                continue
            if self.embedding_dimension is not None and len(embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.embedding_dimension}, "
                    f"got {len(embedding)}."
                )

            metadata = self._metadata(chunk, source_filename)
            records.append(
                {
                    "id": self._chunk_id(chunk, metadata),
                    "document": chunk.get("text", ""),
                    "embedding": [float(value) for value in embedding],
                    "metadata": metadata,
                }
            )
        return records

    def _metadata(self, chunk: dict, source_filename: str | None) -> dict:
        metadata = dict(chunk.get("metadata", {}))
        metadata.setdefault("source", source_filename or "unknown")

        clean_metadata = {}
        for key, value in metadata.items():
            if value is None:
                clean_metadata[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        return clean_metadata

    def _chunk_id(self, chunk: dict, metadata: dict) -> str:
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index")
        section_title = metadata.get("section_title", "")
        text = chunk.get("text", "")

        if chunk_index is not None:
            raw_id = f"{source}:{chunk_index}:{section_title}"
        else:
            raw_id = f"{source}:{text}"
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return f"chunk-{digest}"

    def _chunks_from_query_results(self, results: dict) -> list[dict]:
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks = []
        for index, document in enumerate(documents):
            distance = distances[index] if index < len(distances) else None
            chunks.append(
                {
                    "text": document or "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "score": self._score_from_distance(distance),
                }
            )
        return chunks

    def _score_from_distance(self, distance) -> float:
        if distance is None:
            return 0.0

        score = 1.0 - float(distance)
        return max(0.0, min(1.0, score))
