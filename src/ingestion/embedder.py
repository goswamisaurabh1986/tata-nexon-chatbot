import logging
import os
import time

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class Embedder:
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model_name: str | object = DEFAULT_MODEL,
        api_key: str | None = None,
        dimensions: int | None = None,
        client=None,
        embedding_model=None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        max_input_chars: int = 20000,
    ) -> None:
        if not isinstance(model_name, str):
            embedding_model = model_name
            model_name = self.DEFAULT_MODEL

        self.model_name = model_name
        self.api_key = api_key
        self.dimensions = dimensions
        self.client = client
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_input_chars = max_input_chars
        self._missing_key_warned = False

    @classmethod
    def from_env(cls) -> "Embedder":
        load_dotenv()
        dimensions = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
        return cls(
            model_name=os.getenv("OPENAI_EMBEDDING_MODEL", cls.DEFAULT_MODEL),
            api_key=os.getenv("OPENAI_API_KEY"),
            dimensions=int(dimensions) if dimensions else None,
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        prepared_texts = [self._prepare_text(text) for text in texts]
        indexed_texts = [
            (index, text)
            for index, text in enumerate(prepared_texts)
            if text
        ]
        embeddings = [[] for _ in prepared_texts]
        if not indexed_texts:
            return embeddings

        client = self._client()
        if client is None:
            return embeddings

        response = self._create_embeddings(client, [text for _, text in indexed_texts])
        for (index, _), item in zip(indexed_texts, response.data):
            embeddings[index] = [float(value) for value in item.embedding]

        return embeddings

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        if self.embedding_model is not None:
            embeddings = self._delegate_batch([chunk["text"] for chunk in chunks])
        else:
            embeddings = self.embed_batch([chunk["text"] for chunk in chunks])

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        return chunks

    def _delegate_batch(self, texts: list[str]) -> list[list[float]]:
        batch_method = getattr(self.embedding_model, "embed_batch", None)
        if callable(batch_method):
            embeddings = batch_method(texts)
            if self._valid_embedding_batch(embeddings, len(texts)):
                return embeddings

        embed_method = getattr(self.embedding_model, "embed", None)
        if not callable(embed_method):
            raise TypeError("embedder must provide embed_batch(texts) or embed(text)")

        return [
            embed_method(text) if self._prepare_text(text) else []
            for text in texts
        ]

    def _valid_embedding_batch(self, embeddings, expected_count: int) -> bool:
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            return False

        return all(
            isinstance(embedding, list)
            and all(isinstance(value, (float, int)) for value in embedding)
            for embedding in embeddings
        )

    def _prepare_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()
        if not text:
            return ""

        if len(text) > self.max_input_chars:
            logger.warning(
                "Embedding input exceeded %s characters and was truncated.",
                self.max_input_chars,
            )
            return text[: self.max_input_chars]

        return text

    def _client(self):
        if self.client is not None:
            return self.client

        if not self.api_key:
            if not self._missing_key_warned:
                logger.warning("OPENAI_API_KEY is missing; embeddings will be empty.")
                self._missing_key_warned = True
            return None

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key)
        return self.client

    def _create_embeddings(self, client, texts: list[str]):
        request = {
            "model": self.model_name,
            "input": texts,
        }
        if self.dimensions is not None:
            request["dimensions"] = self.dimensions

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return client.embeddings.create(**request)
            except Exception as error:
                last_error = error
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError("Failed to create embeddings after retries.") from last_error
