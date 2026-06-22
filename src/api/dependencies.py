"""FastAPI dependency and runtime construction helpers.

The API layer owns HTTP concerns and application wiring, not domain behavior.
This module centralizes dependency lookup from ``app.state`` and builds the
long-lived runtime objects used by route handlers: settings, checkpointer,
LangGraph agent, retriever, and ingestion processor.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional

from fastapi import Request

from src.agent.graph import build_agent_graph
from src.agent.memory import get_checkpointer
from src.config.settings import Settings
from src.ingestion.embedder import Embedder
from src.ingestion.processor import IngestionProcessor
from src.ingestion.storer import VectorStorer
from src.retrieval.retriever import Retriever


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiRuntime:
    """Container for long-lived API dependencies.

    Attributes:
        settings: Typed application configuration.
        checkpointer: LangGraph checkpointer used for multi-turn memory.
        llm: ChatOpenAI model, or ``None`` when unavailable.
        embedder: OpenAI embedding component, or ``None`` when unavailable.
        vector_store: ChromaDB-backed vector store component.
        retriever: Retrieval component used by the agent graph.
        agent_graph: Compiled LangGraph application.
        ingestion_processor: Processor used by admin ingestion endpoints.
    """

    settings: Settings
    checkpointer: Any
    llm: Optional[Any]
    embedder: Optional[Embedder]
    vector_store: VectorStorer
    retriever: Retriever
    agent_graph: Any
    ingestion_processor: IngestionProcessor

    def as_state_items(self) -> dict[str, Any]:
        """Return values ready to attach to ``FastAPI.app.state``."""
        return {
            "settings": self.settings,
            "checkpointer": self.checkpointer,
            "llm": self.llm,
            "embedder": self.embedder,
            "vector_store": self.vector_store,
            "retriever": self.retriever,
            "agent_graph": self.agent_graph,
            "ingestion_processor": self.ingestion_processor,
        }

def get_settings(request: Request) -> Settings:
    """Return typed settings stored on FastAPI application state."""
    return request.app.state.settings


def get_agent_graph(request: Request) -> Any:
    """Return the compiled LangGraph application stored on app state."""
    return request.app.state.agent_graph


def get_ingestion_processor(request: Request) -> IngestionProcessor:
    """Return the ingestion processor stored on app state."""
    return request.app.state.ingestion_processor


def get_document_registry(request: Request) -> list:
    """Return the process-local document registry used by admin endpoints."""
    return request.app.state.documents


def build_runtime_dependencies(
    settings: Settings,
    graph: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    ingestion_processor: Optional[IngestionProcessor] = None,
) -> ApiRuntime:
    """Build runtime dependencies used by the FastAPI lifespan hook.

    Args:
        settings: Typed application configuration.
        graph: Optional precompiled graph supplied by tests or callers.
        checkpointer: Optional checkpointer override.
        ingestion_processor: Optional ingestion processor override.

    Returns:
        ApiRuntime with all dependencies initialized exactly once.
    """
    active_checkpointer = checkpointer or _build_checkpointer(settings)
    llm = _build_llm(settings)
    embedder = _build_embedder(settings)
    vector_store = _build_vector_store(settings)
    retriever = _build_retriever(embedder=embedder, vector_store=vector_store)
    active_graph = graph or build_agent_graph(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        settings=settings,
        checkpointer=active_checkpointer,
    )
    active_processor = ingestion_processor or _build_ingestion_processor(
        embedder=embedder,
        storer=vector_store,
    )

    return ApiRuntime(
        settings=settings,
        checkpointer=active_checkpointer,
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        agent_graph=active_graph,
        ingestion_processor=active_processor,
    )


def _build_checkpointer(settings: Settings) -> Any:
    """Build the configured LangGraph checkpointer."""
    return get_checkpointer(
        db_path=settings.memory.sqlite_path,
        backend=settings.memory.backend,
    )


def _build_llm(settings: Settings) -> Optional[Any]:
    """Build the real ChatOpenAI model when credentials exist."""
    if not settings.llm.api_key:
        logger.warning("OPENAI_API_KEY is missing; ChatOpenAI will not be loaded.")
        return None

    from langchain_openai import ChatOpenAI

    chat_model = ChatOpenAI(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        temperature=0,
        timeout=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
    )
    logger.info("Loaded ChatOpenAI model '%s'.", settings.llm.chat_model)
    return StructuredChatOpenAI(chat_model)


class StructuredChatOpenAI:
    """ChatOpenAI adapter that uses schema-compatible structured output.

    LangChain's default OpenAI structured-output method is strict JSON schema.
    Some of this project's Pydantic models include defaults that OpenAI's strict
    parser rejects. ``method="function_calling"`` keeps the same real
    ChatOpenAI model while supporting the existing schemas.
    """

    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model

    def with_structured_output(self, schema: type[Any]) -> Any:
        """Return a structured-output runnable for the supplied schema."""
        return self.chat_model.with_structured_output(
            schema,
            method="function_calling",
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped ChatOpenAI instance."""
        return getattr(self.chat_model, name)


def _build_embedder(settings: Settings) -> Optional[Embedder]:
    """Build the real OpenAI embedder from typed settings."""
    if not settings.llm.api_key:
        logger.warning("OPENAI_API_KEY is missing; Embedder will not be loaded.")
        return None

    embedder = Embedder(
        model_name=settings.llm.embedding_model,
        api_key=settings.llm.api_key,
        dimensions=settings.llm.embedding_dimensions,
    )
    logger.info("Loaded OpenAI embedder model '%s'.", settings.llm.embedding_model)
    return embedder


def _build_vector_store(settings: Settings) -> VectorStorer:
    """Build the real ChromaDB vector store from typed settings."""
    vector_store = VectorStorer(
        collection_name=settings.retrieval.collection_name,
        persist_directory=settings.retrieval.persist_directory,
        embedding_dimension=settings.retrieval.embedding_dimension,
    )
    logger.info(
        "Loaded Chroma vector store collection='%s' persist_directory='%s' count=%s.",
        settings.retrieval.collection_name,
        settings.retrieval.persist_directory,
        _collection_count(vector_store),
    )
    return vector_store


def _build_retriever(
    embedder: Optional[Embedder],
    vector_store: VectorStorer,
) -> Retriever:
    """Build the retrieval component from injected real components."""
    if embedder is None:
        logger.warning("Retriever loaded without embedder; retrieval will use fallback results.")
        return Retriever()

    logger.info(
        "Loaded Retriever with %s and %s.",
        type(embedder).__name__,
        type(vector_store).__name__,
    )
    return Retriever(embedder=embedder, vector_store=vector_store)


def _build_ingestion_processor(
    embedder: Optional[Embedder],
    storer: VectorStorer,
) -> IngestionProcessor:
    """Build the admin ingestion processor from injected real components."""
    logger.info(
        "Loaded IngestionProcessor with embedder=%s storer=%s.",
        type(embedder).__name__ if embedder is not None else "None",
        type(storer).__name__,
    )
    return IngestionProcessor(embedder=embedder, storer=storer)


def _collection_count(vector_store: VectorStorer) -> str:
    """Return the current collection count for startup diagnostics."""
    try:
        return str(vector_store.collection.count())
    except Exception as error:
        return f"unavailable ({error})"
