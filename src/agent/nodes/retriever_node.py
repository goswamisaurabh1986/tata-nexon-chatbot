"""Retrieval node for the LangGraph agent.

Purpose:
    Call the retrieval/search layer with the current user query and attach the
    returned ranked chunks to the graph state.

Inputs:
    AgentState containing ``query`` and a ``RetrieverProtocol`` dependency,
    plus an optional ``top_k`` limit configured on the node wrapper.

Outputs:
    AgentState with ``retrieved_chunks``, ``citations``, ``route``,
    ``reasoning_steps``, and ``error`` updated.

Graph role:
    This node bridges the agent and the retrieval module. It does not rank or
    grade chunks itself; it captures retrieval results and prepares citations
    for downstream grading, answer generation, and grounding.
"""

import logging
from typing import Any, Protocol

from src.agent.state import AgentState


logger = logging.getLogger(__name__)


class RetrieverProtocol(Protocol):
    """Protocol for the retrieval dependency used by RetrieverNode.

    Implementations are expected to embed the query, search the vector store,
    and return chunks with text, metadata, score, and citation IDs.
    """

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return up to ``top_k`` retrieved chunks for a query.

        Args:
            query: User query to search for.
            top_k: Maximum number of chunks to return.

        Returns:
            A list of chunk dictionaries sorted by relevance.
        """


class RetrieverNode:
    """Object wrapper for the retrieval node.

    The wrapper stores the retriever dependency and runtime configuration so the
    graph can call ``run(state)`` without re-passing those dependencies.
    """

    def __init__(self, retriever: RetrieverProtocol, top_k: int = 5) -> None:
        """Create a retriever node.

        Args:
            retriever: Retrieval component implementing ``RetrieverProtocol``.
            top_k: Maximum chunks requested from the retrieval layer.
        """
        self.retriever = retriever
        self.top_k = top_k

    def run(self, state: AgentState) -> AgentState:
        """Run retrieval for the current graph state."""
        return retriever_node(state, retriever=self.retriever, top_k=self.top_k)


def retriever_node(
    state: AgentState,
    retriever: RetrieverProtocol,
    top_k: int = 5,
) -> AgentState:
    """Retrieve relevant chunks and attach retrieval metadata to state.

    Args:
        state: Current graph state containing ``query``.
        retriever: Search dependency used to retrieve ranked chunks.
        top_k: Maximum number of chunks to request.

    Returns:
        Updated AgentState containing retrieval results and route metadata.
    """
    query = state["query"]
    try:
        logger.info("Retrieving context for query: %s", query[:100])
        chunks = retriever.retrieve(query, top_k=top_k)
        return _state_with_retrieval_results(state, chunks)
    except Exception as error:
        logger.error("Retriever node failed: %s", error, exc_info=True)
        return _state_with_retrieval_error(state, str(error))


def _state_with_retrieval_results(
    state: AgentState,
    chunks: list[dict[str, Any]],
) -> AgentState:
    """Build state updates for successful retrieval results."""
    citations = _extract_citations(chunks)
    return {
        **state,
        "retrieved_chunks": chunks,
        "citations": citations,
        "route": _route_for_chunks(chunks),
        "reasoning_steps": _append_reasoning_step(state, chunks),
        "error": None,
    }


def _state_with_retrieval_error(state: AgentState, error: str) -> AgentState:
    """Build a conservative state update when retrieval raises an error."""
    return {
        **state,
        "retrieved_chunks": [],
        "citations": [],
        "route": "clarify",
        "reasoning_steps": [
            *state.get("reasoning_steps", []),
            "Retrieval failed; clarification may be required.",
        ],
        "error": error,
    }


def _extract_citations(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract citation IDs from retrieved chunks for downstream nodes."""
    return [
        {"citation_id": chunk["citation_id"]}
        for chunk in chunks
        if chunk.get("citation_id")
    ]


def _route_for_chunks(chunks: list[dict[str, Any]]) -> str:
    """Choose the next route based on whether retrieval returned context."""
    return "retrieval" if chunks else "clarify"


def _append_reasoning_step(
    state: AgentState,
    chunks: list[dict[str, Any]],
) -> list[str]:
    """Append a retrieval summary to existing reasoning steps."""
    return [
        *state.get("reasoning_steps", []),
        f"Retrieved {len(chunks)} relevant chunks.",
    ]
