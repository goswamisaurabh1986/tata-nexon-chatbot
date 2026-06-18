"""Answer generation node for the LangGraph agent.

Purpose:
    Produce the user-facing answer from graded retrieval context using a
    structured ``AgentResponse`` schema.

Inputs:
    AgentState containing ``query`` plus either ``graded_chunks`` or
    ``retrieved_chunks``. The node requires an injected LLM that supports
    ``with_structured_output(AgentResponse)``.

Outputs:
    AgentState with ``generation``, structured ``response``, ``citations``,
    ``confidence``, ``is_grounded``, ``route``, and ``reasoning_steps`` updated.

Graph role:
    This node is the first answer-producing step. It is intentionally grounded
    to retrieved context and leaves final hallucination verification to the
    GroundingChecker node.
"""

import logging
from typing import Any

from src.agent.schemas import AgentResponse
from src.agent.state import AgentState


logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate a cited answer from graded retrieval context.

    The wrapper stores the LLM dependency so the graph can call ``run(state)``
    while tests can inject deterministic fake LLMs.
    """

    SYSTEM_PROMPT = (
        "You are a helpful Tata Nexon brochure assistant. Answer only from the "
        "provided context. Include citations from the supplied source IDs. Prefer "
        "specific facts and feature names over broad marketing claims, but be "
        "conservative with exact numbers. Do not invent horsepower, torque, "
        "mileage, acceleration, price, variant, or safety counts unless those "
        "numbers appear explicitly in the provided context. If exact metrics are "
        "not available, say so plainly and provide only the general information "
        "that is supported by the chunks. Do not add superlatives, summaries, or "
        "inferred benefits unless the provided context directly supports them. If "
        "the context is insufficient for the user's requested detail, refuse or "
        "ask for clarification instead of guessing."
    )

    def __init__(self, llm: Any) -> None:
        """Create an answer generator.

        Args:
            llm: LangChain-style chat model supporting structured output with
                the ``AgentResponse`` Pydantic schema.
        """
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        """Generate an answer for the current graph state."""
        return answer_generator_node(state, self.llm)


def answer_generator_node(state: AgentState, llm: Any) -> AgentState:
    """Generate a final answer using retrieved context and structured output.

    Args:
        state: Current graph state containing a query and relevant chunks.
        llm: Chat model used to produce an ``AgentResponse``.

    Returns:
        Updated AgentState. If no chunks are available, the node returns a
        structured refusal and routes to ``clarify``.
    """
    query = state["query"]
    chunks = state.get("graded_chunks", []) or state.get("retrieved_chunks", [])

    if not chunks:
        return _refusal_state(state)

    generation_attempts = int(state.get("generation_attempts", 0)) + 1
    citations = _citation_ids(state, chunks)
    context = _context_from_chunks(chunks)
    response = _generate_structured_response(
        llm=llm,
        query=query,
        context=context,
        citations=citations,
        existing_steps=state.get("reasoning_steps", []),
    )

    logger.info(
        "Generated answer with confidence %.2f from %d chunks.",
        response.confidence,
        len(chunks),
    )
    return {
        **state,
        "generation_attempts": generation_attempts,
        "generation": response.answer,
        "response": response,
        "citations": [{"citation_id": citation_id} for citation_id in response.sources],
        "is_grounded": response.is_grounded,
        "confidence": response.confidence,
        "reasoning_steps": [
            *state.get("reasoning_steps", []),
            "Generated grounded answer from graded chunks.",
        ],
        "route": "final",
    }


def _refusal_state(state: AgentState) -> AgentState:
    """Build a low-confidence structured refusal for missing context."""
    message = "I don't have enough information to answer this question."
    reasoning_steps = [
        *state.get("reasoning_steps", []),
        "No relevant chunks found; refusal generated.",
    ]
    response = AgentResponse(
        answer=message,
        sources=[],
        confidence=0.0,
        is_grounded=False,
        reasoning_steps=reasoning_steps,
        refusal_reason="No relevant chunks were available.",
        route="clarify",
    )
    return {
        **state,
        "generation": message,
        "response": response,
        "citations": [],
        "is_grounded": False,
        "confidence": response.confidence,
        "route": "clarify",
        "reasoning_steps": reasoning_steps,
    }


def _generate_structured_response(
    llm: Any,
    query: str,
    context: str,
    citations: list[str],
    existing_steps: list[str],
) -> AgentResponse:
    """Call the LLM and normalize its response to AgentResponse.

    The LLM is instructed to answer only from context. The node then overwrites
    sources with the citation IDs available in state, which keeps citations
    deterministic and prevents the model from inventing source IDs.
    """
    structured_llm = llm.with_structured_output(AgentResponse)
    raw_response = structured_llm.invoke(
        [
            ("system", AnswerGenerator.SYSTEM_PROMPT),
            (
                "human",
                (
                    f"Query: {query}\n\n"
                    f"Available citation IDs: {citations}\n\n"
                    f"Context:\n{context}"
                ),
            ),
        ]
    )
    response = _coerce_agent_response(raw_response)
    return response.model_copy(
        update={
            "sources": citations,
            "confidence": _clamp_confidence(response.confidence),
            "reasoning_steps": [
                *existing_steps,
                "Generated grounded answer from graded chunks.",
            ],
        }
    )


def _coerce_agent_response(response: Any) -> AgentResponse:
    """Convert raw structured-output content into AgentResponse."""
    if isinstance(response, AgentResponse):
        return response
    return AgentResponse.model_validate(response)


def _context_from_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a compact context block for the LLM."""
    return "\n\n".join(
        f"Source: {chunk.get('citation_id')}\n{chunk.get('text', '')}"
        for chunk in chunks
    )


def _citation_ids(state: AgentState, chunks: list[dict[str, Any]]) -> list[str]:
    """Return citation IDs from state first, falling back to chunk IDs."""
    state_citations = state.get("citations", [])
    if state_citations:
        return [
            citation["citation_id"] if isinstance(citation, dict) else str(citation)
            for citation in state_citations
        ]

    return [
        chunk["citation_id"]
        for chunk in chunks
        if chunk.get("citation_id")
    ]


def _clamp_confidence(confidence: float) -> float:
    """Clamp confidence into the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, confidence))
