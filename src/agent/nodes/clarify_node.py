"""Clarification node for the LangGraph agent.

Purpose:
    Generate a polite, specific clarification message when the graph cannot
    safely or confidently answer the user's query.

Inputs:
    AgentState containing the original ``query`` and optional context such as
    ``retrieved_chunks`` and ``reasoning_steps``. The node can use an injected
    LLM supporting ``with_structured_output(ClarificationResponse)``.

Outputs:
    AgentState with ``clarification``, ``generation``, structured ``response``,
    ``route``, and ``reasoning_steps`` updated.

Graph role:
    This is a terminal support node for ``clarify`` routes. It keeps failure
    states helpful by asking the user for the exact Tata Nexon model, feature,
    variant, or brochure detail needed to continue.
"""

import logging
from typing import Any, Optional

from src.agent.schemas import AgentResponse, ClarificationResponse
from src.agent.state import AgentState


logger = logging.getLogger(__name__)


class ClarifyNode:
    """Generate natural clarification prompts for incomplete agent states."""

    SYSTEM_PROMPT = """You are the clarification node for a Tata Nexon brochure assistant.

When the agent cannot answer confidently, write a concise, polite clarification
message. Ask for the missing detail that would help answer from Tata Nexon
brochure/product knowledge.

Guidelines:
- Do not answer the user's original question directly.
- Ask one clear clarification question.
- Mention Tata Nexon when the query is vague or about Tata cars generally.
- Use available reasoning steps and retrieved context to infer what is missing.
- Include 1-3 useful suggested follow-up questions.

Return a structured ClarificationResponse.
"""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """Create a clarification node.

        Args:
            llm: Optional LangChain-style chat model for structured
                clarification generation.
        """
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        """Generate clarification for the current graph state.

        Args:
            state: Agent state routed to clarification.

        Returns:
            Updated AgentState from ``clarify_node``.
        """
        return clarify_node(state, self.llm)


def clarify_node(state: AgentState, llm: Optional[Any] = None) -> AgentState:
    """Generate a clarification response using an LLM or fallback template.

    Args:
        state: Current graph state that needs clarification.
        llm: Optional structured-output LLM. When omitted or failing, a
            deterministic clarification is used.

    Returns:
        Updated AgentState with clarification message and structured response.
    """
    active_llm = llm or state.get("llm")
    clarification = _generate_clarification(state, active_llm)
    reasoning_steps = [
        *state.get("reasoning_steps", []),
        f"Clarification generated: {clarification.reason}",
    ]
    response = AgentResponse(
        answer=clarification.message,
        sources=[],
        confidence=0.0,
        is_grounded=False,
        reasoning_steps=reasoning_steps,
        refusal_reason=clarification.reason,
        route="clarify",
    )

    return {
        **state,
        "clarification": clarification,
        "generation": clarification.message,
        "response": response,
        "route": "clarify",
        "reasoning_steps": reasoning_steps,
    }


def _generate_clarification(
    state: AgentState,
    llm: Optional[Any],
) -> ClarificationResponse:
    """Generate clarification through structured LLM output when possible.

    Args:
        state: Agent state containing query, context, and reasoning.
        llm: Optional structured-output LLM.

    Returns:
        ClarificationResponse from router-supplied message, LLM, or fallback.
    """
    if state.get("clarification_message"):
        return ClarificationResponse(
            message=str(state["clarification_message"]),
            suggested_questions=[
                "Tell me about Tata Nexon safety features.",
                "What is Tata Nexon performance like?",
                "What is Tata Nexon warranty coverage?",
            ],
            reason="The query asks for comparison data outside the Tata Nexon-only scope.",
        )

    if llm is None:
        return _fallback_clarification(state)

    try:
        structured_llm = llm.with_structured_output(ClarificationResponse)
        raw_response = structured_llm.invoke(_messages_for_clarification(state))
        return _coerce_clarification(raw_response)
    except Exception as error:
        logger.warning(
            "Clarification LLM failed; using fallback clarification. Error: %s",
            error,
            exc_info=True,
        )
        return _fallback_clarification(state)


def _messages_for_clarification(state: AgentState) -> list[tuple[str, str]]:
    """Build the structured clarification prompt from query and context.

    Args:
        state: Agent state needing clarification.

    Returns:
        Chat message tuples for a structured-output LLM call.
    """
    return [
        ("system", ClarifyNode.SYSTEM_PROMPT),
        (
            "human",
            (
                f"Original query: {state.get('query', '')}\n\n"
                f"Retrieved context:\n{_context_summary(state)}\n\n"
                f"Reasoning steps:\n{state.get('reasoning_steps', [])}\n\n"
                f"Current error: {state.get('error')}"
            ),
        ),
    ]


def _fallback_clarification(state: AgentState) -> ClarificationResponse:
    """Create a useful clarification without an LLM.

    Args:
        state: Agent state needing clarification.

    Returns:
        Deterministic ClarificationResponse tailored to the query shape.
    """
    query = (state.get("query") or "").strip()
    lower_query = query.lower()

    if "tata cars" in lower_query or ("tata" in lower_query and "nexon" not in lower_query):
        return ClarificationResponse(
            message=(
                "Could you specify whether you mean the Tata Nexon, and which "
                "feature, variant, or specification you want to know about?"
            ),
            suggested_questions=[
                "What are the Tata Nexon safety features?",
                "Which Tata Nexon variant should I compare?",
                "Tell me about Tata Nexon engine specifications.",
            ],
            reason="The query mentions Tata cars broadly and needs a specific model or feature.",
        )

    return ClarificationResponse(
        message=(
            "I need a few more details to answer from Tata Nexon brochure context. "
            "Could you specify the feature, variant, or section you want?"
        ),
        suggested_questions=[
            "What are the Tata Nexon safety features?",
            "What is the Tata Nexon mileage?",
            "Tell me about Tata Nexon performance features.",
        ],
        reason="Insufficient Tata Nexon context is available to answer reliably.",
    )


def _context_summary(state: AgentState) -> str:
    """Return a compact summary of retrieved or graded chunks.

    Args:
        state: Agent state containing retrieved or graded chunks.

    Returns:
        Short context string suitable for clarification prompting.
    """
    chunks = state.get("graded_chunks", []) or state.get("retrieved_chunks", [])
    if not chunks:
        return "No retrieved chunks available."

    return "\n\n".join(
        (
            f"Source: {chunk.get('citation_id')}\n"
            f"Text: {chunk.get('text', '')}\n"
            f"Metadata: {chunk.get('metadata', {})}"
        )
        for chunk in chunks[:3]
    )


def _coerce_clarification(response: Any) -> ClarificationResponse:
    """Convert raw structured-output content into ClarificationResponse.

    Args:
        response: Pydantic model or mapping returned by the LLM.

    Returns:
        Validated ClarificationResponse instance.
    """
    if isinstance(response, ClarificationResponse):
        return response
    return ClarificationResponse.model_validate(response)
