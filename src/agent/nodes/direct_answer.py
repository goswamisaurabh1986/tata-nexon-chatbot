"""Direct answer node for simple conversational turns.

Purpose:
    Answer safe non-retrieval messages such as greetings, thanks,
    acknowledgements, goodbyes, and capability questions without touching the
    retrieval pipeline.

Inputs:
    AgentState containing ``query`` and, optionally, an injected LLM. The LLM
    may support ``with_structured_output(AgentResponse)`` for natural direct
    replies.

Outputs:
    AgentState with fresh ``generation`` and structured ``response`` fields,
    cleared retrieval/citation state, and ``route`` set to ``final``.

Graph role:
    This node handles the router's ``direct_answer`` route. It prevents simple
    follow-up turns like "Okay, thanks" from reusing a previous retrieval
    answer stored in memory.
"""

import logging
from typing import Any, Optional

from src.agent.schemas import AgentResponse
from src.agent.state import AgentState


logger = logging.getLogger(__name__)


class DirectAnswer:
    """Generate concise direct responses for simple conversation."""

    SYSTEM_PROMPT = """You are the direct-answer node for a Tata Nexon assistant.

Reply briefly to greetings, thanks, acknowledgements, goodbyes, and capability
questions. Do not retrieve brochure context and do not make factual vehicle
claims. Invite the user to ask about Tata Nexon topics such as safety,
features, performance, warranty, service, price, variants, or specifications.

Return an AgentResponse. Use no sources for these simple conversational turns.
"""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """Create a direct answer node.

        Args:
            llm: Optional LangChain-style chat model supporting structured
                output with ``AgentResponse``.
        """
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        """Generate a direct answer for the current graph state.

        Args:
            state: Current graph state routed to direct answering.

        Returns:
            Updated AgentState from ``direct_answer_node``.
        """
        return direct_answer_node(state, self.llm)


def direct_answer_node(state: AgentState, llm: Optional[Any] = None) -> AgentState:
    """Answer a simple conversational turn without retrieval.

    Args:
        state: Current graph state containing the user query.
        llm: Optional structured-output LLM. When omitted or failing, a
            deterministic direct answer is used.

    Returns:
        Updated AgentState with stale retrieval and answer fields replaced by a
        fresh direct response.
    """
    active_llm = llm or state.get("llm")
    reasoning_steps = [
        *state.get("reasoning_steps", []),
        "Direct answer generated for simple conversational turn.",
    ]
    response = _generate_response(
        query=state.get("query", ""),
        llm=active_llm,
        reasoning_steps=reasoning_steps,
    )

    logger.info("Generated direct answer for simple conversational turn.")
    return {
        **state,
        "direct_answer": True,
        "generation": response.answer,
        "response": response,
        "citations": [],
        "retrieved_chunks": [],
        "graded_chunks": [],
        "is_grounded": True,
        "hallucination_pass": True,
        "confidence": response.confidence,
        "route": "final",
        "reasoning_steps": reasoning_steps,
    }


def _generate_response(
    query: str,
    llm: Optional[Any],
    reasoning_steps: list[str],
) -> AgentResponse:
    """Generate a structured direct response.

    Args:
        query: User query text.
        llm: Optional structured-output LLM.
        reasoning_steps: Reasoning trail to attach to the response.

    Returns:
        AgentResponse with no citations and a direct conversational answer.
    """
    if llm is not None:
        try:
            structured_llm = llm.with_structured_output(AgentResponse)
            raw_response = structured_llm.invoke(
                [
                    ("system", DirectAnswer.SYSTEM_PROMPT),
                    ("human", f"User query: {query}"),
                ]
            )
            return _normalize_response(_coerce_response(raw_response), reasoning_steps)
        except Exception as error:
            logger.warning(
                "Direct answer LLM failed; using fallback response. Error: %s",
                error,
                exc_info=True,
            )

    return _fallback_response(query, reasoning_steps)


def _fallback_response(query: str, reasoning_steps: list[str]) -> AgentResponse:
    """Build a deterministic response for common conversational turns.

    Args:
        query: User query text.
        reasoning_steps: Reasoning trail to attach to the response.

    Returns:
        AgentResponse with a short direct answer and no sources.
    """
    lower_query = (query or "").strip().lower()
    if any(word in lower_query for word in ("thank", "thanks", "thx")):
        answer = "You're welcome. Happy to help with Tata Nexon questions."
    elif any(word in lower_query for word in ("bye", "goodbye", "see you")):
        answer = "Goodbye. Feel free to come back with more Tata Nexon questions."
    elif "what can you do" in lower_query or "who are you" in lower_query or lower_query == "help":
        answer = (
            "I can help with Tata Nexon questions about safety, features, "
            "performance, warranty, service, price, variants, and specifications."
        )
    else:
        answer = "Hi. Ask me anything about the Tata Nexon."

    return AgentResponse(
        answer=answer,
        sources=[],
        confidence=0.85,
        is_grounded=True,
        reasoning_steps=reasoning_steps,
        route="final",
    )


def _normalize_response(response: AgentResponse, reasoning_steps: list[str]) -> AgentResponse:
    """Constrain LLM output to the direct-answer contract.

    Args:
        response: Raw structured response.
        reasoning_steps: Reasoning trail to attach.

    Returns:
        AgentResponse with no sources, clamped confidence, and final route.
    """
    return response.model_copy(
        update={
            "sources": [],
            "confidence": _clamp_confidence(response.confidence),
            "is_grounded": True,
            "reasoning_steps": reasoning_steps,
            "route": "final",
        }
    )


def _coerce_response(raw_response: Any) -> AgentResponse:
    """Convert raw structured-output content into AgentResponse.

    Args:
        raw_response: Pydantic model or mapping returned by the LLM.

    Returns:
        Validated AgentResponse.
    """
    if isinstance(raw_response, AgentResponse):
        return raw_response
    return AgentResponse.model_validate(raw_response)


def _clamp_confidence(confidence: float) -> float:
    """Clamp confidence into the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, confidence))
