"""Output guardrail node for the LangGraph agent.

Purpose:
    Validate the final answer immediately before it leaves the graph. This node
    performs a secondary safety, citation, and grounding check after answer
    generation and grounding.

Inputs:
    AgentState containing ``generation`` and/or ``response`` plus optional
    citations, grounding flags, and an optional LLM supporting
    ``with_structured_output(GuardrailDecision)``.

Outputs:
    AgentState with ``output_guardrail``, ``guardrail_status``, ``route``,
    ``response``, ``generation``, and ``reasoning_steps`` updated.

Graph role:
    This is the final safety gate. Safe answers continue as ``final``. Blocked
    answers are replaced with a polite refusal so unsafe or ungrounded text is
    never returned to the user.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.agent.schemas import AgentResponse, GuardrailDecision, OutputGuardrailResult
from src.agent.state import AgentState


logger = logging.getLogger(__name__)


class OutputGuardrail:
    """Validate generated answers before final delivery."""

    SYSTEM_PROMPT = """You are the output guardrail for a Tata Nexon RAG assistant.

Review the final answer before it is shown to the user.

Approve concise, factual, cited Tata Nexon answers that are grounded in the
retrieved context. Reject output only when there is a clear problem:
- toxic, hateful, abusive, harassing, biased, or discriminatory content
- harmful instructions or unsafe operational content
- obvious hallucinations, source fabrication, or contradictions
- factual Tata Nexon claims without citation IDs
- empty, unusable, or malformed output

If blocking, provide a short reason and severity. Do not rewrite the user's
answer yourself; the application will replace blocked output with a safe
refusal.

Return a GuardrailDecision with is_safe, reason, severity, blocked_reason,
category, is_blocked, and confidence.
"""

    TOXIC_PATTERNS = (
        r"\babusive\b",
        r"\bhateful\b",
        r"\bhate speech\b",
        r"\binsult",
        r"\bharass",
        r"\bslur\b",
    )
    HARMFUL_PATTERNS = (
        r"\bbuild (?:a )?bomb\b",
        r"\bmake (?:a )?bomb\b",
        r"\bsteps to (?:build|make) (?:a )?bomb\b",
        r"\bmalware\b",
        r"\bexploit\b",
        r"\bsteal\b.*\bpassword\b",
        r"\bpoison\b",
    )
    BIAS_PATTERNS = (
        r"\bonly idiots\b",
        r"\bthat group\b",
        r"\binferior\b",
        r"\bsuperior race\b",
        r"\bdiscriminat",
        r"\bbigoted\b",
    )
    HALLUCINATION_PATTERNS = (
        r"\bi made up\b",
        r"\bwithout any source\b",
        r"\bfabricated\b",
        r"\bcan fly\b",
        r"\bflying car\b",
        r"\bautonomous racing\b",
        r"\bracing mode\b",
    )
    FACTUAL_CLAIM_TERMS = (
        "tata nexon",
        "nexon",
        "airbags",
        "engine",
        "mileage",
        "features",
        "safety",
        "variant",
        "price",
        "warranty",
        "coverage",
        "service",
        "includes",
        "has ",
        "offers",
    )

    def __init__(self, llm: Optional[Any] = None) -> None:
        """Create an output guardrail.

        Args:
            llm: Optional LangChain-style chat model for structured safety
                review after deterministic checks pass.
        """
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        """Run final output safety checks for the current graph state.

        Args:
            state: Agent state containing generated output.

        Returns:
            Updated AgentState from ``output_guardrail_node``.
        """
        return output_guardrail_node(state, self.llm)


def output_guardrail_node(
    state: AgentState,
    llm: Optional[Any] = None,
) -> AgentState:
    """Validate generated output and update final routing.

    Args:
        state: Current graph state containing generation or response.
        llm: Optional structured-output LLM. When omitted, local rules are used.

    Returns:
        Updated AgentState. Blocked output is replaced with a safe refusal
        response because output_guardrail is terminal in the graph.
    """
    output_text = _output_text(state)
    decision = _decide_output_safety(output_text, state, llm=llm or state.get("llm"))
    result = OutputGuardrailResult.model_validate(decision.model_dump())
    safe = result.is_safe and not result.is_blocked
    reasoning_steps = [
        *state.get("reasoning_steps", []),
        _reasoning_step("Output guardrail", result),
    ]
    guardrail_status = {
        **state.get("guardrail_status", {}),
        "output_safe": safe,
    }

    logger.info(
        "Output guardrail %s: category=%s severity=%s",
        "passed" if safe else "blocked",
        result.category,
        result.severity,
    )

    if not safe:
        return _blocked_output_state(
            state,
            result=result,
            guardrail_status=guardrail_status,
            reasoning_steps=reasoning_steps,
        )

    return {
        **state,
        "output_guardrail": result,
        "route": "final",
        "guardrail_status": guardrail_status,
        "reasoning_steps": reasoning_steps,
    }


def _decide_output_safety(
    output_text: str,
    state: AgentState,
    llm: Optional[Any],
) -> GuardrailDecision:
    """Decide whether generated output is safe to return.

    Args:
        output_text: Final answer text to validate.
        state: Agent state containing citations and grounding status.
        llm: Optional structured-output LLM used after local checks pass.

    Returns:
        GuardrailDecision describing whether the output is safe or blocked.
    """
    local_decision = _local_output_decision(output_text, state)
    if local_decision.is_blocked:
        return local_decision
    if state.get("direct_answer"):
        return local_decision

    if llm is None:
        return local_decision

    try:
        structured_llm = llm.with_structured_output(GuardrailDecision)
        raw_decision = structured_llm.invoke(
            [
                ("system", OutputGuardrail.SYSTEM_PROMPT),
                (
                    "human",
                    (
                        f"Final answer:\n{output_text}\n\n"
                        f"Grounded: {_is_grounded(state)}\n"
                        f"Citations: {_source_ids(state)}"
                    ),
                ),
            ]
        )
        return _coerce_decision(raw_decision)
    except Exception as error:
        logger.warning(
            "Output guardrail LLM failed; using local decision. Error: %s",
            error,
            exc_info=True,
        )
        return local_decision


def _local_output_decision(output_text: str, state: AgentState) -> GuardrailDecision:
    """Evaluate final output with deterministic safety and quality checks.

    Args:
        output_text: Final answer text to validate.
        state: Agent state containing citations and grounding status.

    Returns:
        GuardrailDecision from local output checks.
    """
    text = (output_text or "").strip()
    lower_text = text.lower()

    if not text:
        return _blocked(
            "empty_output",
            "Output is empty or unusable.",
            "Output is empty or unusable.",
            "medium",
            1.0,
        )
    if _matches_any(lower_text, OutputGuardrail.HARMFUL_PATTERNS):
        return _blocked(
            "harmful",
            "Unsafe harmful content detected in output.",
            "Unsafe harmful content detected in output.",
            "high",
            0.95,
        )
    if _matches_any(lower_text, OutputGuardrail.TOXIC_PATTERNS):
        return _blocked(
            "toxic",
            "Toxic or abusive content detected in output.",
            "Toxic or abusive content detected in output.",
            "high",
            0.9,
        )
    if _matches_any(lower_text, OutputGuardrail.BIAS_PATTERNS):
        return _blocked(
            "bias",
            "Biased or discriminatory content detected in output.",
            "Biased or discriminatory content detected in output.",
            "high",
            0.9,
        )
    if _matches_any(lower_text, OutputGuardrail.HALLUCINATION_PATTERNS):
        return _blocked(
            "hallucination",
            "Suspicious unsupported output detected.",
            "Suspicious unsupported output detected.",
            "high",
            0.88,
        )
    if state.get("direct_answer"):
        return GuardrailDecision(
            is_safe=True,
            is_blocked=False,
            category="safe",
            reason="Direct conversational answer is safe.",
            severity="low",
            blocked_reason=None,
            confidence=0.85,
        )
    if _is_grounded(state) is False:
        return _blocked(
            "hallucination",
            "Final answer is not grounded in retrieved context.",
            "Secondary hallucination check failed.",
            "high",
            0.88,
        )
    if _has_factual_claims(text) and not _source_ids(state):
        return _blocked(
            "citation_missing",
            "Factual answer is missing citations.",
            "Final answer contains factual claims but no citation IDs.",
            "medium",
            0.86,
        )

    return GuardrailDecision(
        is_safe=True,
        is_blocked=False,
        category="safe",
        reason="Output is safe.",
        severity="low",
        blocked_reason=None,
        confidence=0.85,
    )


def _blocked_output_state(
    state: AgentState,
    result: OutputGuardrailResult,
    guardrail_status: dict[str, bool],
    reasoning_steps: list[str],
) -> AgentState:
    """Replace blocked output with a clean refusal response.

    Args:
        state: Current graph state.
        result: Structured output guardrail result.
        guardrail_status: Updated guardrail status flags.
        reasoning_steps: Reasoning trace including the block reason.

    Returns:
        AgentState with unsafe output replaced by a safe refusal.
    """
    refusal_message = _refusal_message(result)
    response = AgentResponse(
        answer=refusal_message,
        sources=[],
        confidence=0.0,
        is_grounded=False,
        reasoning_steps=reasoning_steps,
        guardrail_status=guardrail_status,
        refusal_reason=result.blocked_reason or result.reason,
        route="refuse",
    )
    return {
        **state,
        "output_guardrail": result,
        "generation": refusal_message,
        "response": response,
        "citations": [],
        "is_grounded": False,
        "hallucination_pass": False,
        "confidence": 0.0,
        "route": "refuse",
        "guardrail_status": guardrail_status,
        "reasoning_steps": reasoning_steps,
    }


def _refusal_message(result: OutputGuardrailResult) -> str:
    """Build a polite refusal message for blocked final output.

    Args:
        result: Structured output guardrail result.

    Returns:
        User-facing refusal text tailored to the output failure category.
    """
    if result.category == "citation_missing":
        return "I can't return that answer because it is missing the citations needed to support its claims."
    if result.category == "hallucination":
        return "I can't return that answer because it was not sufficiently supported by the retrieved Tata Nexon context."
    if result.category == "empty_output":
        return "I can't return an answer because the generated response was empty."
    return "I can't return that answer because it did not pass the final safety check."


def _output_text(state: AgentState) -> str:
    """Extract final answer text from generation or response fields.

    Args:
        state: Current graph state.

    Returns:
        Final answer text, or an empty string when none exists.
    """
    generation = state.get("generation")
    if generation:
        return str(generation)

    response = state.get("response")
    if isinstance(response, AgentResponse):
        return response.answer
    return str(response or "")


def _source_ids(state: AgentState) -> list[str]:
    """Extract citation IDs from response or citation state.

    Args:
        state: Current graph state.

    Returns:
        Citation IDs attached to the generated answer.
    """
    response = state.get("response")
    if isinstance(response, AgentResponse) and response.sources:
        return response.sources

    sources: list[str] = []
    for citation in state.get("citations", []):
        if isinstance(citation, dict):
            citation_id = citation.get("citation_id") or citation.get("source")
            if citation_id:
                sources.append(str(citation_id))
        elif citation:
            sources.append(str(citation))
    return sources


def _is_grounded(state: AgentState) -> Optional[bool]:
    """Return grounding status from state or structured response.

    Args:
        state: Current graph state.

    Returns:
        Grounding status when available, otherwise ``None``.
    """
    if "is_grounded" in state:
        return bool(state.get("is_grounded"))

    response = state.get("response")
    if isinstance(response, AgentResponse):
        return response.is_grounded
    return None


def _blocked(
    category: str,
    reason: str,
    blocked_reason: str,
    severity: str,
    confidence: float,
) -> GuardrailDecision:
    """Build a blocked guardrail decision.

    Args:
        category: Guardrail category.
        reason: Short reason for the decision.
        blocked_reason: Detailed block reason.
        severity: Severity level.
        confidence: Confidence score from zero to one.

    Returns:
        GuardrailDecision marked unsafe and blocked.
    """
    return GuardrailDecision(
        is_safe=False,
        is_blocked=True,
        category=category,
        reason=reason,
        severity=severity,
        blocked_reason=blocked_reason,
        confidence=confidence,
    )


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Check text against a list of regex patterns.

    Args:
        text: Text to inspect.
        patterns: Case-insensitive regex patterns.

    Returns:
        True when any pattern matches.
    """
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _has_factual_claims(text: str) -> bool:
    """Return whether output appears to make factual Tata Nexon claims.

    Args:
        text: Output text to inspect.

    Returns:
        True when the text contains likely factual Tata Nexon claim terms.
    """
    lower_text = text.lower()
    return any(term in lower_text for term in OutputGuardrail.FACTUAL_CLAIM_TERMS)


def _coerce_decision(raw_decision: Any) -> GuardrailDecision:
    """Convert raw structured-output content into GuardrailDecision.

    Args:
        raw_decision: Pydantic model or mapping returned by the LLM.

    Returns:
        Validated GuardrailDecision instance.
    """
    if isinstance(raw_decision, GuardrailDecision):
        return raw_decision
    return GuardrailDecision.model_validate(raw_decision)


def _reasoning_step(prefix: str, decision: GuardrailDecision) -> str:
    """Build a concise reasoning step for the guardrail decision.

    Args:
        prefix: Human-readable node name.
        decision: Guardrail decision to summarize.

    Returns:
        One-line reasoning trace entry.
    """
    status = "passed" if decision.is_safe and not decision.is_blocked else "blocked"
    return f"{prefix} {status}: {decision.reason}"
