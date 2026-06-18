"""Grounding and hallucination checker node for the LangGraph agent.

Purpose:
    Verify that a generated answer is fully supported by retrieved context and
    route unsupported answers back to generation.

Inputs:
    AgentState containing ``generation`` and either ``graded_chunks`` or
    ``retrieved_chunks``. The node requires an injected LLM that supports
    ``with_structured_output(GroundingCheck)``.

Outputs:
    AgentState with ``grounding_check``, ``is_grounded``,
    ``hallucination_pass``, ``guardrail_status``, ``route``, and
    ``reasoning_steps`` updated.

Graph role:
    This node is the quality gate after answer generation. Supported answers
    continue to ``final`` while unsupported or unverifiable answers return to
    ``generate`` for a retry.
"""

import logging
import re
from typing import Any

from src.agent.schemas import GroundingCheck
from src.agent.state import AgentState


logger = logging.getLogger(__name__)

# Temporary UX relaxation: the grounding checker should not block useful answers
# just because the model paraphrased brochure text or declines to provide exact
# metrics that are not in the context. Keep these thresholds low enough to
# accept reasonably supported answers, while still rejecting clear hallucinations
# or major contradictions.
REASONABLE_SUPPORT_OVERLAP_THRESHOLD = 0.2
MIN_GROUNDED_CONFIDENCE = 0.25


class GroundingChecker:
    """Validate that a generated answer is grounded in retrieved context.

    The checker uses a structured LLM response so downstream graph logic can
    rely on boolean grounding status, claim lists, and a textual rationale.
    """

    SYSTEM_PROMPT = (
        "You are a grounding and hallucination checker for a Tata Nexon RAG "
        "assistant. Compare the generated answer against the provided retrieved "
        "chunks. Use a practical, user-friendly RAG standard: accept answers that "
        "are reasonably supported by the chunks, including faithful paraphrases, "
        "general topic summaries, and minor wording differences. Allow general "
        "performance descriptions when retrieved chunks support the performance, "
        "engine, driving mode, powertrain, or feature topic. Do not fail an answer "
        "just because exact metrics are unavailable if the answer clearly says "
        "the provided context does not include those exact numbers. Fail grounding "
        "only for clear hallucinations, invented specific metrics, major "
        "unsupported claims, or important contradictions. List only major "
        "unsupported claims."
    )

    def __init__(self, llm: Any) -> None:
        """Create a grounding checker.

        Args:
            llm: LangChain-style chat model supporting structured output with
                the ``GroundingCheck`` Pydantic schema.
        """
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        """Run grounding validation for the current graph state."""
        return grounding_checker_node(state, self.llm)


def grounding_checker_node(
    state: AgentState,
    llm: Any,
    max_generation_attempts: int = 2,
) -> AgentState:
    """Check generated answer grounding and route to final or regeneration.

    Args:
        state: Current graph state containing generated answer and context.
        llm: Chat model used to produce a ``GroundingCheck``.
        max_generation_attempts: Maximum answer generation attempts before the
            graph stops retrying and asks the user for clarification.

    Returns:
        Updated AgentState. Missing context or generation is treated as a failed
        grounding check so the graph retries generation or clarifies upstream.
    """
    generation = state.get("generation", "")
    chunks = state.get("graded_chunks", []) or state.get("retrieved_chunks", [])

    if not generation or not chunks:
        check = GroundingCheck(
            is_grounded=False,
            confidence=0.0,
            supported_claims=[],
            unsupported_claims=["Missing generated answer or grounding context."],
            reasoning="Missing generation or grounding chunks.",
        )
        return _state_with_grounding_result(
            state,
            check,
            max_generation_attempts=max_generation_attempts,
        )

    try:
        check = _run_structured_grounding_check(
            llm=llm,
            query=state.get("query", ""),
            generation=generation,
            chunks=chunks,
        )
    except Exception as error:
        logger.warning(
            "Grounding check failed; using conservative failed result. Error: %s",
            error,
            exc_info=True,
        )
        check = GroundingCheck(
            is_grounded=False,
            confidence=0.0,
            supported_claims=[],
            unsupported_claims=["Grounding checker could not validate the answer."],
            reasoning="Grounding checker failed; answer needs regeneration.",
        )

    return _state_with_grounding_result(
        state,
        check,
        max_generation_attempts=max_generation_attempts,
    )


def _run_structured_grounding_check(
    llm: Any,
    query: str,
    generation: str,
    chunks: list[dict[str, Any]],
) -> GroundingCheck:
    """Ask the LLM to compare answer claims against retrieved chunks."""
    structured_llm = llm.with_structured_output(GroundingCheck)
    raw_check = structured_llm.invoke(
        [
            ("system", GroundingChecker.SYSTEM_PROMPT),
            (
                "human",
                (
                    f"User query: {query}\n\n"
                    f"Generated answer:\n{generation}\n\n"
                    f"Retrieved chunks:\n{_context_from_chunks(chunks)}"
                ),
            ),
        ]
    )
    return _coerce_grounding_check(raw_check)


def _state_with_grounding_result(
    state: AgentState,
    check: GroundingCheck,
    max_generation_attempts: int,
) -> AgentState:
    """Attach a grounding result and choose the next graph route."""
    chunks = state.get("graded_chunks", []) or state.get("retrieved_chunks", [])
    passed = _grounding_passed(check, chunks)
    attempts = int(state.get("generation_attempts", 0))
    route = _route_for_grounding_result(
        passed=passed,
        attempts=attempts,
        max_generation_attempts=max_generation_attempts,
    )
    return {
        **state,
        "grounding_check": check,
        "is_grounded": passed,
        "hallucination_pass": passed,
        "route": route,
        "guardrail_status": {
            **state.get("guardrail_status", {}),
            "grounded": passed,
            "hallucination_free": passed,
        },
        "reasoning_steps": [
            *state.get("reasoning_steps", []),
            _reasoning_step(check, passed),
            *_retry_reasoning_steps(
                passed=passed,
                attempts=attempts,
                max_generation_attempts=max_generation_attempts,
            ),
        ],
    }


def _grounding_passed(
    check: GroundingCheck,
    chunks: list[dict[str, Any]] | None = None,
) -> bool:
    """Return whether the grounding result is good enough to finalize.

    This is intentionally relaxed for now: a result can pass when the answer is
    reasonably supported by retrieved chunks, even if the LLM checker lists minor
    paraphrase-level unsupported claims. Clear hallucinations still fail.
    """
    if check.is_grounded and not check.unsupported_claims:
        return True

    if not check.unsupported_claims:
        return check.confidence >= MIN_GROUNDED_CONFIDENCE or bool(check.supported_claims)

    if not chunks:
        return False

    context_text = " ".join(
        [
            *check.supported_claims,
            *[chunk.get("text", "") for chunk in chunks],
        ]
    )
    context_tokens = _tokens(context_text)
    major_unsupported_claims = [
        claim
        for claim in check.unsupported_claims
        if _is_major_unsupported_claim(claim, context_tokens, context_text)
    ]
    if major_unsupported_claims:
        return False

    return (
        check.is_grounded
        or check.confidence >= MIN_GROUNDED_CONFIDENCE
        or bool(check.supported_claims)
        or _has_topic_support(context_tokens)
    )


def _is_major_unsupported_claim(
    claim: str,
    context_tokens: set[str],
    context_text: str,
) -> bool:
    """Return whether a claim has too little support to be accepted."""
    if _is_missing_metric_disclaimer(claim):
        return False
    if _looks_like_invented_capability(claim):
        return True
    if _is_general_topic_claim(claim) and _has_topic_support(context_tokens):
        return False
    if _has_clear_contradiction_language(claim, context_text):
        return True

    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return False

    overlap = claim_tokens.intersection(context_tokens)
    return len(overlap) / len(claim_tokens) < REASONABLE_SUPPORT_OVERLAP_THRESHOLD


def _looks_like_invented_capability(claim: str) -> bool:
    """Return whether a claim names a suspicious unsupported capability."""
    lower_claim = claim.lower()
    invented_markers = (
        "racing mode",
        "autonomous racing",
        "flying",
        "self-driving",
        "self driving",
        "autopilot",
        "bulletproof",
    )
    return any(marker in lower_claim for marker in invented_markers)


def _is_missing_metric_disclaimer(claim: str) -> bool:
    """Return whether a claim safely says exact details are unavailable."""
    lower_claim = claim.lower()
    unavailable_markers = (
        "not available",
        "not provided",
        "not specified",
        "does not include",
        "do not include",
        "not in the provided context",
        "unavailable in the provided context",
        "exact numbers",
        "exact metrics",
        "specific metrics",
        "specific numbers",
    )
    has_unavailable_marker = any(marker in lower_claim for marker in unavailable_markers)
    metric_terms = (
        "horsepower",
        "torque",
        "mileage",
        "acceleration",
        "top speed",
        "power",
        "numbers",
        "metrics",
        "figures",
        "specifications",
    )
    return has_unavailable_marker and any(term in lower_claim for term in metric_terms)


def _is_general_topic_claim(claim: str) -> bool:
    """Return whether a claim is a broad automotive topic summary."""
    claim_tokens = _tokens(claim)
    return bool(
        claim_tokens.intersection(
            {
                "performance",
                "engine",
                "turbo",
                "powertrain",
                "drive",
                "driving",
                "mode",
                "modes",
                "safety",
                "features",
                "variant",
                "variants",
            }
        )
    )


def _has_topic_support(context_tokens: set[str]) -> bool:
    """Return whether context has enough car-topic words for broad support."""
    topic_terms = {
        "nexon",
        "performance",
        "engine",
        "turbo",
        "revotron",
        "revotorq",
        "powertrain",
        "drive",
        "driving",
        "mode",
        "modes",
        "eco",
        "city",
        "sports",
        "safety",
        "features",
    }
    return len(context_tokens.intersection(topic_terms)) >= 2


def _has_clear_contradiction_language(claim: str, context_text: str) -> bool:
    """Return whether a claim appears to contradict context outright."""
    lower_claim = claim.lower()
    lower_context = context_text.lower()
    contradiction_markers = (
        "contradict",
        "contradicted",
        "opposite",
        "conflicts with",
        "incorrectly states",
    )
    return any(marker in lower_claim for marker in contradiction_markers) and bool(lower_context)


def _tokens(text: str) -> set[str]:
    """Tokenize text for lightweight grounding fallback checks."""
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "for",
        "in",
        "is",
        "it",
        "of",
        "or",
        "such",
        "that",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _reasoning_step(check: GroundingCheck, passed: bool) -> str:
    """Build a human-readable grounding decision for reasoning history."""
    status = "passed" if passed else "failed"
    return f"Grounding check {status}: {check.reasoning}"


def _route_for_grounding_result(
    passed: bool,
    attempts: int,
    max_generation_attempts: int,
) -> str:
    """Choose final, retry, or clarify after grounding."""
    if passed:
        return "final"
    if attempts >= max_generation_attempts:
        return "clarify"
    return "generate"


def _retry_reasoning_steps(
    passed: bool,
    attempts: int,
    max_generation_attempts: int,
) -> list[str]:
    """Explain retry decisions after failed grounding."""
    if passed:
        return []
    if attempts >= max_generation_attempts:
        return [
            (
                "Maximum generation attempts reached after grounding failure; "
                "routing to clarification."
            )
        ]
    return [
        (
            "Grounding failed; retrying answer generation "
            f"(attempt {attempts + 1} of {max_generation_attempts})."
        )
    ]


def _coerce_grounding_check(raw_check: Any) -> GroundingCheck:
    """Convert raw structured-output content into GroundingCheck."""
    if isinstance(raw_check, GroundingCheck):
        return raw_check
    return GroundingCheck.model_validate(raw_check)


def _context_from_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format chunks with source, text, and metadata for claim checking."""
    return "\n\n".join(
        (
            f"Source: {chunk.get('citation_id')}\n"
            f"Text: {chunk.get('text', '')}\n"
            f"Metadata: {chunk.get('metadata', {})}"
        )
        for chunk in chunks
    )
