"""Context grading node for the LangGraph agent.

Purpose:
    Evaluate retrieved chunks for relevance before the agent spends tokens on
    answer generation.

Inputs:
    AgentState containing ``query`` and ``retrieved_chunks``. The node also
    accepts an optional structured-output LLM grader and relevance thresholds.

Outputs:
    AgentState with ``graded_chunks``, ``route``, and ``reasoning_steps``
    updated. Chunks receive ``relevance_score``, ``is_relevant``, and
    ``grade_explanation`` fields.

Graph role:
    This node is the retrieval quality gate. It sends the graph to
    ``generate`` when enough relevant context is available, or to ``clarify``
    when retrieval did not provide sufficient support.
"""

import logging
from typing import Any

from src.agent.schemas import ChunkGradeResult
from src.agent.state import AgentState


logger = logging.getLogger(__name__)

# Temporarily relaxed for better UX while maintaining basic safety.
DEFAULT_MIN_RELEVANCE_SCORE = 0.45


class GradeNode:
    """Grade retrieved chunks for relevance before answer generation.

    The node prefers LLM-based structured grading. When no grader is supplied,
    or when grading fails, it falls back to the retriever similarity score so
    the graph remains usable in fast unit tests and offline runs.
    """

    SYSTEM_PROMPT = """You are the retrieval grader for a Tata Nexon brochure assistant.

Evaluate retrieved chunks against the user's query using a lenient RAG standard.
Accept chunks that are reasonably related to the same Tata Nexon feature, variant,
system, or brochure topic, even if they only partially answer the query. The goal
is to reduce false negatives before answer generation while still filtering clearly
unrelated chunks.

Use these scoring guidelines:
- 0.90-1.00: directly answers the query with highly useful Tata Nexon information
- 0.60-0.89: clearly related and useful, even if incomplete
- 0.40-0.59: reasonably related to the query topic and worth keeping
- 0.00-0.39: unrelated, generic, or not useful for the query

Set is_relevant=true for chunks scoring 0.40 or higher, or whenever the chunk
mentions the same feature/topic the user asked about. Return only a structured
ChunkGradeResult with relevance_score, is_relevant, and explanation.
"""

    def __init__(
        self,
        grader_llm: Any | None,
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
        min_relevant_chunks: int = 1,
        filter_threshold: float = 0.0,
    ) -> None:
        """Create a grade node.

        Args:
            grader_llm: Optional LangChain-style model supporting
                ``with_structured_output(ChunkGradeResult)``.
            min_relevance_score: Score a chunk must meet to count as useful.
            min_relevant_chunks: Number of useful chunks required to generate.
            filter_threshold: Minimum score required to keep a chunk at all.
        """
        self.grader_llm = grader_llm
        self.min_relevance_score = min_relevance_score
        self.min_relevant_chunks = min_relevant_chunks
        self.filter_threshold = filter_threshold

    def run(self, state: AgentState) -> AgentState:
        """Grade chunks for the current graph state."""
        return grade_node(
            state,
            self.grader_llm,
            min_relevance_score=self.min_relevance_score,
            min_relevant_chunks=self.min_relevant_chunks,
            filter_threshold=self.filter_threshold,
        )


def grade_node(
    state: AgentState,
    grader_llm: Any | None,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    min_relevant_chunks: int = 1,
    filter_threshold: float = 0.0,
) -> AgentState:
    """Grade retrieved chunks and decide whether generation can proceed.

    Args:
        state: Current graph state containing query and retrieved chunks.
        grader_llm: Optional structured-output LLM used for relevance grading.
        min_relevance_score: Passing score for a useful chunk.
        min_relevant_chunks: Minimum passing chunks required for generation.
        filter_threshold: Low-score chunks below this value are discarded.

    Returns:
        Updated AgentState with graded chunks and next route.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return _state_without_chunks(state)

    query = state.get("query", "")
    graded_chunks = _grade_chunks(
        chunks,
        query=query,
        grader_llm=grader_llm,
        min_relevance_score=min_relevance_score,
    )
    filtered_chunks = _filter_low_relevance(graded_chunks, filter_threshold)
    passing_count = _passing_count(filtered_chunks, min_relevance_score)
    route = _route_for_grade(passing_count, min_relevant_chunks)

    logger.info(
        "Graded %d chunks; %d passed relevance threshold.",
        len(graded_chunks),
        passing_count,
    )
    return {
        **state,
        "graded_chunks": filtered_chunks,
        "route": route,
        "reasoning_steps": [
            *state.get("reasoning_steps", []),
            f"Graded retrieval context: {passing_count} chunks passed relevance threshold.",
        ],
    }


def _state_without_chunks(state: AgentState) -> AgentState:
    """Build a clarify route when no chunks are available to grade."""
    return {
        **state,
        "graded_chunks": [],
        "route": "clarify",
        "reasoning_steps": [
            *state.get("reasoning_steps", []),
            "No retrieved chunks available for grading.",
        ],
    }


def _grade_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    grader_llm: Any | None,
    min_relevance_score: float,
) -> list[dict[str, Any]]:
    """Grade all chunks with the LLM when possible, otherwise use fallback."""
    if grader_llm is None:
        return [
            _chunk_with_grade(
                chunk,
                _fallback_grade(
                    chunk,
                    "LLM grader unavailable; used retriever similarity score.",
                    min_relevance_score=min_relevance_score,
                ),
            )
            for chunk in chunks
        ]

    try:
        return _grade_chunks_with_llm(
            chunks,
            query=query,
            grader_llm=grader_llm,
            min_relevance_score=min_relevance_score,
        )
    except Exception as error:
        logger.warning(
            "LLM chunk grading failed; falling back to retriever scores. Error: %s",
            error,
            exc_info=True,
        )
        return [
            _chunk_with_grade(
                chunk,
                _fallback_grade(
                    chunk,
                    "LLM grader failed; used retriever similarity score.",
                    min_relevance_score=min_relevance_score,
                ),
            )
            for chunk in chunks
        ]


def _grade_chunks_with_llm(
    chunks: list[dict[str, Any]],
    query: str,
    grader_llm: Any,
    min_relevance_score: float,
) -> list[dict[str, Any]]:
    """Run structured LLM grading, using batch mode when the model supports it."""
    structured_llm = grader_llm.with_structured_output(ChunkGradeResult)
    prompts = [_messages_for_chunk(query, chunk) for chunk in chunks]

    # LangChain chat models often expose batch(); simple fakes in unit tests may
    # only implement invoke(), so both paths are supported.
    batch = getattr(structured_llm, "batch", None)
    if callable(batch):
        grade_results = batch(prompts)
    else:
        grade_results = [structured_llm.invoke(prompt) for prompt in prompts]

    return _merge_llm_grades(
        chunks,
        grade_results,
        min_relevance_score=min_relevance_score,
    )


def _merge_llm_grades(
    chunks: list[dict[str, Any]],
    grade_results: list[Any],
    min_relevance_score: float,
) -> list[dict[str, Any]]:
    """Attach LLM grades to chunks while preserving one output per input chunk."""
    graded_chunks = []
    for index, chunk in enumerate(chunks):
        try:
            grade = _coerce_grade_result(grade_results[index])
        except (IndexError, TypeError, ValueError) as error:
            logger.warning(
                "Invalid LLM grade for chunk %d; falling back to retriever score. Error: %s",
                index,
                error,
                exc_info=True,
            )
            grade = _fallback_grade(
                chunk,
                "Invalid LLM grade; used retriever similarity score.",
                min_relevance_score=min_relevance_score,
            )
        graded_chunks.append(_chunk_with_grade(chunk, grade))

    return graded_chunks


def _messages_for_chunk(query: str, chunk: dict[str, Any]) -> list[tuple[str, str]]:
    """Create the structured grading prompt for one retrieved chunk."""
    return [
        ("system", GradeNode.SYSTEM_PROMPT),
        (
            "human",
            (
                f"User query: {query}\n\n"
                f"Retrieved chunk text:\n{chunk.get('text', '')}\n\n"
                f"Chunk metadata: {chunk.get('metadata', {})}\n"
                f"Retriever similarity score: {chunk.get('score', 0.0)}"
            ),
        ),
    ]


def _coerce_grade_result(result: Any) -> ChunkGradeResult:
    """Convert a raw structured-output response into ChunkGradeResult."""
    if isinstance(result, ChunkGradeResult):
        return result
    return ChunkGradeResult.model_validate(result)


def _fallback_grade(
    chunk: dict[str, Any],
    explanation: str,
    min_relevance_score: float,
) -> ChunkGradeResult:
    """Create a grade from the retriever similarity score."""
    relevance_score = _retriever_score(chunk)
    return ChunkGradeResult(
        relevance_score=relevance_score,
        is_relevant=relevance_score >= min_relevance_score,
        explanation=explanation,
    )


def _chunk_with_grade(
    chunk: dict[str, Any],
    grade: ChunkGradeResult,
) -> dict[str, Any]:
    """Return a chunk copy enriched with grading fields."""
    return {
        **chunk,
        "relevance_score": _clamp_score(grade.relevance_score),
        "is_relevant": grade.is_relevant,
        "grade_explanation": grade.explanation,
    }


def _filter_low_relevance(
    chunks: list[dict[str, Any]],
    filter_threshold: float,
) -> list[dict[str, Any]]:
    """Remove chunks below the configured filter threshold."""
    return [
        chunk
        for chunk in chunks
        if chunk["relevance_score"] >= filter_threshold
    ]


def _route_for_grade(passing_count: int, min_relevant_chunks: int) -> str:
    """Route to generation only when enough chunks pass relevance checks."""
    return "generate" if passing_count >= min_relevant_chunks else "clarify"


def _retriever_score(chunk: dict[str, Any]) -> float:
    """Read and clamp the retriever score from a chunk."""
    return _clamp_score(float(chunk.get("score", 0.0)))


def _clamp_score(score: float) -> float:
    """Clamp a numeric score into the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, score))


def _passing_count(
    chunks: list[dict[str, Any]],
    min_relevance_score: float,
) -> int:
    """Count chunks that meet the generation relevance threshold."""
    return sum(
        1
        for chunk in chunks
        if chunk.get("is_relevant") or chunk["relevance_score"] >= min_relevance_score
    )
