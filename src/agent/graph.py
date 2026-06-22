"""Main LangGraph assembly for the Tata Nexon agent.

Purpose:
    Build and compile the complete agent workflow from the tested node
    functions. The graph wires safety, query analysis, retrieval, grading,
    answer generation, grounding, and final output validation into one
    executable LangGraph application.

Inputs:
    ``build_agent_graph`` accepts optional dependencies such as an LLM,
    retriever, grader LLM, checkpointer, and threshold configuration. Runtime
    state must use ``AgentState``.

Outputs:
    A compiled LangGraph application by default. Callers can request the graph
    and active checkpointer tuple by passing ``return_checkpointer=True``.
    Tests can request the raw ``StateGraph`` by passing ``compile_graph=False``.

Graph role:
    This module is the orchestration layer. Individual nodes own domain logic;
    this file owns sequencing and conditional routing.
"""

import logging
from typing import Any, Optional

from src.config.settings import Settings, load_settings
from src.agent.memory import get_checkpointer
from src.agent.nodes.answer_generator import answer_generator_node
from src.agent.nodes.clarify_node import clarify_node
from src.agent.nodes.grade_node import DEFAULT_MIN_RELEVANCE_SCORE, grade_node
from src.agent.nodes.grounding_checker import grounding_checker_node
from src.agent.nodes.input_guardrail import input_guardrail_node
from src.agent.nodes.output_guardrail import output_guardrail_node
from src.agent.nodes.retriever_node import retriever_node
from src.agent.nodes.router_node import router_node
from src.agent.state import AgentState


logger = logging.getLogger(__name__)

INPUT_GUARDRAIL = "input_guardrail"
ROUTER_NODE = "router_node"
RETRIEVER_NODE = "retriever_node"
GRADE_NODE = "grade_node"
ANSWER_GENERATOR_NODE = "answer_generator_node"
GROUNDING_CHECKER = "grounding_checker"
OUTPUT_GUARDRAIL = "output_guardrail"
CLARIFY_NODE = "clarify_node"
END_ROUTE = "end"


def build_agent_graph(
    llm: Optional[Any] = None,
    embedder: Optional[Any] = None,
    vector_store: Optional[Any] = None,
    retriever: Optional[Any] = None,
    grader_llm: Optional[Any] = None,
    top_k: Optional[int] = None,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    min_relevant_chunks: int = 1,
    filter_threshold: float = 0.0,
    max_generation_attempts: Optional[int] = None,
    checkpointer: Optional[Any] = None,
    settings: Optional[Settings] = None,
    use_memory: bool = True,
    return_checkpointer: bool = False,
    compile_graph: bool = True,
):
    """Build the complete LangGraph agent workflow.

    Args:
        llm: Optional shared chat model for analyzer, generation, grounding,
            and guardrails. State-level ``llm`` takes precedence at runtime.
        embedder: Optional query embedder used to construct a retriever when
            ``retriever`` is not supplied.
        vector_store: Optional vector store used to construct a retriever when
            ``retriever`` is not supplied.
        retriever: Optional retrieval dependency used by ``retriever_node``.
        grader_llm: Optional structured-output LLM for context grading. When
            omitted, the graph falls back to ``llm`` or retriever scores.
        top_k: Default number of chunks requested from the retriever. When
            omitted, the value comes from ``Settings``.
        min_relevance_score: Minimum score for a chunk to count as relevant.
        min_relevant_chunks: Minimum relevant chunks required before generation.
        filter_threshold: Minimum score required to keep a graded chunk.
        max_generation_attempts: Reserved upper bound for rewrite/regeneration loops.
            When omitted, the value comes from ``Settings``.
        checkpointer: Optional LangGraph checkpointer. When omitted and
            ``use_memory`` is true, the default local checkpointer is created.
        settings: Optional typed configuration object.
        use_memory: Compile the graph with a checkpointer for multi-turn state.
        return_checkpointer: Return ``(graph, checkpointer)`` when true.
        compile_graph: Return a compiled graph when true; otherwise return the
            mutable ``StateGraph`` for introspection.

    Returns:
        A compiled LangGraph application, ``(application, checkpointer)`` when
        requested, an uncompiled StateGraph, or ``None`` when LangGraph is not
        installed.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        logger.warning("LangGraph is not installed; agent graph cannot be built.")
        return None

    active_settings = settings or load_settings()
    active_top_k = top_k if top_k is not None else active_settings.retrieval.top_k
    active_retriever = retriever or _retriever_from_components(embedder, vector_store)
    active_max_generation_attempts = (
        max_generation_attempts
        if max_generation_attempts is not None
        else active_settings.app.max_generation_attempts
    )

    graph = StateGraph(AgentState)

    graph.add_node(
        INPUT_GUARDRAIL,
        lambda state: input_guardrail_node(state, llm=_state_llm(state, llm)),
    )
    graph.add_node(
        ROUTER_NODE,
        lambda state: router_node(state, llm=_state_llm(state, llm)),
    )
    graph.add_node(
        RETRIEVER_NODE,
        lambda state: _run_retriever_node(
            state,
            retriever=active_retriever,
            top_k=active_top_k,
        ),
    )
    graph.add_node(
        GRADE_NODE,
        lambda state: grade_node(
            state,
            grader_llm=_state_llm(state, grader_llm or llm),
            min_relevance_score=min_relevance_score,
            min_relevant_chunks=min_relevant_chunks,
            filter_threshold=filter_threshold,
        ),
    )
    graph.add_node(
        ANSWER_GENERATOR_NODE,
        lambda state: _run_answer_generator_node(
            state,
            llm=_state_llm(state, llm),
        ),
    )
    graph.add_node(
        GROUNDING_CHECKER,
        lambda state: grounding_checker_node(
            state,
            llm=_state_llm(state, llm),
            max_generation_attempts=active_max_generation_attempts,
        ),
    )
    graph.add_node(
        OUTPUT_GUARDRAIL,
        lambda state: output_guardrail_node(state, llm=_state_llm(state, llm)),
    )
    graph.add_node(
        CLARIFY_NODE,
        lambda state: clarify_node(state, llm=_state_llm(state, llm)),
    )

    # Absolute security entry point. No router, retriever, or generator can run
    # until input_guardrail marks the query safe.
    graph.set_entry_point(INPUT_GUARDRAIL)
    graph.add_conditional_edges(
        INPUT_GUARDRAIL,
        _route_after_input_guardrail,
        {
            ROUTER_NODE: ROUTER_NODE,
            END_ROUTE: END,
        },
    )

    # Router is the main decision maker after input safety. It sends valid
    # Tata Nexon questions into retrieval, unclear safe queries to one terminal
    # clarification, and explicit refusals directly to END.
    graph.add_conditional_edges(
        ROUTER_NODE,
        _route_after_router,
        {
            RETRIEVER_NODE: RETRIEVER_NODE,
            CLARIFY_NODE: CLARIFY_NODE,
            END_ROUTE: END,
        },
    )

    # Retrieval path: retrieve context, grade it, then generate only when the
    # context is sufficient. Poor context asks for clarification once.
    graph.add_edge(RETRIEVER_NODE, GRADE_NODE)
    graph.add_conditional_edges(
        GRADE_NODE,
        _route_after_grade,
        {
            ANSWER_GENERATOR_NODE: ANSWER_GENERATOR_NODE,
            CLARIFY_NODE: CLARIFY_NODE,
            END_ROUTE: END,
        },
    )

    # Generation path: answers are always grounded before the output guardrail.
    # If generation cannot run, it routes to terminal clarification.
    graph.add_conditional_edges(
        ANSWER_GENERATOR_NODE,
        _route_after_generation,
        {
            GROUNDING_CHECKER: GROUNDING_CHECKER,
            CLARIFY_NODE: CLARIFY_NODE,
            END_ROUTE: END,
        },
    )

    # Grounding controls retry. Ungrounded answers retry generation while
    # generation_attempts remains below the configured max, then clarify.
    graph.add_conditional_edges(
        GROUNDING_CHECKER,
        lambda state: _route_after_grounding(
            state,
            max_generation_attempts=active_max_generation_attempts,
        ),
        {
            OUTPUT_GUARDRAIL: OUTPUT_GUARDRAIL,
            ANSWER_GENERATOR_NODE: ANSWER_GENERATOR_NODE,
            CLARIFY_NODE: CLARIFY_NODE,
            END_ROUTE: END,
        },
    )

    # Terminal nodes. The output guardrail is the final safety gate; clarify
    # asks the user for more detail and ends the current turn.
    graph.add_edge(OUTPUT_GUARDRAIL, END)
    graph.add_edge(CLARIFY_NODE, END)

    if not compile_graph:
        return (graph, None) if return_checkpointer else graph

    active_checkpointer = checkpointer
    if use_memory and active_checkpointer is None:
        active_checkpointer = get_checkpointer(
            db_path=active_settings.memory.sqlite_path,
            backend=active_settings.memory.backend,
        )

    compiled_graph = graph.compile(checkpointer=active_checkpointer)
    if return_checkpointer:
        return compiled_graph, active_checkpointer
    return compiled_graph


def _state_llm(state: AgentState, default_llm: Optional[Any]) -> Optional[Any]:
    """Return the runtime LLM dependency.

    Args:
        state: Current graph state, which may contain an injected ``llm``.
        default_llm: LLM supplied when the graph was built.

    Returns:
        State-level LLM when present, otherwise the graph-level default.
    """
    return state.get("llm") or default_llm


def _retriever_from_components(embedder: Optional[Any], vector_store: Optional[Any]) -> Optional[Any]:
    """Create a Retriever from low-level components.

    Args:
        embedder: Query embedder dependency.
        vector_store: Vector store dependency.

    Returns:
        Retriever instance when both dependencies are present; otherwise
        ``None`` so callers can handle missing retrieval explicitly.
    """
    if embedder is None or vector_store is None:
        return None

    from src.retrieval.retriever import Retriever

    logger.info(
        "Injecting real retriever components into graph: embedder=%s, vector_store=%s.",
        type(embedder).__name__,
        type(vector_store).__name__,
    )
    return Retriever(embedder=embedder, vector_store=vector_store)


def _route_after_input_guardrail(state: AgentState) -> str:
    """Route immediately after the input guardrail.

    Args:
        state: Agent state after ``input_guardrail`` has executed.

    Returns:
        ``end`` when input was blocked or unsafe; ``router_node`` otherwise.
    """
    guardrail = state.get("input_guardrail")
    input_safe = state.get("guardrail_status", {}).get("input_safe", True)
    guardrail_blocked = bool(
        getattr(guardrail, "is_blocked", False)
        if guardrail is not None
        else False
    )
    if state.get("route") == "refuse" or guardrail_blocked or not input_safe:
        return END_ROUTE
    return ROUTER_NODE


def _route_after_router(state: AgentState) -> str:
    """Translate router decisions into graph destinations.

    Args:
        state: Agent state after ``router_node``.

    Returns:
        ``retriever_node`` for retrieval, ``clarify_node`` for safe unresolved
        queries, or ``end`` for explicit refusals.
    """
    route = state.get("route")
    if route == "refuse":
        return END_ROUTE
    if route == "retrieval":
        return RETRIEVER_NODE
    return CLARIFY_NODE


def _route_after_grade(state: AgentState) -> str:
    """Choose the next node after context grading.

    Args:
        state: Agent state after ``grade_node``.

    Returns:
        ``answer_generator_node`` for sufficient context, ``clarify_node`` for
        weak context, or ``end`` for refusal routes.
    """
    route = state.get("route")
    if route == "refuse":
        return END_ROUTE
    if route == "generate":
        return ANSWER_GENERATOR_NODE
    return CLARIFY_NODE


def _route_after_generation(state: AgentState) -> str:
    """Choose the next node after answer generation.

    Args:
        state: Agent state after ``answer_generator_node``.

    Returns:
        ``grounding_checker`` for generated answers, ``clarify_node`` when
        generation asks for more detail, or ``end`` on refusal.
    """
    if state.get("route") == "refuse":
        return END_ROUTE
    if state.get("route") == "clarify":
        return CLARIFY_NODE
    return GROUNDING_CHECKER


def _route_after_grounding(
    state: AgentState,
    max_generation_attempts: int,
) -> str:
    """Choose the next node after grounding checks.

    Args:
        state: Agent state after ``grounding_checker``.
        max_generation_attempts: Maximum allowed generation attempts for the
            current turn.

    Returns:
        ``output_guardrail`` for grounded answers, ``answer_generator_node``
        for retryable ungrounded answers, ``clarify_node`` when retries are
        exhausted, or ``end`` on refusal.
    """
    route = state.get("route")
    if route == "refuse":
        return END_ROUTE
    if route == "final" and state.get("hallucination_pass"):
        return OUTPUT_GUARDRAIL
    if route == "generate" and _has_generation_attempts_remaining(
        state,
        max_generation_attempts=max_generation_attempts,
    ):
        return ANSWER_GENERATOR_NODE
    return CLARIFY_NODE


def _has_generation_attempts_remaining(
    state: AgentState,
    max_generation_attempts: int,
) -> bool:
    """Return whether answer generation can be retried.

    Args:
        state: Current graph state.
        max_generation_attempts: Configured retry limit.

    Returns:
        True when the current attempt count is below the retry limit.
    """
    return _generation_attempts(state) < max_generation_attempts


def _generation_attempts(state: AgentState) -> int:
    """Read ``generation_attempts`` defensively from graph state.

    Args:
        state: Current graph state.

    Returns:
        Integer attempt count, defaulting to zero for invalid values.
    """
    try:
        return int(state.get("generation_attempts", 0))
    except (TypeError, ValueError):
        return 0


def _run_retriever_node(
    state: AgentState,
    retriever: Optional[Any],
    top_k: int,
) -> AgentState:
    """Run retrieval with an injected or state-provided retriever.

    Args:
        state: Current graph state.
        retriever: Graph-level retriever dependency.
        top_k: Default number of chunks to request.

    Returns:
        Updated AgentState from ``retriever_node`` or a clarification route
        when no retriever is configured.
    """
    active_retriever = retriever or state.get("retriever")
    if active_retriever is None:
        logger.warning("No retriever configured for retrieval route.")
        return {
            **state,
            "retrieved_chunks": [],
            "citations": [],
            "route": "clarify",
            "error": "Retriever is not configured.",
        }

    return retriever_node(
        state,
        retriever=active_retriever,
        top_k=int(state.get("top_k", top_k)),
    )


def _run_answer_generator_node(state: AgentState, llm: Optional[Any]) -> AgentState:
    """Run answer generation with an injected or state-provided LLM.

    Args:
        state: Current graph state.
        llm: Active LLM dependency resolved for this turn.

    Returns:
        Updated AgentState from ``answer_generator_node`` or a clarification
        state when no LLM is configured.
    """
    if llm is None:
        logger.warning("No LLM configured for answer generation.")
        return clarify_node(
            {
                **state,
                "error": "LLM is not configured.",
            },
            llm=None,
        )

    return answer_generator_node(state, llm=llm)
