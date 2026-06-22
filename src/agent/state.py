from typing import Annotated, Any, Literal, Optional, TypedDict, Union

try:
    from langgraph.graph.message import add_messages
except ImportError:

    def add_messages(left: Optional[list], right: Optional[list]) -> list:
        """Fallback reducer used when LangGraph is not installed."""
        return [*(left or []), *(right or [])]

from src.agent.schemas import (
    AgentResponse,
    GroundingCheck,
    InputGuardrailResult,
    OutputGuardrailResult,
    QueryAnalysis,
)


Route = Literal[
    "simple",
    "retrieval",
    "clarify",
    "refuse",
    "generate",
    "final",
    "rewrite",
]


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph agent nodes."""

    # Conversation input
    messages: Annotated[list, add_messages]
    query: str

    # Dependencies supplied by graph construction or tests
    llm: Any
    retriever: Any

    # Routing and control
    route: Route
    query_analysis: QueryAnalysis
    generation_attempts: int
    rewrite_count: int
    reasoning_steps: list[str]

    # Retrieval
    retrieved_chunks: list[dict[str, Any]]
    graded_chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    top_k: int
    similarity_threshold: float

    # Generation
    generation: str
    response: Union[AgentResponse, str]

    # Guardrails and quality
    input_guardrail: InputGuardrailResult
    output_guardrail: OutputGuardrailResult
    grounding_check: GroundingCheck
    hallucination_pass: bool
    is_grounded: bool
    guardrail_status: dict[str, bool]

    # Error handling
    error: Optional[str]


def initial_agent_state(query: str, messages: Optional[list] = None) -> AgentState:
    """Create a state object with sensible defaults for graph execution."""
    return {
        "messages": messages or [],
        "query": query,
        "route": "simple",
        "generation_attempts": 0,
        "rewrite_count": 0,
        "reasoning_steps": [],
        "retrieved_chunks": [],
        "graded_chunks": [],
        "citations": [],
        "generation": "",
        "guardrail_status": {},
        "hallucination_pass": False,
        "is_grounded": False,
        "error": None,
    }
