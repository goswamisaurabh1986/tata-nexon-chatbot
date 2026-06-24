from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured analysis used to decide how the agent should route a query."""

    intent: str = Field(
        default="unknown",
        description="High-level intent, such as vehicle_features or out_of_scope.",
    )
    is_answerable: bool = Field(
        default=False,
        description="Whether the query can be answered by the Tata Nexon assistant.",
    )
    needs_retrieval: bool = Field(
        default=False,
        description="Whether the query requires retrieved brochure/product context.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the routing decision.",
    )
    required_topics: list[str] = Field(
        default_factory=list,
        description="Product topics needed to answer the query.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation for the classification.",
    )


class GroundingCheck(BaseModel):
    """Structured verification that a generated answer is supported by context."""

    is_grounded: bool = Field(
        default=False,
        description="Whether the answer is supported by retrieved chunks.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the grounding decision.",
    )
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Brief grounding rationale.")


class GuardrailDecision(BaseModel):
    """Structured safety decision returned by input and output guardrails."""

    is_safe: bool = Field(
        default=False,
        description="Whether the content is safe to continue through the graph.",
    )
    is_blocked: bool = Field(
        default=True,
        description="Whether the guardrail should block or reject the content.",
    )
    category: Literal[
        "safe",
        "prompt_injection",
        "harmful",
        "abusive",
        "nsfw",
        "illegal",
        "off_topic",
        "toxic",
        "bias",
        "hallucination",
        "citation_missing",
        "empty_output",
        "unsafe",
        "comparison",
    ] = Field(
        default="unsafe",
        description="Primary guardrail category for the decision.",
    )
    reason: str = Field(
        default="",
        description="Concise explanation for the guardrail decision.",
    )
    severity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Severity of the guardrail concern.",
    )
    blocked_reason: Optional[str] = Field(
        default=None,
        description="Detailed blocking reason when the content is rejected.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the guardrail decision.",
    )


class InputGuardrailResult(GuardrailDecision):
    """Structured input-specific guardrail result."""


class OutputGuardrailResult(GuardrailDecision):
    """Structured output-specific guardrail result."""


class ClarificationResponse(BaseModel):
    """Structured clarification message generated when the agent needs more detail."""

    message: str = Field(
        description="Polite user-facing clarification question or message.",
    )
    suggested_questions: list[str] = Field(
        default_factory=list,
        description="Helpful examples the user can ask next.",
    )
    reason: str = Field(
        default="",
        description="Why clarification is needed.",
    )


class ChunkGradeResult(BaseModel):
    """Structured LLM assessment of a retrieved chunk's usefulness."""

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How relevant the chunk is to the specific user query.",
    )
    is_relevant: bool = Field(
        default=False,
        description="Whether the chunk is useful enough to support answer generation.",
    )
    explanation: str = Field(
        default="",
        validation_alias=AliasChoices("explanation", "reasoning"),
        description="Brief reason for the relevance score.",
    )

    @property
    def reasoning(self) -> str:
        """Backward-compatible alias for older grading code."""
        return self.explanation


class AgentResponse(BaseModel):
    """Final structured response returned by the LangGraph agent."""

    answer: str = Field(description="User-facing answer or refusal message.")
    sources: list[str] = Field(
        default_factory=list,
        description="Citation IDs used to ground the answer.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall response confidence.",
    )
    is_grounded: bool = Field(
        default=False,
        description="Whether the response passed grounding checks.",
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Internal steps taken to produce the answer.",
    )
    guardrail_status: dict[str, bool] = Field(
        default_factory=dict,
        description="Boolean guardrail outcomes keyed by guardrail name.",
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Reason for refusal when the agent cannot answer.",
    )
    route: Optional[Literal[
        "simple",
        "retrieval",
        "direct_answer",
        "clarify",
        "refuse",
        "generate",
        "final",
        "rewrite",
    ]] = Field(
        default=None,
        description="Final route that produced the response.",
    )
