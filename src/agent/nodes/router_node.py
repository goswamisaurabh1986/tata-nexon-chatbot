"""Router node for the LangGraph agent.

Purpose:
    Classify the user's query and decide the next graph route after the input
    guardrail has approved the request.

Inputs:
    AgentState with a ``query`` field and, optionally, an injected LangChain
    compatible chat model under ``llm``. A direct ``llm`` argument can also be
    supplied to ``router_node`` for tests or graph construction.

Outputs:
    AgentState with ``query_analysis`` populated as a ``QueryAnalysis`` model
    and ``route`` set to ``retrieval`` or ``clarify``.

Graph role:
    This node is the first decision point after the input guardrail. It decides
    whether a safe query should retrieve brochure context, ask for
    clarification.
"""

import logging
from typing import Any, Optional

from src.agent.nodes.scope import comparison_response, comparison_target
from src.agent.schemas import AgentResponse, QueryAnalysis
from src.agent.state import AgentState


logger = logging.getLogger(__name__)


class Router:
    """Analyze user queries and translate analysis into graph routes.

    This class first attempts an LLM-backed structured classification when an
    LLM is provided. If the LLM is missing or fails, it falls back to local
    product-scope rules so the graph still has a deterministic route decision.
    """

    SYSTEM_PROMPT = """You are the router for a Tata Nexon brochure assistant.

Classify the user's query for a retrieval-augmented Tata Nexon chatbot and
decide whether the graph should retrieve context or clarify.

Answerable queries must be about Tata Nexon brochure or product knowledge, such as
safety, features, specifications, variants, price, engine, mileage, performance,
comfort, interior, exterior, warranty, service schedule, maintenance, booking,
dealer, ownership, warranty coverage, or brochure content. These usually need
retrieval. Because this chatbot is exclusively about Tata Nexon, short queries
like "Tell me about the warranty" or "Tell me the coverage" should be treated
as Tata Nexon questions.

Comparison queries asking to compare Tata Nexon with any other car or model
are out of scope and must not go to retrieval or answer generation. Examples:
"Compare Tata Nexon with Tata Sierra", "Nexon vs Sierra", "Nexon versus
Hyundai Creta", "Is Nexon better than Kia Sonet?", or "difference between
Tata Nexon and Maruti Brezza". Variant-only comparisons within Tata Nexon are
allowed.

Unanswerable queries include weather, sports, cooking, politics, personal advice,
and general knowledge outside the Tata Nexon document scope.

Return a QueryAnalysis object with:
- is_answerable: true only for Tata Nexon product/document questions
- needs_retrieval: true when brochure/product context is needed
- confidence: 0.0 to 1.0
- reasoning: brief reason for the routing decision
"""

    PRODUCT_TERMS = (
        "tata nexon",
        "nexon",
    )
    PRODUCT_TOPICS = (
        "safety",
        "features",
        "spec",
        "specs",
        "specification",
        "specifications",
        "price",
        "variant",
        "variants",
        "engine",
        "mileage",
        "performance",
        "comfort",
        "interior",
        "exterior",
        "brochure",
        "warranty",
        "coverage",
        "covered",
        "cover",
        "service",
        "servicing",
        "schedule",
        "interval",
        "maintenance",
        "cost",
        "pricing",
        "on-road",
        "on road",
        "ex-showroom",
        "ex showroom",
        "booking",
        "dealer",
        "dealership",
        "showroom",
        "delivery",
        "ownership",
        "insurance",
        "emi",
        "finance",
    )

    def __init__(self, llm: Optional[Any] = None) -> None:
        """Create a router.

        Args:
            llm: Optional LangChain-style chat model supporting
                ``with_structured_output(QueryAnalysis)``.
        """
        self.llm = llm

    def analyze(self, query: str) -> QueryAnalysis:
        """Return structured query analysis using the LLM or fallback rules.

        Args:
            query: User query text from the conversation state.

        Returns:
            QueryAnalysis containing answerability, retrieval need, confidence,
            required topics, and a brief rationale.
        """
        query_text = self._normalize_query(query)
        comparison_analysis = self._comparison_analysis(query_text)
        if comparison_analysis is not None:
            logger.info("Router defensively blocked external comparison query.")
            return comparison_analysis

        if self.llm is not None and query_text:
            try:
                logger.debug("Routing query with structured LLM output.")
                return self._analyze_with_llm(query_text)
            except Exception as error:
                logger.warning(
                    "Structured routing failed; using fallback rules. Error: %s",
                    error,
                    exc_info=True,
                )

        logger.debug("Routing query with fallback rules.")
        return self._fallback_analysis(query_text)

    def route_for(self, analysis: QueryAnalysis) -> str:
        """Translate query analysis into the next graph route.

        Args:
            analysis: Structured query analysis produced by ``analyze``.

        Returns:
            ``retrieval`` for answerable Tata Nexon queries; otherwise
            ``clarify`` so the graph can end with a helpful message.
        """
        if not analysis.is_answerable:
            return "clarify"
        return "retrieval"

    def _analyze_with_llm(self, query: str) -> QueryAnalysis:
        """Run structured LLM analysis and coerce the response.

        Args:
            query: Normalized user query.

        Returns:
            QueryAnalysis returned by the structured LLM.
        """
        structured_llm = self.llm.with_structured_output(QueryAnalysis)
        response = structured_llm.invoke(
            [
                ("system", self.SYSTEM_PROMPT),
                ("human", query),
            ]
        )

        if isinstance(response, QueryAnalysis):
            return response
        return QueryAnalysis.model_validate(response)

    def _comparison_analysis(self, query: str) -> Optional[QueryAnalysis]:
        """Build out-of-scope analysis for external comparison queries.

        Args:
            query: Normalized user query.

        Returns:
            QueryAnalysis for comparison requests, or ``None`` when the query
            can continue through normal routing.
        """
        target = comparison_target(query)
        if target is None:
            return None

        return QueryAnalysis(
            intent="comparison_out_of_scope",
            is_answerable=False,
            needs_retrieval=False,
            confidence=0.95,
            required_topics=[target],
            reasoning=(
                "The query asks to compare Tata Nexon with another model, "
                "which is outside this Tata Nexon-only assistant scope."
            ),
        )

    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Classify a query with deterministic local product-scope rules.

        Args:
            query: Normalized user query.

        Returns:
            QueryAnalysis describing the local routing decision.
        """
        if not query:
            return self._empty_query_analysis()

        lower_query = query.lower()
        required_topics = self._required_topics(lower_query)
        if not self._is_product_query(lower_query):
            return self._out_of_scope_analysis()

        if not required_topics:
            required_topics = ["overview"]

        return QueryAnalysis(
            intent="vehicle_features",
            is_answerable=True,
            needs_retrieval=True,
            confidence=0.9,
            required_topics=required_topics,
            reasoning="Query is answerable using Tata Nexon product knowledge.",
        )

    def _empty_query_analysis(self) -> QueryAnalysis:
        """Build the structured response for an empty query.

        Returns:
            QueryAnalysis for empty or whitespace-only input.
        """
        return QueryAnalysis(
            intent="unknown",
            is_answerable=False,
            needs_retrieval=False,
            confidence=0.0,
            required_topics=[],
            reasoning="Empty query cannot be answered.",
        )

    def _out_of_scope_analysis(self) -> QueryAnalysis:
        """Build the structured response for a query outside document scope.

        Returns:
            QueryAnalysis for safe but unsupported topics.
        """
        return QueryAnalysis(
            intent="out_of_scope",
            is_answerable=False,
            needs_retrieval=False,
            confidence=0.2,
            required_topics=[],
            reasoning="Query is outside the Tata Nexon document scope.",
        )

    def _required_topics(self, lower_query: str) -> list[str]:
        """Extract known Tata Nexon product topics from query text.

        Args:
            lower_query: Lower-cased user query.

        Returns:
            Matching product topic keywords.
        """
        return [
            topic
            for topic in self.PRODUCT_TOPICS
            if topic in lower_query
        ]

    def _is_product_query(self, lower_query: str) -> bool:
        """Return whether the query is in scope for Tata Nexon retrieval.

        Args:
            lower_query: Lower-cased user query.

        Returns:
            True when explicit product terms or known product topics appear.
        """
        return any(term in lower_query for term in self.PRODUCT_TERMS) or bool(
            self._required_topics(lower_query)
        )

    def _normalize_query(self, query: Optional[str]) -> str:
        """Normalize optional query text to a stripped string.

        Args:
            query: Raw optional query.

        Returns:
            Trimmed string suitable for routing checks.
        """
        return (query or "").strip()


def router_node(state: AgentState, llm: Optional[Any] = None) -> AgentState:
    """Analyze the user query and attach routing metadata to state.

    Args:
        state: Current graph state containing the user query.
        llm: Optional explicit LLM dependency. When omitted, the node looks for
            ``state["llm"]``.

    Returns:
        Updated AgentState with ``query_analysis`` and ``route`` attached.
    """
    router = Router(llm=llm or state.get("llm"))
    analysis = router.analyze(state.get("query", ""))
    route = router.route_for(analysis)
    updates: AgentState = {
        **state,
        "query_analysis": analysis,
        "route": route,
    }

    if analysis.intent == "comparison_out_of_scope":
        target = analysis.required_topics[0] if analysis.required_topics else None
        message = comparison_response(target)
        reasoning_steps = [
            *state.get("reasoning_steps", []),
            "Router blocked external comparison query before retrieval.",
        ]
        updates.update(
            {
                "clarification_message": message,
                "generation": message,
                "response": AgentResponse(
                    answer=message,
                    sources=[],
                    confidence=0.0,
                    is_grounded=False,
                    reasoning_steps=reasoning_steps,
                    guardrail_status=state.get("guardrail_status", {}),
                    refusal_reason=analysis.reasoning,
                    route="clarify",
                ),
                "reasoning_steps": reasoning_steps,
            }
        )

    return updates
