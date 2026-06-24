import pytest

from src.agent.nodes.router_node import router_node
from src.agent.schemas import QueryAnalysis


class FakeStructuredLLM:
    def __init__(self):
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return {
            "intent": "out_of_scope",
            "is_answerable": False,
            "needs_retrieval": False,
            "confidence": 0.1,
            "required_topics": [],
            "reasoning": "This query is outside the document scope.",
        }


class FakeLLM:
    def __init__(self):
        self.structured_llm = FakeStructuredLLM()
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_llm


def test_router_returns_structured_output_and_answerability():
    state = {"query": "What are the safety features of Tata Nexon?"}

    result = router_node(state)

    analysis = result["query_analysis"]
    assert isinstance(analysis, QueryAnalysis)
    assert analysis.is_answerable is True
    assert analysis.intent == "vehicle_features"
    assert "safety" in analysis.required_topics


def test_router_routes_retrieval_needed_query_to_retrieval():
    state = {"query": "What are the safety features of Tata Nexon?"}

    result = router_node(state)

    analysis = result["query_analysis"]
    assert isinstance(analysis, QueryAnalysis)
    assert analysis.is_answerable is True
    assert analysis.needs_retrieval is True
    assert analysis.confidence >= 0.8
    assert result["route"] == "retrieval"


@pytest.mark.parametrize("query", ["Tell me about Nexon", "Give me a Tata Nexon overview"])
def test_router_routes_normal_tata_nexon_queries_to_retrieval(query):
    result = router_node({"query": query})

    analysis = result["query_analysis"]
    assert analysis.is_answerable is True
    assert analysis.needs_retrieval is True
    assert result["route"] == "retrieval"


@pytest.mark.parametrize(
    "query",
    ["Tell me about the warranty", "What is the service schedule?", "What is the price?"],
)
def test_router_routes_reasonable_ownership_queries_to_retrieval(query):
    result = router_node({"query": query})

    analysis = result["query_analysis"]
    assert analysis.is_answerable is True
    assert analysis.needs_retrieval is True
    assert result["route"] == "retrieval"


@pytest.mark.parametrize(
    "query",
    ["Okay, thanks", "Thanks", "Hi", "Hello", "Bye", "What can you do?"],
)
def test_router_routes_simple_conversation_to_direct_answer(query):
    result = router_node({"query": query})

    analysis = result["query_analysis"]
    assert isinstance(analysis, QueryAnalysis)
    assert analysis.is_answerable is True
    assert analysis.needs_retrieval is False
    assert result["route"] == "direct_answer"


def test_router_keeps_nexon_only_variant_comparison_in_scope():
    result = router_node({"query": "Compare Tata Nexon variants"})

    analysis = result["query_analysis"]
    assert analysis.is_answerable is True
    assert analysis.needs_retrieval is True
    assert result["route"] == "retrieval"


@pytest.mark.parametrize(
    ("query", "expected_model"),
    [
        ("Compare Tata Nexon with Tata Sierra", "Tata Sierra"),
        ("Nexon vs Sierra performance", "Sierra"),
        ("Nexon versus Hyundai Creta", "Hyundai Creta"),
        ("Is Nexon better than Kia Sonet?", "Kia Sonet"),
        ("Difference between Tata Nexon and Maruti Brezza", "Maruti Brezza"),
        ("Compare this car with Mahindra XUV 3XO", "Mahindra XUV 3XO"),
    ],
)
def test_router_defensively_routes_other_car_comparisons_to_clarify(query, expected_model):
    result = router_node({"query": query})

    analysis = result["query_analysis"]
    assert analysis.intent == "comparison_out_of_scope"
    assert analysis.is_answerable is False
    assert analysis.needs_retrieval is False
    assert result["route"] == "clarify"
    assert expected_model in result["generation"]
    assert result["response"].route == "clarify"


@pytest.mark.parametrize("query", ["What is the weather today?", "How to cook biryani?"])
def test_router_identifies_out_of_scope_queries(query):
    result = router_node({"query": query})

    analysis = result["query_analysis"]
    assert isinstance(analysis, QueryAnalysis)
    assert analysis.is_answerable is False
    assert analysis.needs_retrieval is False
    assert analysis.confidence < 0.5
    assert result["route"] == "clarify"


@pytest.mark.parametrize(
    "query",
    ["What is the weather in Delhi today?", "Who won the cricket match yesterday?"],
)
def test_router_handles_off_topic_queries_using_llm(query):
    llm = FakeLLM()

    result = router_node({"query": query, "llm": llm})

    analysis = result["query_analysis"]
    assert llm.schema is QueryAnalysis
    assert len(llm.structured_llm.invocations) == 1
    assert isinstance(analysis, QueryAnalysis)
    assert analysis.is_answerable is False
    assert analysis.needs_retrieval is False
    assert analysis.confidence < 0.3
    assert "outside" in analysis.reasoning.lower()
    assert "document scope" in analysis.reasoning.lower()
    assert result["route"] == "clarify"
