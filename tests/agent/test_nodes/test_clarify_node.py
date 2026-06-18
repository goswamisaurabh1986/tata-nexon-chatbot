from src.agent.schemas import AgentResponse, ClarificationResponse


class FakeStructuredLLM:
    def __init__(self, response: ClarificationResponse | dict | None = None) -> None:
        self.response = response or ClarificationResponse(
            message="Could you specify which Tata Nexon feature or variant you want to know about?",
            suggested_questions=[
                "What are the Tata Nexon safety features?",
                "Which Tata Nexon variant should I compare?",
            ],
            reason="The original query is broad and needs a more specific Tata Nexon topic.",
        )
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: ClarificationResponse | dict | None = None) -> None:
        self.schema = None
        self.structured_llm = FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_llm


def test_clarify_node_uses_structured_llm_output():
    from src.agent.nodes.clarify_node import ClarifyNode

    llm = FakeLLM()
    node = ClarifyNode(llm=llm)

    result = node.run({"query": "Tell me about Tata cars"})

    assert llm.schema is ClarificationResponse
    assert len(llm.structured_llm.invocations) == 1
    assert isinstance(result["clarification"], ClarificationResponse)


def test_clarify_node_generates_natural_specific_message_for_vague_query():
    from src.agent.nodes.clarify_node import ClarifyNode

    node = ClarifyNode(llm=FakeLLM())

    result = node.run({"query": "Tell me about Tata cars"})

    assert "tata nexon" in result["generation"].lower()
    assert "feature" in result["generation"].lower() or "variant" in result["generation"].lower()
    assert result["route"] == "clarify"


def test_clarify_node_prompt_includes_query_and_context():
    from src.agent.nodes.clarify_node import ClarifyNode

    llm = FakeLLM()
    node = ClarifyNode(llm=llm)
    state = {
        "query": "What about safety?",
        "retrieved_chunks": [
            {
                "text": "Tata Nexon safety content was insufficient.",
                "citation_id": "Tata_Nexon_Brochure.pdf:9",
            }
        ],
        "reasoning_steps": ["Retrieval returned weak context."],
    }

    node.run(state)

    messages = str(llm.structured_llm.invocations[0])
    assert "What about safety?" in messages
    assert "Tata Nexon safety content was insufficient." in messages
    assert "Retrieval returned weak context." in messages


def test_clarify_node_wraps_message_in_agent_response():
    from src.agent.nodes.clarify_node import ClarifyNode

    node = ClarifyNode(llm=FakeLLM())

    result = node.run({"query": "Tell me about Tata cars"})

    assert isinstance(result["response"], AgentResponse)
    assert result["response"].answer == result["generation"]
    assert result["response"].route == "clarify"
    assert result["response"].refusal_reason is not None


def test_clarify_node_preserves_router_supplied_comparison_message():
    from src.agent.nodes.clarify_node import ClarifyNode

    message = (
        "I'm a specialized assistant for the Tata Nexon only. I don't have "
        "information or comparison data for other models like Tata Sierra. Would you "
        "like to know more about any specific feature of the Tata Nexon?"
    )
    node = ClarifyNode(llm=FakeLLM())

    result = node.run(
        {
            "query": "Nexon vs Sierra",
            "clarification_message": message,
        }
    )

    assert result["generation"] == message
    assert result["response"].answer == message
    assert result["route"] == "clarify"


def test_clarify_node_fallback_handles_insufficient_context_without_llm():
    from src.agent.nodes.clarify_node import ClarifyNode

    node = ClarifyNode(llm=None)

    result = node.run(
        {
            "query": "What are the safety features?",
            "reasoning_steps": ["No retrieved chunks available for grading."],
        }
    )

    assert "more details" in result["generation"].lower()
    assert "tata nexon" in result["generation"].lower()
    assert result["route"] == "clarify"


def test_clarify_node_appends_reasoning_step():
    from src.agent.nodes.clarify_node import ClarifyNode

    node = ClarifyNode(llm=FakeLLM())
    state = {
        "query": "Tell me about Tata cars",
        "reasoning_steps": ["Query was too broad."],
    }

    result = node.run(state)

    assert result["reasoning_steps"] == [
        "Query was too broad.",
        "Clarification generated: The original query is broad and needs a more specific Tata Nexon topic.",
    ]
