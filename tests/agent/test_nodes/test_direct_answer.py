from typing import Optional, Union

from src.agent.schemas import AgentResponse


class FakeStructuredLLM:
    def __init__(self, response: Optional[Union[AgentResponse, dict]] = None) -> None:
        self.response = response or AgentResponse(
            answer="You're welcome. Happy to help with Tata Nexon questions.",
            sources=[],
            confidence=0.9,
            is_grounded=True,
            reasoning_steps=["Direct answer generated."],
            route="final",
        )
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: Optional[Union[AgentResponse, dict]] = None) -> None:
        self.schema = None
        self.structured_llm = FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_llm


def test_direct_answer_handles_thanks_without_retrieval():
    from src.agent.nodes.direct_answer import direct_answer_node

    result = direct_answer_node(
        {
            "query": "Okay, thanks",
            "generation": "Old Tata Nexon safety answer.",
            "response": "Old response",
            "retrieved_chunks": [{"text": "old chunk"}],
            "graded_chunks": [{"text": "old graded chunk"}],
            "citations": [{"citation_id": "old.pdf:1"}],
            "reasoning_steps": ["Previous answer generated."],
        },
        llm=None,
    )

    assert result["route"] == "final"
    assert result["generation"] != "Old Tata Nexon safety answer."
    assert "welcome" in result["generation"].lower() or "happy to help" in result["generation"].lower()
    assert result["retrieved_chunks"] == []
    assert result["graded_chunks"] == []
    assert result["citations"] == []
    assert isinstance(result["response"], AgentResponse)
    assert result["response"].sources == []


def test_direct_answer_uses_structured_llm_output():
    from src.agent.nodes.direct_answer import direct_answer_node

    llm = FakeLLM(
        {
            "answer": "Hi! Ask me anything about the Tata Nexon.",
            "sources": [],
            "confidence": 0.88,
            "is_grounded": True,
            "reasoning_steps": ["Direct greeting answer generated."],
            "route": "final",
        }
    )

    result = direct_answer_node({"query": "Hi"}, llm=llm)

    assert llm.schema is AgentResponse
    assert len(llm.structured_llm.invocations) == 1
    assert result["generation"] == "Hi! Ask me anything about the Tata Nexon."
    assert result["response"].confidence == 0.88


def test_direct_answer_fallback_describes_capabilities():
    from src.agent.nodes.direct_answer import direct_answer_node

    result = direct_answer_node({"query": "What can you do?"}, llm=None)

    assert "Tata Nexon" in result["generation"]
    assert "safety" in result["generation"].lower()
    assert result["response"].sources == []
