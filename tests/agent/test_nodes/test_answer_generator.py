from typing import Optional, Union

from src.agent.schemas import AgentResponse


class FakeStructuredLLM:
    def __init__(self, response: Optional[Union[AgentResponse, dict]] = None) -> None:
        self.response = response or AgentResponse(
            answer="The Tata Nexon includes advanced safety features.",
            sources=["Tata_Nexon_Brochure.pdf:0"],
            confidence=0.86,
            is_grounded=True,
            reasoning_steps=["Generated an answer from graded context."],
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


def graded_chunk(index: int = 0, score: float = 0.91) -> dict:
    return {
        "citation_id": f"Tata_Nexon_Brochure.pdf:{index}",
        "text": "Tata Nexon includes advanced safety features including 6 airbags.",
        "metadata": {
            "source": "Tata_Nexon_Brochure.pdf",
            "section_title": "Safety",
            "chunk_index": index,
        },
        "score": score,
        "relevance_score": score,
    }


def test_answer_generator_produces_generation_and_response():
    from src.agent.nodes.answer_generator import AnswerGenerator

    node = AnswerGenerator(llm=FakeLLM())
    state = {
        "query": "What are the safety features of Tata Nexon?",
        "graded_chunks": [graded_chunk()],
    }

    result = node.run(state)

    assert result["generation"] == "The Tata Nexon includes advanced safety features."
    assert isinstance(result["response"], AgentResponse)
    assert result["response"].answer == result["generation"]


def test_answer_generator_calls_llm_with_structured_agent_response_schema():
    from src.agent.nodes.answer_generator import AnswerGenerator

    llm = FakeLLM()
    node = AnswerGenerator(llm=llm)

    node.run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "graded_chunks": [graded_chunk()],
        }
    )

    assert llm.schema is AgentResponse
    assert len(llm.structured_llm.invocations) == 1


def test_answer_generator_includes_citations_from_state():
    from src.agent.nodes.answer_generator import AnswerGenerator

    node = AnswerGenerator(llm=FakeLLM())
    state = {
        "query": "Tata Nexon safety",
        "graded_chunks": [graded_chunk(0), graded_chunk(2)],
        "citations": [
            {"citation_id": "Tata_Nexon_Brochure.pdf:0"},
            {"citation_id": "Tata_Nexon_Brochure.pdf:2"},
        ],
    }

    result = node.run(state)

    assert result["response"].sources == [
        "Tata_Nexon_Brochure.pdf:0",
        "Tata_Nexon_Brochure.pdf:2",
    ]


def test_answer_generator_sets_reasonable_confidence_score():
    from src.agent.nodes.answer_generator import AnswerGenerator

    node = AnswerGenerator(llm=FakeLLM())

    result = node.run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "graded_chunks": [graded_chunk()],
        }
    )

    assert 0.5 <= result["response"].confidence <= 1.0


def test_answer_generator_appends_reasoning_step():
    from src.agent.nodes.answer_generator import AnswerGenerator

    node = AnswerGenerator(llm=FakeLLM())
    state = {
        "query": "Tata Nexon safety",
        "graded_chunks": [graded_chunk()],
        "reasoning_steps": ["Graded retrieval context."],
    }

    result = node.run(state)

    assert result["reasoning_steps"] == [
        "Graded retrieval context.",
        "Generated grounded answer from graded chunks.",
    ]


def test_answer_generator_handles_no_relevant_chunks_gracefully():
    from src.agent.nodes.answer_generator import AnswerGenerator

    node = AnswerGenerator(llm=FakeLLM())

    result = node.run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "graded_chunks": [],
        }
    )

    assert result["generation"]
    assert isinstance(result["response"], AgentResponse)
    assert result["response"].confidence <= 0.3
    assert result["response"].refusal_reason is not None
    assert result["route"] == "clarify"


def test_answer_generator_increments_generation_attempts_when_answer_is_generated():
    from src.agent.nodes.answer_generator import AnswerGenerator

    node = AnswerGenerator(llm=FakeLLM())

    result = node.run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "graded_chunks": [graded_chunk()],
            "generation_attempts": 1,
        }
    )

    assert result["generation_attempts"] == 2
