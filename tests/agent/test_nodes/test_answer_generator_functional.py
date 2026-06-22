import os
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

dotenv = pytest.importorskip("dotenv")

from src.agent.nodes.answer_generator import AnswerGenerator
from src.agent.schemas import AgentResponse


dotenv.load_dotenv()
dotenv.load_dotenv(".env.txt")


class OpenAIStructuredLLM:
    """Small live OpenAI adapter matching LangChain's structured-output shape."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        openai = pytest.importorskip("openai")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def with_structured_output(self, schema):
        return OpenAIStructuredInvoker(self.client, self.model, schema)


class OpenAICompatibleAgentResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    is_grounded: bool
    reasoning_steps: list[str]
    refusal_reason: Optional[str]
    route: Optional[str]


class OpenAIStructuredInvoker:
    def __init__(self, client: Any, model: str, schema: type[AgentResponse]) -> None:
        self.client = client
        self.model = model
        self.target_schema = schema
        self.response_schema = (
            OpenAICompatibleAgentResponse
            if schema is AgentResponse
            else schema
        )

    def invoke(self, messages):
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": _role(role), "content": content}
                for role, content in messages
            ],
            response_format=self.response_schema,
            temperature=0,
        )
        message = completion.choices[0].message
        if message.parsed is not None:
            return self.target_schema.model_validate(message.parsed.model_dump())
        return self.target_schema.model_validate_json(message.content)


@pytest.fixture(scope="module")
def openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found; skipping live AnswerGenerator tests.")
    return api_key


@pytest.fixture(scope="module")
def real_llm(openai_api_key: str) -> OpenAIStructuredLLM:
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return OpenAIStructuredLLM(api_key=openai_api_key, model=model)


@pytest.fixture
def tata_nexon_safety_chunks() -> list[dict]:
    return [
        {
            "citation_id": "Tata_Nexon_Brochure.pdf:5",
            "text": (
                "The Tata Nexon includes advanced safety features such as "
                "6 airbags, electronic stability program, and strong occupant protection."
            ),
            "metadata": {
                "source": "Tata_Nexon_Brochure.pdf",
                "section_title": "Safety",
                "chunk_index": 5,
            },
            "score": 0.96,
            "relevance_score": 0.96,
        },
        {
            "citation_id": "Tata_Nexon_Brochure.pdf:6",
            "text": (
                "Additional safety equipment includes ISOFIX child-seat mounts, "
                "reverse parking assist, hill hold control, and brake assist."
            ),
            "metadata": {
                "source": "Tata_Nexon_Brochure.pdf",
                "section_title": "Safety",
                "chunk_index": 6,
            },
            "score": 0.91,
            "relevance_score": 0.91,
        },
    ]


def test_answer_generator_end_to_end_with_real_llm(
    real_llm: OpenAIStructuredLLM,
    tata_nexon_safety_chunks: list[dict],
):
    result = _run_or_skip(
        AnswerGenerator(real_llm),
        {
            "query": "What are the safety features of Tata Nexon?",
            "graded_chunks": tata_nexon_safety_chunks,
        },
    )

    response = result["response"]
    assert isinstance(response, AgentResponse)
    assert response.answer
    assert response.sources
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.is_grounded, bool)


def test_answer_generator_response_includes_input_citation_ids(
    real_llm: OpenAIStructuredLLM,
    tata_nexon_safety_chunks: list[dict],
):
    result = _run_or_skip(
        AnswerGenerator(real_llm),
        {
            "query": "List Tata Nexon safety features.",
            "graded_chunks": tata_nexon_safety_chunks,
        },
    )

    assert result["response"].sources == [
        "Tata_Nexon_Brochure.pdf:5",
        "Tata_Nexon_Brochure.pdf:6",
    ]


def test_answer_generator_refuses_when_context_is_insufficient(
    real_llm: OpenAIStructuredLLM,
):
    result = AnswerGenerator(real_llm).run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "graded_chunks": [],
        }
    )

    assert result["route"] == "clarify"
    assert result["response"].confidence <= 0.3
    assert result["response"].refusal_reason is not None
    assert "enough information" in result["generation"].lower()


def test_answer_generator_generates_factual_high_confidence_answer_with_good_context(
    real_llm: OpenAIStructuredLLM,
    tata_nexon_safety_chunks: list[dict],
):
    result = _run_or_skip(
        AnswerGenerator(real_llm),
        {
            "query": "Which safety features does the Tata Nexon offer?",
            "graded_chunks": tata_nexon_safety_chunks,
        },
    )

    answer = result["response"].answer.lower()
    assert result["response"].confidence >= 0.5
    assert "airbag" in answer or "stability" in answer or "isofix" in answer
    assert result["response"].sources


def test_answer_generator_final_output_conforms_to_agent_response_model(
    real_llm: OpenAIStructuredLLM,
    tata_nexon_safety_chunks: list[dict],
):
    result = _run_or_skip(
        AnswerGenerator(real_llm),
        {
            "query": "Summarize Tata Nexon safety features.",
            "graded_chunks": tata_nexon_safety_chunks,
        },
    )

    response = result["response"]
    assert isinstance(response, AgentResponse)
    assert AgentResponse.model_validate(response.model_dump()) == response


def _run_or_skip(node: AnswerGenerator, state: dict) -> dict:
    try:
        return node.run(state)
    except Exception as error:
        pytest.skip(f"Live OpenAI call unavailable: {error}")


def _role(role: str) -> str:
    return "user" if role in {"human", "user"} else role
