from src.agent.schemas import GroundingCheck


class FakeStructuredLLM:
    def __init__(self, response: GroundingCheck | dict | None = None) -> None:
        self.response = response or GroundingCheck(
            is_grounded=True,
            confidence=0.91,
            supported_claims=["Tata Nexon includes 6 airbags."],
            unsupported_claims=[],
            reasoning="All answer claims are supported by the provided chunks.",
        )
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: GroundingCheck | dict | None = None) -> None:
        self.schema = None
        self.structured_llm = FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_llm


def grounded_chunk(index: int = 0) -> dict:
    return {
        "citation_id": f"Tata_Nexon_Brochure.pdf:{index}",
        "text": "The Tata Nexon includes 6 airbags and electronic stability program.",
        "metadata": {
            "source": "Tata_Nexon_Brochure.pdf",
            "section_title": "Safety",
            "chunk_index": index,
        },
        "relevance_score": 0.92,
    }


def grounded_state() -> dict:
    return {
        "query": "What are the safety features of Tata Nexon?",
        "generation": "The Tata Nexon includes 6 airbags and electronic stability program.",
        "graded_chunks": [grounded_chunk()],
    }


def test_grounding_checker_evaluates_generation_against_graded_chunks():
    from src.agent.nodes.grounding_checker import GroundingChecker

    llm = FakeLLM()
    node = GroundingChecker(llm=llm)

    result = node.run(grounded_state())

    assert isinstance(result["grounding_check"], GroundingCheck)
    messages = llm.structured_llm.invocations[0]
    assert "6 airbags" in str(messages)
    assert "electronic stability program" in str(messages)


def test_grounding_checker_routes_supported_answer_to_final():
    from src.agent.nodes.grounding_checker import GroundingChecker

    node = GroundingChecker(llm=FakeLLM())

    result = node.run(grounded_state())

    assert result["is_grounded"] is True
    assert result["hallucination_pass"] is True
    assert result["route"] == "final"


def test_grounding_checker_accepts_practical_performance_answer_without_exact_metrics():
    from src.agent.nodes.grounding_checker import GroundingChecker

    llm = FakeLLM(
        GroundingCheck(
            is_grounded=False,
            confidence=0.28,
            supported_claims=[
                "The Tata Nexon has performance-related context about turbo engines and drive modes."
            ],
            unsupported_claims=[
                "Exact horsepower numbers are not available in the provided context."
            ],
            reasoning=(
                "The answer gives a general performance summary and clearly says "
                "exact metrics are unavailable."
            ),
        )
    )
    state = {
        "query": "Tell me about Tata Nexon performance.",
        "generation": (
            "The Tata Nexon performance is described through its turbocharged "
            "engine options and drive modes. Exact horsepower numbers are not "
            "available in the provided context."
        ),
        "graded_chunks": [
            {
                "citation_id": "Nexon Brochure.pdf:8",
                "text": (
                    "Performance features include turbocharged engine options, "
                    "Multi-Drive modes: Eco, City, Sports, and driving convenience."
                ),
                "metadata": {"source": "Nexon Brochure.pdf", "chunk_index": 8},
                "relevance_score": 0.82,
            }
        ],
    }

    result = GroundingChecker(llm=llm).run(state)

    assert result["is_grounded"] is True
    assert result["hallucination_pass"] is True
    assert result["route"] == "final"


def test_grounding_checker_routes_unsupported_answer_to_generate():
    from src.agent.nodes.grounding_checker import GroundingChecker

    llm = FakeLLM(
        GroundingCheck(
            is_grounded=False,
            confidence=0.22,
            supported_claims=["Tata Nexon includes 6 airbags."],
            unsupported_claims=["Tata Nexon has autonomous racing mode."],
            reasoning="The racing mode claim is not supported by the chunks.",
        )
    )
    state = {
        **grounded_state(),
        "generation": "The Tata Nexon includes 6 airbags and autonomous racing mode.",
    }
    node = GroundingChecker(llm=llm)

    result = node.run(state)

    assert result["is_grounded"] is False
    assert result["hallucination_pass"] is False
    assert result["route"] == "generate"


def test_grounding_checker_uses_structured_grounding_check_output():
    from src.agent.nodes.grounding_checker import GroundingChecker

    llm = FakeLLM(
        {
            "is_grounded": True,
            "confidence": 0.84,
            "supported_claims": ["Safety claims are supported."],
            "unsupported_claims": [],
            "reasoning": "The answer is supported by retrieved context.",
        }
    )
    node = GroundingChecker(llm=llm)

    result = node.run(grounded_state())

    assert llm.schema is GroundingCheck
    assert isinstance(result["grounding_check"], GroundingCheck)
    assert result["grounding_check"].confidence == 0.84


def test_grounding_checker_adds_reasoning_step_for_decision():
    from src.agent.nodes.grounding_checker import GroundingChecker

    node = GroundingChecker(llm=FakeLLM())
    state = {
        **grounded_state(),
        "reasoning_steps": ["Generated grounded answer from graded chunks."],
    }

    result = node.run(state)

    assert result["reasoning_steps"] == [
        "Generated grounded answer from graded chunks.",
        "Grounding check passed: All answer claims are supported by the provided chunks.",
    ]


def test_grounding_checker_handles_missing_chunks_or_generation_gracefully():
    from src.agent.nodes.grounding_checker import GroundingChecker

    node = GroundingChecker(llm=FakeLLM())

    no_chunks = node.run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "generation": "The Tata Nexon includes 6 airbags.",
            "graded_chunks": [],
        }
    )
    empty_generation = node.run(
        {
            "query": "What are the safety features of Tata Nexon?",
            "generation": "",
            "graded_chunks": [grounded_chunk()],
        }
    )

    for result in [no_chunks, empty_generation]:
        assert result["grounding_check"].is_grounded is False
        assert result["is_grounded"] is False
        assert result["hallucination_pass"] is False
        assert result["route"] == "generate"


def test_grounding_checker_retries_when_attempts_remain():
    from src.agent.nodes.grounding_checker import GroundingChecker

    llm = FakeLLM(
        GroundingCheck(
            is_grounded=False,
            confidence=0.35,
            supported_claims=[],
            unsupported_claims=["Unsupported safety claim."],
            reasoning="The answer contains an unsupported claim.",
        )
    )
    node = GroundingChecker(llm=llm)

    result = node.run({**grounded_state(), "generation_attempts": 1})

    assert result["route"] == "generate"
    assert result["reasoning_steps"][-1] == (
        "Grounding failed; retrying answer generation (attempt 2 of 2)."
    )


def test_grounding_checker_routes_to_clarify_after_max_generation_attempts():
    from src.agent.nodes.grounding_checker import GroundingChecker

    llm = FakeLLM(
        GroundingCheck(
            is_grounded=False,
            confidence=0.2,
            supported_claims=[],
            unsupported_claims=["Unsupported claim."],
            reasoning="The answer is still not grounded.",
        )
    )
    node = GroundingChecker(llm=llm)

    result = node.run({**grounded_state(), "generation_attempts": 2})

    assert result["route"] == "clarify"
    assert result["reasoning_steps"][-1] == (
        "Maximum generation attempts reached after grounding failure; routing to clarification."
    )
