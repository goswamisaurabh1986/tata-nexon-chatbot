from src.agent.schemas import AgentResponse, GuardrailDecision


class FakeStructuredLLM:
    def __init__(self, response: GuardrailDecision | dict | None = None) -> None:
        self.response = response or GuardrailDecision(
            is_safe=True,
            is_blocked=False,
            category="safe",
            reason="The answer is safe and appropriate.",
            severity="low",
            blocked_reason=None,
            confidence=0.9,
        )
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: GuardrailDecision | dict | None = None) -> None:
        self.schema = None
        self.structured_llm = FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_llm


def safe_response() -> AgentResponse:
    return AgentResponse(
        answer="The Tata Nexon includes 6 airbags and electronic stability program.",
        sources=["Tata_Nexon_Brochure.pdf:5"],
        confidence=0.86,
        is_grounded=True,
        reasoning_steps=["Generated grounded answer from graded chunks."],
    )


def test_output_guardrail_approves_safe_final_response():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)

    result = node.run({"response": safe_response(), "generation": safe_response().answer})

    assert result["output_guardrail"].is_safe is True
    assert result["guardrail_status"]["output_safe"] is True
    assert result["route"] == "final"


def test_output_guardrail_rejects_toxic_or_abusive_output():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)

    result = node.run(
        {
            "generation": "This answer contains abusive hateful insults.",
            "response": safe_response().model_copy(
                update={"answer": "This answer contains abusive hateful insults."}
            ),
        }
    )

    assert result["output_guardrail"].is_blocked is True
    assert result["output_guardrail"].category == "toxic"
    assert result["route"] == "refuse"
    assert "can't return" in result["generation"].lower()
    assert result["response"].route == "refuse"


def test_output_guardrail_rejects_unsafe_content():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)

    result = node.run({"generation": "Here are steps to build a bomb."})

    assert result["output_guardrail"].is_blocked is True
    assert result["output_guardrail"].category == "harmful"
    assert result["route"] == "refuse"
    assert "can't return" in result["generation"].lower()


def test_output_guardrail_uses_structured_llm_output():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    llm = FakeLLM(
        {
            "is_safe": True,
            "is_blocked": False,
            "category": "safe",
            "reason": "Final answer is safe and grounded.",
            "severity": "low",
            "blocked_reason": None,
            "confidence": 0.93,
        }
    )
    node = OutputGuardrail(llm=llm)

    result = node.run({"response": safe_response(), "generation": safe_response().answer})

    assert llm.schema is GuardrailDecision
    assert len(llm.structured_llm.invocations) == 1
    assert result["output_guardrail"].confidence == 0.93


def test_output_guardrail_handles_empty_response_gracefully():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)

    result = node.run({"generation": "", "response": ""})

    assert result["output_guardrail"].is_blocked is True
    assert result["output_guardrail"].category == "empty_output"
    assert result["route"] == "refuse"


def test_output_guardrail_adds_reasoning_step_for_decision():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)
    state = {
        "response": safe_response(),
        "generation": safe_response().answer,
        "reasoning_steps": ["Grounding check passed."],
    }

    result = node.run(state)

    assert result["reasoning_steps"] == [
        "Grounding check passed.",
        "Output guardrail passed: Output is safe.",
    ]


def test_output_guardrail_rejects_missing_citations_for_factual_claims():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)
    response = safe_response().model_copy(update={"sources": []})

    result = node.run(
        {
            "generation": response.answer,
            "response": response,
            "is_grounded": True,
        }
    )

    assert result["output_guardrail"].is_blocked is True
    assert result["output_guardrail"].category == "citation_missing"
    assert result["route"] == "refuse"
    assert result["response"].sources == []


def test_output_guardrail_rejects_ungrounded_final_answer():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)

    result = node.run(
        {
            "generation": "The Tata Nexon can fly over traffic.",
            "response": safe_response().model_copy(
                update={"answer": "The Tata Nexon can fly over traffic."}
            ),
            "is_grounded": False,
        }
    )

    assert result["output_guardrail"].category == "hallucination"
    assert result["output_guardrail"].severity == "high"
    assert result["route"] == "refuse"


def test_output_guardrail_rejects_biased_output():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)

    result = node.run({"generation": "Only idiots from that group would buy this car."})

    assert result["output_guardrail"].is_blocked is True
    assert result["output_guardrail"].category == "bias"
    assert result["route"] == "refuse"


def test_output_guardrail_blocked_output_replaces_unsafe_answer_with_refusal():
    from src.agent.nodes.output_guardrail import OutputGuardrail

    node = OutputGuardrail(llm=None)
    result = node.run(
        {
            "generation": "The Tata Nexon can fly and this is fabricated.",
            "response": safe_response().model_copy(
                update={"answer": "The Tata Nexon can fly and this is fabricated."}
            ),
            "citations": [{"citation_id": "old.pdf:1"}],
            "is_grounded": True,
        }
    )

    assert result["route"] == "refuse"
    assert result["generation"] != "The Tata Nexon can fly and this is fabricated."
    assert "can't return" in result["generation"].lower()
    assert result["response"].route == "refuse"
    assert result["response"].sources == []
    assert result["citations"] == []
