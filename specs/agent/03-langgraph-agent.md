# 03 - LangGraph Agent

## Overview
The core reasoning engine of the chatbot using LangGraph. It handles query processing, retrieval, reasoning, and safe response generation with **structured LLM outputs**.

## Guardrails (Safety & Quality Controls)

### Input Guardrails
- Block harmful, abusive, or off-topic queries
- Query intent classification

### Retrieval Guardrails
- Minimum relevance threshold
- Minimum context sufficiency

### Output Guardrails
- Hallucination detection
- Grounding verification
- Toxicity/safety check
- Citation enforcement

## Structured Output Requirement

All LLM calls (especially Answer Generator and Grounding Checker) **must** use structured output via Pydantic models.

### Main Output Schema (`AgentResponse`)
```python
class AgentResponse(BaseModel):
    answer: str
    sources: list[str]                    # citation_ids e.g. ["Tata_Nexon_Brochure.pdf:5"]
    confidence: float                     # 0.0 to 1.0
    is_grounded: bool
    reasoning_steps: list[str]
    guardrail_status: dict[str, bool]     # e.g. {"input_safe": True, "hallucination_free": True}
    refusal_reason: str | None = None
Intermediate Structured Outputs

Query Analysis: Intent, answerability, required topics
Context Evaluation: Sufficiency score, missing information
Grounding Check: List of supported claims vs unsupported claims

### LangGraph Nodes

Input Guardrail
Query Analyzer (structured output)
Retriever
Context Sufficiency Checker (structured)
Answer Generator (uses structured output)
Grounding & Hallucination Checker (structured)
Output Guardrail
Response Formatter

Key Requirements

Use with_structured_output() from LangChain or PydanticOutputParser
All critical LLM nodes must return Pydantic models, not raw text
Fallback mechanism if structured output fails
Clear error handling for parsing failures
Final response must always conform to AgentResponse schema

Acceptance Criteria

LLM responses are reliably parsed into structured format
Agent always returns consistent, typed output
Citations are properly included when answer is given
Guardrails work effectively with structured validation
Easy to extend with new fields in the future