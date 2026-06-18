# 04 - Guardrails

## Overview
Safety, security, and quality control layers that protect the entire chatbot system.

## Input Guardrails
- Block harmful, abusive, NSFW, or illegal content
- Detect and block prompt injection / jailbreak attempts
- Detect off-topic queries (not related to Tata Nexon)
- Early rejection with polite responses

## Output Guardrails
- Detect toxicity, bias, or unsafe language in final answer
- Secondary hallucination / grounding validation
- Ensure proper use of citations
- Block low-quality or fabricated responses

## Key Principles
- Fail-safe design (better to refuse than give bad answer)
- Low latency
- Comprehensive logging
- Use structured output (GuardrailDecision model)
- Strong focus on prompt injection defense

## Integration Points
- Input Guardrail: Runs at the very beginning
- Output Guardrail: Runs after Answer Generation
- Should influence routing decisions (e.g., "refuse", "rewrite")