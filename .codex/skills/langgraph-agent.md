# LangGraph Agent Skill

Description: Strictly follow Test Driven Development for building the LangGraph Agent with structured outputs and guardrails.

When to use: For all agent nodes, graph construction, reasoning logic, and output handling.

## Rules
- Always follow Red → Green → Refactor
- Test each LangGraph node independently when possible
- Use mocks for LLM, Retriever, and external services
- **All critical LLM calls must use structured output** (Pydantic models)
- Test all decision paths: unanswerable query, insufficient context, hallucination, success, guardrail violations
- Prefer structured output over raw text for reliability

## Structured Output Guidelines
- Use Pydantic models (`BaseModel`) for all major LLM responses
- Use `with_structured_output()` from LangChain wherever possible
- Define clear output schemas (e.g., `AgentResponse`, `QueryAnalysis`, `GroundingCheck`)
- Implement fallback parsing in case structured output fails
- All final outputs must conform to the defined `AgentResponse` schema
- Test both successful structured parsing and fallback scenarios

## Core Components to Test
1. Input Guardrail Node
2. Query Analyzer (structured)
3. Retriever Node
4. Context Sufficiency Checker (structured)
5. Answer Generator (structured output)
6. Grounding & Hallucination Checker (structured)
7. Output Guardrail Node
8. Final Response Formatter

## Development Phases

### Phase 1: Basic Structure & Query Handling
- Input guardrails
- Query validation with structured output

### Phase 2: Retrieval Integration
- Connect with Retrieval module
- Context sufficiency check

### Phase 3: Answer Generation & Validation
- Structured answer generation
- Grounding & hallucination detection

### Phase 4: Full Guardrails & Error Handling
- Complete safety checks
- Structured error/refusal responses

## Workflow
1. Read 03-langgraph-agent.md spec first
2. Add one failing test at a time (focus on one node/behavior)
3. Make it pass with minimal code
4. Refactor while keeping tests green
5. Always prefer structured output in LLM nodes

## Best Practices
- Keep graph state clean and typed
- Use LangGraph's state management effectively
- Make nodes single-responsibility
- Log important decisions and reasoning steps
- Ensure high test coverage for guardrail paths