# API Layer Skill

Description: Strictly follow Test Driven Development for the FastAPI API layer.

When to use: For all work related to FastAPI app construction, HTTP routes, API request/response models, dependency injection, API error handling, CORS, authentication hooks, and integration with the Config Layer or LangGraph Agent.

## Rules
- Always follow strict Red -> Green -> Refactor cycle
- Write one small, focused endpoint or model test at a time
- Do not implement endpoint logic before a failing test
- Keep API code thin; domain logic belongs in agent, ingestion, retrieval, and config modules
- Use Pydantic models for all request and response bodies
- Use dependency injection for graph, settings, retriever, embedder, storer, and ingestion processor
- Do not log or return secrets, stack traces, hidden prompts, or raw provider errors

## Core Components to Test

### 1. App Factory
- Creates a FastAPI app without side effects
- Registers all routes
- Loads settings through the Config Layer
- Supports dependency overrides for tests

### 2. Health Endpoint
- Returns status, service name, and version
- Does not expose secrets
- Works without live OpenAI or ChromaDB dependencies

### 3. Chat Endpoint
- Validates non-empty user message
- Supports optional `thread_id`
- Generates a new thread ID when missing
- Invokes the LangGraph Agent with `graph_config(thread_id)`
- Returns answer, citations, confidence, route, and grounding status
- Includes reasoning only when requested

### 4. Ingest Endpoint
- Validates file path or upload input
- Calls the ingestion pipeline
- Returns chunk and metadata summary
- Handles unsupported files and missing files cleanly

### 5. Error Handling
- Converts validation and application errors into structured JSON
- Returns safe messages for unexpected exceptions
- Does not expose tracebacks to clients

### 6. Security and Runtime Concerns
- Prepare auth dependency hooks even if local mode is open
- Configure CORS through settings
- Keep prompt injection defense in the agent guardrails
- Add future rate-limit seams without hardcoding policy in routes

## Development Phases

### Phase 1: API Skeleton
- Create `src/api/` package
- Add app factory test
- Add `/health` test and implementation

### Phase 2: Chat Route
- Add request/response models
- Test empty message validation
- Test graph invocation with mocked graph
- Test thread ID behavior

### Phase 3: Ingest Route
- Add ingest models
- Test ingestion dependency call
- Test unsupported or missing input errors

### Phase 4: Error Handling and Security Hooks
- Add structured error model
- Add exception handlers
- Add CORS config
- Add placeholder auth dependency for future production mode

### Phase 5: Refactor and Integration
- Reuse Config Layer everywhere
- Keep app factory deterministic and testable
- Run full test suite

## Workflow
1. Read `specs/06-api.md` first
2. Add one failing API test
3. Run the test and confirm Red
4. Implement the smallest API code needed for Green
5. Refactor while tests remain green
6. Run relevant existing tests after integration changes

## Best Practices
- Prefer `create_app(settings: Settings | None = None)` over module-level app construction
- Keep route handlers short and readable
- Normalize agent state into API response models in one helper
- Use FastAPI dependency overrides in tests instead of real external services
- Keep live OpenAI and ChromaDB calls out of unit tests
- Include `thread_id` in every chat response
- Make production security stricter by config, not route rewrites
