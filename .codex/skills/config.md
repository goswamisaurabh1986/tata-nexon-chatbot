# Config Layer Skill

Description: Strictly follow Test Driven Development for the chatbot configuration layer.

When to use: For all work related to environment loading, typed settings, runtime configuration, secrets handling, logging configuration, and dependency initialization.

## Rules
- Always follow strict Red -> Green -> Refactor cycle
- Write one small, focused test at a time
- Do not introduce implementation before a failing test
- Keep configuration typed and explicit
- Do not scatter new `os.getenv()` calls through domain code
- Never log, print, snapshot, or assert real secret values
- Prefer dependency injection over global mutable settings

## Core Components to Test

### 1. Environment Loading
- Loads `.env` when present
- Works when `.env` is missing
- Supports local overrides in a controlled way
- Does not leak secrets into output

### 2. Settings Models
- Validates required and optional settings
- Provides sensible defaults
- Groups settings by domain
- Fails with clear errors for invalid values

### 3. OpenAI Configuration
- Loads API key, chat model, embedding model, and embedding dimensions
- Allows tests to skip live calls when API key is absent
- Keeps chat and embedding settings separate

### 4. Retrieval and Storage Configuration
- Loads ChromaDB collection and persist directory
- Loads retriever top_k and similarity threshold
- Supports test-specific collection names

### 5. Memory Configuration
- Loads checkpoint backend and SQLite path
- Supports persistent memory for CLI usage
- Allows in-memory checkpointer injection for tests

### 6. Logging and Safety
- Configures log level from settings
- Redacts secret values in representations and errors
- Keeps CLI output user-friendly

## Development Phases

### Phase 1: Settings Skeleton
- Create settings package structure
- Add tests for defaults and env loading
- Add typed settings models

### Phase 2: Validation
- Add tests for invalid values
- Add validation for numbers, paths, and required secrets
- Add secret redaction tests

### Phase 3: Integration
- Wire settings into `main.py`
- Wire settings into embedder, vector store, retriever, and graph initialization
- Keep all existing tests passing

### Phase 4: Refactor
- Remove scattered env reads where practical
- Keep backward compatibility for existing constructors
- Document new config usage

## Workflow
1. Read `specs/05-config.md` first
2. Add a failing test for one config behavior
3. Run the test and confirm Red
4. Implement the minimal setting or loader behavior
5. Run tests and confirm Green
6. Refactor only while tests stay green

## Best Practices
- Treat environment variables as input, not global application state
- Keep defaults close to the settings schema
- Use temporary env vars or monkeypatch in tests
- Keep live-service requirements opt-in
- Make errors actionable for CLI users
