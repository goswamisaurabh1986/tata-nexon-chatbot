# 05 - Configuration Layer

## Overview
Create a single, reliable configuration layer for the Tata Nexon chatbot. The layer must load environment variables, validate required settings, centralize defaults, and provide typed configuration objects to ingestion, retrieval, memory, and the LangGraph agent.

The goal is to remove scattered `os.getenv()` usage from production code and make the application easy to run in local development, tests, and future deployment environments.

## Core Components

### 1. Environment Loader
- Loads `.env` and optional local env files
- Does not fail when optional env files are missing
- Keeps secrets out of logs, test output, and error messages
- Supports explicit override behavior for local development

### 2. Settings Schema
- Provides typed settings for the full application
- Validates required values such as `OPENAI_API_KEY` when live OpenAI components are enabled
- Supplies safe defaults for local development
- Groups related settings by domain: app, OpenAI, ingestion, retrieval, vector store, memory, logging

### 3. Model Configuration
- Centralizes chat model, embedding model, dimensions, timeout, and retry settings
- Supports environment-based model selection
- Keeps generation and embedding settings separate
- Makes functional tests able to use real models only when credentials are available

### 4. Retrieval and Vector Store Configuration
- Configures ChromaDB collection name and persistence directory
- Configures top_k and similarity threshold defaults
- Supports test collection overrides
- Avoids hardcoded paths inside retrieval and ingestion components

### 5. Memory and Session Configuration
- Configures checkpoint backend and SQLite database path
- Configures default user/session ID behavior for the CLI
- Supports persistent local memory for `main.py`
- Allows tests to inject in-memory checkpointers

### 6. Logging and Runtime Configuration
- Centralizes log level and runtime flags
- Supports quiet defaults for CLI users
- Enables debug logging for development without code changes
- Ensures sensitive values are redacted before display

## Functional Requirements

- Load configuration from environment variables and `.env`
- Provide a clean `Settings` object for application code
- Avoid direct env lookups across unrelated modules after the config layer is introduced
- Validate required settings with clear, actionable errors
- Support test-friendly overrides without mutating global environment unexpectedly
- Keep local defaults useful for development
- Never print secrets such as API keys

## Recommended File Structure

```text
src/config/
  __init__.py
  settings.py
  validators.py
```

## Expected Settings Groups

- `AppSettings`: app name, environment, debug flag
- `OpenAISettings`: API key, chat model, embedding model, dimensions
- `IngestionSettings`: chunk size, overlap, supported file types
- `RetrievalSettings`: top_k, similarity threshold
- `VectorStoreSettings`: Chroma collection and persist directory
- `MemorySettings`: checkpoint backend, SQLite path, default user ID
- `LoggingSettings`: log level and formatting controls

## Acceptance Criteria

- App can load settings from `.env` without scattered `os.getenv()` calls
- Missing required secrets produce clear validation errors only when that feature is used
- Tests can override settings using monkeypatch or temporary env files
- `main.py` can initialize graph, retriever, embedder, vector store, and checkpointer from config
- No secret value is logged or displayed
- Existing ingestion, retrieval, and agent tests remain green after adopting the config layer
