# Tata Nexon Agentic RAG Chatbot

An end-to-end Retrieval-Augmented Generation chatbot for Tata Nexon brochure and ownership questions. The project combines a production-style ingestion pipeline, ChromaDB vector retrieval, a LangGraph agent with guardrails and grounding checks, a FastAPI backend, and a React + Vite chat UI.

The assistant is intentionally scoped to Tata Nexon. It answers product, brochure, feature, safety, warranty, price, service, and ownership questions using ingested documents, while refusing unsafe, prompt-injection, off-topic, and external comparison requests.

## Highlights

- Document ingestion for PDF, TXT, and Markdown files
- PyMuPDF PDF extraction with document metadata
- Recursive text chunking with overlap and section metadata
- OpenAI embeddings using `text-embedding-3-small` by default
- ChromaDB persistent local vector store
- LangGraph orchestration with multi-turn memory
- Input guardrails, router, retrieval, grading, answer generation, grounding, and output guardrails
- FastAPI API with chat, streaming SSE chat, health, and admin ingestion endpoints
- React + Vite + Tailwind frontend with streaming chat support
- Typed configuration with Pydantic Settings
- Unit and functional tests for ingestion, retrieval, agent nodes, API, and config

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI, Uvicorn |
| Agent orchestration | LangGraph |
| LLM integration | LangChain OpenAI, OpenAI API |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | ChromaDB persistent client |
| PDF parsing | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Configuration | Pydantic Settings, python-dotenv |
| Memory | LangGraph checkpointers, synchronous SQLite saver |
| Frontend | React, Vite, TypeScript, Tailwind CSS |
| Tests | Pytest |

## Project Structure

```text
.
|-- README.md
|-- architecture.md
|-- main.py                         # CLI chatbot entry point
|-- ingest.py                       # Batch ingestion script
|-- debug_chunks.py                 # Vector DB keyword diagnostics
|-- debug_vector_db.py              # Vector DB collection diagnostics
|-- requirements.txt
|-- data/
|   |-- ingestion_docs/             # Source documents for offline ingestion
|   `-- admin_uploads/              # Files uploaded through admin API
|-- specs/                          # Feature and architecture specs
|-- src/
|   |-- api/                        # FastAPI app, routes, schemas, dependencies
|   |-- agent/                      # LangGraph state, schemas, graph, nodes, memory
|   |-- config/                     # Typed settings by domain
|   |-- ingestion/                  # Scanner, parser, chunker, embedder, storer, processor
|   `-- retrieval/                  # Retriever abstraction over embedder/vector store
|-- tests/                          # Unit and functional tests
`-- frontend/                       # React + Vite + Tailwind chat UI
```

## Architecture Overview

The system has two main flows.

1. Ingestion flow:
   `DocumentScanner -> TextParser -> DocumentChunker -> Embedder -> VectorStorer`

2. Chat flow:
   `FastAPI -> LangGraph Agent -> Retriever -> ChromaDB -> Answer Generator -> Guardrails -> API Response`

The LangGraph agent is the core reasoning layer. Its graph starts with the input guardrail as the absolute first node, then routes safe Tata Nexon questions through retrieval, grading, answer generation, grounding checks, and output validation.

For a deeper walkthrough, see [architecture.md](architecture.md).

## Prerequisites

- Python 3.9 or newer
- Node.js and npm for the frontend
- OpenAI API key for real embeddings and LLM responses

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_COLLECTION_NAME=tata_nexon_chunks
CHROMA_PERSIST_DIRECTORY=runtime/chroma
CHECKPOINTER_BACKEND=sqlite
CHATBOT_MEMORY_DB=runtime/chatbot_memory.db
CHATBOT_SESSION_REGISTRY=runtime/chatbot_sessions.json
LOG_LEVEL=INFO
```

The settings layer loads both `.env` and `.env.txt`, but `.env` is the recommended local file. Do not commit API keys.

## Ingest Documents

Place PDF, TXT, or MD files in:

```text
data/ingestion_docs/
```

Run:

```bash
python ingest.py
```

The script scans supported files, extracts text, chunks content, creates embeddings, and stores chunks in ChromaDB.

## Run the API

```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Useful endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health check |
| POST | `/chat` | Non-streaming chat |
| POST | `/chat?stream=true` | Streaming chat over SSE |
| GET | `/chat?message=...&stream=true` | EventSource-compatible streaming |
| POST | `/admin/ingest` | Admin document ingestion |
| GET | `/admin/documents` | List ingested documents for current API process |
| GET | `/admin/stats` | Admin ingestion stats |

## Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend talks to:

```text
http://127.0.0.1:8000
```

You can override it with:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## CLI Chat

The root `main.py` script provides a local command-line chatbot with thread memory:

```bash
python main.py
```

It supports multiple conversations, new thread IDs, and optional reasoning display.

## Testing

Run all tests:

```bash
python -m pytest tests/ -q
```

Run focused test suites:

```bash
python -m pytest tests/ingestion/ -q
python -m pytest tests/retrieval/ -q
python -m pytest tests/agent/ -q
python -m pytest tests/api/ -q
python -m pytest tests/config/ -q
```

Functional tests that use real OpenAI services require `OPENAI_API_KEY`.

## Configuration

Configuration is organized by domain under `src/config/`:

- `settings.py` - root settings and app/memory settings
- `llm.py` - OpenAI chat and embedding settings
- `retrieval.py` - ChromaDB collection and retrieval settings
- `ingestion.py` - chunk size, overlap, and ingestion defaults

All settings are loaded through Pydantic Settings and are immutable after loading.

## Guardrails and Scope

The chatbot is only for Tata Nexon. The input guardrail is the first node in the LangGraph workflow. It blocks:

- prompt injection and jailbreak attempts
- unsafe, abusive, NSFW, harmful, or illegal requests
- completely off-topic questions
- external comparison queries such as `Nexon vs Sierra` or `compare Nexon with Creta`

Reasonable Tata Nexon questions are allowed even when phrased implicitly, such as:

- `What is the performance of this car?`
- `Tell me about the warranty`
- `What is the service schedule?`
- `Safety features of this vehicle`

## Repository Hygiene

Recommended files to keep out of Git:

- `.env`
- `.env.txt`
- `runtime/`
- `.chroma/` from older local runs
- `chatbot_memory.db` and `chatbot_sessions.json` from older local runs
- uploaded/admin documents if they are private
- logs and Python/pytest caches

## Current Status

This repository is structured as a local, developer-friendly RAG application with production-oriented boundaries:

- ingestion components are separated and testable
- retrieval is abstracted behind a clean retriever
- agent behavior is node-based and covered by tests
- API wiring injects real dependencies from the config layer
- frontend consumes the API with streaming support

See [architecture.md](architecture.md) for the detailed architecture and data flow.
