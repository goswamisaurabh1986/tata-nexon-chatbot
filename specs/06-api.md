# 06 - API Layer

## Overview
RESTful API layer built with FastAPI that exposes the chatbot's functionality to external clients and provides admin controls for document ingestion.

## Purpose
- Provide chat interface for end users
- Allow secure document ingestion with admin controls
- Enable system monitoring and management

## Core Endpoints

### 1. Chat Endpoint (User-facing)
- **POST** `/chat`
- Supports multi-turn conversations using `thread_id`
- Applies Input Guardrails automatically

### 2. Admin Ingestion Controls
- **POST** `/admin/ingest`
  - Restricted endpoint for administrators
  - Supports single or batch document upload (PDF, TXT, MD)
  - Options:
    - `force_reprocess`: bool (default: false)
    - `collection_name`: str (optional)
    - `metadata_overrides`: object
  - Returns ingestion status and statistics

- **GET** `/admin/documents`
  - List all ingested documents with metadata
  - Filter by source, date, status

- **DELETE** `/admin/documents/{source}`
  - Delete all chunks related to a specific document
  - Requires confirmation

- **POST** `/admin/rebuild-index`
  - Trigger full re-indexing of all documents

### 3. Health & Monitoring
- **GET** `/health`
- **GET** `/admin/stats` (ingestion stats, vector DB stats, etc.)

## Security & Access Control
- `/chat` → Public (with Input Guardrails)
- All `/admin/*` endpoints → Protected (API Key / JWT / Role-based access)
- Rate limiting on chat endpoint
- File size and type validation on ingestion

## Request / Response Models
- Use Pydantic models for all endpoints
- Clear error responses with error codes and messages
- Structured response for chat using `AgentResponse`

## Non-Functional Requirements
- High performance on chat endpoint
- Secure admin operations
- Comprehensive logging for admin actions
- Support for future streaming responses
- Good Swagger documentation

## Future Enhancements
- Bulk ingestion with progress tracking
- Document status dashboard
- Versioning of ingested documents
- Audit logging for admin actions

## Duplicate Document Handling (Admin Ingestion)

### Detection Strategy
- Calculate SHA-256 hash of the raw file content
- Store hash in vector DB metadata for each document
- Check for existing hash before processing

### Behavior Options (Configurable via `force_reprocess` flag)
- `force_reprocess=false` (default): Skip if hash exists, return "Document already exists"
- `force_reprocess=true`: Delete old chunks and re-ingest fresh version
- Always return clear status:
  ```json
  {
    "status": "skipped" | "ingested" | "reprocessed",
    "message": "...",
    "document_hash": "...",
    "chunks_created": 45
  }