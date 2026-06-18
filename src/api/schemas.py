"""API schema definitions.

This module contains the Pydantic models used at the HTTP boundary. These
schemas intentionally stay small and transport-focused: request validation,
response shape, and safe error payloads live here, while ingestion, retrieval,
and agent logic remain in their domain modules.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    """Response returned by ``GET /health``.

    Attributes:
        status: Machine-readable service status.
        service: Human-readable service name.
        version: Application/API version.
    """

    status: str = "ok"
    service: str
    version: str


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``.

    Attributes:
        message: User message to send through the LangGraph agent.
        thread_id: Optional conversation ID for multi-turn memory.
        user_id: Optional user/session owner used when creating a new thread.
        top_k: Optional retrieval result limit for this request.
        include_reasoning: Whether to include internal reasoning steps.
    """

    message: str = Field(
        min_length=1,
        description="User message to send to the agent.",
    )
    thread_id: str | None = Field(default=None, description="Conversation thread ID.")
    user_id: str | None = Field(default=None, description="User/session owner.")
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_reasoning: bool = False

    @field_validator("message")
    @classmethod
    def _message_must_have_text(cls, value: str) -> str:
        """Reject whitespace-only chat messages."""
        message = value.strip()
        if not message:
            raise ValueError("message must not be empty.")
        return message


class ChatResponse(BaseModel):
    """Response body for ``POST /chat``.

    Attributes:
        answer: User-facing answer, clarification, or refusal.
        thread_id: Conversation thread used for memory.
        sources: Citation IDs returned by the agent.
        confidence: Agent confidence score from 0.0 to 1.0.
        is_grounded: Whether the answer passed grounding checks.
        route: Final graph route.
        reasoning_steps: Optional reasoning trail, included only on request.
    """

    answer: str
    thread_id: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_grounded: bool = False
    route: str | None = None
    reasoning_steps: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Request body for ``POST /admin/ingest``.

    Attributes:
        file_path: Local document path for ingestion.
        source_filename: Optional display/source name for the document.
        force_reprocess: Whether callers intend to reprocess an existing source.
        collection_name: Optional target collection override for future support.
        metadata_overrides: Metadata to merge into the ingestion summary.
    """

    file_path: str | None = None
    source_filename: str | None = None
    force_reprocess: bool = False
    collection_name: str | None = None
    metadata_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _must_have_input(self) -> "IngestRequest":
        """Require a document input for ingestion."""
        if not self.file_path or not self.file_path.strip():
            raise ValueError("file_path is required for ingestion.")
        self.file_path = self.file_path.strip()
        return self


class IngestResponse(BaseModel):
    """Response body for ``POST /admin/ingest``."""

    status: str
    source: str
    chunks_created: int
    chunks_stored: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    """In-process summary of an ingested document."""

    source: str
    chunks_created: int
    chunks_stored: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentsResponse(BaseModel):
    """Response body for ``GET /admin/documents``."""

    documents: list[DocumentSummary] = Field(default_factory=list)


class AdminStatsResponse(BaseModel):
    """Response body for ``GET /admin/stats``."""

    documents_count: int
    chunks_count: int


class ErrorDetail(BaseModel):
    """Structured error details used by all API error responses."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Top-level structured error response."""

    error: ErrorDetail
