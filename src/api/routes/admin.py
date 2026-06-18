"""Admin endpoints for ingestion and document metadata.

These routes expose operational controls for local/admin workflows. They remain
thin adapters around the ingestion pipeline: request validation and response
summaries live here, while parsing, chunking, embedding, and storage stay in
the ingestion components.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.api.dependencies import (
    get_document_registry,
    get_ingestion_processor,
    get_settings,
)
from src.api.schemas import (
    AdminStatsResponse,
    DocumentSummary,
    DocumentsResponse,
    IngestRequest,
    IngestResponse,
)
from src.config.settings import Settings
from src.ingestion.processor import IngestionProcessor


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])
ADMIN_UPLOAD_DIR = Path("data/admin_uploads")
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    http_request: Request,
    settings: Settings = Depends(get_settings),
    processor: IngestionProcessor = Depends(get_ingestion_processor),
    documents: list[DocumentSummary] = Depends(get_document_registry),
) -> IngestResponse:
    """Ingest one local document into the retrieval index.

    Args:
        http_request: Raw HTTP request, either JSON or multipart upload.
        settings: Typed runtime settings.
        processor: Configured ingestion processor dependency.
        documents: Process-local document summary registry.

    Returns:
        IngestResponse with chunk counts and merged metadata.

    Raises:
        HTTPException: Safe client/server errors for known ingestion failures.
    """
    request = await _ingest_request_from_http(http_request)
    logger.info(
        "Admin ingestion requested for source=%s file_path=%s force=%s collection=%s",
        request.source_filename,
        request.file_path,
        request.force_reprocess,
        request.collection_name or "default",
    )
    document_hash = _ensure_document_hash(request)
    duplicate = _find_duplicate_document(processor, document_hash)

    if duplicate["ids"] and not request.force_reprocess:
        logger.info(
            "Skipping duplicate admin ingestion for source=%s hash=%s chunks=%d.",
            _source_from_request(request),
            document_hash,
            len(duplicate["ids"]),
        )
        return _duplicate_document_response(request, document_hash, duplicate)

    deleted_chunks = 0
    if request.force_reprocess:
        deleted_chunks = _delete_existing_document_chunks(
            processor=processor,
            duplicate=duplicate,
            source=_source_from_request(request),
        )

    chunks = _process_document(request, settings, processor)
    summary = _document_summary_from_chunks(request, chunks)
    if deleted_chunks:
        summary.metadata["deleted_chunks"] = deleted_chunks
    _upsert_document_summary(documents, summary)
    logger.info("Ingested %s with %d chunks.", summary.source, summary.chunks_stored)

    return IngestResponse(
        status="reprocessed" if deleted_chunks else "completed",
        source=summary.source,
        chunks_created=summary.chunks_created,
        chunks_stored=summary.chunks_stored,
        metadata=summary.metadata,
    )


async def _ingest_request_from_http(http_request: Request) -> IngestRequest:
    """Build an IngestRequest from JSON or multipart form data."""
    content_type = http_request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        return await _ingest_request_from_multipart(http_request)

    try:
        payload = await http_request.json()
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ingestion request body.",
        ) from error

    try:
        return IngestRequest.model_validate(payload)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


async def _ingest_request_from_multipart(http_request: Request) -> IngestRequest:
    """Persist an uploaded file and return an ingestion request for it."""
    try:
        form = await http_request.form()
    except Exception as error:
        logger.exception("Failed to parse multipart admin ingestion request.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid multipart ingestion request.",
        ) from error

    uploaded_file = form.get("file")
    if not isinstance(uploaded_file, (UploadFile, StarletteUploadFile)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multipart ingestion requires a file field.",
        )

    source_filename = Path(uploaded_file.filename or "uploaded-document").name
    _validate_upload_filename(source_filename)
    file_path, document_hash = await _save_uploaded_file(uploaded_file, source_filename)
    return IngestRequest(
        file_path=str(file_path),
        source_filename=str(form.get("source_filename") or source_filename),
        force_reprocess=_form_bool(form.get("force_reprocess")),
        collection_name=_optional_form_string(form.get("collection_name")),
        metadata_overrides={
            "uploaded_via": "frontend",
            "document_hash": document_hash,
        },
    )


async def _save_uploaded_file(uploaded_file: UploadFile, source_filename: str) -> tuple[Path, str]:
    """Save an uploaded document and return its SHA-256 content hash."""
    ADMIN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ADMIN_UPLOAD_DIR / source_filename
    if target_path.exists():
        target_path = _unique_upload_path(target_path)

    try:
        content = await uploaded_file.read()
        document_hash = hashlib.sha256(content).hexdigest()
        target_path.write_bytes(content)
    except Exception as error:
        logger.exception("Failed to save uploaded admin document %s.", source_filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save uploaded document.",
        ) from error

    logger.info("Saved uploaded admin document %s to %s.", source_filename, target_path)
    return target_path, document_hash


def _validate_upload_filename(source_filename: str) -> None:
    """Validate uploaded document filename and extension."""
    suffix = Path(source_filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported upload type. Use PDF, TXT, or MD files.",
        )


def _unique_upload_path(path: Path) -> Path:
    """Return a non-conflicting upload path near the requested path."""
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Could not create a unique upload filename.",
    )


def _form_bool(value: Any) -> bool:
    """Parse common multipart boolean values."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_form_string(value: Any) -> str | None:
    """Return a stripped form string or None."""
    text = str(value or "").strip()
    return text or None


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(
    documents: list[DocumentSummary] = Depends(get_document_registry),
) -> DocumentsResponse:
    """List documents ingested during the current API process."""
    return DocumentsResponse(documents=documents)


@router.get("/stats", response_model=AdminStatsResponse)
def admin_stats(
    documents: list[DocumentSummary] = Depends(get_document_registry),
) -> AdminStatsResponse:
    """Return aggregate ingestion statistics for the current process."""
    return AdminStatsResponse(
        documents_count=len(documents),
        chunks_count=sum(document.chunks_stored for document in documents),
    )


def _process_document(
    request: IngestRequest,
    settings: Settings,
    processor: IngestionProcessor,
) -> list[dict[str, Any]]:
    """Run the ingestion processor and translate known failures."""
    try:
        return processor.process(
            file_path=request.file_path,
            source_filename=request.source_filename,
            chunk_size=settings.ingestion.chunk_size,
            overlap=settings.ingestion.overlap,
            metadata_overrides=request.metadata_overrides,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document file was not found.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except Exception as error:
        logger.exception(
            "Document ingestion failed for source=%s file_path=%s.",
            request.source_filename,
            request.file_path,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from error


def _document_summary_from_chunks(
    request: IngestRequest,
    chunks: list[dict[str, Any]],
) -> DocumentSummary:
    """Create a document summary from processed chunks and request metadata."""
    metadata = _metadata_from_chunks(chunks)
    metadata.update(request.metadata_overrides)
    metadata["force_reprocess"] = request.force_reprocess
    if request.collection_name:
        metadata["collection_name"] = request.collection_name
    metadata = _safe_metadata(metadata)

    return DocumentSummary(
        source=_source_from_request(request),
        chunks_created=len(chunks),
        chunks_stored=len(chunks),
        metadata=metadata,
    )


def _source_from_request(request: IngestRequest) -> str:
    """Resolve the display source for an ingestion request."""
    if request.source_filename:
        return request.source_filename
    return Path(request.file_path or "unknown").name


def _metadata_from_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy representative metadata from the first chunk."""
    if not chunks:
        return {}
    metadata = chunks[0].get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata containing only JSON-serializable values."""
    return {str(key): _safe_metadata_value(value) for key, value in metadata.items()}


def _safe_metadata_value(value: Any) -> Any:
    """Convert unsafe metadata values into compact JSON-safe placeholders."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary metadata omitted>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_metadata_value(item) for item in value]
    return str(value)


def _upsert_document_summary(
    documents: list[DocumentSummary],
    summary: DocumentSummary,
) -> None:
    """Insert or replace a document summary by source."""
    for index, document in enumerate(documents):
        if document.source == summary.source:
            documents[index] = summary
            return
    documents.append(summary)


def _ensure_document_hash(request: IngestRequest) -> str | None:
    """Ensure request metadata contains a SHA-256 hash when bytes are available."""
    existing_hash = request.metadata_overrides.get("document_hash")
    if isinstance(existing_hash, str) and existing_hash:
        return existing_hash

    if not request.file_path:
        return None

    path = Path(request.file_path)
    if not path.is_file():
        logger.info("Skipping duplicate hash check because %s is not a readable file.", path)
        return None

    document_hash = _file_sha256(path)
    request.metadata_overrides["document_hash"] = document_hash
    return document_hash


def _file_sha256(path: Path) -> str:
    """Calculate SHA-256 for a local document without loading huge files twice."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_duplicate_document(
    processor: IngestionProcessor,
    document_hash: str | None,
) -> dict[str, list[Any]]:
    """Find existing Chroma chunks with the same document hash."""
    if not document_hash:
        return {"ids": [], "metadatas": []}

    collection = _collection_from_processor(processor)
    if collection is None:
        logger.info("Skipping duplicate lookup because no vector collection is available.")
        return {"ids": [], "metadatas": []}

    try:
        results = collection.get(
            where={"document_hash": document_hash},
            include=["metadatas"],
        )
    except Exception:
        logger.exception("Duplicate lookup failed for document hash %s.", document_hash)
        return {"ids": [], "metadatas": []}

    return {
        "ids": list(results.get("ids") or []),
        "metadatas": list(results.get("metadatas") or []),
    }


def _delete_existing_document_chunks(
    processor: IngestionProcessor,
    duplicate: dict[str, list[Any]],
    source: str,
) -> int:
    """Delete existing chunks for a duplicated or force-reprocessed source."""
    collection = _collection_from_processor(processor)
    if collection is None:
        logger.info("Skipping forced delete for %s because no vector collection is available.", source)
        return 0

    delete_source = _source_from_duplicate(duplicate) or source
    ids = list(duplicate.get("ids") or [])

    if delete_source:
        try:
            existing_by_source = collection.get(
                where={"source": delete_source},
                include=[],
            )
            ids.extend(existing_by_source.get("ids") or [])
        except Exception:
            logger.exception("Failed to look up existing chunks by source=%s.", delete_source)

    ids = _unique_list(ids)
    if not ids:
        return 0

    try:
        collection.delete(ids=ids)
    except Exception as error:
        logger.exception("Failed to delete existing chunks for source=%s.", delete_source)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete existing document chunks before reprocessing.",
        ) from error

    logger.info("Deleted %d existing chunks for source=%s.", len(ids), delete_source)
    return len(ids)


def _duplicate_document_response(
    request: IngestRequest,
    document_hash: str | None,
    duplicate: dict[str, list[Any]],
) -> IngestResponse:
    """Build a clear, JSON-safe response for skipped duplicate uploads."""
    metadata = _safe_metadata(
        {
            **request.metadata_overrides,
            "document_hash": document_hash,
            "duplicate": True,
            "existing_chunks": len(duplicate.get("ids") or []),
            "existing_source": _source_from_duplicate(duplicate),
            "message": "Document already exists. Use force_reprocess=true to re-ingest.",
        }
    )
    return IngestResponse(
        status="already_exists",
        source=_source_from_request(request),
        chunks_created=0,
        chunks_stored=0,
        metadata=metadata,
    )


def _collection_from_processor(processor: IngestionProcessor) -> Any | None:
    """Return the processor's Chroma collection when available."""
    storer = getattr(processor, "storer", None)
    return getattr(storer, "collection", None)


def _source_from_duplicate(duplicate: dict[str, list[Any]]) -> str | None:
    """Return the first source metadata value from duplicate lookup results."""
    for metadata in duplicate.get("metadatas") or []:
        if isinstance(metadata, dict) and metadata.get("source"):
            return str(metadata["source"])
    return None


def _unique_list(values: list[Any]) -> list[Any]:
    """Deduplicate values while preserving their original order."""
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
