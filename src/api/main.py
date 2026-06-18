"""FastAPI application factory for the Tata Nexon chatbot API.

The module exposes ``create_app`` for tests, ASGI servers, and future deployment
entry points. Runtime dependencies are initialized inside FastAPI's lifespan
hook so importing this module does not eagerly connect to external services.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import ApiRuntime, build_runtime_dependencies
from src.api.routes.admin import router as admin_router
from src.api.routes.chat import router as chat_router
from src.api.schemas import ErrorResponse, HealthResponse
from src.agent.memory import checkpointer_context
from src.config.settings import Settings, load_settings


logger = logging.getLogger(__name__)
BINARY_ERROR_PLACEHOLDER = "<binary data omitted: {size} bytes>"
MAX_ERROR_STRING_LENGTH = 500


def create_app(
    settings: Settings | None = None,
    graph: Any | None = None,
    checkpointer: Any | None = None,
    ingestion_processor: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional typed settings override, primarily used by tests.
        graph: Optional precompiled LangGraph override.
        checkpointer: Optional LangGraph checkpointer override.
        ingestion_processor: Optional ingestion processor override.

    Returns:
        Configured FastAPI application with routes, CORS, lifespan, and error
        handlers registered.
    """
    active_settings = settings or load_settings()
    _configure_logging(active_settings)

    app = FastAPI(
        title=active_settings.app.app_name,
        version=active_settings.app.version,
        lifespan=_lifespan_factory(
            settings=active_settings,
            graph=graph,
            checkpointer=checkpointer,
            ingestion_processor=ingestion_processor,
        ),
    )
    _configure_cors(app, active_settings)
    _register_exception_handlers(app)
    _register_routes(app, active_settings)
    return app


def _lifespan_factory(
    settings: Settings,
    graph: Any | None,
    checkpointer: Any | None,
    ingestion_processor: Any | None,
):
    """Create the FastAPI lifespan context manager for runtime setup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            active_checkpointer = checkpointer
            if active_checkpointer is None:
                active_checkpointer = await stack.enter_async_context(
                    checkpointer_context(
                        db_path=settings.memory.sqlite_path,
                        backend=settings.memory.backend,
                    )
                )

            runtime = build_runtime_dependencies(
                settings=settings,
                graph=graph,
                checkpointer=active_checkpointer,
                ingestion_processor=ingestion_processor,
            )
            _attach_runtime(app, runtime)
            logger.info("%s API started.", settings.app.app_name)
            yield
            logger.info("%s API stopped.", settings.app.app_name)

    return lifespan


def _attach_runtime(app: FastAPI, runtime: ApiRuntime) -> None:
    """Attach long-lived dependencies to ``app.state``."""
    for name, value in runtime.as_state_items().items():
        setattr(app.state, name, value)
    app.state.documents = []


def _configure_logging(settings: Settings) -> None:
    """Configure logging from the typed settings layer."""
    logging.basicConfig(level=getattr(logging, settings.app.log_level, logging.WARNING))


def _configure_cors(app: FastAPI, settings: Settings) -> None:
    """Install CORS middleware using configured allowed origins."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.app.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_routes(app: FastAPI, settings: Settings) -> None:
    """Register public, admin, and health routes."""
    app.include_router(chat_router)
    app.include_router(admin_router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Return basic service health without exposing sensitive settings."""
        return HealthResponse(
            status="ok",
            service=settings.app.app_name,
            version=settings.app.version,
        )


def _register_exception_handlers(app: FastAPI) -> None:
    """Register API-wide structured error handlers."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        safe_detail = _safe_json_value(exc.detail)
        logger.info("HTTP error for %s %s: %s", request.method, request.url.path, safe_detail)
        return _error_response(
            status_code=exc.status_code,
            code="http_error",
            message=_safe_error_message(safe_detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.info(
            "Validation error for %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return _error_response(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error for %s %s", request.method, request.url.path)
        return _error_response(
            status_code=500,
            code="internal_error",
            message="An unexpected API error occurred.",
        )


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a consistent JSON error response."""
    payload = ErrorResponse(
        error={
            "code": code,
            "message": _safe_error_message(message),
            "details": _safe_json_value(details or {}),
        }
    )
    return JSONResponse(status_code=status_code, content=_safe_json_value(payload.model_dump()))


def _safe_json_value(value: Any) -> Any:
    """Recursively convert values into JSON-serializable primitives.

    Multipart validation errors can contain raw bytes or framework objects in
    their ``input`` fields. Returning those directly from ``JSONResponse`` raises
    serialization errors, so the API sanitizes every error payload before it
    leaves the exception handler.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _truncate_error_string(value)
    if isinstance(value, bytes):
        return BINARY_ERROR_PLACEHOLDER.format(size=len(value))
    if isinstance(value, bytearray):
        return BINARY_ERROR_PLACEHOLDER.format(size=len(value))
    if isinstance(value, memoryview):
        return BINARY_ERROR_PLACEHOLDER.format(size=value.nbytes)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return _truncate_error_string(str(value))
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json_value(item) for item in value]
    if hasattr(value, "filename") and hasattr(value, "file"):
        filename = getattr(value, "filename", "uploaded-file")
        return f"<uploaded file omitted: {filename}>"
    return _truncate_error_string(str(value))


def _safe_error_message(value: Any) -> str:
    """Return a short string message without raw binary content."""
    safe_value = _safe_json_value(value)
    if isinstance(safe_value, str):
        return safe_value
    return _truncate_error_string(str(safe_value))


def _truncate_error_string(value: str) -> str:
    """Limit error string length so responses never echo large payloads."""
    if len(value) <= MAX_ERROR_STRING_LENGTH:
        return value
    return f"{value[:MAX_ERROR_STRING_LENGTH]}... <truncated>"


app = create_app()
