"""Conversation memory and session management for the LangGraph agent."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
from typing import Any, Literal, Optional
from uuid import uuid4


logger = logging.getLogger(__name__)

CheckpointerBackend = Literal["sqlite", "memory"]

DEFAULT_MEMORY_DB = "runtime/chatbot_memory.db"
DEFAULT_SESSION_REGISTRY = "runtime/chatbot_sessions.json"


def get_checkpointer(
    db_path: str = DEFAULT_MEMORY_DB,
    backend: CheckpointerBackend = "sqlite",
) -> Any:
    """Return a LangGraph checkpointer for conversation memory.

    SQLite is preferred for persistent local memory and uses LangGraph's
    regular ``SqliteSaver``. The synchronous saver is the compatibility path
    for Python 3.9 deployments on Amazon Linux and avoids async checkpoint API
    mismatches during graph execution.
    """
    if backend == "memory":
        return _memory_checkpointer()
    if backend != "sqlite":
        raise ValueError(f"Unsupported checkpointer backend: {backend}")

    try:
        return _sqlite_checkpointer(db_path)
    except ImportError:
        logger.warning(
            "SQLite checkpointer package is not installed; using in-memory memory."
        )
        return _memory_checkpointer()


def create_checkpointer(
    backend: CheckpointerBackend = "memory",
    sqlite_path: Optional[str] = None,
) -> Any:
    """Backward-compatible checkpointer factory."""
    if backend == "memory":
        return get_checkpointer(backend="memory")
    return get_checkpointer(db_path=sqlite_path or DEFAULT_MEMORY_DB, backend="sqlite")


@asynccontextmanager
async def checkpointer_context(
    db_path: str = DEFAULT_MEMORY_DB,
    backend: CheckpointerBackend = "sqlite",
) -> AsyncIterator[Any]:
    """Yield a checkpointer for FastAPI lifespan management.

    FastAPI lifespan hooks are async, but the checkpointer itself is now the
    synchronous ``SqliteSaver``. This wrapper keeps the SQLite connection open
    for the full application lifespan and closes it cleanly on shutdown.
    """
    if backend == "memory":
        yield _memory_checkpointer()
        return
    if backend != "sqlite":
        raise ValueError(f"Unsupported checkpointer backend: {backend}")

    try:
        saver = _sqlite_checkpointer(db_path)
        try:
            yield saver
        finally:
            _close_sqlite_checkpointer(saver)
    except ImportError:
        logger.warning(
            "SQLite checkpointer package is not installed; using in-memory memory."
        )
        yield _memory_checkpointer()


def generate_thread_id(user_id: str = "default") -> str:
    """Create a unique conversation thread ID for a user/session."""
    safe_user_id = _safe_user_id(user_id)
    return f"{safe_user_id}-{uuid4().hex[:12]}"


def create_thread_id(prefix: str = "session") -> str:
    """Backward-compatible alias for generating a thread ID."""
    return generate_thread_id(prefix)


def get_or_create_thread_id(user_id: str = "default") -> str:
    """Return the active thread ID for a user, creating one when missing."""
    registry = _load_session_registry()
    user_key = _safe_user_id(user_id)
    user_record = registry.setdefault(
        user_key,
        {"active_thread_id": None, "threads": []},
    )

    active_thread_id = user_record.get("active_thread_id")
    if active_thread_id:
        return str(active_thread_id)

    thread_id = generate_thread_id(user_key)
    user_record["active_thread_id"] = thread_id
    user_record.setdefault("threads", []).append(thread_id)
    _save_session_registry(registry)
    return thread_id


def save_thread_id(user_id: str, thread_id: str) -> str:
    """Mark a thread as the active conversation for a user."""
    registry = _load_session_registry()
    user_key = _safe_user_id(user_id)
    clean_thread_id = thread_id.strip()
    user_record = registry.setdefault(
        user_key,
        {"active_thread_id": None, "threads": []},
    )
    threads = user_record.setdefault("threads", [])
    if clean_thread_id not in threads:
        threads.append(clean_thread_id)
    user_record["active_thread_id"] = clean_thread_id
    _save_session_registry(registry)
    return clean_thread_id


def list_thread_ids(user_id: str = "default") -> list[str]:
    """List known thread IDs for a user."""
    registry = _load_session_registry()
    user_record = registry.get(_safe_user_id(user_id), {})
    return list(user_record.get("threads", []))


def graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build the LangGraph runtime config that selects a memory thread."""
    return {"configurable": {"thread_id": thread_id}}


def _sqlite_checkpointer(db_path: str) -> Any:
    """Create a persistent synchronous SQLite checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = _prepare_sqlite_path(db_path)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    logger.info("Loaded synchronous SQLite checkpointer at '%s'.", path)
    return saver


def _close_sqlite_checkpointer(checkpointer: Any) -> None:
    """Close the underlying SQLite connection when one is available."""
    connection = getattr(checkpointer, "conn", None)
    if connection is None:
        return

    try:
        connection.close()
    except Exception:
        logger.warning("Failed to close SQLite checkpointer connection.", exc_info=True)


def _prepare_sqlite_path(db_path: str) -> Path:
    """Ensure the SQLite parent directory exists and return the normalized path."""
    path = Path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _memory_checkpointer() -> Any:
    """Create an in-memory checkpointer."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def _load_session_registry() -> dict[str, dict[str, Any]]:
    """Load the lightweight local session registry."""
    path = Path(DEFAULT_SESSION_REGISTRY)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Session registry is unreadable; starting with an empty registry.")
        return {}

    return data if isinstance(data, dict) else {}


def _save_session_registry(registry: dict[str, dict[str, Any]]) -> None:
    """Persist the local session registry."""
    path = Path(DEFAULT_SESSION_REGISTRY)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_user_id(user_id: str) -> str:
    """Normalize user IDs so they are safe in thread IDs and registry keys."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", (user_id or "default").strip())
    return normalized.strip("-") or "default"
