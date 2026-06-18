"""Application-wide typed settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.ingestion import IngestionSettings
from src.config.llm import LLMSettings
from src.config.retrieval import RetrievalSettings


class AppSettings(BaseSettings):
    """General runtime settings."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.txt"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    app_name: str = Field(default="Tata Nexon Chatbot", validation_alias="APP_NAME")
    version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    environment: str = Field(default="local", validation_alias="APP_ENV")
    app_debug: bool = Field(default=False, validation_alias="APP_DEBUG")
    log_level: str = Field(default="WARNING", validation_alias="LOG_LEVEL")
    cors_allowed_origins: tuple[str, ...] = Field(
        default=(
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:5173",
        ),
        validation_alias="CORS_ALLOWED_ORIGINS",
    )
    max_generation_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        validation_alias="MAX_GENERATION_ATTEMPTS",
    )

    @property
    def debug(self) -> bool:
        """Return whether debug mode is enabled."""
        return self.app_debug

    @field_validator("app_name", "version", "environment", "log_level")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("App settings must not be empty.")
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(sorted(allowed))}.")
        return normalized

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(origin.strip() for origin in value.split(",") if origin.strip())
        return value


class MemorySettings(BaseSettings):
    """LangGraph checkpointer and CLI session settings."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.txt"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    backend: Literal["sqlite", "memory"] = Field(
        default="sqlite",
        validation_alias=AliasChoices("CHECKPOINTER_BACKEND", "MEMORY_BACKEND"),
    )
    sqlite_path: str = Field(
        default="chatbot_memory.db",
        validation_alias=AliasChoices("CHATBOT_MEMORY_DB", "MEMORY_DB_PATH"),
    )
    default_user_id: str = Field(
        default="default",
        validation_alias="CHATBOT_USER_ID",
    )
    session_registry_path: str = Field(
        default="chatbot_sessions.json",
        validation_alias="CHATBOT_SESSION_REGISTRY",
    )

    @field_validator("sqlite_path", "default_user_id", "session_registry_path")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Memory settings must not be empty.")
        return value


class Settings(BaseSettings):
    """Root settings object grouped by application domain."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.txt"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load and cache application settings."""
    return Settings()
