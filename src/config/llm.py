"""LLM and embedding configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """OpenAI chat and embedding settings."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.txt"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        repr=False,
    )
    chat_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("OPENAI_CHAT_MODEL", "CHAT_MODEL"),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    embedding_dimensions: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "OPENAI_EMBEDDING_DIMENSIONS",
            "CHROMA_EMBEDDING_DIMENSION",
        ),
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        validation_alias="OPENAI_TIMEOUT_SECONDS",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        validation_alias="OPENAI_MAX_RETRIES",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=2048,
        validation_alias="OPENAI_EMBEDDING_BATCH_SIZE",
    )

    @property
    def api_key(self) -> Optional[str]:
        """Return the raw OpenAI API key when configured."""
        if self.openai_api_key is None:
            return None
        return self.openai_api_key.get_secret_value()

    @property
    def has_api_key(self) -> bool:
        """Return whether an OpenAI API key is available."""
        return bool(self.api_key)

    @field_validator("chat_model", "embedding_model")
    @classmethod
    def _non_empty_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Model names must not be empty.")
        return value
