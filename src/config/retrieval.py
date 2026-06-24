"""Retrieval and vector store configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    """Retriever and ChromaDB settings."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.txt"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        validation_alias=AliasChoices("AGENT_TOP_K", "RETRIEVAL_TOP_K"),
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        validation_alias="RETRIEVAL_SIMILARITY_THRESHOLD",
    )
    collection_name: str = Field(
        default="tata_nexon_chunks",
        validation_alias="CHROMA_COLLECTION_NAME",
    )
    persist_directory: str = Field(
        default="runtime/chroma",
        validation_alias="CHROMA_PERSIST_DIRECTORY",
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias="CHROMA_EMBEDDING_DIMENSION",
    )

    @field_validator("collection_name", "persist_directory")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Retrieval configuration values must not be empty.")
        return value
