"""Ingestion pipeline configuration."""

from __future__ import annotations

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Scanner, parser, and chunker settings."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.txt"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    chunk_size: int = Field(
        default=1000,
        ge=50,
        validation_alias=AliasChoices("INGESTION_CHUNK_SIZE", "CHUNK_SIZE"),
    )
    overlap: int = Field(
        default=200,
        ge=0,
        validation_alias=AliasChoices("INGESTION_OVERLAP", "CHUNK_OVERLAP"),
    )
    supported_file_types: tuple[str, ...] = Field(
        default=(".pdf", ".txt", ".md"),
        validation_alias="INGESTION_SUPPORTED_FILE_TYPES",
    )
    max_file_size_mb: int = Field(
        default=50,
        gt=0,
        validation_alias="INGESTION_MAX_FILE_SIZE_MB",
    )

    @field_validator("supported_file_types")
    @classmethod
    def _normalize_file_types(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = []
        for value in values:
            file_type = value.strip().lower()
            if not file_type:
                continue
            normalized.append(file_type if file_type.startswith(".") else f".{file_type}")

        if not normalized:
            raise ValueError("At least one supported file type is required.")
        return tuple(normalized)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk_size(self) -> "IngestionSettings":
        if self.overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        return self
