"""Envelope JSON compartilhado. Cada módulo enriquece e preserva os demais atributos."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    text: str
    source: str
    document_id: str
    position: int
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_tags(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        if "tags" not in metadata:
            return {**metadata, "tags": []}
        tags = metadata["tags"]
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("metadata.tags deve ser uma lista de strings.")
        return metadata


class Evidence(Chunk):
    score: float = 0
    scores: dict[str, float] = Field(default_factory=dict)
    context_text: str | None = None
    citation: int | None = None


class Envelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "1.0"
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str = Field(min_length=1, max_length=10000)
    queries: list[str] = Field(default_factory=list)
    retrieval: dict[str, list[Evidence]] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    context: str = ""
    answer: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
