"""Contrato do YAML. Opções desconhecidas são erros, evitando experimentos silenciosos."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

MODULE_ORDER = (
    "multi_query",
    "hyde",
    "decomposicao",
    "tfidf",
    "bm25",
    "embedding",
    "hybrid",
    "reranking",
    "sentence",
    "compressao",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Endpoint(StrictModel):
    provider: Literal["ollama", "openai"]
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    timeout: float = Field(default=120, gt=0, le=600)


class LLMConfig(Endpoint):
    temperature: float = Field(default=0, ge=0, le=2)


class EmbeddingConfig(Endpoint):
    dimensions: int = Field(gt=0, le=16000)


class Switch(StrictModel):
    enabled: StrictBool


class PipelineConfig(StrictModel):
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=12, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    expanded_queries: int = Field(default=3, ge=1, le=10)
    sentence_radius: int = Field(default=1, ge=0, le=10)
    max_context_chars: int = Field(default=12000, ge=100, le=100000)
    max_sentence_chars: int = Field(default=1000, ge=100, le=10000)


class Settings(StrictModel):
    database_url: str = Field(min_length=1, repr=False)
    llm: LLMConfig
    embeddings: EmbeddingConfig
    pipeline: PipelineConfig
    modules: dict[str, Switch]

    @model_validator(mode="after")
    def validate_pipeline(self) -> Settings:
        if set(self.modules) != set(MODULE_ORDER):
            raise ValueError(f"modules deve declarar exatamente: {', '.join(MODULE_ORDER)}")
        retrievers = sum(self.modules[name].enabled for name in ("tfidf", "bm25", "embedding"))
        if self.modules["hybrid"].enabled and retrievers < 2:
            raise ValueError("Hybrid exige ao menos dois recuperadores ativos.")
        if self.modules["hyde"].enabled and not self.modules["embedding"].enabled:
            raise ValueError("HyDE exige Embedding ativo.")
        if self.pipeline.candidate_k < self.pipeline.top_k:
            raise ValueError("candidate_k deve ser maior ou igual a top_k.")
        return self


def _expand(value):
    """Expande variáveis nos valores após ler YAML, sem reinterpretar senhas como YAML."""
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name, default = match.group(1), match.group(2)
        result = os.getenv(name, default)
        if result is None:
            raise ValueError(f"Variável de ambiente ausente: {name}")
        return result

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", replace, value)


def load_settings(path: str | Path | None = None) -> Settings:
    load_dotenv(Path.cwd() / ".env", override=False)
    config_path = Path(path or os.getenv("RAG_CONFIG", "config/lab.yaml"))
    with config_path.open(encoding="utf-8") as file:
        return Settings.model_validate(_expand(yaml.safe_load(file)))
