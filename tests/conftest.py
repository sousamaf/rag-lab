from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from rag_lab.config import Settings
from rag_lab.contracts import Chunk, Evidence
from rag_lab.modules.base import Services

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Leitura pelo mesmo caminho de produção garante que o exemplo versionado seja válido.
    from rag_lab.config import load_settings

    return load_settings(ROOT / "config/lab.yaml")


@pytest.fixture
def config_data():
    return yaml.safe_load((ROOT / "config/lab.yaml").read_text())


class MemoryStore:
    def __init__(self):
        self.chunks = [
            Chunk(
                id="a",
                document_id="d1",
                source="a.txt",
                position=0,
                text="RAG recupera documentos relevantes.",
            ),
            Chunk(
                id="b",
                document_id="d1",
                source="a.txt",
                position=1,
                text="BM25 utiliza busca lexical.",
            ),
            Chunk(
                id="c",
                document_id="d2",
                source="b.txt",
                position=0,
                text="Embeddings representam significados como vetores.",
            ),
        ]
        self.replaced = []

    def initialize(self):
        pass

    def all_chunks(self):
        return deepcopy(self.chunks)

    def search_vector(self, vector, key, limit):
        return [
            Evidence(**c.model_dump(), score=1 / (i + 1)) for i, c in enumerate(self.chunks[:limit])
        ]

    def window(self, document_id, position, radius):
        return " ".join(
            c.text
            for c in self.chunks
            if c.document_id == document_id and abs(c.position - position) <= radius
        )

    def replace_document(self, chunks, vectors, key):
        self.replaced.append((chunks, vectors, key))


class FakeProviders:
    embedding_key = "test-model"

    def __init__(self):
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        if "hipotético" in system:
            return "DOCUMENTO HIPOTÉTICO SEM FONTE"
        if "Reformule" in system or "Decomponha" in system:
            return "1. O que é RAG?\n2. Como documentos são recuperados?"
        return "RAG recupera documentos relevantes [1]."

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def services(settings):
    return Services(settings, MemoryStore(), FakeProviders())


def with_modules(settings, *enabled):
    data = settings.model_dump()
    for name in data["modules"]:
        data["modules"][name]["enabled"] = name in enabled
    return Settings.model_validate(data)
