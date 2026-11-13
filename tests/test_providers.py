from types import SimpleNamespace

import pytest

from rag_lab.providers import Providers


@pytest.mark.parametrize("vectors", [[[1, 2]], [[0] * 768], [[float("nan")] * 768], []])
def test_embedding_dimension_zero_nonfinite_and_count_are_rejected(settings, vectors):
    providers = Providers(settings)
    providers._embeddings = SimpleNamespace(embed_documents=lambda texts: vectors)
    with pytest.raises(ValueError):
        providers.embed(["RAG"])


def test_model_change_changes_embedding_identity(settings):
    providers = Providers(settings)
    original = providers.embedding_key
    settings.embeddings.model = "another-model"
    assert providers.embedding_key != original


def test_openai_compatible_requires_explicit_api_key(settings, monkeypatch):
    monkeypatch.delenv(settings.llm.api_key_env, raising=False)
    with pytest.raises(ValueError, match="Defina"):
        Providers.api_key(settings.llm)
