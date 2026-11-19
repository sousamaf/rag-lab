from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_json_classification_requests_no_thinking(settings, provider):
    settings.llm.provider = provider
    providers = Providers(settings)
    invoke = Mock(return_value=SimpleNamespace(text='{"tags": []}'))
    providers._chat = SimpleNamespace(invoke=invoke)
    schema = {"type": "object"}

    assert providers.chat_json("instruções", "trecho", schema=schema) == '{"tags": []}'
    if provider == "ollama":
        assert invoke.call_args.kwargs == {
            "reasoning": False,
            "format": schema,
            "stream": False,
            "options": {"temperature": settings.llm.temperature, "num_predict": 256},
        }
    else:
        assert invoke.call_args.kwargs == {
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        }

    providers.chat("instruções", "consulta")
    assert invoke.call_args.kwargs == {}
