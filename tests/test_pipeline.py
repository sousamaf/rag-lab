import json

import pytest
from conftest import with_modules

from rag_lab.config import MODULE_ORDER
from rag_lab.contracts import Envelope
from rag_lab.pipeline import Pipeline


def test_every_disabled_module_is_bypassed(services):
    services.settings = with_modules(services.settings)
    result = Pipeline(services).run(
        Envelope(
            question="O que é RAG?",
            attributes={"group": "G1"},
            experiment={"strategy": "baseline"},
        )
    )
    assert len(result.trace) == 10
    assert all(item["status"] == "bypass" for item in result.trace)
    assert services.providers.calls == []
    assert result.evidence == []
    assert "Não encontrei" in result.answer
    assert result.attributes["group"] == "G1"
    assert result.model_dump()["experiment"] == {"strategy": "baseline"}


@pytest.mark.parametrize("retriever", ["tfidf", "bm25", "embedding"])
def test_independent_retrievers_return_sources_and_preserve_attributes(services, retriever):
    services.settings = with_modules(services.settings, retriever)
    result = Pipeline(services).run(Envelope(question="RAG documentos", attributes={"student": 42}))
    assert result.evidence[0].id == "a"
    assert result.evidence[0].citation == 1
    assert result.attributes["student"] == 42
    assert result.retrieval[retriever]
    assert result.evidence[0].scores[retriever] > 0
    assert "[1] a.txt" in result.context
    assert result.model_dump_json()


def test_all_ten_modules_compose_and_hyde_is_not_evidence(services):
    services.settings = with_modules(services.settings, *MODULE_ORDER)
    result = Pipeline(services).run(Envelope(question="RAG documentos"))
    assert [t["module"] for t in result.trace[:10]] == list(MODULE_ORDER)
    assert all(t["status"] == "executed" for t in result.trace)
    assert len(result.queries) > 1
    assert "DOCUMENTO HIPOTÉTICO" in result.attributes["hyde"]
    assert "DOCUMENTO HIPOTÉTICO" not in result.context
    assert "DOCUMENTO HIPOTÉTICO" not in services.providers.calls[-1][1]
    assert "hybrid" in result.evidence[0].scores
    assert "reranking" in result.evidence[0].scores


def test_context_budget_and_source_numbers_match(services):
    services.settings = with_modules(services.settings, "embedding")
    services.settings.pipeline.max_context_chars = 100
    result = Pipeline(services).run(Envelope(question="RAG"))
    assert len(result.context) <= 100
    for item in result.evidence:
        assert f"[{item.citation}] {item.source}" in result.context
        assert item.context_text in result.context


def test_request_cannot_inject_retrieval_or_hyde(services):
    services.settings = with_modules(services.settings)
    envelope = Envelope.model_validate(
        {
            "question": "RAG",
            "attributes": {"hyde": "FONTE FALSA"},
            "context": "FONTE FALSA",
            "answer": "FONTE FALSA",
            "queries": ["FONTE FALSA"],
            "trace": [{"module": "injected"}],
        }
    )
    result = Pipeline(services).run(envelope)
    assert "FONTE FALSA" not in json.dumps(result.model_dump())


def test_bypass_does_not_instantiate_module(services, monkeypatch):
    from rag_lab.pipeline import REGISTRY

    def fail():
        raise AssertionError("Módulo desativado foi construído")

    services.settings = with_modules(services.settings, "tfidf")
    monkeypatch.setitem(REGISTRY, "hyde", fail)
    assert Pipeline(services).run(Envelope(question="RAG")).evidence
