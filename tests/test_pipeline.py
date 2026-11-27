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


@pytest.mark.parametrize("hybrid", [False, True])
def test_tags_follow_chunk_id_through_fusion_and_context_steps(services, hybrid):
    enabled = ["tfidf", "bm25", "embedding", "reranking", "sentence", "compressao"]
    if hybrid:
        enabled.append("hybrid")
    services.settings = with_modules(services.settings, *enabled)
    neighbor = services.store.chunks[1]
    neighbor.text = "RAG consulta bancos de dados com SQL."
    neighbor.metadata["tags"] = ["bancos-de-dados"]

    result = Pipeline(services).run(Envelope(question="RAG"))
    payload = json.loads(result.model_dump_json())
    assert set(payload["retrieval"]) == {"tfidf", "bm25", "embedding"}
    expected = ["dados-e-inteligencia-artificial"]
    for ranking in payload["retrieval"].values():
        item = next(item for item in ranking if item["id"] == "a")
        assert item["metadata"]["tags"] == expected

    evidence = {item["id"]: item for item in payload["evidence"]}
    assert "com SQL" in evidence["a"]["context_text"]
    assert evidence["a"]["metadata"]["tags"] == expected
    assert evidence["b"]["metadata"]["tags"] == ["bancos-de-dados"]
    assert evidence["c"]["metadata"]["tags"] == []


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


@pytest.mark.parametrize(
    ("strategy", "weights"),
    [
        ("rrf", {}),
        ("weighted_rrf", {"bm25": 1.0, "embedding": 2.0}),
        ("convex", {}),
        ("dbsf", {"bm25": 1.0, "embedding": 1.0}),
    ],
)
def test_pipeline_hybrid_strategies_emit_telemetry_and_preserve_contract(
    services, strategy, weights
):
    services.settings = with_modules(services.settings, "bm25", "embedding", "hybrid")
    services.settings.hybrid.strategy = strategy
    services.settings.hybrid.weights = weights
    services.settings.pipeline.top_k = 2

    result = Pipeline(services).run(
        Envelope(question="RAG documentos", attributes={"grupo": "G7"})
    )
    trace_by_module = {entry["module"]: entry for entry in result.trace}

    assert trace_by_module["hybrid"]["status"] == "executed"
    assert result.attributes["grupo"] == "G7"
    telemetry = result.attributes["hybrid"]
    assert telemetry["schema"] == "g7-hybrid/1"
    assert telemetry["strategy"] == strategy
    assert len(telemetry["ranking"]) >= len(result.evidence)
    assert len(result.evidence) <= 2
    assert "hybrid" in result.evidence[0].scores
    assert result.model_dump_json()


def test_pipeline_without_hybrid_uses_round_robin_and_leaves_hybrid_bypassed(services):
    services.settings = with_modules(services.settings, "tfidf", "bm25", "embedding")
    result = Pipeline(services).run(Envelope(question="RAG documentos"))
    trace_by_module = {entry["module"]: entry for entry in result.trace}

    assert trace_by_module["hybrid"]["status"] == "bypass"
    assert "hybrid" not in result.attributes
    assert result.evidence
    assert "hybrid" not in result.evidence[0].scores

