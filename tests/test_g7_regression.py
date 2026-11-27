"""Regressão G7: desligamento seguro (bypass) e invariantes do Hybrid RAG."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from conftest import with_modules

from rag_lab.config import Settings, load_settings
from rag_lab.contracts import Chunk, Envelope, Evidence
from rag_lab.modules.hybrid import Hybrid
from rag_lab.pipeline import REGISTRY, Pipeline, select_without_fusion


def _make_evidence(chunk: Chunk, retriever: str, score: float) -> dict:
    return Evidence(**chunk.model_dump(), score=score, scores={retriever: score}).model_dump()


def test_regression_default_lab_yaml_is_untouched_and_bypasses_hybrid(
    services, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RAG_TAG_CLASSIFIER", "llm")
    default_settings = load_settings(root / "config" / "lab.yaml")

    assert default_settings.modules["embedding"].enabled is True
    for name, switch in default_settings.modules.items():
        if name != "embedding":
            assert switch.enabled is False

    def fail_if_called():
        raise AssertionError("Hybrid não deve ser instanciado quando desabilitado.")

    monkeypatch.setitem(REGISTRY, "hybrid", fail_if_called)
    services.settings = default_settings
    result = Pipeline(services).run(Envelope(question="RAG documentos"))

    hyb_trace = next(t for t in result.trace if t["module"] == "hybrid")
    assert hyb_trace["status"] == "bypass"
    assert "hybrid" not in result.attributes
    assert result.evidence
    for item in result.evidence:
        assert "hybrid" not in item.scores
        assert isinstance(item.metadata.get("tags"), list)
    assert result.evidence[0].metadata["tags"] == ["dados-e-inteligencia-artificial"]


@pytest.mark.parametrize(
    "retrievers",
    [
        ("bm25", "embedding"),
        ("tfidf", "bm25", "embedding"),
    ],
)
def test_regression_hybrid_disabled_with_2_and_3_retrievers_matches_round_robin(
    services, monkeypatch, retrievers
):
    services.settings = with_modules(services.settings, *retrievers)
    services.store.chunks[1].metadata["tags"] = ["bancos-de-dados"]

    def fail_if_called():
        raise AssertionError("Hybrid não deve ser instanciado em bypass.")

    monkeypatch.setitem(REGISTRY, "hybrid", fail_if_called)
    result = Pipeline(services).run(Envelope(question="RAG busca lexical vetores"))

    hyb_trace = next(t for t in result.trace if t["module"] == "hybrid")
    assert hyb_trace["status"] == "bypass"
    assert "hybrid" not in result.attributes

    expected_payload = {"retrieval": deepcopy(result.model_dump()["retrieval"]), "evidence": []}
    select_without_fusion(expected_payload)
    expected_ids = [
        item["id"] for item in expected_payload["evidence"][: services.settings.pipeline.top_k]
    ]
    assert [item.id for item in result.evidence] == expected_ids

    tags_by_id = {item.id: item.metadata.get("tags") for item in result.evidence}
    assert tags_by_id["a"] == ["dados-e-inteligencia-artificial"]
    assert tags_by_id["b"] == ["bancos-de-dados"]
    for item in result.evidence:
        assert "hybrid" not in item.scores


def test_regression_yaml_hybrid_section_and_request_hybrid_config_ignored_when_disabled(
    services, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    raw_dbsf = yaml.safe_load((root / "config" / "lab.yaml").read_text(encoding="utf-8"))
    raw_dbsf["tagging_method"] = "llm"
    raw_dbsf["modules"]["bm25"]["enabled"] = True
    raw_dbsf["modules"]["embedding"]["enabled"] = True
    raw_dbsf["modules"]["hybrid"]["enabled"] = False
    raw_dbsf["hybrid"] = {"strategy": "dbsf", "weights": {"bm25": 1.0, "embedding": 2.0}}
    cfg_2ret = Settings.model_validate(raw_dbsf)

    raw_1ret = deepcopy(raw_dbsf)
    raw_1ret["modules"]["bm25"]["enabled"] = False
    raw_1ret["hybrid"] = {"strategy": "weighted_rrf", "weights": {"embedding": 1.0}}
    cfg_1ret = Settings.model_validate(raw_1ret)

    def fail_if_called():
        raise AssertionError("Hybrid não deve ser instanciado quando disabled.")

    monkeypatch.setitem(REGISTRY, "hybrid", fail_if_called)

    for cfg in (cfg_2ret, cfg_1ret):
        services.settings = cfg
        result = Pipeline(services).run(
            Envelope(
                question="RAG busca lexical",
                attributes={"hybrid_config": {"strategy": "dbsf"}, "grupo": "G7"},
            )
        )
        hyb_trace = next(t for t in result.trace if t["module"] == "hybrid")
        assert hyb_trace["status"] == "bypass"
        assert result.attributes["grupo"] == "G7"
        assert "hybrid" not in result.attributes
        for item in result.evidence:
            assert "hybrid" not in item.scores


@pytest.mark.parametrize(
    ("strategy", "weights"),
    [
        ("rrf", {}),
        ("weighted_rrf", {"bm25": 1.0, "embedding": 2.0}),
        ("convex", {}),
        ("dbsf", {"bm25": 1.0, "embedding": 1.0}),
    ],
)
def test_regression_hybrid_enabled_preserves_tags_attributes_scores_and_retrieval_immutability(
    services, strategy, weights
):
    services.settings = with_modules(services.settings, "bm25", "embedding", "hybrid")
    services.settings.hybrid.strategy = strategy
    services.settings.hybrid.weights = weights
    services.store.chunks[1].metadata["tags"] = ["bancos-de-dados"]

    result = Pipeline(services).run(
        Envelope(
            question="RAG busca lexical",
            attributes={"grupo": "G7", "req_tag": "ok"},
        )
    )
    hyb_trace = next(t for t in result.trace if t["module"] == "hybrid")
    assert hyb_trace["status"] == "executed"
    assert result.attributes["grupo"] == "G7"
    assert result.attributes["req_tag"] == "ok"
    assert result.attributes["hybrid"]["schema"] == "g7-hybrid/1"
    assert result.attributes["hybrid"]["strategy"] == strategy

    for item in result.evidence:
        assert set(item.scores).issubset({"bm25", "embedding", "hybrid"})
        assert "hybrid" in item.scores
        assert isinstance(item.metadata.get("tags"), list)

    dumped = result.model_dump(mode="json")
    serialized = json.dumps(dumped, allow_nan=False)
    assert Envelope.model_validate(json.loads(serialized))


def test_regression_round_robin_vs_rrf_regimes_and_scale_invariant_tiebreak(services):
    def make_chunk(cid: str, pos: int) -> Chunk:
        return Chunk(id=cid, document_id="d1", source="a.txt", position=pos, text=cid)

    # Cenário C=12: "a_iso1" é 1º só em bm25; "b_cons12" é 12º em bm25 e 12º em embedding.
    bm25_list = [_make_evidence(make_chunk("a_iso1", 0), "bm25", 50.0)]
    for i in range(1, 11):
        bm25_list.append(_make_evidence(make_chunk(f"z_b_{i:02d}", i), "bm25", 40.0 - i))
    bm25_list.append(_make_evidence(make_chunk("b_cons12", 99), "bm25", 1.0))

    emb_list = [
        _make_evidence(make_chunk(f"z_e_{i:02d}", 100 + i), "embedding", 0.9 - i * 0.02)
        for i in range(1, 12)
    ] + [_make_evidence(make_chunk("b_cons12", 99), "embedding", 0.1)]

    retrieval = {"bm25": bm25_list, "embedding": emb_list}

    # Round-robin preserva ordem de inserção (1º de bm25 vence independente de consenso)
    rr_payload = {"retrieval": deepcopy(retrieval), "evidence": []}
    select_without_fusion(rr_payload)
    assert rr_payload["evidence"][0]["id"] == "a_iso1"

    # RRF com k <= 10 (k=1, k=10): 1º isolado vence (ou empata por id "a_iso1" < "b_cons12")
    for k_low in (1, 10):
        services.settings.pipeline.rrf_k = k_low
        p_low = Envelope(question="RAG").model_dump()
        p_low["retrieval"] = deepcopy(retrieval)
        res_low = Hybrid().run(p_low, services)
        assert res_low["evidence"][0]["id"] == "a_iso1"

    # RRF com k > 21 (k=60, k=120): consenso m=2 na 12ª posição vence 1º isolado
    for k_high in (60, 120):
        services.settings.pipeline.rrf_k = k_high
        p_high = Envelope(question="RAG").model_dump()
        p_high["retrieval"] = deepcopy(retrieval)
        res_high = Hybrid().run(p_high, services)
        assert res_high["evidence"][0]["id"] == "b_cons12"

    # Desempate invariante à escala em r=3 (1/(60+3)): id do embedding vence mesmo com score menor
    services.settings.pipeline.rrf_k = 60
    tie_payload = Envelope(question="RAG").model_dump()
    tie_payload["retrieval"] = {
        "bm25": [
            _make_evidence(make_chunk("01_b", 1), "bm25", 50.0),
            _make_evidence(make_chunk("02_b", 2), "bm25", 40.0),
            _make_evidence(make_chunk("z_bm25_r3", 3), "bm25", 30.0),
        ],
        "embedding": [
            _make_evidence(make_chunk("01_e", 11), "embedding", 0.95),
            _make_evidence(make_chunk("02_e", 12), "embedding", 0.85),
            _make_evidence(make_chunk("a_emb_r3", 13), "embedding", 0.68),
        ],
    }
    tie_res = Hybrid().run(tie_payload, services)
    pair = [e["id"] for e in tie_res["evidence"] if e["id"] in {"a_emb_r3", "z_bm25_r3"}]
    assert pair == ["a_emb_r3", "z_bm25_r3"]


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), float("-inf")])
def test_regression_hybrid_rejects_non_finite_scores_and_emits_strict_json(services, bad_score):
    chunk = services.store.all_chunks()[0]
    bad_item = chunk.model_dump()
    bad_item["score"] = bad_score
    bad_item["scores"] = {"bm25": bad_score}
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {"bm25": [bad_item], "embedding": []}

    with pytest.raises(ValueError, match="Score bruto inválido"):
        Hybrid().run(payload, services)

    valid_payload = Envelope(question="RAG").model_dump()
    valid_payload["retrieval"] = {
        "bm25": [_make_evidence(chunk, "bm25", 12.0)],
        "embedding": [_make_evidence(chunk, "embedding", 0.9)],
    }
    ok_res = Hybrid().run(valid_payload, services)
    serialized = json.dumps(ok_res, allow_nan=False)
    assert Envelope.model_validate(json.loads(serialized))
