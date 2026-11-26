"""Testes unitários e de integração para o ferramental G7 (Fase F9: rodar.py e avaliar.py)."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest
from conftest import with_modules

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.g7 import avaliar, rodar  # noqa: E402
from rag_lab.contracts import Chunk, Envelope, Evidence  # noqa: E402
from rag_lab.pipeline import Pipeline  # noqa: E402


def _make_evidence_dict(chunk: Chunk, retriever: str, score: float) -> dict:
    return Evidence(**chunk.model_dump(), score=score, scores={retriever: score}).model_dump()


def test_document_and_snippet_metrics_manual_values_and_empty_lists():
    ranking = [
        {"id": "c1", "source": "01_arq.md", "position": 1, "text": "Proxy Envoy na porta 9443."},
        {
            "id": "c2",
            "source": "03_banco.md",
            "position": 4,
            "text": "O motor aborta a busca imediatamente com o código de erro ERR-AUR-417.",
        },
        {
            "id": "c3",
            "source": "03_banco.md",
            "position": 5,
            "text": "Outro trecho do mesmo documento 03_banco.md sem o segundo fato.",
        },
        {
            "id": "c4",
            "source": "06_seg.md",
            "position": 5,
            "text": "Convertida em erro genérico 503 sem exibir URLs ou chaves de API.",
        },
    ]
    rel_docs = ["03_banco.md", "06_seg.md"]
    rel_snippets = [
        {
            "fonte": "03_banco.md",
            "trecho": "aborta a busca imediatamente com o codigo de erro ERR-AUR-417",
            "essencial": True,
        },
        {
            "fonte": "06_seg.md",
            "trecho": "convertida em erro generico 503 sem exibir URLs ou chaves de API",
            "essencial": True,
        },
    ]

    doc_m = avaliar.compute_document_metrics(ranking, rel_docs)
    assert doc_m["doc_recall_1"] == pytest.approx(0.0)
    assert doc_m["doc_recall_3"] == pytest.approx(0.5)
    assert doc_m["doc_recall_5"] == pytest.approx(1.0)
    assert doc_m["doc_recall_10"] == pytest.approx(1.0)
    assert doc_m["doc_hit_5"] == pytest.approx(1.0)
    assert doc_m["doc_mrr"] == pytest.approx(0.5)  # 1º relevante na posição 2

    # nDCG@5 doc: acertos únicos nas posições 2 (03_banco.md) e 4 (06_seg.md); pos 3 é duplicata.
    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert doc_m["doc_ndcg_5"] == pytest.approx(expected_dcg / expected_idcg)

    snp_m = avaliar.compute_snippet_metrics(ranking, rel_snippets)
    assert snp_m["trecho_recall_1"] == pytest.approx(0.0)
    assert snp_m["trecho_recall_3"] == pytest.approx(0.5)
    assert snp_m["trecho_recall_5"] == pytest.approx(1.0)
    assert snp_m["trecho_hit_5"] == pytest.approx(1.0)
    assert snp_m["trecho_mrr"] == pytest.approx(0.5)
    assert snp_m["trecho_ndcg_5"] == pytest.approx(expected_dcg / expected_idcg)

    # Listas vazias e perguntas sem_resposta (documentos_relevantes=[] / trechos_relevantes=[])
    assert avaliar.compute_document_metrics([], rel_docs)["doc_mrr"] == 0.0
    assert avaliar.compute_document_metrics(ranking, [])["doc_recall_5"] == 0.0
    assert avaliar.compute_snippet_metrics(ranking, [])["trecho_ndcg_5"] == 0.0


def test_ndcg_weights_essential_snippets_and_jaccard_edge_cases():
    ranking = [
        {"id": "c1", "source": "a.md", "position": 0, "text": "Trecho complementar secundário."},
        {"id": "c2", "source": "a.md", "position": 1, "text": "Trecho essencial principal."},
    ]
    snippets = [
        {"fonte": "a.md", "trecho": "Trecho essencial principal", "essencial": True},
        {"fonte": "a.md", "trecho": "Trecho complementar secundário", "essencial": False},
    ]
    snp_m = avaliar.compute_snippet_metrics(ranking, snippets)
    # Posição 1 tem ganho 1.0 (não essencial), posição 2 tem ganho 2.0 (essencial)
    expected_dcg = (1.0 / math.log2(2)) + (2.0 / math.log2(3))
    expected_idcg = (2.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert snp_m["trecho_ndcg_5"] == pytest.approx(expected_dcg / expected_idcg)

    assert avaliar.jaccard_index(set(), set()) == 0.0
    assert avaliar.jaccard_index({"a", "b"}, {"b", "c", "d"}) == pytest.approx(0.25)
    assert avaliar.normalize_text("  Vazamento   de   Topologia e Segredos! ") == (
        "vazamento de topologia e segredos!"
    )


def test_citation_parser_validity_precision_and_abstention_detection():
    evidence = [
        {
            "id": "e1",
            "source": "06_seguranca_criptografia.md",
            "position": 5,
            "text": "Qualquer exceção é convertida em erro genérico 503 sem exibir URLs.",
        },
        {
            "id": "e2",
            "source": "01_arquitetura_geral.md",
            "position": 2,
            "text": "Proxy reverso Envoy na porta 9443.",
        },
    ]
    q_spec = {
        "id": "S02",
        "responder": True,
        "documentos_relevantes": ["06_seguranca_criptografia.md"],
        "trechos_relevantes": [
            {
                "fonte": "06_seguranca_criptografia.md",
                "trecho": "convertida em erro genérico 503 sem exibir URLs",
                "essencial": True,
            }
        ],
    }

    # 1) Citações válidas [1] e [2] (1 de 2 aponta para a fonte relevante)
    res_valid = avaliar.evaluate_citations(
        "A borda converte em erro 503 [1] após passar pelo Envoy [2].", evidence, q_spec
    )
    assert res_valid["citations"] == [1, 2]
    assert res_valid["syntax_valid"] is True
    assert res_valid["out_of_bounds_citations"] == []
    assert res_valid["cited_doc_precision"] == pytest.approx(0.5)
    assert res_valid["cited_snippet_precision"] == pytest.approx(0.5)

    # 2) Citação fora do intervalo [99]
    res_invalid = avaliar.evaluate_citations("Fato inventado [99] e [1].", evidence, q_spec)
    assert res_invalid["syntax_valid"] is False
    assert res_invalid["out_of_bounds_citations"] == [99]

    # 3) Ausência de citações
    res_none = avaliar.evaluate_citations("Resposta sem colchetes.", evidence, q_spec)
    assert res_none["has_citations"] is False
    assert res_none["syntax_valid"] is True
    assert res_none["cited_doc_precision"] == 0.0

    # 4) Detecção de abstenção
    assert avaliar.detect_abstention(
        "Não encontrei evidências na base para responder à pergunta."
    )
    assert avaliar.detect_abstention(
        "As evidências fornecidas não mencionam parâmetros HNSW."
    )
    assert not avaliar.detect_abstention(
        "O motor aborta a busca com o código ERR-AUR-417 [1]."
    )


def test_fusion_tax_detector_captures_s02_like_degradation_and_clears_weighted_rrf():
    # Simula exatamente a geometria de S02: alvo está em #3 no embedding e ausente no bm25.
    target = Chunk(
        id="e59f_target",
        document_id="d6",
        source="06_seguranca_criptografia.md",
        position=5,
        text="Convertida em erro genérico 503 sem exibir URLs ou chaves de API.",
    )
    cons1 = Chunk(id="52cd_c1", document_id="d1", source="01_arq.md", position=2, text="Envoy.")
    cons2 = Chunk(id="24a9_c2", document_id="d1", source="01_arq.md", position=5, text="JSON.")
    bm25_1 = Chunk(id="336f_b1", document_id="d3", source="03_db.md", position=3, text="SHA.")
    bm25_2 = Chunk(id="2387_b2", document_id="d11", source="11_dep.md", position=2, text="CLI.")
    bm25_3 = Chunk(id="5483_b3", document_id="d7", source="07_bkp.md", position=7, text="Eletrica.")
    emb_2 = Chunk(id="86cf_e2", document_id="d10", source="10_sla.md", position=0, text="SLA.")

    retrieval = {
        "bm25": [
            _make_evidence_dict(bm25_1, "bm25", 10.2),
            _make_evidence_dict(bm25_2, "bm25", 10.1),
            _make_evidence_dict(bm25_3, "bm25", 9.9),
            _make_evidence_dict(cons1, "bm25", 9.0),
            _make_evidence_dict(cons2, "bm25", 8.1),
        ],
        "embedding": [
            _make_evidence_dict(cons1, "embedding", 0.72),
            _make_evidence_dict(emb_2, "embedding", 0.71),
            _make_evidence_dict(target, "embedding", 0.69),
            _make_evidence_dict(cons2, "embedding", 0.66),
        ],
    }
    q_spec = {
        "id": "S02",
        "perfil": "semantica",
        "responder": True,
        "documentos_relevantes": ["06_seguranca_criptografia.md"],
        "trechos_relevantes": [
            {
                "fonte": "06_seguranca_criptografia.md",
                "trecho": "erro genérico 503 sem exibir URLs ou chaves de API",
                "essencial": True,
            }
        ],
    }

    # 1) RRF simétrico (k=60): target cai de #3 (embedding) para #7 (fora do Top-5) -> Fusion Tax!
    rrf_rep = avaliar.replay_from_retrieval(retrieval, mode="hybrid", strategy="rrf", rrf_k=60)
    env_rrf = {
        "retrieval": retrieval,
        "evidence": rrf_rep["evidence_top_k"],
        "attributes": {"hybrid": rrf_rep["telemetry"]},
        "trace": [{"module": "hybrid", "status": "executed", "duration_ms": 1.0}],
    }
    tax = avaliar.detect_fusion_tax_for_envelope(env_rrf, q_spec, config_name="e2_rrf_k60")
    assert tax is not None
    assert tax["severidade"] == "perda_top5"
    assert tax["best_single_retriever"] == "embedding"
    assert tax["best_single_rank"] == 3
    assert tax["fused_rank"] == 7
    assert tax["fused_recall_5"] == 0.0
    assert tax["delta_recall5_vs_best_single"] == pytest.approx(-1.0)

    # 2) Weighted RRF (bm25=1.0, embedding=2.0): target sobe para #4 (Recall@5 = 1.0, MRR 1/4 < 1/3)
    w_rep = avaliar.replay_from_retrieval(
        retrieval,
        mode="hybrid",
        strategy="weighted_rrf",
        weights={"bm25": 1.0, "embedding": 2.0},
        rrf_k=60,
    )
    assert w_rep["telemetry"]["ranking"][3]["id"] == "e59f_target"
    env_w = {
        "retrieval": retrieval,
        "evidence": w_rep["evidence_top_k"],
        "attributes": {"hybrid": w_rep["telemetry"]},
        "trace": [{"module": "hybrid", "status": "executed", "duration_ms": 1.0}],
    }
    tax_w = avaliar.detect_fusion_tax_for_envelope(env_w, q_spec, config_name="e4_weighted_rrf")
    assert tax_w is not None
    assert tax_w["severidade"] == "queda_mrr_no_top5"
    assert tax_w["fused_rank"] == 4
    assert tax_w["fused_recall_5"] == 1.0
    assert tax_w["delta_recall5_vs_best_single"] == pytest.approx(0.0)
    assert tax_w["delta_mrr_vs_best_single"] == pytest.approx(0.25 - 1.0 / 3.0, abs=1e-6)

    # 3) E1 / bypass_rr (hybrid: false) e E0 (1 recuperador) NUNCA são listados como Fusion Tax
    env_bypass = {
        "retrieval": retrieval,
        "evidence": rrf_rep["evidence_top_k"],
        "attributes": {},
        "trace": [{"module": "hybrid", "status": "bypass", "duration_ms": 0.0}],
    }
    assert avaliar.detect_fusion_tax_for_envelope(env_bypass, q_spec, config_name="e1_rr") is None
    env_e0 = {
        "retrieval": {"embedding": retrieval["embedding"]},
        "evidence": retrieval["embedding"][:5],
        "attributes": {},
        "trace": [{"module": "hybrid", "status": "bypass", "duration_ms": 0.0}],
    }
    assert avaliar.detect_fusion_tax_for_envelope(env_e0, q_spec, config_name="e0_emb") is None


def test_extract_full_ranking_reconstructs_e0_and_e1_beyond_top_k_for_recall10_and_mrr():
    # Cria 8 chunks distintos; o alvo relevante estará exatamente na posição #6 (fora do top_k=5).
    chunks = [
        Chunk(
            id=f"c{i}",
            document_id=f"d{i}",
            source=f"doc_{i}.md",
            position=i,
            text=f"Sentença {i} do documento {i}.",
        )
        for i in range(1, 9)
    ]
    rel_docs = ["doc_6.md"]
    rel_snippets = [{"fonte": "doc_6.md", "trecho": "Sentença 6 do documento 6", "essencial": True}]

    # Caso A: Bloco E0 (1 único recuperador ativo, candidate_k=8, mas evidence truncado em top_k=5)
    e0_list = [_make_evidence_dict(c, "bm25", float(20 - i)) for i, c in enumerate(chunks)]
    env_e0 = {
        "retrieval": {"bm25": e0_list},
        "evidence": e0_list[:5],  # truncado em top_k = 5 por build_context
        "attributes": {},
        "trace": [{"module": "hybrid", "status": "bypass", "duration_ms": 0.0}],
    }
    full_e0 = avaliar.extract_full_ranking(env_e0)
    assert len(full_e0) == 8
    doc_m_e0 = avaliar.compute_document_metrics(full_e0, rel_docs)
    snp_m_e0 = avaliar.compute_snippet_metrics(full_e0, rel_snippets)
    assert doc_m_e0["doc_recall_5"] == 0.0
    assert doc_m_e0["doc_recall_10"] == 1.0
    assert doc_m_e0["doc_mrr"] == pytest.approx(1.0 / 6.0)
    assert snp_m_e0["trecho_recall_5"] == 0.0
    assert snp_m_e0["trecho_recall_10"] == 1.0
    assert snp_m_e0["trecho_mrr"] == pytest.approx(1.0 / 6.0)

    # Caso B: Bloco E1 / bypass_rr (2 recuperadores, round-robin coloca doc_6.md na 6ª posição)
    # Ordem round-robin: bm25#1(c1), emb#1(c2), bm25#2(c3), emb#2(c4), bm25#3(c5), emb#3(c6=ALVO)
    bm25_rr = [_make_evidence_dict(chunks[i], "bm25", 10.0 - i) for i in (0, 2, 4, 6)]
    emb_rr = [_make_evidence_dict(chunks[i], "embedding", 0.9 - i * 0.05) for i in (1, 3, 5, 7)]
    env_e1 = {
        "retrieval": {"bm25": bm25_rr, "embedding": emb_rr},
        "evidence": [bm25_rr[0], emb_rr[0], bm25_rr[1], emb_rr[1], bm25_rr[2]],  # top_k = 5
        "attributes": {},
        "trace": [{"module": "hybrid", "status": "bypass", "duration_ms": 0.0}],
    }
    full_e1 = avaliar.extract_full_ranking(env_e1)
    assert [item["id"] for item in full_e1] == ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    doc_m_e1 = avaliar.compute_document_metrics(full_e1, rel_docs)
    snp_m_e1 = avaliar.compute_snippet_metrics(full_e1, rel_snippets)
    assert doc_m_e1["doc_recall_5"] == 0.0
    assert doc_m_e1["doc_recall_10"] == 1.0
    assert doc_m_e1["doc_mrr"] == pytest.approx(1.0 / 6.0)
    assert snp_m_e1["trecho_recall_5"] == 0.0
    assert snp_m_e1["trecho_recall_10"] == 1.0
    assert snp_m_e1["trecho_mrr"] == pytest.approx(1.0 / 6.0)


@pytest.mark.parametrize(
    ("hybrid_enabled", "strategy", "weights", "rrf_k"),
    [
        (False, "rrf", {}, 60),
        (True, "rrf", {}, 10),
        (True, "rrf", {}, 60),
        (True, "weighted_rrf", {"tfidf": 0.5, "bm25": 1.0, "embedding": 2.0}, 60),
        (True, "convex", {"tfidf": 1.0, "bm25": 1.0, "embedding": 2.0}, 60),
        (True, "dbsf", {}, 60),
    ],
)
def test_offline_replay_matches_online_pipeline_across_all_strategies_and_rrf_k(
    services, hybrid_enabled, strategy, weights, rrf_k
):
    enabled = ["tfidf", "bm25", "embedding"]
    if hybrid_enabled:
        enabled.append("hybrid")
    services.settings = with_modules(services.settings, *enabled)
    services.settings.pipeline.rrf_k = rrf_k
    services.settings.pipeline.top_k = 2
    services.settings.hybrid.strategy = strategy
    services.settings.hybrid.weights = weights

    online_env = Pipeline(services).run(Envelope(question="RAG busca lexical vetores")).model_dump(
        mode="json"
    )
    record = {
        "run_id": "test_run",
        "config_name": f"{strategy}_{rrf_k}",
        "question_id": "Q1",
        "perfil": "literal",
        "rep": 1,
        "envelope": online_env,
    }
    check = avaliar.verify_replay_equivalence([record])
    assert check["total_checked"] == 1
    assert check["all_matched"] is True
    assert check["mismatches"] == []


def test_instrumented_providers_and_manifest_secret_sanitization(services, tmp_path):
    services.settings.database_url = "postgresql://rag:SUPER_SECRET_PASS_123@db.internal:5432/rag_g7"
    services.settings.llm.base_url = "http://user:TOKEN_SECRET_456@localhost:11434"

    sanitized = rodar.sanitize_settings(services.settings)
    serialized = json.dumps(sanitized)
    assert "SUPER_SECRET_PASS_123" not in serialized
    assert "TOKEN_SECRET_456" not in serialized
    assert "database_url" not in sanitized
    assert sanitized["database_name"] == "rag_g7"

    inst = rodar.InstrumentedProviders(services.providers)
    assert inst.chat("sys", "user")
    assert inst.embed(["t1", "t2", "t3"])
    assert inst.snapshot_counts() == {
        "chat_calls": 1,
        "embed_calls": 1,
        "embed_texts": 3,
    }
    inst.reset_counts()
    assert inst.snapshot_counts() == {
        "chat_calls": 0,
        "embed_calls": 0,
        "embed_texts": 0,
    }

    # Verifica determinismo de file_sha256 entre LF e CRLF (sha256-lf)
    lf_file = tmp_path / "lf.yaml"
    crlf_file = tmp_path / "crlf.yaml"
    lf_file.write_bytes(b"id: L01\npergunta: teste\n")
    crlf_file.write_bytes(b"id: L01\r\npergunta: teste\r\n")
    assert rodar.file_sha256(lf_file) == rodar.file_sha256(crlf_file)


def test_end_to_end_rodar_and_avaliar_generates_t1_to_t6_and_human_csv(services, tmp_path):
    run_out = tmp_path / "runs"
    eval_out = tmp_path / "results"
    config_path = REPO_ROOT / "experiments" / "g7" / "configs" / "e2_rrf_k60_bm25_emb.yaml"
    gabarito_path = REPO_ROOT / "experiments" / "g7" / "perguntas_gabarito.yaml"
    corpus_dir = REPO_ROOT / "experiments" / "g7" / "corpus"

    args_collect = rodar.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--gabarito",
            str(gabarito_path),
            "--corpus-dir",
            str(corpus_dir),
            "--out-dir",
            str(run_out),
            "--run-id",
            "piloto_mem",
            "--ids",
            "L02",
            "S02",
            "N01",
            "--llm-model",
            "lfm3:1.7b-q4_K_M",
        ]
    )
    manifest = rodar.run_collection(args_collect, services_override=services)
    assert manifest["run_id"] == "piloto_mem"
    assert manifest["overrides"]["llm_model"] == "lfm3:1.7b-q4_K_M"
    assert manifest["summary"]["completed_runs"] == 3
    assert "database_url" not in json.dumps(manifest)

    args_eval = avaliar.build_parser().parse_args(
        [
            "--runs-dir",
            str(run_out),
            "--gabarito",
            str(gabarito_path),
            "--out-dir",
            str(eval_out),
        ]
    )
    summary = avaliar.run_evaluation(args_eval)
    assert summary["ok"] is True
    assert summary["records_evaluated"] == 3
    assert summary["replay_verification"]["all_matched"] is True

    for stem in (
        "t1_recuperacao",
        "t2_curva_rrf_k",
        "t3_geracao_citacoes",
        "t4_latencia_chamadas",
        "t5_fusion_tax",
        "t6_sobreposicao_empates",
    ):
        assert (eval_out / f"{stem}.csv").exists()
        assert (eval_out / f"{stem}.md").exists()

    human_csv = eval_out / "avaliacao_humana_template.csv"
    assert human_csv.exists()
    with human_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {r["question_id"] for r in rows} == {"L02", "S02", "N01"}
