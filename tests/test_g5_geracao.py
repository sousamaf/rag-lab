"""Métricas de geração, citações e chamadas do G5 (mesmo molde das tabelas T3/T4 do G7)."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from conftest import with_modules

FOLDER = Path(__file__).resolve().parents[1] / "experimentos"
sys.path.insert(0, str(FOLDER))
spec = importlib.util.spec_from_file_location("g5_geracao", FOLDER / "g5_geracao.py")
g5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g5)
sys.path.remove(str(FOLDER))

EVIDENCE = [
    {"source": "a.txt", "position": 0},
    {"source": "a.txt", "position": 1},
    {"source": "b.txt", "position": 0},
]


def test_extract_citations_keeps_order_and_repeats():
    assert g5.extract_citations("Veja [2] e [1], de novo [2].") == [2, 1, 2]
    assert g5.extract_citations(None) == []


def test_evaluate_citations_precision_and_syntax():
    result = g5.evaluate_citations("A [1] e B [3].", EVIDENCE, ["a.txt:0"])
    assert result["has_citations"] and result["syntax_valid"]
    assert result["doc_precision"] == 0.5
    assert result["snippet_precision"] == 0.5


def test_evaluate_citations_out_of_bounds_and_missing():
    assert g5.evaluate_citations("Algo [9].", EVIDENCE, ["a.txt:0"])["syntax_valid"] is False
    none = g5.evaluate_citations("Sem citação.", EVIDENCE, ["a.txt:0"])
    assert not none["has_citations"] and none["doc_precision"] == 0.0


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Não encontrei evidências na base para responder à pergunta.", True),
        ("As evidências fornecidas não trazem essa informação.", True),
        ("", True),
        ("O parâmetro b normaliza o tamanho [1].", False),
    ],
)
def test_detect_abstention(answer, expected):
    assert g5.detect_abstention(answer) is expected


def test_fact_coverage_is_lexical_and_bounded():
    facts = ["b controla a normalização pelo tamanho do documento"]
    assert (
        g5.fact_coverage("A normalização pelo tamanho do documento é controlada por b.", facts)
        >= 0.8
    )
    assert g5.fact_coverage("Nada a ver.", facts) == 0.0
    assert g5.fact_coverage("qualquer texto", []) == 0.0


def questions():
    return [
        {
            "id": "L1",
            "perfil": "literal",
            "pergunta": "RAG documentos",
            "relevantes": ["a.txt:0"],
            "deve_conter": ["RAG recupera documentos relevantes"],
        },
        {
            "id": "N1",
            "perfil": "sem_resposta",
            "pergunta": "xyz",
            "relevantes": [],
            "deve_conter": ["a base não contém a informação"],
        },
    ]


class CitingProvider:
    embedding_key = None

    def chat(self, system, user):
        return "RAG recupera documentos relevantes [1]."


def test_collect_counts_calls_and_builds_t3_t4(services, tmp_path):
    settings = with_modules(services.settings, "bm25")
    records = g5.collect("cfg", settings, services.store, CitingProvider(), questions())
    assert [r["calls"]["chat_calls"] for r in records] == [1, 0]
    assert all(r["calls"]["embed_calls"] == 0 for r in records)

    [t3] = g5.t3_rows(records)
    assert t3["n_com_resposta"] == 1 and t3["n_sem_resposta"] == 1
    assert t3["taxa_respostas_com_citacao"] == 1.0
    assert t3["precisao_fontes_citadas_trecho"] == 1.0
    assert t3["taxa_abstencao_sem_resposta"] == 1.0
    assert t3["fidelidade_humana_status"] == "pendente_rubrica_humana_0_2"

    [t4] = g5.t4_rows(records)
    assert t4["n_consultas"] == 2 and t4["chat_calls_total"] == 1
    assert t4["wall_time_ms_p95"] >= t4["wall_time_ms_mediana"] > 0

    g5.write_annotation_template(tmp_path, records)
    with (tmp_path / "anotacao_humana.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["pergunta"] == "L1" and rows[0]["citacoes_extraidas"] == "[1]"
    assert rows[0]["fidelidade_0_2"] == ""
