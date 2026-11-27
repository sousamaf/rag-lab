"""Verifica as métricas do experimento do G5 (recall, MRR e médias por perfil)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experimentos/g5_bm25.py"
spec = importlib.util.spec_from_file_location("g5_bm25", SCRIPT)
g5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g5)


def top(*labels):
    return [{"id": label, "score": 10.0 - i} for i, label in enumerate(labels)]


def test_evaluate_hit_at_first_position():
    result = g5.evaluate(top("a.txt:0", "b.txt:1"), ["a.txt:0"])
    assert result == {"recall_trecho": 1.0, "recall_documento": 1.0, "mrr": 1.0, "primeiro": 1}


def test_evaluate_partial_recall_and_reciprocal_rank():
    result = g5.evaluate(top("x.txt:0", "a.txt:1", "z.txt:2"), ["a.txt:1", "a.txt:3"])
    assert result["recall_trecho"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["primeiro"] == 2


def test_evaluate_document_recall_counts_other_chunks_of_same_document():
    result = g5.evaluate(top("a.txt:0"), ["a.txt:3"])
    assert result["recall_trecho"] == 0.0
    assert result["recall_documento"] == 1.0
    assert result["mrr"] == 0.0
    assert result["primeiro"] is None


def test_evaluate_miss_everything():
    result = g5.evaluate(top("x.txt:0"), ["a.txt:1"])
    assert result == {"recall_trecho": 0.0, "recall_documento": 0.0, "mrr": 0.0, "primeiro": None}


def test_evaluate_unanswerable_reports_results_and_first_score():
    assert g5.evaluate(top("a.txt:0", "b.txt:0"), []) == {"resultados": 2, "score1": 10.0}
    assert g5.evaluate([], []) == {"resultados": 0, "score1": 0.0}


def test_summarize_averages_by_profile():
    questions = [
        {"id": "L1", "perfil": "literal", "relevantes": ["a:0"]},
        {"id": "L2", "perfil": "literal", "relevantes": ["a:1"]},
        {"id": "N1", "perfil": "sem_resposta", "relevantes": []},
        {"id": "N2", "perfil": "sem_resposta", "relevantes": []},
    ]
    per_question = {
        "L1": {"recall_trecho": 1.0, "recall_documento": 1.0, "mrr": 1.0, "primeiro": 1},
        "L2": {"recall_trecho": 0.0, "recall_documento": 1.0, "mrr": 0.0, "primeiro": None},
        "N1": {"resultados": 3, "score1": 4.0},
        "N2": {"resultados": 0, "score1": 0.0},
    }
    summary = g5.summarize(per_question, questions)
    assert summary["literal"] == {"recall_trecho": 0.5, "recall_documento": 1.0, "mrr": 0.5}
    assert summary["sem_resposta"] == {"com_resultado": 0.5, "score_1o": 2.0}
    assert summary["com_resposta"]["mrr"] == 0.5
