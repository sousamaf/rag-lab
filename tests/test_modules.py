import pytest

from rag_lab.contracts import Envelope, Evidence
from rag_lab.modules.bm25 import BM25
from rag_lab.modules.hybrid import Hybrid
from rag_lab.modules.sentence import SentenceWindow
from rag_lab.modules.tfidf import TFIDF


def test_rrf_rewards_agreement_without_adding_raw_scores(services):
    chunks = services.store.all_chunks()
    a = Evidence(**chunks[0].model_dump(), score=1000, scores={"bm25": 1000}).model_dump()
    b = Evidence(**chunks[1].model_dump(), score=1, scores={"bm25": 1}).model_dump()
    c = Evidence(**chunks[1].model_dump(), score=0.9, scores={"embedding": 0.9}).model_dump()
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {"bm25": [a, b], "embedding": [c]}
    result = Hybrid().run(payload, services)
    assert result["evidence"][0]["id"] == "b"
    assert result["evidence"][0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert result["evidence"][0]["scores"] == {
        "bm25": 1,
        "embedding": 0.9,
        "hybrid": pytest.approx(1 / 62 + 1 / 61),
    }


def test_sentence_window_keeps_document_boundary(services):
    item = Evidence(**services.store.chunks[0].model_dump()).model_dump()
    payload = Envelope(question="RAG").model_dump()
    payload["evidence"] = [item]
    result = SentenceWindow().run(payload, services)
    assert "BM25" in result["evidence"][0]["context_text"]
    assert "Embeddings" not in result["evidence"][0]["context_text"]


def window_payload(services, chunk_indexes, radius):
    """Executa a janela sobre trechos do MemoryStore: d1 = [a, b] e d2 = [c]."""
    services.settings.pipeline.sentence_radius = radius
    payload = Envelope(question="RAG").model_dump()
    payload["evidence"] = [
        Evidence(**services.store.chunks[i].model_dump()).model_dump() for i in chunk_indexes
    ]
    return SentenceWindow().run(payload, services)


def run_window(services, chunk_indexes, radius):
    return window_payload(services, chunk_indexes, radius)["evidence"]


def test_sentence_window_radius_zero_keeps_only_the_sentence(services):
    [item] = run_window(services, [0], radius=0)
    assert item["context_text"] == item["text"]


def test_sentence_window_large_radius_stays_in_document(services):
    [item] = run_window(services, [0], radius=10)
    assert item["context_text"] == "RAG recupera documentos relevantes. BM25 utiliza busca lexical."


def test_sentence_window_preserves_original_text(services):
    original = [chunk.text for chunk in services.store.chunks]
    evidence = run_window(services, [0, 1, 2], radius=1)
    assert [item["text"] for item in evidence] == original


@pytest.mark.xfail(
    strict=True,
    reason="Problema conhecido: janelas de sentenças vizinhas repetem texto no contexto.",
)
def test_sentence_window_neighbors_do_not_repeat_sentences(services):
    evidence = run_window(services, [0, 1], radius=1)
    sentences = [chunk.text for chunk in services.store.chunks[:2]]
    for sentence in sentences:
        assert sum(sentence in item["context_text"] for item in evidence) == 1


def test_sentence_window_metrics_count_overlap_only_in_same_document(services):
    payload = window_payload(services, [0, 1, 2], radius=1)
    metrics = payload["attributes"]["sentence_window"]
    assert metrics == {
        "radius": 1,
        "evidence_count": 3,
        "original_chars": sum(len(item["text"]) for item in payload["evidence"]),
        "window_chars": sum(len(item["context_text"]) for item in payload["evidence"]),
        "overlapping_pairs": 1,
    }
    assert metrics["window_chars"] > metrics["original_chars"]


def test_sentence_window_metrics_radius_zero_has_no_expansion(services):
    metrics = window_payload(services, [0, 1, 2], radius=0)["attributes"]["sentence_window"]
    assert metrics["window_chars"] == metrics["original_chars"]
    assert metrics["overlapping_pairs"] == 0


def test_sentence_window_metrics_consider_only_top_k(services):
    services.settings.pipeline.top_k = 1
    metrics = window_payload(services, [0, 1], radius=1)["attributes"]["sentence_window"]
    assert metrics["evidence_count"] == 1
    assert metrics["overlapping_pairs"] == 0


@pytest.mark.parametrize("module", [TFIDF, BM25])
def test_lexical_empty_and_unmatched_queries(module, services):
    payload = Envelope(question="xyz", queries=["xyz"]).model_dump()
    result = module().run(payload, services)
    assert not any(result["retrieval"].values())
    services.store.chunks = []
    assert not any(module().run(payload, services)["retrieval"].values())


def test_bm25_single_document_still_finds_term(services):
    services.store.chunks = services.store.chunks[:1]
    payload = Envelope(question="RAG", queries=["RAG"]).model_dump()
    result = BM25().run(payload, services)
    assert result["retrieval"]["bm25"][0]["score"] > 0
