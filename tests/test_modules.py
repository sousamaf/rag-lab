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
