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


def run_bm25(services, *queries):
    payload = Envelope(question=queries[0], queries=list(queries)).model_dump()
    return BM25().run(payload, services)


def test_bm25_ranks_chunk_with_query_term_first(services):
    result = run_bm25(services, "busca lexical")
    ranking = result["retrieval"]["bm25"]
    assert [item["id"] for item in ranking] == ["b"]
    assert ranking[0]["scores"]["bm25"] == ranking[0]["score"] > 0


def test_bm25_multi_query_keeps_best_score_per_chunk(services):
    result = run_bm25(services, "RAG", "busca lexical")
    assert {item["id"] for item in result["retrieval"]["bm25"]} == {"a", "b"}


def test_bm25_respects_candidate_k(services):
    services.settings.pipeline.candidate_k = 1
    services.settings.pipeline.top_k = 1
    result = run_bm25(services, "RAG busca embeddings")
    assert len(result["retrieval"]["bm25"]) == 1


def bm25_with(services, **options):
    data = services.settings.model_dump()
    data["modules"]["bm25"].update(options)
    services.settings = type(services.settings).model_validate(data)


def test_bm25_defaults_are_reported_in_attributes(services):
    metrics = run_bm25(services, "RAG")["attributes"]["bm25"]
    assert (metrics["k1"], metrics["b"], metrics["idf"]) == (1.5, 0.75, "lucene")
    assert metrics["chunks"] == 3 and metrics["hits"] == 1


def test_bm25_parameters_change_scores(services):
    base = run_bm25(services, "RAG")["retrieval"]["bm25"][0]["score"]
    bm25_with(services, k1=3.0)
    assert run_bm25(services, "RAG")["retrieval"]["bm25"][0]["score"] != base


def test_bm25_b_zero_disables_length_normalization(services):
    bm25_with(services, b=0.0)
    scores = {
        item["id"]: item["score"] for item in run_bm25(services, "RAG busca")["retrieval"]["bm25"]
    }
    assert scores["a"] == scores["b"]


def test_bm25_idf_variants_differ_and_lucene_stays_positive(services):
    services.store.chunks = services.store.chunks[:1]
    lucene = run_bm25(services, "RAG")["retrieval"]["bm25"]
    bm25_with(services, idf="okapi")
    okapi = run_bm25(services, "RAG")["retrieval"]["bm25"]
    assert lucene and lucene[0]["score"] > 0
    assert not okapi or okapi[0]["score"] != lucene[0]["score"]


def test_bm25_classic_idf_is_zero_for_term_in_every_document(services):
    services.store.chunks = services.store.chunks[:1]
    bm25_with(services, idf="classic")
    assert run_bm25(services, "RAG")["retrieval"]["bm25"] == []


def test_bm25_classic_idf_scores_rare_terms(services):
    bm25_with(services, idf="classic")
    ranking = run_bm25(services, "lexical")["retrieval"]["bm25"]
    assert [item["id"] for item in ranking] == ["b"]
    assert ranking[0]["score"] > 0


def common_word_chunks(services):
    from rag_lab.contracts import Chunk

    texts = [
        "de busca rara zebra",
        "de outra frase comum",
        "de mais uma frase",
        "de texto qualquer aqui",
    ]
    services.store.chunks = [
        Chunk(id=str(i), document_id="d", source="d.txt", position=i, text=t)
        for i, t in enumerate(texts)
    ]


def test_bm25_min_idf_zero_keeps_every_query_term(services):
    common_word_chunks(services)
    result = run_bm25(services, "de")
    assert len(result["retrieval"]["bm25"]) == 4
    assert result["attributes"]["bm25"]["termos_ignorados"] == 0


def test_bm25_min_idf_ignores_common_terms_but_keeps_rare_ones(services):
    common_word_chunks(services)
    bm25_with(services, min_idf=1.0)
    only_common = run_bm25(services, "de")
    assert only_common["retrieval"]["bm25"] == []
    assert only_common["attributes"]["bm25"]["termos_ignorados"] == 1
    mixed = run_bm25(services, "de zebra")
    assert [item["id"] for item in mixed["retrieval"]["bm25"]] == ["0"]
    assert mixed["attributes"]["bm25"]["min_idf"] == 1.0
    assert mixed["attributes"]["bm25"]["termos_ignorados"] == 1
