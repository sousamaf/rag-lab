import pytest
from conftest import with_modules

from rag_lab.ingestion import ingest_directory, split_document


def test_ingestion_sentence_order_and_stable_identity():
    chunks = split_document("a.txt", "Primeira sentença. Segunda sentença!\nTerceira.", 100)
    assert [c.position for c in chunks] == [0, 1, 2]
    assert chunks == split_document("a.txt", "Primeira sentença. Segunda sentença!\nTerceira.", 100)
    assert (
        chunks[0].document_id != split_document("b.txt", "Primeira sentença.", 100)[0].document_id
    )
    assert (
        chunks[0].document_id == split_document("a.txt", "Conteúdo atualizado.", 100)[0].document_id
    )


def test_ingestion_respects_embedding_bypass(services, tmp_path):
    services.settings = with_modules(services.settings, "bm25")
    (tmp_path / "a.md").write_text("RAG recupera documentos.", encoding="utf-8")
    result = ingest_directory(tmp_path, services.settings, services.store, services.providers)
    assert result["total_chunks"] == 1
    assert services.store.replaced[0][1:] == (None, None)


def test_missing_or_empty_corpus_fails(services, tmp_path):
    with pytest.raises(ValueError, match="Nenhum documento"):
        ingest_directory(tmp_path, services.settings, services.store, services.providers)


def test_provider_failure_does_not_replace_document(services, tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("RAG.", encoding="utf-8")

    def fail(texts):
        raise RuntimeError("Endpoint indisponível")

    monkeypatch.setattr(services.providers, "embed", fail)
    with pytest.raises(RuntimeError):
        ingest_directory(tmp_path, services.settings, services.store, services.providers)
    assert not services.store.replaced
