import pytest
from conftest import ROOT, with_modules

from rag_lab.ingestion import ingest_directory, split_document
from rag_lab.tagging import TagCatalogError, TagClassificationError, load_tag_catalog


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


def test_ingestion_respects_embedding_bypass(services, tmp_path, monkeypatch):
    services.settings = with_modules(services.settings, "bm25")
    (tmp_path / "a.md").write_text(
        "O SQL consulta tabelas. RAG usa embeddings. O dragão voou.", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text("O SQL consulta tabelas.", encoding="utf-8")
    catalogs_loaded = []

    def track_catalog(path):
        catalogs_loaded.append(path)
        return load_tag_catalog(path)

    services.providers.chat_json.side_effect = [
        '{"tags": ["bancos-de-dados"]}',
        '{"tags": ["dados-e-inteligencia-artificial"]}',
        '{"tags": []}',
        '{"tags": ["bancos-de-dados"]}',
    ]
    monkeypatch.setattr("rag_lab.ingestion.load_tag_catalog", track_catalog)
    config_path = ROOT / "config/lab.yaml"
    result = ingest_directory(
        tmp_path, services.settings, services.store, services.providers, config_path=config_path
    )
    assert catalogs_loaded == [config_path]
    assert services.providers.chat_json.call_count == 4
    assert result["total_chunks"] == 4
    chunks, vectors, key = services.store.replaced[0]
    assert [chunk.metadata["tags"] for chunk in chunks] == [
        ["bancos-de-dados"],
        ["dados-e-inteligencia-artificial"],
        [],
    ]
    assert chunks[0].metadata["splitter"] == "sentence-v1"
    assert vectors is None and key is None


def test_missing_or_empty_corpus_fails(services, tmp_path):
    with pytest.raises(ValueError, match="Nenhum documento"):
        ingest_directory(tmp_path, services.settings, services.store, services.providers)


def test_ingestion_selects_terms_without_asking_the_llm(services, tmp_path):
    (tmp_path / "a.txt").write_text("Uma consulta SQL. Um fragmento.", encoding="utf-8")
    services.settings.tagging_method = "terms"
    ingest_directory(tmp_path, services.settings, services.store, services.providers)
    chunks, _, _ = services.store.replaced[0]
    assert [chunk.metadata["tags"] for chunk in chunks] == [["bancos-de-dados"], []]
    services.providers.chat_json.assert_not_called()


def test_embedding_failure_does_not_replace_document(services, tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("RAG.", encoding="utf-8")

    def fail(*args):
        raise RuntimeError("Falha na preparação")

    monkeypatch.setattr(services.providers, "embed", fail)
    with pytest.raises(RuntimeError, match="Falha na preparação"):
        ingest_directory(
            tmp_path,
            services.settings,
            services.store,
            services.providers,
            config_path=ROOT / "config/lab.yaml",
        )
    assert not services.store.replaced


@pytest.mark.parametrize("reply", ['{"tags": ["inventado"]}', TimeoutError("privado")])
def test_classification_failure_does_not_replace_document(services, tmp_path, monkeypatch, reply):
    (tmp_path / "a.txt").write_text("SQL. RAG.", encoding="utf-8")
    services.providers.chat_json.side_effect = ['{"tags": ["bancos-de-dados"]}', reply]

    def unexpected_call(*args):
        pytest.fail("A classificação deve terminar antes dos embeddings.")

    monkeypatch.setattr(services.providers, "embed", unexpected_call)
    with pytest.raises(TagClassificationError, match=r"a\.txt \(trecho 1\)") as failure:
        ingest_directory(tmp_path, services.settings, services.store, services.providers)
    assert "privado" not in str(failure.value)
    assert not services.store.replaced


def test_invalid_catalog_fails_before_storage_or_embeddings(services, tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("RAG.", encoding="utf-8")
    (tmp_path / "tags-si.yaml").write_text("rotulos: []", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def unexpected_call(*args):
        pytest.fail("A ingestão deve validar o catálogo primeiro.")

    monkeypatch.setattr(services.store, "initialize", unexpected_call)
    monkeypatch.setattr(services.providers, "embed", unexpected_call)
    with pytest.raises(TagCatalogError, match="tags-si.yaml"):
        ingest_directory(
            tmp_path,
            services.settings,
            services.store,
            services.providers,
            config_path=tmp_path / "lab.yaml",
        )
    services.providers.chat_json.assert_not_called()
