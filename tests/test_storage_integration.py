"""Use banco dedicado: TEST_DATABASE_URL=postgresql://... pytest -m integration."""

import json
import os
from uuid import uuid4

import pytest
import yaml
from conftest import ROOT, with_modules

from rag_lab.contracts import Chunk, Envelope
from rag_lab.ingestion import ingest_directory, split_document
from rag_lab.modules.base import Services
from rag_lab.pipeline import Pipeline
from rag_lab.storage import PostgresStore


@pytest.fixture
def postgres_store():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Defina TEST_DATABASE_URL para testar PostgreSQL/pgvector reais")
    store = PostgresStore(url)
    store.initialize()
    return store


@pytest.mark.integration
def test_pgvector_search_replace_and_mismatch(postgres_store):
    store = postgres_store
    document_id = uuid4().hex
    chunk = Chunk(
        id=document_id,
        document_id=document_id,
        source="test.txt",
        position=0,
        text="Documento de integração.",
    )
    try:
        store.replace_document([chunk], [[1.0, 0.0, 0.0]], "test-model")
        assert store.window(document_id, 0, 1) == chunk.text
        result = store.search_vector([1.0, 0.0, 0.0], "test-model", 5)
        assert result[0].id == chunk.id
        assert result[0].score == pytest.approx(1)
        chunk.text = "Texto atualizado."
        store.replace_document([chunk], [[0.0, 1.0, 0.0]], "test-model")
        assert store.window(document_id, 0, 0) == "Texto atualizado."
        with pytest.raises(ValueError, match="Reingira"):
            store.search_vector([1.0, 0.0, 0.0], "other-model", 5)
    finally:
        with store.connect() as connection:
            connection.execute("DELETE FROM rag_chunks WHERE document_id = %s", (document_id,))


@pytest.mark.integration
def test_example_tags_survive_pgvector_and_reingestion(
    postgres_store, services, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store = postgres_store
    settings = with_modules(services.settings, "tfidf", "bm25", "embedding", "hybrid")
    providers = services.providers
    source = f"{uuid4().hex}.txt"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = (ROOT / "examples/documents/rag.txt").read_text(encoding="utf-8")
    (corpus / source).write_text(text, encoding="utf-8")
    config_path = tmp_path / "lab.yaml"
    catalog_path = config_path.with_name("tags-si.yaml")
    catalog_path.write_bytes((ROOT / "config/tags-si.yaml").read_bytes())
    document_id = split_document(source, text, settings.pipeline.max_sentence_chars)[0].document_id

    def model_reply(system, user, *, schema):
        if not json.loads(user)["trecho"].startswith("Embeddings"):
            return '{"tags": []}'
        if "Catálogo revisado para a atividade" in system:
            return '{"tags": ["bancos-de-dados"]}'
        return '{"tags": ["dados-e-inteligencia-artificial"]}'

    providers.chat_json.side_effect = model_reply

    try:
        ingest_directory(corpus, settings, store, providers, config_path=config_path)
        stored = {
            chunk.id: chunk for chunk in store.all_chunks() if chunk.document_id == document_id
        }
        chunk = next(item for item in stored.values() if item.text.startswith("Embeddings"))
        assert chunk.metadata == {
            "tags": ["dados-e-inteligencia-artificial"],
            "unit": "sentence",
            "splitter": "sentence-v1",
        }

        result = Pipeline(Services(settings, store, providers)).run(Envelope(question="Embeddings"))
        assert set(result.retrieval) == {"tfidf", "bm25", "embedding"}
        for items in (*result.retrieval.values(), result.evidence):
            item = next(item for item in items if item.id == chunk.id)
            assert item.metadata == chunk.metadata

        legacy = next(item for item in stored.values() if item.id != chunk.id)
        with store.connect() as connection:
            connection.execute(
                "UPDATE rag_chunks SET metadata = metadata - 'tags' WHERE id = %s", (legacy.id,)
            )
        lexical = next(item for item in store.all_chunks() if item.id == legacy.id)
        vector = next(
            item
            for item in store.search_vector([1.0, 0.0, 0.0], providers.embedding_key, 100)
            if item.id == legacy.id
        )
        assert (
            lexical.metadata
            == vector.metadata
            == {
                "tags": [],
                "unit": "sentence",
                "splitter": "sentence-v1",
            }
        )

        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        banks = next(label for label in catalog["rotulos"] if label["id"] == "bancos-de-dados")
        banks["descricao"] += " Catálogo revisado para a atividade."
        catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
        ingest_directory(corpus, settings, store, providers, config_path=config_path)

        updated = {item.id: item for item in store.all_chunks() if item.document_id == document_id}
        assert updated.keys() == stored.keys()
        assert updated[chunk.id].metadata["tags"] == ["bancos-de-dados"]
    finally:
        with store.connect() as connection:
            connection.execute("DELETE FROM rag_chunks WHERE document_id = %s", (document_id,))
