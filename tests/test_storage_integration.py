"""Use banco dedicado: TEST_DATABASE_URL=postgresql://... pytest -m integration."""

import os
from uuid import uuid4

import pytest

from rag_lab.contracts import Chunk
from rag_lab.storage import PostgresStore


@pytest.mark.integration
def test_pgvector_search_replace_and_mismatch():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Defina TEST_DATABASE_URL para testar PostgreSQL/pgvector reais")
    store = PostgresStore(url)
    store.initialize()
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
