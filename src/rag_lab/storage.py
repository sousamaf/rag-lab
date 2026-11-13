"""Persistência explícita em PostgreSQL. <=> é a distância por cosseno do pgvector."""

from __future__ import annotations

import json
from collections.abc import Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rag_lab.contracts import Chunk, Evidence

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_chunks (
    id text PRIMARY KEY,
    document_id text NOT NULL,
    source text NOT NULL,
    position integer NOT NULL,
    text text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    embedding vector,
    embedding_key text,
    UNIQUE (document_id, position)
);
CREATE INDEX IF NOT EXISTS rag_chunks_document_idx ON rag_chunks (document_id, position);
"""


class PostgresStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5, row_factory=dict_row)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(SCHEMA)

    def replace_document(
        self,
        chunks: Sequence[Chunk],
        vectors: list[list[float]] | None,
        embedding_key: str | None,
    ) -> None:
        if not chunks:
            raise ValueError("Documento sem conteúdo indexável.")
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("Quantidade de embeddings diferente da quantidade de trechos.")
        # Uma transação por documento: falhas não deixam metade do arquivo indexada.
        with self.connect() as connection:
            # Serializa reingestões concorrentes do mesmo documento.
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (chunks[0].document_id,)
            )
            connection.execute(
                "DELETE FROM rag_chunks WHERE document_id = %s",
                (chunks[0].document_id,),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO rag_chunks
                       (id, document_id, source, position, text, metadata, embedding, embedding_key)
                       VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)""",
                    [
                        (
                            c.id,
                            c.document_id,
                            c.source,
                            c.position,
                            c.text,
                            Jsonb(c.metadata),
                            json.dumps(vectors[i]) if vectors is not None else None,
                            embedding_key,
                        )
                        for i, c in enumerate(chunks)
                    ],
                )

    def all_chunks(self) -> list[Chunk]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, document_id, source, position, text, metadata "
                "FROM rag_chunks ORDER BY document_id, position",
            ).fetchall()
        return [Chunk.model_validate(row) for row in rows]

    def search_vector(self, vector: list[float], key: str, limit: int) -> list[Evidence]:
        with self.connect() as connection:
            missing = connection.execute(
                "SELECT count(*) AS n FROM rag_chunks "
                "WHERE embedding IS NULL OR embedding_key IS DISTINCT FROM %s",
                (key,),
            ).fetchone()["n"]
            if missing:
                raise ValueError(
                    "A base contém embeddings ausentes ou de outra configuração. "
                    "Reingira todos os documentos com a configuração atual.",
                )
            # Busca exata, propositalmente sem ANN: adequada a uma base pequena de laboratório.
            rows = connection.execute(
                """SELECT id, document_id, source, position, text, metadata,
                          1 - (embedding <=> %s::vector) AS score
                   FROM rag_chunks WHERE embedding_key = %s
                   ORDER BY embedding <=> %s::vector, id LIMIT %s""",
                (json.dumps(vector), key, json.dumps(vector), limit),
            ).fetchall()
        return [Evidence.model_validate(row) for row in rows]

    def window(self, document_id: str, position: int, radius: int) -> str:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT text FROM rag_chunks WHERE document_id = %s "
                "AND position BETWEEN %s AND %s ORDER BY position",
                (document_id, position - radius, position + radius),
            ).fetchall()
        return " ".join(row["text"] for row in rows)
