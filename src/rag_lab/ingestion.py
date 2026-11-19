"""Ingestão UTF-8 de .txt e .md; uma sentença por unidade, com divisão das muito longas."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_lab.config import Settings
from rag_lab.contracts import Chunk
from rag_lab.tagging import (
    TagClassificationError,
    TagClassifier,
    TermTagClassifier,
    load_tag_catalog,
)


def split_document(source: str, text: str, max_chars: int) -> list[Chunk]:
    # Heurística simples e visível: abreviações e limites linguísticos são uma missão de melhoria.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=0)
    parts = [
        part for sentence in sentences for part in splitter.split_text(sentence) if part.strip()
    ]
    document_id = hashlib.sha256(source.encode()).hexdigest()
    return [
        Chunk(
            id=hashlib.sha256(f"{document_id}:{i}:{part}".encode()).hexdigest(),
            document_id=document_id,
            source=source,
            position=i,
            text=part,
            metadata={"unit": "sentence", "splitter": "sentence-v1"},
        )
        for i, part in enumerate(parts)
    ]


def iter_document_chunks(directory: Path, max_chars: int) -> Iterator[list[Chunk]]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError("Informe um diretório de documentos existente.")
    paths = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in {".txt", ".md"} and p.is_file()
    )
    if not paths:
        raise ValueError("Nenhum documento .txt ou .md encontrado.")

    def read_chunks(path: Path) -> list[Chunk]:
        source = path.relative_to(directory).as_posix()
        chunks = split_document(source, path.read_text(encoding="utf-8"), max_chars)
        if not chunks:
            raise ValueError(f"Documento vazio: {source}")
        return chunks

    return (read_chunks(path) for path in paths)


def ingest_directory(
    directory: Path, settings: Settings, store, providers, *, config_path: Path | None = None
) -> dict:
    documents_to_ingest = iter_document_chunks(directory, settings.pipeline.max_sentence_chars)
    catalog = load_tag_catalog(config_path)
    classifier = (
        TermTagClassifier(catalog)
        if settings.tagging_method == "terms"
        else TagClassifier(catalog, providers)
    )
    store.initialize()
    documents = []
    for chunks in documents_to_ingest:
        source = chunks[0].source
        for chunk in chunks:
            try:
                chunk.metadata["tags"] = classifier.classify(chunk.text)
            except TagClassificationError as error:
                message = f"{source} (trecho {chunk.position}) — {error}"
                raise TagClassificationError(message) from error
        vectors = None
        if settings.modules["embedding"].enabled:
            # Lotes pequenos reduzem uso de memória e tamanho dos requests de modelos locais.
            vectors = []
            for start in range(0, len(chunks), 32):
                vectors.extend(providers.embed([c.text for c in chunks[start : start + 32]]))
        store.replace_document(chunks, vectors, providers.embedding_key if vectors else None)
        documents.append({"source": source, "chunks": len(chunks)})
    return {"documents": documents, "total_chunks": sum(d["chunks"] for d in documents)}
