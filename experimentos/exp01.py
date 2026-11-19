"""Compara os classificadores temáticos do RAG Lab em um corpus pareado."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from pydantic import BaseModel, TypeAdapter

from rag_lab.config import configuration_path, load_settings
from rag_lab.contracts import Chunk
from rag_lab.ingestion import iter_document_chunks
from rag_lab.providers import Providers
from rag_lab.tagging import (
    TagClassificationError,
    TagClassifier,
    TermTagClassifier,
    load_tag_catalog,
)


class GoldEntry(BaseModel):
    id: str
    tags: list[str]


def describe_chunk(chunk: Chunk) -> dict:
    return {"id": chunk.id, "source": chunk.source, "position": chunk.position, "text": chunk.text}


def read_gold(path: Path, chunks: list[Chunk], allowed_ids: set[str]) -> dict[str, set[str]]:
    entries = TypeAdapter(list[GoldEntry]).validate_json(path.read_text(encoding="utf-8"))
    gold = {entry.id: set(entry.tags) for entry in entries}
    if len(gold) != len(entries) or gold.keys() != {chunk.id for chunk in chunks}:
        raise ValueError("O gabarito precisa ter exatamente uma entrada para cada ID do corpus.")
    if any(tags - allowed_ids for tags in gold.values()):
        raise ValueError("O gabarito contém IDs de rótulo fora do catálogo.")
    return gold


def measure(text: str, classifier: TagClassifier | TermTagClassifier) -> dict:
    start = perf_counter_ns()
    tags = classifier.classify(text)
    return {"tags": tags, "latency_ms": (perf_counter_ns() - start) / 1_000_000}


def summarize(rows: list[dict], method: str, gold: dict[str, set[str]] | None) -> dict:
    latencies = [row[method]["latency_ms"] for row in rows]
    summary = {"median_ms": median(latencies), "total_ms": sum(latencies)}
    if gold is not None:
        comparisons = [(set(row[method]["tags"]), gold[row["id"]]) for row in rows]
        hits = sum(len(predicted & expected) for predicted, expected in comparisons)
        predicted_total = sum(len(predicted) for predicted, _ in comparisons)
        expected_total = sum(len(expected) for _, expected in comparisons)
        summary.update(
            precision_micro=hits / predicted_total if predicted_total else 0.0,
            recall_micro=hits / expected_total if expected_total else 0.0,
            f1_micro=2 * hits / (predicted_total + expected_total)
            if predicted_total + expected_total
            else 0.0,
            exact_match=sum(predicted == expected for predicted, expected in comparisons)
            / len(comparisons),
        )
    return summary


def run_experiment(args: argparse.Namespace) -> dict | list[dict]:
    settings = load_settings(args.config)
    config_path = configuration_path(args.config)
    chunks = [
        chunk
        for document in iter_document_chunks(args.corpus, settings.pipeline.max_sentence_chars)
        for chunk in document
    ]
    if args.prepare_gold:
        return [dict(describe_chunk(chunk), tags=None) for chunk in chunks]

    catalog = load_tag_catalog(config_path)
    allowed_ids = {label.id for label in catalog.rotulos}
    gold = read_gold(args.gold, chunks, allowed_ids) if args.gold else None
    terms = TermTagClassifier(catalog)
    llm = TagClassifier(catalog, Providers(settings))
    rows = []
    for chunk in chunks:
        row = describe_chunk(chunk)
        row["terms"] = measure(chunk.text, terms)
        try:
            row["llm"] = measure(chunk.text, llm)
        except TagClassificationError as error:
            raise TagClassificationError(
                f"{chunk.source} (trecho {chunk.position}) — {error}"
            ) from error
        if gold is not None:
            row["gold_tags"] = sorted(gold[chunk.id])
        rows.append(row)

    return {
        "experiment": "exp01",
        "corpus": str(args.corpus),
        "max_sentence_chars": settings.pipeline.max_sentence_chars,
        "catalog_sha256": sha256(catalog.model_dump_json().encode()).hexdigest(),
        "labels": sorted(allowed_ids),
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
        },
        "chunks": len(chunks),
        "agreement_exact": sum(row["terms"]["tags"] == row["llm"]["tags"] for row in rows)
        / len(rows),
        "gold_provided": gold is not None,
        "methods": {method: summarize(rows, method, gold) for method in ("terms", "llm")},
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("examples/documents"))
    parser.add_argument("--config", type=Path, help="Configuração do RAG Lab")
    parser.add_argument("--output", type=Path, help="Grava o JSON no arquivo indicado")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gold", type=Path, help="Gabarito JSON com rótulos atribuídos por humanos")
    mode.add_argument("--prepare-gold", action="store_true", help="Exporta os chunks para anotação")
    args = parser.parse_args()

    try:
        output = json.dumps(run_experiment(args), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Falha no experimento — {error}\n")


if __name__ == "__main__":
    main()
