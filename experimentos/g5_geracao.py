"""G5: geração, citações, latência e chamadas aos modelos — baseline × modo ativo.

Mesmo molde das tabelas T3 (geração e citações) e T4 (latência e chamadas) do G7. Executa o
pipeline completo com o LLM configurado no YAML, com as configs versionadas em
experimentos/g5-configs/, sobre as perguntas do gabarito.

    export DATABASE_URL=...
    python experimentos/g5_geracao.py [--configs experimentos/g5-configs/baseline.yaml ...]

Saída em data/results/g5/geracao/: t3_geracao_citacoes.csv/.md, t4_latencia_chamadas.csv/.md,
envelopes/<config>/<pergunta>.json e anotacao_humana.csv (modelo para a rubrica 0–2 de
fidelidade e a conferência manual das citações).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from pathlib import Path
from time import perf_counter

from rag_lab.config import load_settings
from rag_lab.contracts import Envelope
from rag_lab.modules.base import Services
from rag_lab.pipeline import Pipeline
from rag_lab.providers import Providers
from rag_lab.storage import PostgresStore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = ["experimentos/g5-configs/baseline.yaml", "experimentos/g5-configs/ativo.yaml"]
ABSTENTION = (
    "nao encontrei evidencia",
    "nao encontrei informac",
    "nao ha evidencia",
    "nao ha informac",
    "nao consta",
    "nao contem",
    "nenhuma informacao",
    "evidencias fornecidas nao",
    "fontes fornecidas nao",
    "insuficiente para responder",
    "insuficientes para responder",
    "nao e possivel responder",
)


class InstrumentedProviders:
    """Conta chamadas ao chat e aos embeddings por consulta."""

    def __init__(self, inner):
        self._inner = inner
        self.chat_calls = self.embed_calls = self.embed_texts = 0

    def reset(self) -> None:
        self.chat_calls = self.embed_calls = self.embed_texts = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "embed_texts": self.embed_texts,
        }

    def chat(self, system: str, user: str) -> str:
        self.chat_calls += 1
        return self._inner.chat(system, user)

    def chat_json(self, *args, **kwargs) -> str:
        self.chat_calls += 1
        return self._inner.chat_json(*args, **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch = list(texts)
        self.embed_calls += 1
        self.embed_texts += len(batch)
        return self._inner.embed(batch)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def extract_citations(answer: str | None) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d+)\]", answer or "")]


def evaluate_citations(answer: str | None, evidence: list[dict], relevant: list[str]) -> dict:
    """Sintaxe [n] contra 1..len(evidência) e precisão das fontes citadas (documento e trecho)."""
    cited = extract_citations(answer)
    valid = list(dict.fromkeys(n for n in cited if 1 <= n <= len(evidence)))
    docs = {r.rsplit(":", 1)[0] for r in relevant}
    snippets = set(relevant)
    doc_hits = sum(evidence[n - 1]["source"] in docs for n in valid)
    snippet_hits = sum(
        f"{evidence[n - 1]['source']}:{evidence[n - 1]['position']}" in snippets for n in valid
    )
    return {
        "has_citations": bool(cited),
        "syntax_valid": all(1 <= n <= len(evidence) for n in cited),
        "doc_precision": doc_hits / len(valid) if valid else 0.0,
        "snippet_precision": snippet_hits / len(valid) if valid else 0.0,
    }


def detect_abstention(answer: str | None) -> bool:
    norm = normalize(answer or "")
    return not norm or any(pattern in norm for pattern in ABSTENTION)


def fact_coverage(answer: str | None, facts: list[str]) -> float:
    """Cobertura lexical heurística de `deve_conter` (complementa a rubrica humana)."""
    answer_tokens = set(re.findall(r"[a-z0-9]{4,}", normalize(answer or "")))
    scores = []
    for fact in facts:
        tokens = set(re.findall(r"[a-z0-9]{4,}", normalize(fact)))
        if tokens:
            scores.append(len(tokens & answer_tokens) / len(tokens))
    return statistics.mean(scores) if scores and answer_tokens else 0.0


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0


def t3_rows(records: list[dict]) -> list[dict]:
    rows = []
    for config in sorted({r["config"] for r in records}):
        items = [r for r in records if r["config"] == config]
        answerable = [r for r in items if r["question"]["relevantes"]]
        unanswerable = [r for r in items if not r["question"]["relevantes"]]
        cites = [
            evaluate_citations(
                r["envelope"]["answer"], r["envelope"]["evidence"], r["question"]["relevantes"]
            )
            for r in answerable
        ]

        def mean(values):
            return round(statistics.mean(values), 4) if values else 0.0

        rows.append(
            {
                "config": config,
                "n_com_resposta": len(answerable),
                "taxa_respostas_com_citacao": mean([c["has_citations"] for c in cites]),
                "taxa_citacoes_sintaxe_valida": mean([c["syntax_valid"] for c in cites]),
                "precisao_fontes_citadas_doc": mean([c["doc_precision"] for c in cites]),
                "precisao_fontes_citadas_trecho": mean([c["snippet_precision"] for c in cites]),
                "cobertura_fatos_heuristica": mean(
                    [
                        fact_coverage(r["envelope"]["answer"], r["question"]["deve_conter"])
                        for r in answerable
                    ]
                ),
                "n_sem_resposta": len(unanswerable),
                "taxa_abstencao_sem_resposta": mean(
                    [detect_abstention(r["envelope"]["answer"]) for r in unanswerable]
                ),
                "fidelidade_humana_status": "pendente_rubrica_humana_0_2",
            }
        )
    return rows


def t4_rows(records: list[dict]) -> list[dict]:
    rows = []
    for config in sorted({r["config"] for r in records}):
        items = [r for r in records if r["config"] == config]
        durations: dict[str, list[float]] = {"bm25": [], "generation": []}
        for r in items:
            for step in r["envelope"]["trace"]:
                if step["module"] in durations:
                    durations[step["module"]].append(float(step["duration_ms"]))
        wall = [r["wall_ms"] for r in items]
        rows.append(
            {
                "config": config,
                "n_consultas": len(items),
                "bm25_ms_mediana": round(statistics.median(durations["bm25"]), 3),
                "generation_ms_mediana": round(
                    statistics.median(durations["generation"] or [0.0]), 3
                ),
                "generation_ms_p95": round(percentile95(durations["generation"]), 3),
                "wall_time_ms_mediana": round(statistics.median(wall), 3),
                "wall_time_ms_p95": round(percentile95(wall), 3),
                "chat_calls_total": sum(r["calls"]["chat_calls"] for r in items),
                "embed_calls_total": sum(r["calls"]["embed_calls"] for r in items),
                "embed_texts_total": sum(r["calls"]["embed_texts"] for r in items),
            }
        )
    return rows


def collect(config_name: str, settings, store, providers, questions: list[dict]) -> list[dict]:
    instrumented = InstrumentedProviders(providers)
    pipeline = Pipeline(Services(settings, store, instrumented))
    records = []
    for question in questions:
        instrumented.reset()
        start = perf_counter()
        envelope = pipeline.run(
            Envelope(
                question=question["pergunta"], attributes={"grupo": "G5", "config": config_name}
            )
        ).model_dump(mode="json")
        records.append(
            {
                "config": config_name,
                "question": question,
                "envelope": envelope,
                "wall_ms": (perf_counter() - start) * 1000,
                "calls": instrumented.snapshot(),
            }
        )
    return records


def write_table(out_dir: Path, name: str, title: str, rows: list[dict]) -> None:
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in row.values()) + " |\n" for row in rows)
    (out_dir / f"{name}.md").write_text(f"# {title}\n\n{header}{body}", encoding="utf-8")


def write_annotation_template(out_dir: Path, records: list[dict]) -> None:
    fields = [
        "config",
        "pergunta",
        "resposta_llm",
        "citacoes_extraidas",
        "fidelidade_0_2",
        "citacoes_corretas_s_n",
        "observacoes",
    ]
    with (out_dir / "anotacao_humana.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in records:
            answer = r["envelope"]["answer"]
            writer.writerow(
                {
                    "config": r["config"],
                    "pergunta": r["question"]["id"],
                    "resposta_llm": answer,
                    "citacoes_extraidas": " ".join(f"[{n}]" for n in extract_citations(answer)),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--gold", default=str(ROOT / "experimentos/g5_gold.json"))
    parser.add_argument("--out", default=str(ROOT / "data/results/g5/geracao"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    questions = json.loads(Path(args.gold).read_text(encoding="utf-8"))["perguntas"]
    records = []
    for path in args.configs:
        settings = load_settings(path)
        name = Path(path).stem
        records += collect(
            name, settings, PostgresStore(settings.database_url), Providers(settings), questions
        )
        folder = out_dir / "envelopes" / name
        folder.mkdir(parents=True, exist_ok=True)
        for r in records:
            if r["config"] == name:
                (folder / f"{r['question']['id']}.json").write_text(
                    json.dumps(r["envelope"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
    write_table(out_dir, "t3_geracao_citacoes", "Tabela T3 — Geração e citações", t3_rows(records))
    write_table(
        out_dir, "t4_latencia_chamadas", "Tabela T4 — Latência e chamadas", t4_rows(records)
    )
    write_annotation_template(out_dir, records)
    print((out_dir / "t3_geracao_citacoes.md").read_text(encoding="utf-8"))
    print((out_dir / "t4_latencia_chamadas.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
