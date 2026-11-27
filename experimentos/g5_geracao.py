"""G5: geração, citações, latência e chamadas aos modelos — baseline × modo ativo.

Executa o pipeline completo com o LLM configurado no YAML, usando as configs versionadas em
experimentos/g5-configs/, sobre as perguntas do gabarito, e resume o que o modelo respondeu.

    export DATABASE_URL=...
    python experimentos/g5_geracao.py [--configs experimentos/g5-configs/baseline.yaml ...]

Saída em data/results/g5/geracao/: geracao_citacoes.csv/.md, latencia_chamadas.csv/.md,
envelopes/<config>/<pergunta>.json e anotacao_humana.csv (planilha para conferir, uma a uma,
se a resposta é fiel às fontes e se as citações estão corretas).
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
# Trechos (sem acento, minúsculos) que indicam que o modelo recusou responder.
DECLINE_PHRASES = (
    "nao encontrei evidencia",
    "nao encontrei informac",
    "nao ha evidencia",
    "nao ha nenhuma evidencia",
    "nao ha informac",
    "nao ha nenhuma informac",
    "nao ha evidencias suficientes",
    "nenhuma evidencia",
    "nenhuma informacao",
    "nao consta",
    "nao contem",
    "evidencias fornecidas nao",
    "fontes fornecidas nao",
    "insuficiente para responder",
    "insuficientes para responder",
    "nao e possivel responder",
    "nao e possivel determinar",
)


class CallCounter:
    """Envolve os provedores e conta as chamadas ao chat e aos embeddings."""

    def __init__(self, inner):
        self._inner = inner
        self.reset()

    def reset(self) -> None:
        self.counts = {"chat": 0, "embedding": 0, "textos": 0}

    def chat(self, system: str, user: str) -> str:
        self.counts["chat"] += 1
        return self._inner.chat(system, user)

    def chat_json(self, *args, **kwargs) -> str:
        self.counts["chat"] += 1
        return self._inner.chat_json(*args, **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch = list(texts)
        self.counts["embedding"] += 1
        self.counts["textos"] += len(batch)
        return self._inner.embed(batch)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def plain(text: str) -> str:
    """Minúsculas, sem acentos e com espaços normalizados."""
    text = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def cited_numbers(answer: str | None) -> list[int]:
    """Números entre colchetes, na ordem em que aparecem: [1], [3]..."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", answer or "")]


def check_citations(answer: str | None, evidence: list[dict], relevant: list[str]) -> dict:
    """Confere cada [n] com as evidências enviadas e com os trechos relevantes do gabarito."""
    numbers = cited_numbers(answer)
    in_range = [n for n in dict.fromkeys(numbers) if 1 <= n <= len(evidence)]
    docs = {label.rsplit(":", 1)[0] for label in relevant}
    spans = set(relevant)

    def share(hits: int) -> float:
        return hits / len(in_range) if in_range else 0.0

    return {
        "tem_citacao": bool(numbers),
        "todas_validas": all(1 <= n <= len(evidence) for n in numbers),
        "precisao_documento": share(sum(evidence[n - 1]["source"] in docs for n in in_range)),
        "precisao_trecho": share(
            sum(
                f"{evidence[n - 1]['source']}:{evidence[n - 1]['position']}" in spans
                for n in in_range
            )
        ),
    }


def declined(answer: str | None) -> bool:
    """Heurística: a resposta é vazia ou contém uma frase de recusa."""
    text = plain(answer or "")
    return not text or any(phrase in text for phrase in DECLINE_PHRASES)


def facts_covered(answer: str | None, facts: list[str]) -> float:
    """Fração média das palavras (4+ letras) de cada fato esperado que aparece na resposta."""
    words = set(re.findall(r"[a-z0-9]{4,}", plain(answer or "")))
    shares = []
    for fact in facts:
        fact_words = set(re.findall(r"[a-z0-9]{4,}", plain(fact)))
        if fact_words:
            shares.append(len(fact_words & words) / len(fact_words))
    return statistics.mean(shares) if shares and words else 0.0


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0


def generation_rows(records: list[dict]) -> list[dict]:
    rows = []
    for config in sorted({r["config"] for r in records}):
        mine = [r for r in records if r["config"] == config]
        answerable = [r for r in mine if r["question"]["relevantes"]]
        unanswerable = [r for r in mine if not r["question"]["relevantes"]]
        checks = [
            check_citations(
                r["envelope"]["answer"], r["envelope"]["evidence"], r["question"]["relevantes"]
            )
            for r in answerable
        ]

        def mean(values):
            return round(statistics.mean(values), 4) if values else 0.0

        rows.append(
            {
                "config": config,
                "perguntas_com_resposta": len(answerable),
                "respostas_com_citacao": mean([c["tem_citacao"] for c in checks]),
                "citacoes_validas": mean([c["todas_validas"] for c in checks]),
                "precisao_citacao_documento": mean([c["precisao_documento"] for c in checks]),
                "precisao_citacao_trecho": mean([c["precisao_trecho"] for c in checks]),
                "fatos_cobertos": mean(
                    [
                        facts_covered(r["envelope"]["answer"], r["question"]["deve_conter"])
                        for r in answerable
                    ]
                ),
                "perguntas_sem_resposta": len(unanswerable),
                "recusas_sem_resposta": mean(
                    [declined(r["envelope"]["answer"]) for r in unanswerable]
                ),
                "avaliacao_humana": "pendente",
            }
        )
    return rows


def cost_rows(records: list[dict]) -> list[dict]:
    rows = []
    for config in sorted({r["config"] for r in records}):
        mine = [r for r in records if r["config"] == config]
        steps: dict[str, list[float]] = {"bm25": [], "generation": []}
        for r in mine:
            for step in r["envelope"]["trace"]:
                if step["module"] in steps:
                    steps[step["module"]].append(float(step["duration_ms"]))
        total = [r["total_ms"] for r in mine]
        rows.append(
            {
                "config": config,
                "consultas": len(mine),
                "tempo_bm25_ms_mediana": round(statistics.median(steps["bm25"]), 3),
                "tempo_geracao_ms_mediana": round(
                    statistics.median(steps["generation"] or [0.0]), 3
                ),
                "tempo_geracao_ms_p95": round(p95(steps["generation"]), 3),
                "tempo_total_ms_mediana": round(statistics.median(total), 3),
                "tempo_total_ms_p95": round(p95(total), 3),
                "chamadas_chat": sum(r["calls"]["chat"] for r in mine),
                "chamadas_embedding": sum(r["calls"]["embedding"] for r in mine),
                "textos_embedding": sum(r["calls"]["textos"] for r in mine),
            }
        )
    return rows


def collect(config_name: str, settings, store, providers, questions: list[dict]) -> list[dict]:
    counter = CallCounter(providers)
    pipeline = Pipeline(Services(settings, store, counter))
    records = []
    for question in questions:
        counter.reset()
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
                "total_ms": (perf_counter() - start) * 1000,
                "calls": dict(counter.counts),
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


def write_annotation_sheet(out_dir: Path, records: list[dict]) -> None:
    """Planilha com as respostas e colunas vazias (s/n) para conferência manual."""
    fields = [
        "config",
        "pergunta",
        "resposta",
        "citacoes",
        "fiel_as_fontes",
        "citacoes_corretas",
        "obs",
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
                    "resposta": answer,
                    "citacoes": " ".join(f"[{n}]" for n in cited_numbers(answer)),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.configs:
        settings = load_settings(path)
        name = Path(path).stem
        done = collect(
            name, settings, PostgresStore(settings.database_url), Providers(settings), questions
        )
        records += done
        folder = out_dir / "envelopes" / name
        folder.mkdir(parents=True, exist_ok=True)
        for r in done:
            (folder / f"{r['question']['id']}.json").write_text(
                json.dumps(r["envelope"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
    write_table(out_dir, "geracao_citacoes", "Geração e citações", generation_rows(records))
    write_table(out_dir, "latencia_chamadas", "Latência e chamadas aos modelos", cost_rows(records))
    write_annotation_sheet(out_dir, records)
    print((out_dir / "geracao_citacoes.md").read_text(encoding="utf-8"))
    print((out_dir / "latencia_chamadas.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
