"""G5: compara variantes do BM25 (k1, b, IDF) nas mesmas perguntas e no mesmo corpus.

Só o BM25 fica ativo e a geração é substituída por um texto fixo: o experimento mede a
recuperação, não o LLM. Saída em data/results/g5/ (ignorada pelo Git):
  - manifesto.json: commit, data, hardware, versões, hash do gabarito, critérios;
  - <variante>/: YAML usado e um envelope JSON por pergunta (variantes nomeadas);
  - resumo.json, perguntas.csv e grade.csv: métricas por perfil, métricas e posição do 1º
    relevante por pergunta (variantes nomeadas) e a varredura k1 × b × IDF;
  - ablacao.json: BM25 desligado (`bypass` no trace).

    export DATABASE_URL=... RAG_TAG_CLASSIFIER=terms
    rag-lab --config experimentos/g5-configs/baseline.yaml ingest experimentos/g5-corpus
    python experimentos/g5_bm25.py [--config experimentos/g5-configs/baseline.yaml]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml

from rag_lab.config import Settings, load_settings
from rag_lab.contracts import Envelope
from rag_lab.modules.base import Services
from rag_lab.pipeline import Pipeline
from rag_lab.storage import PostgresStore

ROOT = Path(__file__).resolve().parents[1]
TOP = 5
VARIANTS = {  # variantes nomeadas: envelopes completos são salvos
    "baseline": {},  # lucene, k1=1.5, b=0.75
    "idf-okapi": {"idf": "okapi"},
    "b-0": {"b": 0.0},
    "b-1": {"b": 1.0},
    "k1-0.5": {"k1": 0.5},
    "k1-3": {"k1": 3.0},
    "min-idf-1.0": {"min_idf": 1.0},
    "min-idf-1.5": {"min_idf": 1.5},
}
GRID = {"k1": [0.5, 0.9, 1.2, 1.5, 2.0, 3.0], "b": [0.0, 0.25, 0.5, 0.75, 1.0]}
IDFS = ["lucene", "okapi", "classic"]
PROFILES = ["literal", "semantica", "composta", "sem_resposta"]


class NoLLM:
    embedding_key = None

    def chat(self, system, user):
        return "(geração desativada no experimento G5)"


def with_bm25(settings: Settings, enabled: bool = True, **options) -> Settings:
    data = settings.model_dump()
    for name in data["modules"]:
        data["modules"][name] = {"enabled": False}
    data["modules"]["bm25"] = {"enabled": enabled, **options}
    return Settings.model_validate(data)


def label(item: dict) -> str:
    return f"{item['source']}:{item['position']}"


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "indisponível"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconhecido"


def manifest(gold_path: Path, config_path: str, store: PostgresStore) -> dict:
    sizes = [len(c.text) for c in store.all_chunks()]
    return {
        "data_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "codigo": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "alteracoes_nao_commitadas": bool(git("status", "--porcelain")),
        },
        "config_base": config_path,
        "gabarito_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "corpus": {
            "trechos": len(sizes),
            "caracteres_min_mediana_max": [min(sizes), statistics.median(sizes), max(sizes)],
        },
        "hardware": {"cpu": cpu_name(), "sistema": platform.platform()},
        "versoes": {"python": platform.python_version(), "rank_bm25": version("rank_bm25")},
        "modelos": "nenhum (só BM25; geração e embeddings desativados)",
        "variantes": VARIANTS,
        "grade": {**GRID, "idf": IDFS},
        "criterios": (
            f"recall@{TOP} por trecho e por documento; MRR; latência do módulo no trace; "
            "para sem_resposta: fração com algum resultado e score médio do 1º"
        ),
    }


def evaluate(top: list[dict], relevant: list[str]) -> dict:
    """top: [{'id': 'arquivo:pos', 'score': float}]. Métricas por trecho e por documento."""
    labels = [t["id"] for t in top]
    if not relevant:
        return {"resultados": len(top), "score1": top[0]["score"] if top else 0.0}
    rel, found = set(relevant), set(labels)
    rel_docs = {r.split(":")[0] for r in rel}
    first = next((i + 1 for i, t in enumerate(labels) if t in rel), None)
    return {
        "recall_trecho": len(rel & found) / len(rel),
        "recall_documento": len(rel_docs & {t.split(":")[0] for t in labels}) / len(rel_docs),
        "mrr": 1 / first if first else 0.0,
        "primeiro": first,
    }


def summarize(per_question: dict[str, dict], questions: list[dict]) -> dict:
    """Média por perfil. per_question: id -> métricas de evaluate()."""
    summary = {}
    for profile in PROFILES:
        rows = [per_question[q["id"]] for q in questions if q["perfil"] == profile]
        if not rows:
            continue
        if profile == "sem_resposta":
            summary[profile] = {
                "com_resultado": statistics.mean(r["resultados"] > 0 for r in rows),
                "score_1o": statistics.mean(r["score1"] for r in rows),
            }
        else:
            summary[profile] = {
                key: statistics.mean(r[key] for r in rows)
                for key in ("recall_trecho", "recall_documento", "mrr")
            }
    answered = [per_question[q["id"]] for q in questions if q["relevantes"]]
    summary["com_resposta"] = {
        key: statistics.mean(r[key] for r in answered)
        for key in ("recall_trecho", "recall_documento", "mrr")
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experimentos/g5-configs/baseline.yaml")
    parser.add_argument("--gold", default=str(ROOT / "experimentos/g5_gold.json"))
    parser.add_argument("--out", default=str(ROOT / "data/results/g5"))
    args = parser.parse_args()
    out_dir, gold_path = Path(args.out), Path(args.gold)
    base = load_settings(args.config)
    store = PostgresStore(base.database_url)
    questions = json.loads(gold_path.read_text(encoding="utf-8"))["perguntas"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifesto.json").write_text(
        json.dumps(manifest(gold_path, args.config, store), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def run(settings: Settings, question: dict, variant: str) -> dict:
        envelope = Envelope(
            question=question["pergunta"], attributes={"grupo": "G5", "experimento": variant}
        )
        return Pipeline(Services(settings, store, NoLLM())).run(envelope).model_dump(mode="json")

    def top_of(envelope: dict) -> list[dict]:
        found = envelope["retrieval"].get("bm25", [])[:TOP]
        return [{"id": label(e), "score": e["score"]} for e in found]

    def bm25_ms(envelope: dict) -> float:
        return next(t["duration_ms"] for t in envelope["trace"] if t["module"] == "bm25")

    # 1) Variantes nomeadas: envelopes completos, resumo por perfil e latência.
    summaries, latency, tops, question_rows = {}, {}, {}, []
    for variant, options in VARIANTS.items():
        settings = with_bm25(base, **options)
        folder = out_dir / variant
        folder.mkdir(parents=True, exist_ok=True)
        dump = json.loads(settings.model_dump_json(exclude={"database_url"}))
        (folder / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")
        per_question, latency[variant], tops[variant] = {}, [], {}
        for q in questions:
            envelope = run(settings, q, variant)
            (folder / f"{q['id']}.json").write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tops[variant][q["id"]] = top_of(envelope)
            per_question[q["id"]] = evaluate(tops[variant][q["id"]], q["relevantes"])
            latency[variant].append(bm25_ms(envelope))
            question_rows.append(
                {
                    "variante": variant,
                    "pergunta": q["id"],
                    "perfil": q["perfil"],
                    "posicao_1o_relevante": per_question[q["id"]].get("primeiro"),
                    "recall_trecho": per_question[q["id"]].get("recall_trecho"),
                    "recall_documento": per_question[q["id"]].get("recall_documento"),
                    "resultados": len(tops[variant][q["id"]]),
                    "score_1o_resultado": (tops[variant][q["id"]] or [{"score": None}])[0]["score"],
                    "latencia_ms": latency[variant][-1],
                }
            )
        summaries[variant] = summarize(per_question, questions)

    print("== Variantes nomeadas, por perfil de pergunta ==")
    print(f"{'variante':<10} {'perfil':<13} {'rec.tr.':>7} {'rec.doc':>7} {'MRR':>5} {'ms':>6}")
    for variant, summary in summaries.items():
        ms = statistics.median(latency[variant])
        for profile in PROFILES:
            s = summary[profile]
            if profile == "sem_resposta":
                extra = f"{s['com_resultado']:>7.0%} c/ resultado, score do 1º {s['score_1o']:.2f}"
                print(f"{variant:<10} {profile:<13} {extra}")
            else:
                print(
                    f"{variant:<10} {profile:<13} {s['recall_trecho']:>7.2f} "
                    f"{s['recall_documento']:>7.2f} {s['mrr']:>5.2f} {ms:>6.1f}"
                )

    with (out_dir / "perguntas.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(question_rows[0]))
        writer.writeheader()
        writer.writerows(question_rows)

    print("\n== Posição do 1º relevante no top-5 (- = não apareceu) ==")
    print(f"{'pergunta':<9}" + "".join(f"{v:>13}" for v in VARIANTS))
    for q in questions:
        if q["relevantes"]:
            cells = [
                next(
                    r["posicao_1o_relevante"]
                    for r in question_rows
                    if r["variante"] == v and r["pergunta"] == q["id"]
                )
                for v in VARIANTS
            ]
            print(f"{q['id']:<9}" + "".join(f"{('-' if c is None else c)!s:>13}" for c in cells))

    # 2) Varredura k1 × b × IDF (só métricas; sem salvar envelopes).
    rows = []
    for idf in IDFS:
        for k1 in GRID["k1"]:
            for b in GRID["b"]:
                settings = with_bm25(base, k1=k1, b=b, idf=idf)
                per_question = {
                    q["id"]: evaluate(top_of(run(settings, q, "grade")), q["relevantes"])
                    for q in questions
                }
                s = summarize(per_question, questions)
                rows.append(
                    {
                        "idf": idf,
                        "k1": k1,
                        "b": b,
                        "recall_trecho": s["com_resposta"]["recall_trecho"],
                        "recall_documento": s["com_resposta"]["recall_documento"],
                        "mrr": s["com_resposta"]["mrr"],
                        "mrr_literal": s["literal"]["mrr"],
                        "mrr_semantica": s["semantica"]["mrr"],
                        "mrr_composta": s["composta"]["mrr"],
                        "sem_resposta_com_resultado": s["sem_resposta"]["com_resultado"],
                    }
                )
    with (out_dir / "grade.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for metric in ("recall_trecho", "mrr"):
        for idf in IDFS:
            print(f"\n== Grade {metric} (com resposta), IDF {idf}; linhas k1, colunas b ==")
            print("k1\\b   " + " ".join(f"{b:>5}" for b in GRID["b"]))
            for k1 in GRID["k1"]:
                cells = [
                    next(r for r in rows if (r["idf"], r["k1"], r["b"]) == (idf, k1, b))[metric]
                    for b in GRID["b"]
                ]
                print(f"{k1:<6} " + " ".join(f"{c:>5.2f}" for c in cells))
    best = max(rows, key=lambda r: (r["mrr"], r["recall_trecho"]))
    worst = min(rows, key=lambda r: (r["mrr"], r["recall_trecho"]))
    spread = max(r["mrr"] for r in rows) - min(r["mrr"] for r in rows)
    print(f"\nMelhor MRR: {best['mrr']:.3f} (idf={best['idf']}, k1={best['k1']}, b={best['b']})")
    print(f"Pior MRR:   {worst['mrr']:.3f} (idf={worst['idf']}, k1={worst['k1']}, b={worst['b']})")
    print(f"Amplitude de MRR na grade inteira: {spread:.3f}")

    (out_dir / "resumo.json").write_text(
        json.dumps(
            {
                "variantes": summaries,
                "latencia_ms_mediana": {k: statistics.median(v) for k, v in latency.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3) Determinismo: repetir a baseline deve devolver os mesmos trechos e scores.
    baseline = with_bm25(base)
    same = all(
        top_of(run(baseline, q, "repeticao")) == tops["baseline"][q["id"]] for q in questions
    )
    print(f"\nRepetição da baseline idêntica (trechos e scores): {same}")

    # 4) Ablação: com o BM25 desligado o trace deve mostrar `bypass`.
    ablated = run(with_bm25(base, enabled=False), questions[0], "ablacao")
    step = next(t for t in ablated["trace"] if t["module"] == "bm25")
    ok = step["status"] == "bypass" and not ablated["retrieval"] and not ablated["evidence"]
    print(f"Ablação (BM25 desligado): status={step['status']}, sem candidatos={ok}")
    (out_dir / "ablacao.json").write_text(
        json.dumps(ablated, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
