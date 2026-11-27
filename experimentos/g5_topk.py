"""G5: baseline (BM25 de referência) × modo ativo (variante escolhida) variando o top_k.

Método pedido pelo professor na aula: rodar a baseline com o valor padrão, variar o
parâmetro (aqui `pipeline.top_k` = 5/7/10/12, limite = `candidate_k` 12), ativar o modo e repetir
os mesmos valores, comparando numa tabela. A baseline é o BM25 de referência (IDF lucene,
k1=1.5, b=0.75); o modo ativo usa a combinação de k1/b/IDF escolhida na metade "treino" das
perguntas e é avaliado também na metade "teste", que não participou da escolha.
Também mede quais parâmetros do YAML alteram o resultado de um BM25 sozinho (sensibilidade).

    export DATABASE_URL=...
    python experimentos/g5_topk.py [--config experimentos/g5-configs/baseline.yaml]

Saída em data/results/g5/topk/ (ignorada pelo Git): tabela_topk.csv/.md, sensibilidade.csv,
ativo.yaml e manifesto.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml
from g5_bm25 import (
    GRID,
    IDFS,
    ROOT,
    NoLLM,
    cpu_name,
    evaluate,
    git,
    label,
    with_bm25,
)

from rag_lab.config import Settings, load_settings
from rag_lab.contracts import Envelope
from rag_lab.modules.base import Services
from rag_lab.pipeline import Pipeline
from rag_lab.storage import PostgresStore

TOPKS = [5, 7, 10, 12]
BASELINE = {"idf": "lucene", "k1": 1.5, "b": 0.75, "min_idf": 0}
DEFAULTS = {"k1": 1.5, "b": 0.75, "idf": "lucene", "min_idf": 0}
MIN_IDFS = [0, 1.0, 1.2, 1.5, 2.0]


def split_questions(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Dentro de cada perfil: posições pares → treino (escolha), ímpares → teste."""
    train, test = [], []
    for profile in {q["perfil"] for q in questions}:
        group = [q for q in questions if q["perfil"] == profile]
        train += group[0::2]
        test += group[1::2]
    order = {q["id"]: i for i, q in enumerate(questions)}
    return sorted(train, key=lambda q: order[q["id"]]), sorted(test, key=lambda q: order[q["id"]])


def distance_from_default(option: dict) -> float:
    """Desempate: entre combinações equivalentes, prefere a mais próxima da referência."""
    return (
        abs(option["k1"] - DEFAULTS["k1"]) / DEFAULTS["k1"]
        + abs(option["b"] - DEFAULTS["b"])
        + (option["idf"] != DEFAULTS["idf"])
        + (option["min_idf"] > 0)
    )


def with_pipeline(settings: Settings, **pipeline) -> Settings:
    data = settings.model_dump()
    data["pipeline"].update(pipeline)
    return Settings.model_validate(data)


def summarize_at(rows: list[dict], questions: list[dict]) -> dict:
    """rows: id -> {'evidence': [{'id','score'}, ...]} já cortado no top_k."""
    answered = [q for q in questions if q["relevantes"]]
    metrics = [evaluate(rows[q["id"]], q["relevantes"]) for q in answered]
    unanswered = [q for q in questions if not q["relevantes"]]
    return {
        "n_com_resposta": len(answered),
        "recall_trecho": statistics.mean(m["recall_trecho"] for m in metrics),
        "recall_documento": statistics.mean(m["recall_documento"] for m in metrics),
        "mrr": statistics.mean(m["mrr"] for m in metrics),
        "sem_resposta_com_evidencia": (
            statistics.mean(bool(rows[q["id"]]) for q in unanswered) if unanswered else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experimentos/g5-configs/baseline.yaml")
    parser.add_argument("--gold", default=str(ROOT / "experimentos/g5_gold.json"))
    parser.add_argument("--out", default=str(ROOT / "data/results/g5/topk"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_settings(args.config)
    store = PostgresStore(base.database_url)
    questions = json.loads(Path(args.gold).read_text(encoding="utf-8"))["perguntas"]
    train, test = split_questions(questions)
    cache: dict[tuple, dict] = {}

    def evidence_for(option: dict, top_k: int, question: dict, **pipeline) -> list[dict]:
        key = (
            tuple(sorted(option.items())),
            top_k,
            question["id"],
            tuple(sorted(pipeline.items())),
        )
        if key not in cache:
            settings = with_pipeline(with_bm25(base, **option), top_k=top_k, **pipeline)
            envelope = Envelope(question=question["pergunta"], attributes={"grupo": "G5"})
            out = Pipeline(Services(settings, store, NoLLM())).run(envelope)
            cache[key] = [{"id": label(e.model_dump()), "score": e.score} for e in out.evidence]
        return cache[key]

    def run_set(option: dict, top_k: int, qs: list[dict], **pipeline) -> dict:
        rows = {q["id"]: evidence_for(option, top_k, q, **pipeline) for q in qs}
        return summarize_at(rows, qs)

    # 1) Escolha do modo ativo na metade de treino (top_k padrão = 5), sem olhar o teste.
    candidates = [
        {"idf": idf, "k1": k1, "b": b, "min_idf": min_idf}
        for idf in IDFS
        for k1 in GRID["k1"]
        for b in GRID["b"]
        for min_idf in MIN_IDFS
    ]
    scored = []
    for option in candidates:
        s = run_set(option, 5, train)
        scored.append((-s["mrr"], -s["recall_trecho"], distance_from_default(option), option))
    scored.sort(key=lambda t: t[:3])
    active = scored[0][3]
    scored_all = sorted(
        (
            (
                -(s := run_set(o, 5, questions))["mrr"],
                -s["recall_trecho"],
                distance_from_default(o),
                o,
            )
            for o in candidates
        ),
        key=lambda t: t[:3],
    )
    best_in_sample = scored_all[0][3]
    ref_train = run_set(BASELINE, 5, train)
    print(f"Baseline (referência): {BASELINE}")
    print(f"Modo ativo escolhido no treino: {active}")
    print(f"  treino top_k=5 — baseline MRR {ref_train['mrr']:.3f}; ativo MRR {-scored[0][0]:.3f}")
    print(f"Melhor da grade em todas as perguntas (em amostra, otimista): {best_in_sample}")
    modes = {"baseline": BASELINE, "ativo": active, "melhor_grade": best_in_sample}
    (out_dir / "ativo.yaml").write_text(
        yaml.safe_dump(
            {**modes, "criterio_ativo": "MRR no treino (desempate: mais perto da referência)"}
        ),
        encoding="utf-8",
    )

    # 2) Tabela baseline × ativo por top_k: todas as perguntas e só as de teste.
    table = []
    for top_k in TOPKS:
        for scope, qs in (("todas", questions), ("teste", test)):
            for mode, option in modes.items():
                table.append(
                    {"top_k": top_k, "escopo": scope, "modo": mode, **run_set(option, top_k, qs)}
                )
    with (out_dir / "tabela_topk.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    def row(top_k, scope, mode):
        return next(
            r for r in table if (r["top_k"], r["escopo"], r["modo"]) == (top_k, scope, mode)
        )

    lines = [
        "| top_k | perguntas | modo | recall trecho | recall doc | MRR | sem resposta c/ evid. |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['top_k']} | {r['escopo']} | {r['modo']} | {r['recall_trecho']:.3f} | "
            f"{r['recall_documento']:.3f} | {r['mrr']:.3f} | "
            f"{r['sem_resposta_com_evidencia']:.0%} |"
        )
    lines += ["", "| top_k | perguntas | modo | Δ recall trecho | Δ recall doc | Δ MRR | melhor? |"]
    lines.append("|---|---|---|---|---|---|---|")
    for top_k in TOPKS:
        for scope in ("todas", "teste"):
            for mode in ("ativo", "melhor_grade"):
                b, a = row(top_k, scope, "baseline"), row(top_k, scope, mode)
                d = {k: a[k] - b[k] for k in ("recall_trecho", "recall_documento", "mrr")}
                if all(abs(v) < 1e-9 for v in d.values()):
                    verdict = "igual"
                else:
                    verdict = "sim" if d["mrr"] > 0 and d["recall_trecho"] >= 0 else "não"
                lines.append(
                    f"| {top_k} | {scope} | {mode} | {d['recall_trecho']:+.3f} | "
                    f"{d['recall_documento']:+.3f} | {d['mrr']:+.3f} | {verdict} |"
                )
    markdown = "\n".join(lines)
    (out_dir / "tabela_topk.md").write_text(markdown + "\n", encoding="utf-8")
    print("\n" + markdown)

    # 3) Sensibilidade: o que, no YAML, altera o resultado de um BM25 sozinho (baseline).
    ref = {q["id"]: evidence_for(BASELINE, 5, q) for q in questions}
    sens = []

    def changed(other: dict[str, list[dict]]) -> int:
        return sum(
            [e["id"] for e in other[q["id"]]] != [e["id"] for e in ref[q["id"]]] for q in questions
        )

    for cand in (5, 12, 30):
        other = {q["id"]: evidence_for(BASELINE, 5, q, candidate_k=cand) for q in questions}
        sens.append(("pipeline.candidate_k", f"{cand} (top_k=5)", changed(other)))
    for chars in (12000, 600, 300, 150):
        other = {q["id"]: evidence_for(BASELINE, 5, q, max_context_chars=chars) for q in questions}
        sens.append(("pipeline.max_context_chars", chars, changed(other)))
    for top_k in TOPKS:
        other = {q["id"]: evidence_for(BASELINE, top_k, q) for q in questions}
        sens.append(("pipeline.top_k", top_k, changed(other)))
    for name, values in (
        ("k1", GRID["k1"]),
        ("b", GRID["b"]),
        ("idf", IDFS),
        ("min_idf", MIN_IDFS),
    ):
        for value in values:
            other = {q["id"]: evidence_for({**BASELINE, name: value}, 5, q) for q in questions}
            sens.append((f"modules.bm25.{name}", value, changed(other)))
    with (out_dir / "sensibilidade.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["parametro", "valor", "perguntas_com_evidencias_diferentes_da_baseline"])
        writer.writerows(sens)
    print(
        f"\nSensibilidade (nº de perguntas, de {len(questions)}, cujas evidências mudam "
        "vs. baseline com top_k=5):"
    )
    for name, value, n in sens:
        print(f"  {name:<28} {value!s:<16} {n:>2}")

    (out_dir / "manifesto.json").write_text(
        json.dumps(
            {
                "codigo": {
                    "commit": git("rev-parse", "HEAD"),
                    "branch": git("branch", "--show-current"),
                    "alteracoes_nao_commitadas": bool(git("status", "--porcelain")),
                },
                "hardware": {"cpu": cpu_name()},
                "baseline": BASELINE,
                "ativo": active,
                "melhor_grade": best_in_sample,
                "top_k": TOPKS,
                "treino": [q["id"] for q in train],
                "teste": [q["id"] for q in test],
                "modelos": "nenhum (só BM25; geração desativada)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
