"""G7 (Fase F9): Avaliador experimental, replay offline exato e gerador de tabelas T1-T6."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path
from statistics import fmean, median
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_lab.modules.hybrid import Hybrid  # noqa: E402
from rag_lab.pipeline import select_without_fusion  # noqa: E402

RRF_K_GRID: tuple[int, ...] = (1, 3, 5, 10, 11, 15, 21, 22, 30, 60, 120, 1000)

ABSTENTION_PATTERNS: tuple[str, ...] = (
    "nao encontrei evidencia",
    "nao encontrei informac",
    "nao ha evidencia",
    "nao ha informac",
    "nao consta",
    "nao mencion",
    "nao especifica",
    "nao informa",
    "nao contem",
    "nao traz",
    "nao abord",
    "nao detalha",
    "nao possui",
    "nao suporta",
    "nao utiliza",
    "bloqueados na triagem",
    "sem indice",
    "nenhuma informacao",
    "fora do escopo",
    "evidencias fornecidas nao",
    "fontes fornecidas nao",
    "documentos fornecidos nao",
    "insuficientes para responder",
    "insuficiente para responder",
    "nao e possivel determinar",
    "nao e possivel informar",
    "nao e possivel responder",
)


def normalize_text(text: str, strip_accents: bool = True) -> str:
    """Normaliza espaços, caixa e (opcionalmente) diacríticos para casamento determinístico."""
    collapsed = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not strip_accents:
        return unicodedata.normalize("NFC", collapsed)
    decomposed = unicodedata.normalize("NFD", collapsed)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def jaccard_index(set_a: set[str], set_b: set[str]) -> float:
    """Índice de Jaccard |A cap B| / |A cup B|; retorna 0.0 quando ambos os conjuntos são vazios."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def extract_citations(answer: str | None) -> list[int]:
    """Extrai referências numéricas no formato [1], [2], preservando a ordem de aparição."""
    if not answer:
        return []
    return [int(m) for m in re.findall(r"\[(\d+)\]", answer)]


def evaluate_citations(
    answer: str | None,
    evidence: list[dict[str, Any]],
    question_spec: dict[str, Any],
) -> dict[str, Any]:
    """Valida sintaxe das citações [n] contra 1..len(evidence) e calcula precisão das fontes."""
    cited = extract_citations(answer)
    max_idx = len(evidence)
    out_of_bounds = [idx for idx in cited if idx < 1 or idx > max_idx]
    valid_indices = [idx for idx in cited if 1 <= idx <= max_idx]
    unique_valid = list(dict.fromkeys(valid_indices))
    syntax_valid = len(out_of_bounds) == 0

    rel_docs = set(question_spec.get("documentos_relevantes") or [])
    rel_snippets = question_spec.get("trechos_relevantes") or []

    if not unique_valid:
        doc_precision = 0.0
        snippet_precision = 0.0
    else:
        doc_hits = 0
        snippet_hits = 0
        for idx in unique_valid:
            chunk = evidence[idx - 1]
            if chunk.get("source") in rel_docs:
                doc_hits += 1
            if any(chunk_matches_snippet(chunk, sn) for sn in rel_snippets):
                snippet_hits += 1
        doc_precision = doc_hits / len(unique_valid)
        snippet_precision = snippet_hits / len(unique_valid)

    return {
        "citations": cited,
        "unique_valid_citations": unique_valid,
        "out_of_bounds_citations": out_of_bounds,
        "has_citations": len(cited) > 0,
        "syntax_valid": syntax_valid,
        "cited_doc_precision": doc_precision,
        "cited_snippet_precision": snippet_precision,
    }


def detect_abstention(answer: str | None) -> bool:
    """Heurística inicial de detecção de abstenção (a avaliação final usa rubrica humana)."""
    norm = normalize_text(answer or "", strip_accents=True)
    if not norm:
        return True
    return any(pat in norm for pat in ABSTENTION_PATTERNS)


def heuristic_fact_coverage(answer: str | None, fatos_esperados: list[str]) -> float:
    """Cobertura lexical heurística dos fatos esperados (complementar à rubrica humana 0-2)."""
    if not fatos_esperados:
        return 0.0
    norm_answer = normalize_text(answer or "", strip_accents=True)
    answer_tokens = set(re.findall(r"[a-z0-9]{4,}", norm_answer))
    if not answer_tokens:
        return 0.0
    scores: list[float] = []
    for fact in fatos_esperados:
        fact_tokens = set(re.findall(r"[a-z0-9]{4,}", normalize_text(fact, strip_accents=True)))
        if not fact_tokens:
            continue
        scores.append(len(fact_tokens & answer_tokens) / len(fact_tokens))
    return fmean(scores) if scores else 0.0


def chunk_matches_snippet(
    chunk: dict[str, Any],
    snippet_spec: dict[str, Any],
    text_by_id: dict[str, str] | None = None,
) -> bool:
    """Verifica se um chunk corresponde a um trecho relevante do gabarito."""
    expected_source = snippet_spec.get("fonte")
    if expected_source and chunk.get("source") != expected_source:
        return False
    raw_text = chunk.get("text")
    if not raw_text and text_by_id is not None:
        raw_text = text_by_id.get(str(chunk.get("id", "")), "")
    if not raw_text:
        return False
    target = normalize_text(str(snippet_spec.get("trecho", "")), strip_accents=True)
    if not target:
        return False
    return target in normalize_text(str(raw_text), strip_accents=True)


def build_text_lookup(envelope: dict[str, Any]) -> dict[str, str]:
    """Constrói mapa id -> text a partir de retrieval e evidence para avaliar ranking completo."""
    lookup: dict[str, str] = {}
    retrieval = envelope.get("retrieval") or {}
    for ranking in retrieval.values():
        if isinstance(ranking, list):
            for item in ranking:
                if isinstance(item, dict) and item.get("id") and item.get("text"):
                    lookup[str(item["id"])] = str(item["text"])
    for item in envelope.get("evidence") or []:
        if isinstance(item, dict) and item.get("id") and item.get("text"):
            lookup[str(item["id"])] = str(item["text"])
    return lookup


def extract_full_ranking(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extrai o ranking completo pós-seleção/fusão (além do corte top_k quando disponível),
    garantindo simetria metodológica para Recall@10 e MRR entre E0, E1, E2 e E4:
      - Com reranking ativo (E5): usa envelope["evidence"].
      - Com hybrid executado (E2/E4): reconstrói a lista completa de attributes.hybrid.ranking.
      - Com hybrid em bypass e 1 recuperador (E0): usa retrieval[unico_recuperador].
      - Com hybrid em bypass e >=2 recuperadores (E1): reconstrói via select_without_fusion.
    """
    text_by_id = build_text_lookup(envelope)
    attributes = envelope.get("attributes") or {}
    hybrid_attr = attributes.get("hybrid") if isinstance(attributes, dict) else None
    trace = envelope.get("trace") or []
    rerank_trace = next(
        (t for t in trace if isinstance(t, dict) and t.get("module") == "reranking"),
        None,
    )
    if rerank_trace is not None and rerank_trace.get("status") == "executed":
        return deepcopy(envelope.get("evidence") or [])

    hybrid_trace = next(
        (t for t in trace if isinstance(t, dict) and t.get("module") == "hybrid"),
        None,
    )
    hybrid_executed = (
        (hybrid_trace is not None and hybrid_trace.get("status") == "executed")
        or (hybrid_trace is None and isinstance(hybrid_attr, dict) and "ranking" in hybrid_attr)
    )

    if hybrid_executed and isinstance(hybrid_attr, dict) and isinstance(
        hybrid_attr.get("ranking"), list
    ):
        full_list: list[dict[str, Any]] = []
        for entry in hybrid_attr["ranking"]:
            cid = str(entry["id"])
            full_list.append(
                {
                    "id": cid,
                    "source": entry["source"],
                    "position": entry["position"],
                    "score": entry["fused"],
                    "text": text_by_id.get(cid, ""),
                }
            )
        return full_list

    retrieval = envelope.get("retrieval") or {}
    if len(retrieval) == 1:
        sole_list = next(iter(retrieval.values()))
        if isinstance(sole_list, list) and sole_list:
            return deepcopy(sole_list)
    elif len(retrieval) >= 2:
        payload_copy = {"retrieval": deepcopy(retrieval), "evidence": []}
        select_without_fusion(payload_copy)
        rr_full = payload_copy["evidence"]
        evidence = envelope.get("evidence") or []
        ev_ids = [item["id"] for item in evidence]
        if not ev_ids or [item["id"] for item in rr_full[: len(ev_ids)]] == ev_ids:
            return rr_full

    return deepcopy(envelope.get("evidence") or [])


def compute_document_metrics(
    ranking: list[dict[str, Any]],
    relevant_docs: list[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
    ndcg_k: int = 5,
) -> dict[str, float]:
    """Calcula Recall@k, Hit@5, MRR e nDCG@5 em nível de documento (sem ganho duplicado)."""
    rel_set = set(relevant_docs)
    if not rel_set:
        return {
            **{f"doc_recall_{k}": 0.0 for k in ks},
            "doc_hit_5": 0.0,
            "doc_mrr": 0.0,
            f"doc_ndcg_{ndcg_k}": 0.0,
        }

    metrics: dict[str, float] = {}
    for k in ks:
        found = {str(item.get("source")) for item in ranking[:k]} & rel_set
        metrics[f"doc_recall_{k}"] = len(found) / len(rel_set)

    found_at_5 = {str(item.get("source")) for item in ranking[:5]} & rel_set
    metrics["doc_hit_5"] = 1.0 if found_at_5 else 0.0

    mrr = 0.0
    for idx, item in enumerate(ranking, start=1):
        if item.get("source") in rel_set:
            mrr = 1.0 / idx
            break
    metrics["doc_mrr"] = mrr

    # nDCG@k binário com deduplicação de documento já creditado:
    # DCG@k = sum_{r=1..k} g_r / log2(r + 1), onde g_r = 1 na 1ª aparição de doc relevante.
    seen_docs: set[str] = set()
    dcg = 0.0
    for idx, item in enumerate(ranking[:ndcg_k], start=1):
        src = str(item.get("source", ""))
        if src in rel_set and src not in seen_docs:
            seen_docs.add(src)
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(ndcg_k, len(rel_set))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    metrics[f"doc_ndcg_{ndcg_k}"] = (dcg / idcg) if idcg > 0 else 0.0
    return metrics


def compute_snippet_metrics(
    ranking: list[dict[str, Any]],
    relevant_snippets: list[dict[str, Any]],
    text_by_id: dict[str, str] | None = None,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    ndcg_k: int = 5,
) -> dict[str, float]:
    """
    Calcula Recall@k, Hit@5, MRR e nDCG@5 em nível de trecho (substring normalizada).

    Fórmula do nDCG@k graduado por `essencial`:
      - Cada trecho relevante m do gabarito tem ganho g(m) = 2.0 se essencial=True, senão 1.0.
      - Cada trecho m do gabarito só é creditado uma vez (no primeiro chunk que o contém).
      - DCG@k = sum_{r=1..k} gain(r) / log2(r + 1).
      - IDCG@k = sum_{r=1..min(k, |T_rel|)} gain_sorted[r-1] / log2(r + 1).
    """
    if not relevant_snippets:
        return {
            **{f"trecho_recall_{k}": 0.0 for k in ks},
            "trecho_hit_5": 0.0,
            "trecho_mrr": 0.0,
            f"trecho_ndcg_{ndcg_k}": 0.0,
        }

    num_snippets = len(relevant_snippets)
    matched_by_rank: list[list[int]] = []
    for item in ranking:
        matched_indices = [
            s_idx
            for s_idx, sn in enumerate(relevant_snippets)
            if chunk_matches_snippet(item, sn, text_by_id)
        ]
        matched_by_rank.append(matched_indices)

    metrics: dict[str, float] = {}
    for k in ks:
        covered: set[int] = set()
        for indices in matched_by_rank[:k]:
            covered.update(indices)
        metrics[f"trecho_recall_{k}"] = len(covered) / num_snippets

    covered_5: set[int] = set()
    for indices in matched_by_rank[:5]:
        covered_5.update(indices)
    metrics["trecho_hit_5"] = 1.0 if covered_5 else 0.0

    mrr = 0.0
    for idx, indices in enumerate(matched_by_rank, start=1):
        if indices:
            mrr = 1.0 / idx
            break
    metrics["trecho_mrr"] = mrr

    gains = [2.0 if sn.get("essencial", True) else 1.0 for sn in relevant_snippets]
    credited: set[int] = set()
    dcg = 0.0
    for idx, indices in enumerate(matched_by_rank[:ndcg_k], start=1):
        new_matches = [s_idx for s_idx in indices if s_idx not in credited]
        if new_matches:
            best_idx = max(new_matches, key=lambda i: gains[i])
            credited.add(best_idx)
            dcg += gains[best_idx] / math.log2(idx + 1)

    ideal_gains = sorted(gains, reverse=True)[:ndcg_k]
    idcg = sum(g / math.log2(idx + 1) for idx, g in enumerate(ideal_gains, start=1))
    metrics[f"trecho_ndcg_{ndcg_k}"] = (dcg / idcg) if idcg > 0 else 0.0
    return metrics


def first_relevant_snippet_rank(
    ranking: list[dict[str, Any]],
    relevant_snippets: list[dict[str, Any]],
    text_by_id: dict[str, str] | None = None,
) -> int | None:
    for idx, item in enumerate(ranking, start=1):
        if any(chunk_matches_snippet(item, sn, text_by_id) for sn in relevant_snippets):
            return idx
    return None


def replay_from_retrieval(
    retrieval: dict[str, list[dict[str, Any]]],
    *,
    mode: str = "hybrid",
    strategy: str = "rrf",
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    top_k: int = 5,
    retriever_order: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Reexecuta select_without_fusion ou Hybrid.run em memória sobre um dicionário `retrieval`.
    Garante 100% de fidelidade com o código de produção do rag_lab.
    """
    if retriever_order is not None:
        filtered_retrieval = {
            r: deepcopy(retrieval[r]) for r in retriever_order if r in retrieval
        }
    else:
        filtered_retrieval = deepcopy(retrieval)

    payload: dict[str, Any] = {
        "question": "replay",
        "retrieval": filtered_retrieval,
        "evidence": [],
        "attributes": {},
    }

    if mode == "round_robin":
        select_without_fusion(payload)
        full_ranking = deepcopy(payload["evidence"])
        return {
            "mode": "round_robin",
            "strategy": "round_robin",
            "evidence_top_k": full_ranking[:top_k],
            "full_ranking": full_ranking,
            "telemetry": None,
        }

    if mode != "hybrid":
        raise ValueError(f"Modo de replay desconhecido: {mode}")

    stub_services = SimpleNamespace(
        settings=SimpleNamespace(
            pipeline=SimpleNamespace(rrf_k=rrf_k, top_k=top_k),
            hybrid=SimpleNamespace(strategy=strategy, weights=dict(weights or {})),
        )
    )
    select_without_fusion(payload)
    result = Hybrid().run(payload, stub_services)
    full_ranking = deepcopy(result["evidence"])
    return {
        "mode": "hybrid",
        "strategy": strategy,
        "evidence_top_k": full_ranking[:top_k],
        "full_ranking": full_ranking,
        "telemetry": result["attributes"]["hybrid"],
    }


def verify_replay_equivalence(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compara o ranking produzido pelo replay offline com os envelopes online salvos."""
    checked = 0
    mismatches: list[dict[str, Any]] = []

    for rec in records:
        env = rec["envelope"]
        retrieval = env.get("retrieval") or {}
        if not retrieval:
            continue
        trace = env.get("trace") or []
        rerank_trace = next(
            (t for t in trace if isinstance(t, dict) and t.get("module") == "reranking"),
            None,
        )
        if rerank_trace and rerank_trace.get("status") == "executed":
            continue

        hybrid_trace = next(
            (t for t in trace if isinstance(t, dict) and t.get("module") == "hybrid"),
            None,
        )
        attrs = env.get("attributes") or {}
        hybrid_telemetry = attrs.get("hybrid") if isinstance(attrs, dict) else None
        hybrid_active = (
            (hybrid_trace is not None and hybrid_trace.get("status") == "executed")
            or (hybrid_trace is None and isinstance(hybrid_telemetry, dict))
        )

        online_ev_ids = [str(item["id"]) for item in (env.get("evidence") or [])]
        top_k = len(online_ev_ids) or 5

        if not hybrid_active:
            replayed = replay_from_retrieval(retrieval, mode="round_robin", top_k=top_k)
            replay_top_ids = [str(item["id"]) for item in replayed["evidence_top_k"]]
            checked += 1
            if replay_top_ids != online_ev_ids:
                mismatches.append(
                    {
                        "config": rec["config_name"],
                        "question_id": rec["question_id"],
                        "rep": rec["rep"],
                        "mode": "round_robin",
                        "online_ids": online_ev_ids,
                        "replay_ids": replay_top_ids,
                    }
                )
        else:
            if not isinstance(hybrid_telemetry, dict):
                continue
            strategy = str(hybrid_telemetry.get("strategy", "rrf"))
            params = hybrid_telemetry.get("params") or {}
            rrf_k = int(params.get("rrf_k", 60))
            raw_weights = params.get("weights") or {}
            weights = dict(raw_weights) if strategy != "rrf" else {}
            replayed = replay_from_retrieval(
                retrieval,
                mode="hybrid",
                strategy=strategy,
                weights=weights,
                rrf_k=rrf_k,
                top_k=top_k,
            )
            replay_top_ids = [str(item["id"]) for item in replayed["evidence_top_k"]]
            replay_full_ids = [str(item["id"]) for item in replayed["full_ranking"]]
            online_full_ids = [
                str(entry["id"]) for entry in (hybrid_telemetry.get("ranking") or [])
            ]
            checked += 1
            if replay_top_ids != online_ev_ids or (
                online_full_ids and replay_full_ids != online_full_ids
            ):
                mismatches.append(
                    {
                        "config": rec["config_name"],
                        "question_id": rec["question_id"],
                        "rep": rec["rep"],
                        "mode": strategy,
                        "online_ids": online_ev_ids,
                        "replay_ids": replay_top_ids,
                    }
                )

    return {
        "total_checked": checked,
        "all_matched": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def detect_fusion_tax_for_envelope(
    envelope: dict[str, Any],
    question_spec: dict[str, Any],
    *,
    config_name: str = "hybrid",
) -> dict[str, Any] | None:
    """
    Identifica se a fusão híbrida sofreu Fusion Tax frente ao melhor recuperador individual
    ou ao round-robin (MRR ou Recall@5 de trecho inferior).
    """
    if not question_spec.get("responder", True):
        return None
    rel_snippets = question_spec.get("trechos_relevantes") or []
    if not rel_snippets:
        return None

    retrieval = envelope.get("retrieval") or {}
    if len(retrieval) < 2:
        return None

    trace = envelope.get("trace") or []
    hybrid_trace = next(
        (t for t in trace if isinstance(t, dict) and t.get("module") == "hybrid"),
        None,
    )
    if hybrid_trace is not None and hybrid_trace.get("status") != "executed":
        return None

    attrs = envelope.get("attributes") or {}
    hybrid_telemetry = attrs.get("hybrid") if isinstance(attrs, dict) else None
    if not isinstance(hybrid_telemetry, dict):
        return None

    text_by_id = build_text_lookup(envelope)
    fused_ranking = extract_full_ranking(envelope)
    fused_metrics = compute_snippet_metrics(fused_ranking, rel_snippets, text_by_id)
    fused_mrr = fused_metrics["trecho_mrr"]
    fused_r5 = fused_metrics["trecho_recall_5"]
    fused_rank = first_relevant_snippet_rank(fused_ranking, rel_snippets, text_by_id)

    best_single_name = ""
    best_single_mrr = -1.0
    best_single_r5 = -1.0
    best_single_rank: int | None = None
    ranks_in_retrievers: dict[str, int | None] = {}

    for r_name, r_list in retrieval.items():
        r_metrics = compute_snippet_metrics(r_list, rel_snippets, text_by_id)
        r_rank = first_relevant_snippet_rank(r_list, rel_snippets, text_by_id)
        ranks_in_retrievers[r_name] = r_rank
        if (r_metrics["trecho_mrr"], r_metrics["trecho_recall_5"]) > (
            best_single_mrr,
            best_single_r5,
        ):
            best_single_name = r_name
            best_single_mrr = r_metrics["trecho_mrr"]
            best_single_r5 = r_metrics["trecho_recall_5"]
            best_single_rank = r_rank

    rr_replay = replay_from_retrieval(retrieval, mode="round_robin", top_k=5)
    rr_metrics = compute_snippet_metrics(rr_replay["full_ranking"], rel_snippets, text_by_id)
    rr_mrr = rr_metrics["trecho_mrr"]
    rr_r5 = rr_metrics["trecho_recall_5"]
    rr_rank = first_relevant_snippet_rank(rr_replay["full_ranking"], rel_snippets, text_by_id)

    ref_mrr = max(best_single_mrr, rr_mrr)
    ref_r5 = max(best_single_r5, rr_r5)

    if fused_mrr + 1e-12 >= ref_mrr and fused_r5 + 1e-12 >= ref_r5:
        return None

    if fused_r5 + 1e-12 < ref_r5:
        severidade = "perda_top5"
    elif fused_r5 > 0.0:
        severidade = "queda_mrr_no_top5"
    else:
        severidade = "queda_mrr_fora_top5"

    target_contribs: dict[str, float] = {}
    if fused_rank is not None and fused_rank <= len(hybrid_telemetry.get("ranking") or []):
        target_entry = hybrid_telemetry["ranking"][fused_rank - 1]
        target_contribs = dict(target_entry.get("contributions") or {})

    ret_names = list(retrieval.keys())
    if len(ret_names) >= 2:
        pair_jaccards = [
            jaccard_index(
                {str(x["id"]) for x in retrieval[r_a]},
                {str(x["id"]) for x in retrieval[r_b]},
            )
            for r_a, r_b in itertools.combinations(ret_names, 2)
        ]
        jacc = fmean(pair_jaccards)
    else:
        jacc = 0.0

    if "bm25" in retrieval and "embedding" in retrieval:
        jacc_bm25_emb: float | str = round(
            jaccard_index(
                {str(x["id"]) for x in retrieval["bm25"]},
                {str(x["id"]) for x in retrieval["embedding"]},
            ),
            4,
        )
    else:
        jacc_bm25_emb = "-"

    return {
        "config": config_name,
        "strategy": str(hybrid_telemetry.get("strategy", "rrf")),
        "question_id": str(question_spec["id"]),
        "perfil": str(question_spec.get("perfil", "")),
        "severidade": severidade,
        "best_single_retriever": best_single_name,
        "best_single_rank": best_single_rank,
        "best_single_mrr": round(best_single_mrr, 6),
        "best_single_recall_5": round(best_single_r5, 6),
        "rr_rank": rr_rank,
        "rr_mrr": round(rr_mrr, 6),
        "rr_recall_5": round(rr_r5, 6),
        "fused_rank": fused_rank,
        "fused_mrr": round(fused_mrr, 6),
        "fused_recall_5": round(fused_r5, 6),
        "delta_mrr_vs_best_single": round(fused_mrr - best_single_mrr, 6),
        "delta_recall5_vs_best_single": round(fused_r5 - best_single_r5, 6),
        "target_ranks_by_retriever": json.dumps(ranks_in_retrievers, ensure_ascii=False),
        "target_contributions": json.dumps(target_contribs, ensure_ascii=False),
        "jaccard_top12": round(jacc, 4),
        "jaccard_bm25_embedding": jacc_bm25_emb,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def load_run_records(
    runs_dir: Path | None,
    run_dirs: list[Path] | None,
    questions_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Carrega envelopes tanto de diretórios `<run_id>/envelopes/*.json` gerados por `rodar.py`
    quanto de diretórios com envelopes diretos (como `data/results/teste_ao_vivo/*.json`).
    """
    candidate_dirs: list[Path] = []
    if run_dirs:
        candidate_dirs.extend(run_dirs)
    elif runs_dir and runs_dir.exists():
        subdirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
        has_direct_json = any(
            p.is_file() and p.suffix == ".json" and p.name != "manifest.json"
            for p in runs_dir.iterdir()
        )
        if has_direct_json:
            candidate_dirs.append(runs_dir)
        candidate_dirs.extend(subdirs)

    questions_by_norm_text = {
        normalize_text(str(q["pergunta"])): q for q in questions_by_id.values()
    }

    records: list[dict[str, Any]] = []
    for d in candidate_dirs:
        if not d.exists() or not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        envelopes_folder = d / "envelopes" if (d / "envelopes").is_dir() else d
        json_files = sorted(
            p
            for p in envelopes_folder.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".json"
            and p.name not in {"manifest.json", "replay_verificacao.json"}
        )

        for jf in json_files:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue

            if "envelope" in raw and isinstance(raw["envelope"], dict):
                env = raw["envelope"]
                qid = str(raw.get("question_id") or "")
                config_name = str(
                    raw.get("config_name")
                    or (manifest.get("config_name") if isinstance(manifest, dict) else None)
                    or d.name
                )
                run_id = str(raw.get("run_id") or d.name)
                rep = int(raw.get("rep") or 1)
                wall_ms = float(raw.get("wall_time_ms") or 0.0)
                calls = dict(
                    raw.get("calls") or {"chat_calls": 0, "embed_calls": 0, "embed_texts": 0}
                )
            elif "question" in raw and "retrieval" in raw:
                env = raw
                attrs = env.get("attributes") or {}
                stem_parts = jf.stem.split("__", 1)
                qid = str(
                    (attrs.get("pergunta_id") if isinstance(attrs, dict) else None)
                    or (stem_parts[0] if stem_parts[0] in questions_by_id else "")
                )
                if not qid:
                    matched_q = questions_by_norm_text.get(
                        normalize_text(str(env.get("question", "")))
                    )
                    if matched_q:
                        qid = str(matched_q["id"])
                config_name = str(
                    (attrs.get("modo") if isinstance(attrs, dict) else None)
                    or (attrs.get("config") if isinstance(attrs, dict) else None)
                    or (stem_parts[1] if len(stem_parts) > 1 else d.name)
                )
                run_id = d.name
                rep = int((attrs.get("rep") if isinstance(attrs, dict) else None) or 1)
                trace = env.get("trace") or []
                wall_ms = round(
                    sum(float(t.get("duration_ms", 0.0)) for t in trace if isinstance(t, dict)),
                    3,
                )
                gen_exec = any(
                    isinstance(t, dict)
                    and t.get("module") == "generation"
                    and t.get("status") == "executed"
                    for t in trace
                )
                emb_exec = any(
                    isinstance(t, dict)
                    and t.get("module") == "embedding"
                    and t.get("status") == "executed"
                    for t in trace
                )
                n_queries = len(env.get("queries") or [1])
                calls = {
                    "chat_calls": 1 if gen_exec else 0,
                    "embed_calls": 1 if emb_exec else 0,
                    "embed_texts": n_queries if emb_exec else 0,
                }
            else:
                continue

            if qid not in questions_by_id:
                continue
            q_spec = questions_by_id[qid]
            records.append(
                {
                    "run_id": run_id,
                    "config_name": config_name,
                    "question_id": qid,
                    "perfil": str(q_spec.get("perfil", "")),
                    "rep": rep,
                    "wall_time_ms": wall_ms,
                    "calls": calls,
                    "envelope": env,
                    "question_spec": q_spec,
                }
            )

    return records


def _write_csv_and_md(
    out_dir: Path,
    stem: str,
    title: str,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_path = out_dir / f"{stem}.md"
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_Nenhum registro encontrado._")
    else:
        header = "| " + " | ".join(fieldnames) + " |"
        sep = "| " + " | ".join("---" for _ in fieldnames) + " |"
        lines.extend([header, sep])
        for row in rows:
            vals = [str(row.get(k, "")).replace("|", "\\|") for k in fieldnames]
            lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def generate_t1_recuperacao(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "config",
        "perfil",
        "n_consultas",
        "doc_recall_1",
        "doc_recall_3",
        "doc_recall_5",
        "doc_recall_10",
        "doc_hit_5",
        "doc_mrr",
        "doc_ndcg_5",
        "trecho_recall_1",
        "trecho_recall_3",
        "trecho_recall_5",
        "trecho_recall_10",
        "trecho_hit_5",
        "trecho_mrr",
        "trecho_ndcg_5",
    ]
    metric_keys = fieldnames[3:]
    by_config: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if not rec["question_spec"].get("responder", True):
            continue
        env = rec["envelope"]
        q_spec = rec["question_spec"]
        full_ranking = extract_full_ranking(env)
        text_by_id = build_text_lookup(env)
        doc_m = compute_document_metrics(full_ranking, q_spec.get("documentos_relevantes") or [])
        snp_m = compute_snippet_metrics(
            full_ranking, q_spec.get("trechos_relevantes") or [], text_by_id
        )
        by_config.setdefault(rec["config_name"], []).append(
            {"perfil": rec["perfil"], **doc_m, **snp_m}
        )

    rows: list[dict[str, Any]] = []
    for cfg in sorted(by_config):
        items = by_config[cfg]
        for perfil_filter in ("TODOS_RESPONDIVEIS", "literal", "semantica", "composta"):
            subset = (
                items
                if perfil_filter == "TODOS_RESPONDIVEIS"
                else [x for x in items if x["perfil"] == perfil_filter]
            )
            if not subset:
                continue
            row: dict[str, Any] = {
                "config": cfg,
                "perfil": perfil_filter,
                "n_consultas": len(subset),
            }
            for mk in metric_keys:
                row[mk] = round(fmean(x[mk] for x in subset), 4)
            rows.append(row)

    return fieldnames, rows


def _rrf_regime_label(rrf_k: int) -> str:
    if rrf_k <= 10:
        return "k<=10 (1o isolado pode vencer m=2)"
    if rrf_k <= 21:
        return "10<k<=21 (m=2 domina m=1)"
    return "k>21 (dominancia cardinal m=3>m=2>m=1)"


def generate_t2_curva_rrf_k_and_replay(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "familia_replay",
        "variante",
        "rrf_k",
        "regime_dominancia",
        "recuperadores",
        "n_perguntas",
        "doc_recall_5",
        "doc_mrr",
        "doc_ndcg_5",
        "trecho_recall_1",
        "trecho_recall_3",
        "trecho_recall_5",
        "trecho_recall_10",
        "trecho_mrr",
        "trecho_ndcg_5",
        "boundary_tie_rate",
    ]
    # Escolhe por pergunta o envelope com maior número de recuperadores em payload["retrieval"]
    best_by_qid: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not rec["question_spec"].get("responder", True):
            continue
        qid = rec["question_id"]
        ret = rec["envelope"].get("retrieval") or {}
        if len(ret) < 2:
            continue
        prev = best_by_qid.get(qid)
        if prev is None or len(ret) > len(prev["envelope"].get("retrieval") or {}):
            best_by_qid[qid] = rec

    if not best_by_qid:
        return fieldnames, []

    base_records = [best_by_qid[qid] for qid in sorted(best_by_qid)]
    available_retrievers = tuple(base_records[0]["envelope"]["retrieval"].keys())

    retriever_groups: list[tuple[str, ...]] = [available_retrievers]
    if (
        len(available_retrievers) >= 3
        and "bm25" in available_retrievers
        and "embedding" in available_retrievers
    ):
        retriever_groups.append(("bm25", "embedding"))

    rows: list[dict[str, Any]] = []

    def _agg_replay(
        familia: str,
        variante: str,
        rrf_k_val: int | str,
        regime: str,
        ret_label: str,
        eval_items: list[dict[str, Any]],
    ) -> None:
        rows.append(
            {
                "familia_replay": familia,
                "variante": variante,
                "rrf_k": rrf_k_val,
                "regime_dominancia": regime,
                "recuperadores": ret_label,
                "n_perguntas": len(eval_items),
                "doc_recall_5": round(fmean(x["doc_recall_5"] for x in eval_items), 4),
                "doc_mrr": round(fmean(x["doc_mrr"] for x in eval_items), 4),
                "doc_ndcg_5": round(fmean(x["doc_ndcg_5"] for x in eval_items), 4),
                "trecho_recall_1": round(fmean(x["trecho_recall_1"] for x in eval_items), 4),
                "trecho_recall_3": round(fmean(x["trecho_recall_3"] for x in eval_items), 4),
                "trecho_recall_5": round(fmean(x["trecho_recall_5"] for x in eval_items), 4),
                "trecho_recall_10": round(fmean(x["trecho_recall_10"] for x in eval_items), 4),
                "trecho_mrr": round(fmean(x["trecho_mrr"] for x in eval_items), 4),
                "trecho_ndcg_5": round(fmean(x["trecho_ndcg_5"] for x in eval_items), 4),
                "boundary_tie_rate": round(fmean(x["boundary_tie"] for x in eval_items), 4),
            }
        )

    for ret_group in retriever_groups:
        retrievers_label = "+".join(ret_group)

        # 1) Varredura de rrf_k (E2)
        for k_val in RRF_K_GRID:
            evals: list[dict[str, Any]] = []
            for rec in base_records:
                env = rec["envelope"]
                q_spec = rec["question_spec"]
                text_by_id = build_text_lookup(env)
                rep = replay_from_retrieval(
                    env["retrieval"],
                    mode="hybrid",
                    strategy="rrf",
                    rrf_k=k_val,
                    top_k=5,
                    retriever_order=ret_group,
                )
                doc_m = compute_document_metrics(
                    rep["full_ranking"], q_spec.get("documentos_relevantes") or []
                )
                snp_m = compute_snippet_metrics(
                    rep["full_ranking"], q_spec.get("trechos_relevantes") or [], text_by_id
                )
                btie = 1.0 if (rep["telemetry"] or {}).get("boundary_tie") else 0.0
                evals.append({**doc_m, **snp_m, "boundary_tie": btie})
            _agg_replay(
                "E2_curva_rrf_k",
                f"rrf_k={k_val}",
                k_val,
                _rrf_regime_label(k_val),
                retrievers_label,
                evals,
            )

        # 2) Permutações de ordem do Round-Robin (E3)
        for perm in itertools.permutations(ret_group):
            evals = []
            for rec in base_records:
                env = rec["envelope"]
                q_spec = rec["question_spec"]
                text_by_id = build_text_lookup(env)
                rep = replay_from_retrieval(
                    env["retrieval"], mode="round_robin", top_k=5, retriever_order=perm
                )
                doc_m = compute_document_metrics(
                    rep["full_ranking"], q_spec.get("documentos_relevantes") or []
                )
                snp_m = compute_snippet_metrics(
                    rep["full_ranking"], q_spec.get("trechos_relevantes") or [], text_by_id
                )
                evals.append({**doc_m, **snp_m, "boundary_tie": 0.0})
            perm_label = "->".join(perm)
            _agg_replay(
                "E3_permutacao_round_robin",
                f"rr({perm_label})",
                "-",
                "sem_fusao_round_robin",
                perm_label,
                evals,
            )

        # 3) Comparação de estratégias de fusão no replay offline (E4)
        w_emb2 = {r: (2.0 if r == "embedding" else 1.0) for r in ret_group}
        for strat_name, strat_weights in (
            ("rrf", {}),
            ("weighted_rrf", w_emb2),
            ("convex", {}),
            ("dbsf", {}),
        ):
            evals = []
            for rec in base_records:
                env = rec["envelope"]
                q_spec = rec["question_spec"]
                text_by_id = build_text_lookup(env)
                rep = replay_from_retrieval(
                    env["retrieval"],
                    mode="hybrid",
                    strategy=strat_name,
                    weights=strat_weights,
                    rrf_k=60,
                    top_k=5,
                    retriever_order=ret_group,
                )
                doc_m = compute_document_metrics(
                    rep["full_ranking"], q_spec.get("documentos_relevantes") or []
                )
                snp_m = compute_snippet_metrics(
                    rep["full_ranking"], q_spec.get("trechos_relevantes") or [], text_by_id
                )
                btie = 1.0 if (rep["telemetry"] or {}).get("boundary_tie") else 0.0
                evals.append({**doc_m, **snp_m, "boundary_tie": btie})
            _agg_replay(
                "E4_estrategias_fusao",
                strat_name,
                60 if "rrf" in strat_name else "-",
                "estrategia_hibrida",
                retrievers_label,
                evals,
            )

    return fieldnames, rows


def generate_t3_geracao_citacoes(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "config",
        "n_respondiveis",
        "taxa_respostas_com_citacao",
        "taxa_citacoes_sintaxe_valida",
        "precisao_fontes_citadas_doc",
        "precisao_fontes_citadas_trecho",
        "cobertura_fatos_heuristica",
        "n_sem_resposta",
        "taxa_abstencao_heuristica_sem_resposta",
        "fidelidade_humana_status",
    ]
    by_config: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_config.setdefault(rec["config_name"], []).append(rec)

    rows: list[dict[str, Any]] = []
    for cfg in sorted(by_config):
        items = by_config[cfg]
        ans_items = [r for r in items if r["question_spec"].get("responder", True)]
        unans_items = [r for r in items if not r["question_spec"].get("responder", True)]

        has_cit_list: list[float] = []
        valid_syn_list: list[float] = []
        doc_prec_list: list[float] = []
        snp_prec_list: list[float] = []
        fact_cov_list: list[float] = []

        for r in ans_items:
            env = r["envelope"]
            q_spec = r["question_spec"]
            cit_eval = evaluate_citations(
                env.get("answer"), env.get("evidence") or [], q_spec
            )
            has_cit_list.append(1.0 if cit_eval["has_citations"] else 0.0)
            valid_syn_list.append(1.0 if cit_eval["syntax_valid"] else 0.0)
            doc_prec_list.append(cit_eval["cited_doc_precision"])
            snp_prec_list.append(cit_eval["cited_snippet_precision"])
            fact_cov_list.append(
                heuristic_fact_coverage(env.get("answer"), q_spec.get("fatos_esperados") or [])
            )

        abst_list = [
            1.0 if detect_abstention(r["envelope"].get("answer")) else 0.0 for r in unans_items
        ]

        rows.append(
            {
                "config": cfg,
                "n_respondiveis": len(ans_items),
                "taxa_respostas_com_citacao": round(fmean(has_cit_list), 4)
                if has_cit_list
                else 0.0,
                "taxa_citacoes_sintaxe_valida": round(fmean(valid_syn_list), 4)
                if valid_syn_list
                else 0.0,
                "precisao_fontes_citadas_doc": round(fmean(doc_prec_list), 4)
                if doc_prec_list
                else 0.0,
                "precisao_fontes_citadas_trecho": round(fmean(snp_prec_list), 4)
                if snp_prec_list
                else 0.0,
                "cobertura_fatos_heuristica": round(fmean(fact_cov_list), 4)
                if fact_cov_list
                else 0.0,
                "n_sem_resposta": len(unans_items),
                "taxa_abstencao_heuristica_sem_resposta": round(fmean(abst_list), 4)
                if abst_list
                else 0.0,
                "fidelidade_humana_status": "pendente_rubrica_humana_0_2",
            }
        )

    return fieldnames, rows


def generate_t4_latencia_chamadas(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "config",
        "n_consultas",
        "hybrid_status",
        "hybrid_ms_mediana",
        "hybrid_ms_p95",
        "tfidf_ms_mediana",
        "bm25_ms_mediana",
        "embedding_ms_mediana",
        "reranking_ms_mediana",
        "generation_ms_mediana",
        "generation_ms_p95",
        "wall_time_ms_mediana",
        "wall_time_ms_p95",
        "chat_calls_total",
        "embed_calls_total",
        "embed_texts_total",
    ]
    by_config: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_config.setdefault(rec["config_name"], []).append(rec)

    rows: list[dict[str, Any]] = []
    for cfg in sorted(by_config):
        items = by_config[cfg]
        mod_durations: dict[str, list[float]] = {
            m: [] for m in ("hybrid", "tfidf", "bm25", "embedding", "reranking", "generation")
        }
        hybrid_statuses: set[str] = set()
        wall_times: list[float] = []
        chat_total = 0
        embed_total = 0
        embed_texts_total = 0

        for r in items:
            wall_times.append(float(r.get("wall_time_ms", 0.0)))
            calls = r.get("calls") or {}
            chat_total += int(calls.get("chat_calls", 0))
            embed_total += int(calls.get("embed_calls", 0))
            embed_texts_total += int(calls.get("embed_texts", 0))
            for t in r["envelope"].get("trace") or []:
                if not isinstance(t, dict):
                    continue
                m_name = str(t.get("module", ""))
                if m_name == "hybrid":
                    hybrid_statuses.add(str(t.get("status", "")))
                if m_name in mod_durations:
                    mod_durations[m_name].append(float(t.get("duration_ms", 0.0)))

        rows.append(
            {
                "config": cfg,
                "n_consultas": len(items),
                "hybrid_status": "/".join(sorted(hybrid_statuses)) or "unknown",
                "hybrid_ms_mediana": round(median(mod_durations["hybrid"]), 3)
                if mod_durations["hybrid"]
                else 0.0,
                "hybrid_ms_p95": round(_percentile(mod_durations["hybrid"], 95), 3)
                if mod_durations["hybrid"]
                else 0.0,
                "tfidf_ms_mediana": round(median(mod_durations["tfidf"]), 3)
                if mod_durations["tfidf"]
                else 0.0,
                "bm25_ms_mediana": round(median(mod_durations["bm25"]), 3)
                if mod_durations["bm25"]
                else 0.0,
                "embedding_ms_mediana": round(median(mod_durations["embedding"]), 3)
                if mod_durations["embedding"]
                else 0.0,
                "reranking_ms_mediana": round(median(mod_durations["reranking"]), 3)
                if mod_durations["reranking"]
                else 0.0,
                "generation_ms_mediana": round(median(mod_durations["generation"]), 3)
                if mod_durations["generation"]
                else 0.0,
                "generation_ms_p95": round(_percentile(mod_durations["generation"], 95), 3)
                if mod_durations["generation"]
                else 0.0,
                "wall_time_ms_mediana": round(median(wall_times), 3) if wall_times else 0.0,
                "wall_time_ms_p95": round(_percentile(wall_times, 95), 3) if wall_times else 0.0,
                "chat_calls_total": chat_total,
                "embed_calls_total": embed_total,
                "embed_texts_total": embed_texts_total,
            }
        )

    return fieldnames, rows


def generate_t5_fusion_tax(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "config",
        "strategy",
        "question_id",
        "perfil",
        "severidade",
        "best_single_retriever",
        "best_single_rank",
        "best_single_mrr",
        "best_single_recall_5",
        "rr_rank",
        "rr_mrr",
        "rr_recall_5",
        "fused_rank",
        "fused_mrr",
        "fused_recall_5",
        "delta_mrr_vs_best_single",
        "delta_recall5_vs_best_single",
        "target_ranks_by_retriever",
        "target_contributions",
        "jaccard_top12",
        "jaccard_bm25_embedding",
    ]
    rows: list[dict[str, Any]] = []
    for rec in records:
        tax = detect_fusion_tax_for_envelope(
            rec["envelope"],
            rec["question_spec"],
            config_name=rec["config_name"],
        )
        if tax is not None:
            rows.append(tax)
    rows.sort(key=lambda r: (r["config"], r["question_id"]))
    return fieldnames, rows


def generate_t6_sobreposicao_empates(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "config",
        "perfil",
        "n_consultas",
        "jaccard_bm25_embedding",
        "jaccard_tfidf_bm25",
        "jaccard_tfidf_embedding",
        "in_all_media",
        "fused_count_media",
        "boundary_tie_count",
        "boundary_tie_rate",
    ]
    by_config: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        env = rec["envelope"]
        ret = env.get("retrieval") or {}
        sets = {
            r: {str(item["id"]) for item in (ret.get(r) or [])}
            for r in ("tfidf", "bm25", "embedding")
            if r in ret and isinstance(ret.get(r), list)
        }
        active_sets = list(sets.values())
        if active_sets:
            union_all = set().union(*active_sets)
            inter_all: set[str] | None = (
                set(active_sets[0]).intersection(*active_sets[1:])
                if len(active_sets) >= 2
                else None
            )
        else:
            union_all = set()
            inter_all = None

        attrs = env.get("attributes") or {}
        hyb = attrs.get("hybrid") if isinstance(attrs, dict) else None
        btie = bool(isinstance(hyb, dict) and hyb.get("boundary_tie"))

        by_config.setdefault(rec["config_name"], []).append(
            {
                "perfil": rec["perfil"],
                "j_bm25_emb": (
                    jaccard_index(sets["bm25"], sets["embedding"])
                    if "bm25" in sets and "embedding" in sets
                    else None
                ),
                "j_tfidf_bm25": (
                    jaccard_index(sets["tfidf"], sets["bm25"])
                    if "tfidf" in sets and "bm25" in sets
                    else None
                ),
                "j_tfidf_emb": (
                    jaccard_index(sets["tfidf"], sets["embedding"])
                    if "tfidf" in sets and "embedding" in sets
                    else None
                ),
                "in_all": float(len(inter_all)) if inter_all is not None else None,
                "fused_count": float(len(union_all)),
                "btie": 1.0 if btie else 0.0,
            }
        )

    def _mean_or_dash(items_subset: list[dict[str, Any]], key: str, digits: int = 4) -> float | str:
        vals = [float(x[key]) for x in items_subset if x[key] is not None]
        return round(fmean(vals), digits) if vals else "-"

    rows: list[dict[str, Any]] = []
    for cfg in sorted(by_config):
        items = by_config[cfg]
        for perfil_filter in ("TODOS", "literal", "semantica", "composta", "sem_resposta"):
            subset = (
                items
                if perfil_filter == "TODOS"
                else [x for x in items if x["perfil"] == perfil_filter]
            )
            if not subset:
                continue
            rows.append(
                {
                    "config": cfg,
                    "perfil": perfil_filter,
                    "n_consultas": len(subset),
                    "jaccard_bm25_embedding": _mean_or_dash(subset, "j_bm25_emb", 4),
                    "jaccard_tfidf_bm25": _mean_or_dash(subset, "j_tfidf_bm25", 4),
                    "jaccard_tfidf_embedding": _mean_or_dash(subset, "j_tfidf_emb", 4),
                    "in_all_media": _mean_or_dash(subset, "in_all", 3),
                    "fused_count_media": round(fmean(x["fused_count"] for x in subset), 3),
                    "boundary_tie_count": int(sum(x["btie"] for x in subset)),
                    "boundary_tie_rate": round(fmean(x["btie"] for x in subset), 4),
                }
            )

    return fieldnames, rows


def export_human_evaluation_template(
    out_dir: Path,
    records: list[dict[str, Any]],
) -> Path:
    fieldnames = [
        "run_id",
        "config",
        "question_id",
        "perfil",
        "rep",
        "pergunta",
        "fatos_esperados",
        "resposta_llm",
        "citacoes_extraidas",
        "nota_fidelidade_0_2",
        "citacoes_corretas_s_n",
        "abstencao_correta_s_n",
        "avaliador",
        "observacoes",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "avaliacao_humana_template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(
            records, key=lambda r: (r["config_name"], r["question_id"], r["rep"])
        ):
            env = rec["envelope"]
            q_spec = rec["question_spec"]
            cits = extract_citations(env.get("answer"))
            fatos = " | ".join(str(x) for x in (q_spec.get("fatos_esperados") or []))
            writer.writerow(
                {
                    "run_id": rec["run_id"],
                    "config": rec["config_name"],
                    "question_id": rec["question_id"],
                    "perfil": rec["perfil"],
                    "rep": rec["rep"],
                    "pergunta": q_spec.get("pergunta", ""),
                    "fatos_esperados": fatos,
                    "resposta_llm": env.get("answer") or "",
                    "citacoes_extraidas": json.dumps(cits),
                    "nota_fidelidade_0_2": "",
                    "citacoes_corretas_s_n": "",
                    "abstencao_correta_s_n": "",
                    "avaliador": "",
                    "observacoes": "",
                }
            )
    return csv_path


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gabarito.exists():
        raise ValueError(f"Gabarito não encontrado: {args.gabarito}")
    gabarito_data = yaml.safe_load(args.gabarito.read_text(encoding="utf-8"))
    perguntas = gabarito_data.get("perguntas") or []
    questions_by_id = {str(q["id"]): q for q in perguntas}

    records = load_run_records(args.runs_dir, args.run_dir, questions_by_id)
    if not records:
        raise ValueError(
            "Nenhum envelope JSON válido encontrado para avaliação. "
            "Verifique --runs-dir ou --run-dir."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f1, r1 = generate_t1_recuperacao(records)
    _write_csv_and_md(out_dir, "t1_recuperacao", "Tabela T1 — Métricas de Recuperação", f1, r1)

    f2, r2 = generate_t2_curva_rrf_k_and_replay(records)
    _write_csv_and_md(
        out_dir,
        "t2_curva_rrf_k",
        "Tabela T2 — Replay Offline: Curva rrf_k, Permutações Round-Robin e Estratégias",
        f2,
        r2,
    )

    f3, r3 = generate_t3_geracao_citacoes(records)
    _write_csv_and_md(
        out_dir, "t3_geracao_citacoes", "Tabela T3 — Diagnóstico de Geração e Citações", f3, r3
    )

    f4, r4 = generate_t4_latencia_chamadas(records)
    _write_csv_and_md(
        out_dir, "t4_latencia_chamadas", "Tabela T4 — Latência por Etapa e Chamadas", f4, r4
    )

    f5, r5 = generate_t5_fusion_tax(records)
    _write_csv_and_md(
        out_dir, "t5_fusion_tax", "Tabela T5 — Casos Detectados de Fusion Tax", f5, r5
    )

    f6, r6 = generate_t6_sobreposicao_empates(records)
    _write_csv_and_md(
        out_dir,
        "t6_sobreposicao_empates",
        "Tabela T6 — Sobreposição (Jaccard Top-12) e Empates na Fronteira",
        f6,
        r6,
    )

    human_csv = export_human_evaluation_template(out_dir, records)
    replay_check = verify_replay_equivalence(records)
    (out_dir / "replay_verificacao.json").write_text(
        json.dumps(replay_check, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "records_evaluated": len(records),
        "configs_evaluated": sorted({r["config_name"] for r in records}),
        "questions_evaluated": sorted({r["question_id"] for r in records}),
        "replay_verification": replay_check,
        "fusion_tax_cases": len(r5),
        "out_dir": out_dir.as_posix(),
        "human_template_csv": human_csv.as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G7: Avaliador de métricas de recuperação/geração e motor de replay offline."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/results/g7"),
        help="Diretório contendo subpastas <run_id> ou envelopes JSON (padrão: data/results/g7)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="*",
        help="Lista explícita opcional de diretórios de execução (ex.: data/results/teste_ao_vivo)",
    )
    parser.add_argument(
        "--gabarito",
        type=Path,
        default=Path("experiments/g7/perguntas_gabarito.yaml"),
        help="Caminho do gabarito YAML (padrão: experiments/g7/perguntas_gabarito.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/g7/results"),
        help="Diretório de saída das tabelas T1-T6 e template humano",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_evaluation(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Erro em avaliar.py ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
