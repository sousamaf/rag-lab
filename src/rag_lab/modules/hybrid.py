"""G7: Fusão híbrida de rankings e scores (RRF, Weighted RRF, Convex Min-Max e DBSF)."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from statistics import fmean, pstdev
from typing import Any

from rag_lab.modules.base import Services

Scorer = Callable[[list[float], float, float, int], list[float]]


def _score_rrf(
    raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
) -> list[float]:
    return [1.0 / (rrf_k + rank) for rank in range(1, len(raw_scores) + 1)]


def _score_weighted_rrf(
    raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
) -> list[float]:
    return [weight / (rrf_k + rank) for rank in range(1, len(raw_scores) + 1)]


def _score_convex(
    raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
) -> list[float]:
    if not raw_scores or weight_sum <= 0.0:
        return []
    factor = weight / weight_sum
    lo = min(raw_scores)
    hi = max(raw_scores)
    span = hi - lo
    if span <= 0.0:
        return [factor for _ in raw_scores]
    return [factor * ((score - lo) / span) for score in raw_scores]


def _score_dbsf(
    raw_scores: list[float], weight: float, weight_sum: float, rrf_k: int
) -> list[float]:
    if not raw_scores or weight_sum <= 0.0:
        return []
    factor = weight / weight_sum
    mean = fmean(raw_scores)
    std = pstdev(raw_scores)
    if std <= 0.0:
        return [factor for _ in raw_scores]
    lo = mean - 3.0 * std
    span = 6.0 * std
    return [factor * min(1.0, max(0.0, (score - lo) / span)) for score in raw_scores]


STRATEGIES: dict[str, Scorer] = {
    "rrf": _score_rrf,
    "weighted_rrf": _score_weighted_rrf,
    "convex": _score_convex,
    "dbsf": _score_dbsf,
}


def _resolve_weights(
    strategy: str, configured_weights: dict[str, float], retrievers: list[str]
) -> dict[str, float]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia híbrida desconhecida: {strategy}")
    if strategy == "rrf" and configured_weights:
        raise ValueError("A estratégia 'rrf' não aceita weights; use 'weighted_rrf'.")
    if strategy == "weighted_rrf" and not configured_weights:
        raise ValueError("A estratégia 'weighted_rrf' exige weights explícitos.")
    if not configured_weights:
        return {r: 1.0 for r in retrievers}
    if set(configured_weights) != set(retrievers):
        raise ValueError(
            "hybrid.weights deve declarar exatamente os recuperadores presentes em retrieval."
        )
    weights: dict[str, float] = {}
    for r in retrievers:
        w = configured_weights[r]
        if (
            isinstance(w, bool)
            or not isinstance(w, (int, float))
            or not math.isfinite(w)
            or not 0 <= w <= 100
        ):
            raise ValueError(f"Peso inválido para o recuperador '{r}'.")
        weights[r] = float(w)
    if retrievers and math.fsum(weights.values()) <= 0.0:
        raise ValueError("hybrid.weights exige ao menos um peso maior que zero.")
    return weights


def _resolve_hybrid_config(payload: dict, settings: Any) -> tuple[str, dict[str, Any]]:
    hybrid_cfg = getattr(settings, "hybrid", None)
    if hybrid_cfg is not None:
        return hybrid_cfg.strategy, hybrid_cfg.weights

    attributes = payload.get("attributes")
    if not isinstance(attributes, dict) or "hybrid_config" not in attributes:
        return "rrf", {}

    raw_cfg = attributes["hybrid_config"]
    if not isinstance(raw_cfg, dict):
        raise ValueError("attributes.hybrid_config deve ser um dicionário.")
    unknown_keys = set(raw_cfg) - {"strategy", "weights"}
    if unknown_keys:
        raise ValueError(
            f"Opções desconhecidas em attributes.hybrid_config: {', '.join(sorted(unknown_keys))}"
        )

    strategy = raw_cfg.get("strategy", "rrf")
    if not isinstance(strategy, str) or strategy not in STRATEGIES:
        raise ValueError(f"Estratégia híbrida desconhecida: {strategy}")

    raw_weights = raw_cfg.get("weights", {})
    if not isinstance(raw_weights, dict):
        raise ValueError("attributes.hybrid_config.weights deve ser um dicionário.")
    weights: dict[str, float] = {}
    for retriever, weight in raw_weights.items():
        if not isinstance(retriever, str) or not retriever:
            raise ValueError("Nome de recuperador inválido em attributes.hybrid_config.weights.")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 100
        ):
            raise ValueError(f"Peso inválido para o recuperador '{retriever}'.")
        weights[retriever] = float(weight)
    return strategy, weights


def _extract_raw_scores(retriever: str, ranking: list[dict[str, Any]]) -> list[float]:
    raw_scores: list[float] = []
    for item in ranking:
        item_scores = item.get("scores") or {}
        raw = item_scores[retriever] if retriever in item_scores else item.get("score")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(
                f"Score bruto inválido em '{retriever}' para o item '{item.get('id')}'."
            )
        raw_scores.append(float(raw))
    return raw_scores


def _retriever_stats(raw_scores: list[float], weight: float) -> dict[str, Any]:
    if not raw_scores:
        return {
            "candidates": 0,
            "weight": weight,
            "raw_min": None,
            "raw_max": None,
            "raw_mean": None,
            "raw_std": None,
        }
    return {
        "candidates": len(raw_scores),
        "weight": weight,
        "raw_min": min(raw_scores),
        "raw_max": max(raw_scores),
        "raw_mean": fmean(raw_scores),
        "raw_std": pstdev(raw_scores),
    }


class Hybrid:
    def run(self, payload: dict, services: Services) -> dict:
        settings = services.settings
        strategy, configured_weights = _resolve_hybrid_config(payload, settings)
        rrf_k = settings.pipeline.rrf_k
        top_k = settings.pipeline.top_k

        retrieval = payload.get("retrieval") or {}
        retrievers = list(retrieval.keys())
        weights = _resolve_weights(strategy, configured_weights, retrievers)
        weight_sum = math.fsum(weights.values())
        scorer = STRATEGIES[strategy]

        merged: dict[str, dict[str, Any]] = {}
        ranks_by_id: dict[str, dict[str, int]] = {}
        contribs_by_id: dict[str, dict[str, float]] = {}
        retriever_telemetry: dict[str, dict[str, Any]] = {}

        for retriever in retrievers:
            ranking = retrieval[retriever]
            weight = weights[retriever]
            raw_scores = _extract_raw_scores(retriever, ranking)
            retriever_telemetry[retriever] = _retriever_stats(raw_scores, weight)
            contributions = scorer(raw_scores, weight, weight_sum, rrf_k)

            for rank, (item, contribution) in enumerate(
                zip(ranking, contributions, strict=True), start=1
            ):
                doc_id = item["id"]
                if doc_id not in merged:
                    merged[doc_id] = deepcopy(item)
                    merged[doc_id]["score"] = 0.0
                    ranks_by_id[doc_id] = {}
                    contribs_by_id[doc_id] = {}
                target = merged[doc_id]
                target["scores"].update(item["scores"])
                ranks_by_id[doc_id].setdefault(retriever, rank)
                contribs_by_id[doc_id][retriever] = (
                    contribs_by_id[doc_id].get(retriever, 0.0) + contribution
                )

        for doc_id, target in merged.items():
            target["score"] = math.fsum(contribs_by_id[doc_id].values())

        ordered = sorted(merged.values(), key=lambda e: (-e["score"], e["id"]))
        for item in ordered:
            item["scores"]["hybrid"] = item["score"]
        payload["evidence"] = ordered

        num_retrievers = len(retrievers)
        in_all = (
            sum(1 for doc_id in merged if len(ranks_by_id[doc_id]) == num_retrievers)
            if num_retrievers > 0
            else 0
        )
        exclusive = {
            r: sum(
                1
                for doc_id in merged
                if len(ranks_by_id[doc_id]) == 1 and r in ranks_by_id[doc_id]
            )
            for r in retrievers
        }
        boundary_tie = len(ordered) > top_k and math.isclose(
            ordered[top_k - 1]["score"],
            ordered[top_k]["score"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        payload.setdefault("attributes", {})["hybrid"] = {
            "schema": "g7-hybrid/1",
            "strategy": strategy,
            "params": {
                "rrf_k": rrf_k,
                "weights": weights,
                "weight_sum": weight_sum,
                "top_k": top_k,
            },
            "retrievers": retriever_telemetry,
            "overlap": {
                "fused_count": len(ordered),
                "in_all": in_all,
                "exclusive": exclusive,
            },
            "boundary_tie": boundary_tie,
            "ranking": [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "position": item["position"],
                    "fused": item["score"],
                    "fused_rank": idx,
                    "ranks": dict(ranks_by_id[item["id"]]),
                    "contributions": dict(contribs_by_id[item["id"]]),
                }
                for idx, item in enumerate(ordered, start=1)
            ],
        }
        return payload
