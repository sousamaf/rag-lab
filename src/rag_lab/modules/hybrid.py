"""G7: Reciprocal Rank Fusion (RRF); evita somar scores de escalas incompatíveis."""

from copy import deepcopy

from rag_lab.modules.base import Services


class Hybrid:
    def run(self, payload: dict, services: Services) -> dict:
        merged = {}
        constant = services.settings.pipeline.rrf_k
        for ranking in payload["retrieval"].values():
            for rank, item in enumerate(ranking, start=1):
                if item["id"] not in merged:
                    merged[item["id"]] = deepcopy(item)
                    merged[item["id"]]["score"] = 0.0
                target = merged[item["id"]]
                target["score"] += 1 / (constant + rank)
                target["scores"].update(item["scores"])
        payload["evidence"] = sorted(merged.values(), key=lambda e: (-e["score"], e["id"]))
        for item in payload["evidence"]:
            item["scores"]["hybrid"] = item["score"]
        return payload
