"""G9: expande cada sentença recuperada com suas vizinhas do MESMO documento."""

from itertools import combinations

from rag_lab.modules.base import Services


def count_overlapping_windows(evidence: list[dict], radius: int) -> int:
    """Pares do mesmo documento cujas janelas [p - r, p + r] compartilham sentenças."""
    return sum(
        a["document_id"] == b["document_id"] and abs(a["position"] - b["position"]) <= 2 * radius
        for a, b in combinations(evidence, 2)
    )


def window_metrics(evidence: list[dict], radius: int, top_k: int) -> dict:
    """Mede só os top_k trechos, os candidatos a ir ao LLM.

    Os valores são anteriores ao corte por max_context_chars e à Compressão, que
    podem encurtar o contexto efetivamente enviado.
    """
    sent = evidence[:top_k]
    return {
        "radius": radius,
        "evidence_count": len(sent),
        "original_chars": sum(len(item["text"]) for item in sent),
        "window_chars": sum(len(item["context_text"]) for item in sent),
        "overlapping_pairs": count_overlapping_windows(sent, radius),
    }


class SentenceWindow:
    def run(self, payload: dict, services: Services) -> dict:
        pipeline = services.settings.pipeline
        radius = pipeline.sentence_radius
        for item in payload["evidence"]:
            item["context_text"] = services.store.window(
                item["document_id"],
                item["position"],
                radius,
            )
        payload["attributes"]["sentence_window"] = window_metrics(
            payload["evidence"], radius, pipeline.top_k
        )
        return payload
