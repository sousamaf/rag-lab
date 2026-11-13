"""G9: expande cada sentença recuperada com suas vizinhas do MESMO documento."""

from rag_lab.modules.base import Services


class SentenceWindow:
    def run(self, payload: dict, services: Services) -> dict:
        radius = services.settings.pipeline.sentence_radius
        for item in payload["evidence"]:
            item["context_text"] = services.store.window(
                item["document_id"],
                item["position"],
                radius,
            )
        return payload
