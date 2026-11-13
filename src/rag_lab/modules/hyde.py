"""G2: o documento hipotético orienta SOMENTE a busca vetorial, nunca vira uma fonte."""

from rag_lab.modules.base import Services


class HyDE:
    def run(self, payload: dict, services: Services) -> dict:
        payload["attributes"]["hyde"] = services.providers.chat(
            "Escreva um parágrafo hipotético que responderia à pergunta em um documento. "
            "Será usado apenas como representação para busca, não como evidência.",
            payload["question"],
        )[:4000]
        return payload
