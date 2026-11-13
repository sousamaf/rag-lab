"""G3: subperguntas adicionadas à busca; a geração final ainda responde à pergunta original."""

from rag_lab.modules.base import Services, add_queries


class Decomposicao:
    def run(self, payload: dict, services: Services) -> dict:
        limit = services.settings.pipeline.expanded_queries
        generated = services.providers.chat(
            f"Decomponha a pergunta em até {limit} subperguntas pesquisáveis, "
            "uma por linha. Não responda.",
            payload["question"],
        )
        add_queries(payload, generated, limit)
        return payload
