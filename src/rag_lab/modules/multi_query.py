"""G1: reformulações preservam a pergunta original como uma das buscas."""

from rag_lab.modules.base import Services, add_queries


class MultiQuery:
    def run(self, payload: dict, services: Services) -> dict:
        limit = services.settings.pipeline.expanded_queries
        generated = services.providers.chat(
            f"Reformule a pergunta para busca documental. Retorne até {limit} alternativas, "
            "uma por linha, sem responder à pergunta.",
            payload["question"],
        )
        add_queries(payload, generated, limit)
        return payload
