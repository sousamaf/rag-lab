"""G6: representação semântica das consultas e busca exata por cosseno no pgvector."""

from rag_lab.modules.base import Services, add_ranking


class Embedding:
    def run(self, payload: dict, services: Services) -> dict:
        queries = list(payload["queries"])
        if hypothetical := payload["attributes"].get("hyde"):
            queries.append(hypothetical)
        vectors = services.providers.embed(queries)
        limit = services.settings.pipeline.candidate_k
        rankings = [
            services.store.search_vector(v, services.providers.embedding_key, limit)
            for v in vectors
        ]
        return add_ranking(payload, "embedding", rankings, limit)
