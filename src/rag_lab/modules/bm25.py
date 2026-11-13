"""G5: BM25 com IDF positivo, inclusive em corpora mínimos de uma aula."""

from rank_bm25 import BM25Okapi

from rag_lab.contracts import Evidence
from rag_lab.modules.base import Services, add_ranking, tokens


class BM25:
    def run(self, payload: dict, services: Services) -> dict:
        chunks = services.store.all_chunks()
        corpus = [tokens(c.text) for c in chunks]
        limit = services.settings.pipeline.candidate_k
        rankings = []
        if chunks and any(corpus):
            model = BM25Okapi(corpus)
            # Variante IDF de Lucene: evita score <= 0 para termos frequentes em bases pequenas.
            import math

            for term in model.idf:
                frequency = sum(term in document for document in model.doc_freqs)
                model.idf[term] = math.log(1 + (len(corpus) - frequency + 0.5) / (frequency + 0.5))
            for query in payload["queries"]:
                scores = model.get_scores(tokens(query))
                rankings.append(
                    [
                        Evidence(**chunks[i].model_dump(), score=float(scores[i]))
                        for i in scores.argsort()[::-1][:limit]
                        if scores[i] > 0
                    ]
                )
        return add_ranking(payload, "bm25", rankings, limit)
