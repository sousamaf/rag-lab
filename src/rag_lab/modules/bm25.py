"""G5: BM25 com parâmetros (k1, b), variantes de IDF e `min_idf` configuráveis em `modules.bm25`."""

import math

from rank_bm25 import BM25Okapi

from rag_lab.config import BM25Switch
from rag_lab.contracts import Evidence
from rag_lab.modules.base import Services, add_ranking, tokens


def lucene_idf(total: int, frequency: int) -> float:
    """IDF de Lucene: sempre positivo, útil em bases pequenas de aula."""
    return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))


def classic_idf(total: int, frequency: int) -> float:
    """IDF clássico log(N/n): zero para um termo presente em todos os documentos."""
    return math.log(total / frequency)


IDF_FORMULAS = {"lucene": lucene_idf, "classic": classic_idf}


def apply_idf(model: BM25Okapi, variant: str) -> None:
    """`okapi` mantém o IDF da biblioteca (pode ser negativo e é limitado por epsilon)."""
    formula = IDF_FORMULAS.get(variant)
    if formula:
        for term in model.idf:
            frequency = sum(term in document for document in model.doc_freqs)
            model.idf[term] = formula(model.corpus_size, frequency)


def bm25_options(services: Services) -> BM25Switch:
    switch = services.settings.modules["bm25"]
    return switch if isinstance(switch, BM25Switch) else BM25Switch(enabled=switch.enabled)


class BM25:
    def run(self, payload: dict, services: Services) -> dict:
        options = bm25_options(services)
        chunks = services.store.all_chunks()
        corpus = [tokens(c.text) for c in chunks]
        limit = services.settings.pipeline.candidate_k
        rankings = []
        ignored = 0
        if chunks and any(corpus):
            model = BM25Okapi(corpus, k1=options.k1, b=options.b)
            apply_idf(model, options.idf)
            for query in payload["queries"]:
                terms = tokens(query)
                if options.min_idf > 0:
                    kept = [t for t in terms if model.idf.get(t, 0) >= options.min_idf]
                    ignored += len(terms) - len(kept)
                    terms = kept
                scores = model.get_scores(terms)
                rankings.append(
                    [
                        Evidence(**chunks[i].model_dump(), score=float(scores[i]))
                        for i in scores.argsort()[::-1][:limit]
                        if scores[i] > 0
                    ]
                )
        result = add_ranking(payload, "bm25", rankings, limit)
        result["attributes"]["bm25"] = {
            "k1": options.k1,
            "b": options.b,
            "idf": options.idf,
            "min_idf": options.min_idf,
            "termos_ignorados": ignored,
            "chunks": len(chunks),
            "avgdl": round(sum(map(len, corpus)) / len(corpus), 2) if corpus else 0,
            "queries": len(payload["queries"]),
            "hits": len(result["retrieval"]["bm25"]),
        }
        return result
