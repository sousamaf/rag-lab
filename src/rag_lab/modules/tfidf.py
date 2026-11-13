"""G4: cosseno entre vetores TF-IDF. Recalcular o índice é aceitável na base didática."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_lab.contracts import Evidence
from rag_lab.modules.base import Services, add_ranking, tokens


class TFIDF:
    def run(self, payload: dict, services: Services) -> dict:
        chunks = services.store.all_chunks()
        limit = services.settings.pipeline.candidate_k
        rankings = []
        if chunks and any(tokens(c.text) for c in chunks):
            vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None)
            matrix = vectorizer.fit_transform(c.text for c in chunks)
            for query in payload["queries"]:
                scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
                rankings.append(
                    [
                        Evidence(**chunks[i].model_dump(), score=float(scores[i]))
                        for i in scores.argsort()[::-1][:limit]
                        if scores[i] > 0
                    ]
                )
        return add_ranking(payload, "tfidf", rankings, limit)
