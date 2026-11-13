"""G8: baseline lexical de segunda passagem. Missão: comparar com um cross-encoder."""

from rag_lab.modules.base import Services, tokens


class Reranking:
    def run(self, payload: dict, services: Services) -> dict:
        query = set(tokens(payload["question"]))
        for item in payload["evidence"]:
            overlap = len(query & set(tokens(item["text"]))) / max(len(query), 1)
            item["scores"]["reranking"] = overlap
        payload["evidence"].sort(key=lambda e: (-e["scores"]["reranking"], e["id"]))
        return payload
