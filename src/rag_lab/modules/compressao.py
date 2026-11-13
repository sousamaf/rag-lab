"""G10: compressão extrativa. A seleção mantém trechos literais e a fonte original."""

import re

from rag_lab.modules.base import Services, tokens


class Compressao:
    def run(self, payload: dict, services: Services) -> dict:
        query = set(tokens(payload["question"]))
        for item in payload["evidence"]:
            text = item.get("context_text") or item["text"]
            sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
            relevant = [s for s in sentences if query & set(tokens(s))]
            item["context_text"] = " ".join(relevant) if relevant else item["text"]
        return payload
