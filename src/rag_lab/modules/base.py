"""Interface de módulo: JSON entra, JSON enriquecido sai; serviços ficam fora do envelope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from rag_lab.config import Settings
from rag_lab.contracts import Evidence


@dataclass
class Services:
    settings: Settings
    store: Any
    providers: Any


class Module(Protocol):
    def run(self, payload: dict, services: Services) -> dict: ...


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def add_queries(payload: dict, generated: str, limit: int) -> None:
    """Mantém a pergunta original e limita a saída não estruturada do LLM."""
    for line in generated.splitlines()[:limit]:
        query = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()[:2000]
        if query and query not in payload["queries"]:
            payload["queries"].append(query)


def add_ranking(payload: dict, name: str, rankings: list[list[Evidence]], limit: int) -> dict:
    """Agrega subconsultas pelo melhor score; não compara escalas de métodos distintos."""
    best = {}
    for ranking in rankings:
        for evidence in ranking:
            if evidence.id not in best or evidence.score > best[evidence.id].score:
                best[evidence.id] = evidence
    selected = sorted(best.values(), key=lambda e: (-e.score, e.id))[:limit]
    for item in selected:
        item.scores[name] = item.score
    payload["retrieval"][name] = [e.model_dump(mode="json") for e in selected]
    return payload
