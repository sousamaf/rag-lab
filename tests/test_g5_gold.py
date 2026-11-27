"""Garante a estrutura do gabarito do G5 (o corpus é local; aqui só o formato é verificado)."""

import json
import re
from pathlib import Path

GOLD = Path(__file__).resolve().parents[1] / "experimentos/g5_gold.json"
PROFILES = {"literal", "semantica", "composta", "sem_resposta"}


def questions():
    return json.loads(GOLD.read_text(encoding="utf-8"))["perguntas"]


def test_gold_has_unique_ids_and_the_four_profiles():
    items = questions()
    assert len({q["id"] for q in items}) == len(items)
    assert {q["perfil"] for q in items} == PROFILES
    for profile in PROFILES:
        assert sum(q["perfil"] == profile for q in items) >= 5


def test_gold_relevant_labels_are_well_formed():
    for q in questions():
        assert q["pergunta"].strip()
        assert all(re.fullmatch(r"[\w\-]+\.(txt|md):\d+", label) for label in q["relevantes"])
        assert (q["perfil"] == "sem_resposta") == (not q["relevantes"])


def test_gold_states_what_a_supported_answer_must_contain():
    for q in questions():
        assert q["deve_conter"] and all(fact.strip() for fact in q["deve_conter"])
