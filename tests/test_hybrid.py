"""Testes de caracterização e unidade para fusão híbrida (G7) e seleção round-robin."""

from copy import deepcopy

import pytest

from rag_lab.contracts import Chunk, Envelope, Evidence
from rag_lab.modules.hybrid import Hybrid
from rag_lab.pipeline import select_without_fusion


def _make_evidence(chunk: Chunk, retriever: str, score: float) -> dict:
    return Evidence(**chunk.model_dump(), score=score, scores={retriever: score}).model_dump()


def test_rrf_classic_reference_behavior_and_exact_scores_keys(services):
    chunks = services.store.all_chunks()
    a = _make_evidence(chunks[0], "bm25", 1000.0)
    b_bm25 = _make_evidence(chunks[1], "bm25", 1.0)
    b_emb = _make_evidence(chunks[1], "embedding", 0.9)
    payload = Envelope(question="RAG", attributes={"grupo": "G7"}).model_dump()
    payload["retrieval"] = {"bm25": [a, b_bm25], "embedding": [b_emb]}

    result = Hybrid().run(payload, services)

    assert [item["id"] for item in result["evidence"]] == ["b", "a"]
    assert result["evidence"][0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert result["evidence"][0]["scores"] == {
        "bm25": 1.0,
        "embedding": 0.9,
        "hybrid": pytest.approx(1 / 62 + 1 / 61),
    }
    assert result["evidence"][1]["scores"] == {
        "bm25": 1000.0,
        "hybrid": pytest.approx(1 / 61),
    }
    assert result["attributes"]["grupo"] == "G7"


@pytest.mark.parametrize(
    ("rrf_k", "expected_first"),
    [
        (1, "a"),   # 1/(1+1) = 0.5 > 2/(1+4) = 0.4 -> 1º isolado vence consenso mediano
        (60, "b"),  # 1/(60+1) < 2/(60+4) -> consenso vence 1º isolado
    ],
)
def test_rrf_k_parameter_shifts_consensus_vs_single_top_winner(services, rrf_k, expected_first):
    services.settings.pipeline.rrf_k = rrf_k
    chunks = {
        cid: Chunk(id=cid, document_id="d1", source="a.txt", position=i, text=f"Texto {cid}")
        for i, cid in enumerate(["a", "b", "c1", "c2", "e1", "e2", "e3"])
    }
    # "a" é 1º apenas em bm25; "b" é 4º em bm25 e 4º em embedding; demais fillers são disjuntos.
    bm25_list = [
        _make_evidence(chunks["a"], "bm25", 10.0),
        _make_evidence(chunks["c1"], "bm25", 9.0),
        _make_evidence(chunks["c2"], "bm25", 8.0),
        _make_evidence(chunks["b"], "bm25", 7.0),
    ]
    emb_list = [
        _make_evidence(chunks["e1"], "embedding", 0.9),
        _make_evidence(chunks["e2"], "embedding", 0.8),
        _make_evidence(chunks["e3"], "embedding", 0.7),
        _make_evidence(chunks["b"], "embedding", 0.6),
    ]
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {"bm25": bm25_list, "embedding": emb_list}

    result = Hybrid().run(payload, services)
    scores_by_id = {item["id"]: item["score"] for item in result["evidence"]}

    assert scores_by_id["a"] == pytest.approx(1 / (rrf_k + 1))
    assert scores_by_id["b"] == pytest.approx(2 / (rrf_k + 4))
    assert result["evidence"][0]["id"] == expected_first


@pytest.mark.parametrize(
    ("rrf_k", "expected_winner"),
    [
        (9, "a_top1"),   # 1/10 = 0.1000 > 2/21 = 0.0952
        (10, "a_top1"),  # 1/11 == 2/22 -> empate exato, desempate por id ("a_top1" < "b_rank12")
        (11, "b_rank12"),  # 2/23 = 0.0870 > 1/12 = 0.0833 (dominância m=2 sobre m=1)
    ],
)
def test_rrf_dominance_threshold_at_candidate_k_12(services, rrf_k, expected_winner):
    services.settings.pipeline.rrf_k = rrf_k
    top1_id = "a_top1"
    rank12_id = "b_rank12"

    def make_chunk(cid: str, pos: int) -> Chunk:
        return Chunk(id=cid, document_id="d1", source="a.txt", position=pos, text=cid)

    bm25_list = [_make_evidence(make_chunk(top1_id, 0), "bm25", 50.0)]
    for i in range(1, 11):
        bm25_list.append(_make_evidence(make_chunk(f"z_bm25_fill_{i:02d}", i), "bm25", 50.0 - i))
    bm25_list.append(_make_evidence(make_chunk(rank12_id, 99), "bm25", 1.0))

    emb_list = []
    for i in range(1, 12):
        emb_list.append(
            _make_evidence(
                make_chunk(f"z_emb_fill_{i:02d}", 100 + i), "embedding", 1.0 - i * 0.05
            )
        )
    emb_list.append(_make_evidence(make_chunk(rank12_id, 99), "embedding", 0.1))

    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {"bm25": bm25_list, "embedding": emb_list}

    result = Hybrid().run(payload, services)
    assert result["evidence"][0]["id"] == expected_winner


def test_hybrid_does_not_truncate_to_top_k_and_does_not_mutate_retrieval(services):
    services.settings.pipeline.top_k = 2
    chunks = [
        Chunk(id=f"c{i}", document_id="d1", source="a.txt", position=i, text=f"Sentença {i}.")
        for i in range(6)
    ]
    retrieval = {
        "bm25": [_make_evidence(c, "bm25", 10.0 - i) for i, c in enumerate(chunks[:4])],
        "embedding": [
            _make_evidence(c, "embedding", 0.9 - i * 0.1) for i, c in enumerate(chunks[2:])
        ],
    }
    snapshot = deepcopy(retrieval)
    payload = Envelope(question="Pergunta original", attributes={"student": "G7"}).model_dump()
    payload["retrieval"] = retrieval

    result = Hybrid().run(payload, services)

    assert len(result["evidence"]) == 6
    assert payload["retrieval"] == snapshot
    assert result["question"] == "Pergunta original"
    assert result["attributes"]["student"] == "G7"
    for item in result["evidence"]:
        assert item["text"].startswith("Sentença ")


def test_hybrid_handles_empty_retriever_lists_and_ties_by_id(services):
    chunks = services.store.all_chunks()
    payload_empty = Envelope(question="RAG").model_dump()
    payload_empty["retrieval"] = {"bm25": [], "embedding": []}
    assert Hybrid().run(payload_empty, services)["evidence"] == []

    # Empate simétrico: "b" é 1º em bm25 e "a" é 1º em embedding -> desempate por id ("a" < "b").
    payload_tie = Envelope(question="RAG").model_dump()
    payload_tie["retrieval"] = {
        "bm25": [_make_evidence(chunks[1], "bm25", 10.0)],
        "embedding": [_make_evidence(chunks[0], "embedding", 0.95)],
        "tfidf": [],
    }
    result = Hybrid().run(payload_tie, services)
    assert [item["id"] for item in result["evidence"]] == ["a", "b"]
    assert result["evidence"][0]["score"] == result["evidence"][1]["score"]


def test_select_without_fusion_round_robin_order_bias_and_score_preservation(services):
    chunks = services.store.all_chunks()
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "tfidf": [_make_evidence(chunks[2], "tfidf", 0.4)],
        "bm25": [
            _make_evidence(chunks[1], "bm25", 12.5),
            _make_evidence(chunks[0], "bm25", 8.0),
        ],
        "embedding": [
            _make_evidence(chunks[0], "embedding", 0.92),
            _make_evidence(chunks[2], "embedding", 0.88),
        ],
    }

    select_without_fusion(payload)

    # Posição 0 intercala na ordem de inserção: tfidf(c) -> bm25(b) -> embedding(a).
    assert [item["id"] for item in payload["evidence"]] == ["c", "b", "a"]
    by_id = {item["id"]: item for item in payload["evidence"]}
    # O score escalar principal permanece o do primeiro recuperador que inseriu o trecho.
    assert by_id["c"]["score"] == pytest.approx(0.4)
    assert by_id["c"]["scores"] == {"tfidf": 0.4, "embedding": 0.88}
    assert by_id["a"]["score"] == pytest.approx(0.92)
    assert by_id["a"]["scores"] == {"embedding": 0.92, "bm25": 8.0}
    assert "hybrid" not in by_id["c"]["scores"]


def test_telemetry_schema_boundary_tie_and_overwrites_injected_attribute(services):
    services.settings.pipeline.top_k = 1
    chunks = services.store.all_chunks()
    # Empate no topo (ambos em 1º lugar de uma lista) com top_k=1 -> boundary_tie deve ser True.
    payload = Envelope(
        question="RAG",
        attributes={"grupo": "G7", "hybrid": {"injected": "fake"}},
    ).model_dump()
    payload["retrieval"] = {
        "bm25": [_make_evidence(chunks[1], "bm25", 10.0)],
        "embedding": [_make_evidence(chunks[0], "embedding", 0.9)],
        "tfidf": [],
    }

    result = Hybrid().run(payload, services)
    telemetry = result["attributes"]["hybrid"]

    assert "injected" not in telemetry
    assert result["attributes"]["grupo"] == "G7"
    assert telemetry["schema"] == "g7-hybrid/1"
    assert telemetry["strategy"] == "rrf"
    assert telemetry["params"] == {
        "rrf_k": 60,
        "weights": {"bm25": 1.0, "embedding": 1.0, "tfidf": 1.0},
        "weight_sum": 3.0,
        "top_k": 1,
    }
    assert telemetry["retrievers"]["tfidf"] == {
        "candidates": 0,
        "weight": 1.0,
        "raw_min": None,
        "raw_max": None,
        "raw_mean": None,
        "raw_std": None,
    }
    assert telemetry["retrievers"]["bm25"]["candidates"] == 1
    assert telemetry["overlap"] == {
        "fused_count": 2,
        "in_all": 0,
        "exclusive": {"bm25": 1, "embedding": 1, "tfidf": 0},
    }
    assert telemetry["boundary_tie"] is True
    assert [entry["id"] for entry in telemetry["ranking"]] == ["a", "b"]
    assert [entry["fused_rank"] for entry in telemetry["ranking"]] == [1, 2]


def test_fsum_prevents_floating_point_order_bias_on_three_way_ties(services):
    import math

    # Ranks (1, 7, 2) vs (2, 7, 1) com rrf_k=60 diferem em 1 ULP na adição float padrão.
    assert ((1 / 61 + 1 / 67) + 1 / 62) < ((1 / 62 + 1 / 67) + 1 / 61)

    def make_list(retriever: str, target_ranks: dict[int, str]) -> list[dict]:
        items = []
        for rank in range(1, 8):
            cid = target_ranks.get(rank, f"z_{retriever}_{rank}")
            chunk = Chunk(id=cid, document_id="d1", source="a.txt", position=rank, text=cid)
            items.append(_make_evidence(chunk, retriever, float(20 - rank)))
        return items

    # "a" acumula nas posições (1, 7, 2) e "b" nas posições (2, 7, 1); na soma comum "b" venceria.
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "r1": make_list("r1", {1: "a", 2: "b"}),
        "r2": make_list("r2", {7: "a"}),
        "r3": make_list("r3", {7: "b"}),
        "r4": make_list("r4", {2: "a", 1: "b"}),
    }

    result = Hybrid().run(payload, services)
    scores = {item["id"]: item["score"] for item in result["evidence"]}
    assert scores["a"] == scores["b"] == math.fsum([1 / 61, 1 / 67, 1 / 62])
    assert [item["id"] for item in result["evidence"][:2]] == ["a", "b"]


@pytest.mark.parametrize(
    "bad_score", [float("nan"), float("inf"), float("-inf"), True, "1.0", None]
)
def test_hybrid_rejects_non_finite_or_invalid_raw_scores(services, bad_score):
    chunk = services.store.all_chunks()[0]
    item = chunk.model_dump()
    item["score"] = bad_score
    item["scores"] = {"bm25": bad_score}
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {"bm25": [item], "embedding": []}

    with pytest.raises(ValueError, match="Score bruto inválido"):
        Hybrid().run(payload, services)


def test_weighted_rrf_unit_weights_match_classic_rrf_and_weights_shift_winner(services):
    # Candidato "lex" está em 11º no tfidf e 12º no bm25; "sem" está em 1º só no embedding.
    lex = Chunk(id="lex", document_id="d1", source="a.txt", position=0, text="lexical")
    sem = Chunk(id="sem", document_id="d2", source="b.txt", position=0, text="semantico")

    tfidf_list = [
        _make_evidence(
            Chunk(id=f"z_t_{i:02d}", document_id="d1", source="a.txt", position=i, text="z"),
            "tfidf",
            1.0 - i * 0.05,
        )
        for i in range(1, 11)
    ] + [_make_evidence(lex, "tfidf", 0.1)]

    bm25_list = [
        _make_evidence(
            Chunk(id=f"z_b_{i:02d}", document_id="d1", source="a.txt", position=i, text="z"),
            "bm25",
            20.0 - i,
        )
        for i in range(1, 12)
    ] + [_make_evidence(lex, "bm25", 1.0)]

    emb_list = [_make_evidence(sem, "embedding", 0.95)]

    base_payload = Envelope(question="RAG").model_dump()
    base_payload["retrieval"] = {"tfidf": tfidf_list, "bm25": bm25_list, "embedding": emb_list}

    # 1) RRF clássico: viés 2:1 faz "lex" (11º e 12º) vencer "sem" (1º isolado).
    rrf_result = Hybrid().run(deepcopy(base_payload), services)
    assert rrf_result["evidence"][0]["id"] == "lex"

    # 2) weighted_rrf com pesos unitários equivale exatamente ao RRF clássico.
    services.settings.hybrid.strategy = "weighted_rrf"
    services.settings.hybrid.weights = {"tfidf": 1.0, "bm25": 1.0, "embedding": 1.0}
    unit_result = Hybrid().run(deepcopy(base_payload), services)
    assert [e["id"] for e in unit_result["evidence"]] == [e["id"] for e in rrf_result["evidence"]]
    assert unit_result["evidence"][0]["score"] == pytest.approx(rrf_result["evidence"][0]["score"])

    # 3) weighted_rrf com peso 2.0 no embedding corrige o viés 2:1: 2/61 > 1/71 + 1/72.
    services.settings.hybrid.weights = {"tfidf": 1.0, "bm25": 1.0, "embedding": 2.0}
    weighted_result = Hybrid().run(deepcopy(base_payload), services)
    assert weighted_result["evidence"][0]["id"] == "sem"
    assert weighted_result["evidence"][0]["score"] == pytest.approx(2.0 / 61.0)

    # 4) Peso 0.0 mantém candidato exclusivo sem truncar, mas com contribuição 0.0 no fim.
    services.settings.hybrid.weights = {"tfidf": 1.0, "bm25": 1.0, "embedding": 0.0}
    zero_result = Hybrid().run(deepcopy(base_payload), services)
    by_id = {e["id"]: e for e in zero_result["evidence"]}
    assert "sem" in by_id
    assert by_id["sem"]["score"] == 0.0


@pytest.mark.parametrize(
    ("strategy", "weights"),
    [
        ("rrf", {"bm25": 1.0, "embedding": 1.0}),
        ("weighted_rrf", {}),
        ("weighted_rrf", {"bm25": 1.0}),
        ("weighted_rrf", {"bm25": 1.0, "embedding": 1.0, "tfidf": 1.0}),
        ("convex", {"bm25": 0.0, "embedding": 0.0}),
    ],
)
def test_hybrid_runtime_weight_validation_errors(services, strategy, weights):
    chunks = services.store.all_chunks()
    services.settings.hybrid.strategy = strategy
    services.settings.hybrid.weights = weights
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "bm25": [_make_evidence(chunks[0], "bm25", 5.0)],
        "embedding": [_make_evidence(chunks[1], "embedding", 0.8)],
    }

    with pytest.raises(ValueError):
        Hybrid().run(payload, services)


def test_convex_scale_invariance_and_outlier_sensitivity(services):
    services.settings.hybrid.strategy = "convex"
    chunks = [
        Chunk(id=cid, document_id="d1", source="a.txt", position=i, text=cid)
        for i, cid in enumerate(["a", "b", "c"])
    ]
    payload_1 = Envelope(question="RAG").model_dump()
    payload_1["retrieval"] = {
        "bm25": [
            _make_evidence(chunks[0], "bm25", 10.0),
            _make_evidence(chunks[1], "bm25", 6.0),
            _make_evidence(chunks[2], "bm25", 2.0),
        ],
        "embedding": [
            _make_evidence(chunks[1], "embedding", 0.90),
            _make_evidence(chunks[2], "embedding", 0.80),
            _make_evidence(chunks[0], "embedding", 0.70),
        ],
    }
    payload_scaled = deepcopy(payload_1)
    for item in payload_scaled["retrieval"]["bm25"]:
        item["score"] *= 1000.0
        item["scores"]["bm25"] *= 1000.0

    res_1 = Hybrid().run(payload_1, services)
    res_scaled = Hybrid().run(payload_scaled, services)

    assert [e["id"] for e in res_1["evidence"]] == [e["id"] for e in res_scaled["evidence"]]
    for e1, e2 in zip(res_1["evidence"], res_scaled["evidence"], strict=True):
        assert e1["score"] == pytest.approx(e2["score"])

    # Em payload_1: "b" tem 0.5 no bm25 e 1.0 no embedding -> (0.5+1.0)/2 = 0.75 (vence "a"=0.5).
    assert res_1["evidence"][0]["id"] == "b"
    assert res_1["evidence"][0]["score"] == pytest.approx(0.75)

    # Se "a" tiver outlier extremo no bm25 (1002.0 vs 6.0 e 2.0), "b" cai para 0.004 no bm25.
    payload_outlier = deepcopy(payload_1)
    payload_outlier["retrieval"]["bm25"][0]["score"] = 1002.0
    payload_outlier["retrieval"]["bm25"][0]["scores"]["bm25"] = 1002.0
    res_outlier = Hybrid().run(payload_outlier, services)
    scores_outlier = {e["id"]: e["score"] for e in res_outlier["evidence"]}
    assert scores_outlier["b"] == pytest.approx((0.004 + 1.0) / 2.0)


def test_convex_degenerate_lists_and_empty_retriever_denominator(services):
    import math

    services.settings.hybrid.strategy = "convex"
    chunks = services.store.all_chunks()
    # Três recuperadores em retrieval: tfidf vazio, bm25 com 1 item, embedding com scores iguais.
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "tfidf": [],
        "bm25": [_make_evidence(chunks[0], "bm25", 5.0)],
        "embedding": [
            _make_evidence(chunks[0], "embedding", 0.8),
            _make_evidence(chunks[1], "embedding", 0.8),
        ],
    }

    result = Hybrid().run(payload, services)
    by_id = {e["id"]: e for e in result["evidence"]}

    # Peso total W = 3.0 (inclui tfidf vazio). "a" aparece em bm25 (1/3) e embedding (1/3) -> 2/3.
    assert by_id["a"]["score"] == pytest.approx(2.0 / 3.0)
    assert by_id["b"]["score"] == pytest.approx(1.0 / 3.0)
    for entry in result["attributes"]["hybrid"]["ranking"]:
        assert math.fsum(entry["contributions"].values()) == entry["fused"]


def test_dbsf_zero_std_outlier_clipping_and_presence_bonus(services):
    import math

    services.settings.hybrid.strategy = "dbsf"
    chunks = [
        Chunk(id=f"c{i:02d}", document_id="d1", source="a.txt", position=i, text=f"Texto {i}")
        for i in range(12)
    ]
    # Em n=12, [100.0] + [1.0]*11 tem z = sqrt(11) ≈ 3.3166 > 3.0 para o 1º item -> clip em 1.0.
    bm25_scores = [100.0] + [1.0] * 11
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "bm25": [_make_evidence(chunks[i], "bm25", bm25_scores[i]) for i in range(12)],
        "embedding": [_make_evidence(chunks[0], "embedding", 0.75)],  # std == 0.0 -> norm = 1.0
    }

    result = Hybrid().run(payload, services)
    by_id = {e["id"]: e for e in result["evidence"]}

    # c00 recebe 0.5 (clip em 1.0 no bm25) + 0.5 (std=0 no embedding) = 1.0
    assert by_id["c00"]["score"] == pytest.approx(1.0)
    # Os demais itens (score=1.0 no bm25) têm z = -1/sqrt(11) -> norm = 0.5 + z/6 > 0 (presença)
    expected_tail = 0.5 * (0.5 - (1.0 / math.sqrt(11.0)) / 6.0)
    assert by_id["c11"]["score"] == pytest.approx(expected_tail)
    assert by_id["c11"]["score"] > 0.0


@pytest.mark.parametrize(
    ("strategy", "weights"),
    [
        ("rrf", {}),
        ("weighted_rrf", {"tfidf": 0.5, "bm25": 1.0, "embedding": 2.0}),
        ("convex", {}),
        ("convex", {"tfidf": 1.0, "bm25": 1.0, "embedding": 2.0}),
        ("dbsf", {}),
        ("dbsf", {"tfidf": 1.0, "bm25": 1.0, "embedding": 2.0}),
    ],
)
def test_all_strategies_roundtrip_strict_json_and_envelope_contract(services, strategy, weights):
    import json
    import math

    services.settings.hybrid.strategy = strategy
    services.settings.hybrid.weights = weights
    chunks = services.store.all_chunks()
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "tfidf": [_make_evidence(chunks[0], "tfidf", 0.6)],
        "bm25": [
            _make_evidence(chunks[1], "bm25", 8.0),
            _make_evidence(chunks[0], "bm25", 4.0),
        ],
        "embedding": [
            _make_evidence(chunks[2], "embedding", 0.9),
            _make_evidence(chunks[1], "embedding", 0.7),
        ],
    }

    result = Hybrid().run(payload, services)
    serialized = json.dumps(result, allow_nan=False)
    validated = Envelope.model_validate(json.loads(serialized))

    assert len(validated.evidence) == 3
    for item in validated.evidence:
        assert "hybrid" in item.scores
        assert set(item.scores).issubset({"tfidf", "bm25", "embedding", "hybrid"})

    telemetry = validated.attributes["hybrid"]
    assert telemetry["strategy"] == strategy
    for entry in telemetry["ranking"]:
        assert math.fsum(entry["contributions"].values()) == entry["fused"]


def test_rrf_tie_breaking_is_strictly_scale_invariant_by_chunk_id(services):
    def make_chunk(cid: str, pos: int) -> Chunk:
        return Chunk(id=cid, document_id="d1", source="a.txt", position=pos, text=cid)

    # Caso 1: chunk do embedding em rank 3 (score bruto 0.68) tem id lexicograficamente menor
    # que chunk do bm25 em rank 3 (score bruto 30.0) -> embedding vence o desempate em 1/63.
    bm25_list = [
        _make_evidence(make_chunk("01_bm25", 1), "bm25", 50.0),
        _make_evidence(make_chunk("02_bm25", 2), "bm25", 40.0),
        _make_evidence(make_chunk("z_bm25_rank3", 3), "bm25", 30.0),
    ]
    emb_list = [
        _make_evidence(make_chunk("01_emb", 10), "embedding", 0.95),
        _make_evidence(make_chunk("02_emb", 11), "embedding", 0.85),
        _make_evidence(make_chunk("a_emb_rank3", 12), "embedding", 0.68),
    ]
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {"bm25": bm25_list, "embedding": emb_list}

    result = Hybrid().run(payload, services)
    rank3_pair = [e for e in result["evidence"] if e["id"] in {"a_emb_rank3", "z_bm25_rank3"}]
    assert [e["id"] for e in rank3_pair] == ["a_emb_rank3", "z_bm25_rank3"]
    assert rank3_pair[0]["score"] == rank3_pair[1]["score"] == pytest.approx(1.0 / 63.0)


@pytest.mark.parametrize(
    ("strategy", "weights"),
    [
        ("rrf", {}),
        ("weighted_rrf", {"bm25": 1.0, "embedding": 2.0}),
        ("convex", {}),
        ("dbsf", {}),
    ],
)
def test_hybrid_preserves_metadata_tags_across_all_strategies(services, strategy, weights):
    services.settings.hybrid.strategy = strategy
    services.settings.hybrid.weights = weights

    c_consensus = Chunk(
        id="c_cons",
        document_id="d1",
        source="01.md",
        position=1,
        text="Consenso",
        metadata={"tags": ["seguranca-e-privacidade", "infraestrutura-e-redes"]},
    )
    c_only_emb = Chunk(
        id="c_emb",
        document_id="d2",
        source="06.md",
        position=5,
        text="Exclusivo embedding",
        metadata={"tags": ["dados-e-inteligencia-artificial"]},
    )
    payload = Envelope(question="RAG").model_dump()
    payload["retrieval"] = {
        "bm25": [_make_evidence(c_consensus, "bm25", 12.0)],
        "embedding": [
            _make_evidence(c_consensus, "embedding", 0.88),
            _make_evidence(c_only_emb, "embedding", 0.77),
        ],
    }

    result = Hybrid().run(payload, services)
    by_id = {item["id"]: item for item in result["evidence"]}
    assert by_id["c_cons"]["metadata"]["tags"] == [
        "seguranca-e-privacidade",
        "infraestrutura-e-redes",
    ]
    assert by_id["c_emb"]["metadata"]["tags"] == ["dados-e-inteligencia-artificial"]


def test_hybrid_config_attribute_fallback_precedence_and_validation(services):
    from types import SimpleNamespace

    chunks = services.store.all_chunks()
    base_retrieval = {
        "bm25": [_make_evidence(chunks[0], "bm25", 10.0)],
        "embedding": [_make_evidence(chunks[1], "embedding", 0.9)],
    }

    # 1) Quando settings.hybrid existe, tem precedência sobre attributes["hybrid_config"].
    services.settings.hybrid.strategy = "rrf"
    services.settings.hybrid.weights = {}
    payload_with_override = Envelope(
        question="RAG",
        attributes={"hybrid_config": {"strategy": "convex"}},
    ).model_dump()
    payload_with_override["retrieval"] = deepcopy(base_retrieval)
    res_precedence = Hybrid().run(payload_with_override, services)
    assert res_precedence["attributes"]["hybrid"]["strategy"] == "rrf"

    # 2) Sem settings.hybrid (contrato upstream/develop), usa attributes["hybrid_config"].
    stub_services = SimpleNamespace(
        settings=SimpleNamespace(pipeline=services.settings.pipeline)
    )
    payload_fallback = Envelope(
        question="RAG",
        attributes={
            "hybrid_config": {
                "strategy": "weighted_rrf",
                "weights": {"bm25": 1, "embedding": 2.0},
            }
        },
    ).model_dump()
    payload_fallback["retrieval"] = deepcopy(base_retrieval)
    res_fallback = Hybrid().run(payload_fallback, stub_services)
    assert res_fallback["attributes"]["hybrid"]["strategy"] == "weighted_rrf"
    assert res_fallback["attributes"]["hybrid"]["params"]["weights"] == {
        "bm25": 1.0,
        "embedding": 2.0,
    }

    # 3) Sem settings.hybrid e sem attributes["hybrid_config"], assume padrão ("rrf", {}).
    payload_default = Envelope(question="RAG").model_dump()
    payload_default["retrieval"] = deepcopy(base_retrieval)
    res_default = Hybrid().run(payload_default, stub_services)
    assert res_default["attributes"]["hybrid"]["strategy"] == "rrf"

    # 4) Validação estrita de tipos e chaves em attributes["hybrid_config"].
    invalid_configs = [
        "not_a_dict",
        {"strategy": "rrf", "extra_key": 1},
        {"strategy": "unknown_strategy"},
        {"strategy": "convex", "weights": [1.0, 2.0]},
        {"strategy": "weighted_rrf", "weights": {"": 1.0, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": True, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": -1.0, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": 100.1, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": float("nan"), "embedding": 1.0}},
    ]
    for bad_cfg in invalid_configs:
        bad_payload = Envelope(question="RAG", attributes={"hybrid_config": bad_cfg}).model_dump()
        bad_payload["retrieval"] = deepcopy(base_retrieval)
        with pytest.raises(ValueError):
            Hybrid().run(bad_payload, stub_services)


def test_boundary_tie_true_and_false_at_top_k_cutoff(services):
    def make_chunk(cid: str, pos: int) -> Chunk:
        return Chunk(id=cid, document_id="d1", source="a.txt", position=pos, text=cid)

    # Reproduz a geometria de S02: 2 itens de consenso no topo (#1 e #2), seguidos de pares
    # exclusivos empatados em rank 1 (#3), rank 2 (#4 e #5) e rank 3 (#6 e #7).
    c_cons1 = make_chunk("01_cons", 1)
    c_cons2 = make_chunk("02_cons", 2)
    bm25_list = [
        _make_evidence(make_chunk("10_bm25_r1", 10), "bm25", 10.2),
        _make_evidence(make_chunk("20_bm25_r2", 20), "bm25", 10.1),
        _make_evidence(make_chunk("30_bm25_r3", 30), "bm25", 9.9),
        _make_evidence(c_cons1, "bm25", 9.0),
        _make_evidence(c_cons2, "bm25", 8.1),
    ]
    emb_list = [
        _make_evidence(c_cons1, "embedding", 0.72),
        _make_evidence(make_chunk("21_emb_r2", 21), "embedding", 0.71),
        _make_evidence(make_chunk("31_emb_r3", 31), "embedding", 0.69),
        _make_evidence(make_chunk("41_emb_r4", 41), "embedding", 0.68),
        _make_evidence(c_cons2, "embedding", 0.67),
    ]
    base_payload = Envelope(question="RAG").model_dump()
    base_payload["retrieval"] = {"bm25": bm25_list, "embedding": emb_list}

    # Com top_k = 5: #4 e #5 empatam em 1/62, e #6 tem 1/63 -> sem empate na fronteira 5|6.
    services.settings.pipeline.top_k = 5
    res_top5 = Hybrid().run(deepcopy(base_payload), services)
    assert res_top5["attributes"]["hybrid"]["boundary_tie"] is False

    # Com top_k = 4: #4 e #5 empatam em 1/62 na fronteira 4|5 -> boundary_tie True.
    services.settings.pipeline.top_k = 4
    res_top4 = Hybrid().run(deepcopy(base_payload), services)
    assert res_top4["attributes"]["hybrid"]["boundary_tie"] is True

    # Com top_k = 6: #6 e #7 empatam em 1/63 na fronteira 6|7 -> boundary_tie True.
    services.settings.pipeline.top_k = 6
    res_top6 = Hybrid().run(deepcopy(base_payload), services)
    assert res_top6["attributes"]["hybrid"]["boundary_tie"] is True


