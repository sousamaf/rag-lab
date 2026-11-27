import pytest
import yaml
from pydantic import ValidationError

from rag_lab.config import MODULE_ORDER, Settings, load_settings


def test_config_reload_and_environment(config_data, tmp_path, monkeypatch):
    path = tmp_path / "lab.yaml"
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:password@host/db")
    path.write_text(yaml.safe_dump(config_data))
    assert load_settings(path).modules["embedding"].enabled
    assert load_settings(path).database_url == "postgresql://rag:password@host/db"
    config_data["modules"]["embedding"]["enabled"] = False
    path.write_text(yaml.safe_dump(config_data))
    assert not load_settings(path).modules["embedding"].enabled


@pytest.mark.parametrize("legacy", [False, True])
def test_classifier_is_selected_by_environment(config_data, tmp_path, monkeypatch, legacy):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_TAG_CLASSIFIER", raising=False)
    if legacy:
        config_data.pop("tagging_method")
    path = tmp_path / "lab.yaml"
    path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    assert load_settings(path).tagging_method == "llm"
    monkeypatch.setenv("RAG_TAG_CLASSIFIER", "terms")
    assert load_settings(path).tagging_method == "terms"
    monkeypatch.setenv("RAG_TAG_CLASSIFIER", "invalid")
    with pytest.raises(ValidationError, match="tagging_method"):
        load_settings(path)


@pytest.mark.parametrize(
    "change", ["unknown", "missing", "hybrid", "hyde", "string_bool", "limits"]
)
def test_invalid_config_rejected(config_data, change):
    if change == "unknown":
        config_data["modules"]["typo"] = {"enabled": True}
    elif change == "missing":
        del config_data["modules"]["sentence"]
    elif change == "hybrid":
        config_data["modules"]["hybrid"]["enabled"] = True
    elif change == "hyde":
        config_data["modules"]["hyde"]["enabled"] = True
        config_data["modules"]["embedding"]["enabled"] = False
    elif change == "string_bool":
        config_data["modules"]["tfidf"]["enabled"] = "false"
    else:
        config_data["pipeline"]["candidate_k"] = 1
    with pytest.raises(ValidationError):
        Settings.model_validate(config_data)


def test_yaml_declares_every_mission(settings):
    assert set(settings.modules) == set(MODULE_ORDER)


def test_hybrid_config_defaults_and_valid_strategies(config_data):
    config_data["tagging_method"] = "llm"
    default_settings = Settings.model_validate(config_data)
    assert default_settings.hybrid.strategy == "rrf"
    assert default_settings.hybrid.weights == {}

    config_data["modules"]["bm25"]["enabled"] = True
    config_data["modules"]["embedding"]["enabled"] = True
    config_data["modules"]["hybrid"]["enabled"] = True

    config_data["hybrid"] = {"strategy": "weighted_rrf", "weights": {"bm25": 1, "embedding": 2.5}}
    weighted = Settings.model_validate(config_data)
    assert weighted.hybrid.strategy == "weighted_rrf"
    assert weighted.hybrid.weights == {"bm25": 1.0, "embedding": 2.5}
    assert all(isinstance(v, float) for v in weighted.hybrid.weights.values())

    for strategy in ("convex", "dbsf"):
        config_data["hybrid"] = {"strategy": strategy}
        parsed = Settings.model_validate(config_data)
        assert parsed.hybrid.strategy == strategy
        assert parsed.hybrid.weights == {}

        config_data["hybrid"] = {"strategy": strategy, "weights": {"bm25": 0, "embedding": 1.0}}
        parsed_weighted = Settings.model_validate(config_data)
        assert parsed_weighted.hybrid.weights == {"bm25": 0.0, "embedding": 1.0}


@pytest.mark.parametrize(
    "hybrid_cfg",
    [
        {"strategy": "rrf", "unknown_option": True},
        {"strategy": "combmnz"},
        {"strategy": "weighted_rrf", "weights": {"bm25": 1.0, "unknown": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": -0.1, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": 100.1, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": float("nan"), "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": float("inf"), "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": True, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": "1.0", "embedding": 1.0}},
        {"strategy": "rrf", "weights": {"bm25": 1.0, "embedding": 1.0}},
        {"strategy": "weighted_rrf"},
        {"strategy": "weighted_rrf", "weights": {"tfidf": 1.0, "bm25": 1.0, "embedding": 1.0}},
        {"strategy": "weighted_rrf", "weights": {"bm25": 1.0}},
        {"strategy": "convex", "weights": {"bm25": 0, "embedding": 0.0}},
    ],
)
def test_invalid_hybrid_config_rejected(config_data, hybrid_cfg):
    config_data["tagging_method"] = "llm"
    config_data["modules"]["bm25"]["enabled"] = True
    config_data["modules"]["embedding"]["enabled"] = True
    config_data["modules"]["hybrid"]["enabled"] = True
    config_data["hybrid"] = hybrid_cfg
    with pytest.raises(ValidationError):
        Settings.model_validate(config_data)


def test_bm25_options_have_defaults_and_are_validated(config_data):
    config_data["tagging_method"] = "llm"
    config_data["database_url"] = "postgresql://rag:rag@localhost/rag"
    config_data["modules"]["bm25"] = {"enabled": True}
    bm25 = Settings.model_validate(config_data).modules["bm25"]
    assert (bm25.k1, bm25.b, bm25.idf, bm25.min_idf) == (1.5, 0.75, "lucene", 0)
    config_data["modules"]["bm25"] = {"enabled": True, "idf": "classic"}
    assert Settings.model_validate(config_data).modules["bm25"].idf == "classic"
    for invalid in ({"b": 1.5}, {"k1": -1}, {"idf": "x"}, {"min_idf": -1}, {"typo": 1}):
        config_data["modules"]["bm25"] = {"enabled": True, **invalid}
        with pytest.raises(ValidationError):
            Settings.model_validate(config_data)


def test_bm25_options_survive_model_dump_roundtrip(config_data):
    config_data["tagging_method"] = "llm"
    config_data["database_url"] = "postgresql://rag:rag@localhost/rag"
    config_data["modules"]["bm25"] = {
        "enabled": True,
        "k1": 2.0,
        "b": 0.1,
        "idf": "okapi",
        "min_idf": 1.2,
    }
    settings = Settings.model_validate(config_data)
    dumped = settings.model_dump()
    assert dumped["modules"]["bm25"] == {
        "enabled": True,
        "k1": 2.0,
        "b": 0.1,
        "idf": "okapi",
        "min_idf": 1.2,
    }
    assert dumped["modules"]["tfidf"] == {"enabled": False}
    again = Settings.model_validate(dumped).modules["bm25"]
    assert (again.k1, again.b, again.idf, again.min_idf) == (2.0, 0.1, "okapi", 1.2)
