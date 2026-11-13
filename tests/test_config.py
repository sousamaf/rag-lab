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
