import json

import pytest
import yaml
from conftest import ROOT

from rag_lab.tagging import (
    TagCatalogError,
    TagClassificationError,
    TagClassifier,
    TermTagClassifier,
    load_tag_catalog,
)

VALID_LABEL = {"id": "tema", "nome": "Tema", "descricao": "Descrição", "termos": ["termo"]}


@pytest.mark.parametrize(
    ("text", "tags"),
    [
        (
            "SQL e POSTGRESQL usam AUTENTICACAO MULTIFATOR.",
            ["bancos-de-dados", "seguranca-e-privacidade"],
        ),
        ("Uma conexão TCP/IP.", ["infraestrutura-e-redes"]),
        ("O fragmento foi revisado.", []),
    ],
)
def test_terms_matches_complete_normalized_words(monkeypatch, text, tags):
    monkeypatch.chdir(ROOT)
    assert TermTagClassifier(load_tag_catalog()).classify(text) == tags


def test_classifies_using_catalog_descriptions_and_model_output(monkeypatch, services):
    monkeypatch.chdir(ROOT)
    classifier = TagClassifier(load_tag_catalog(), services.providers)
    text = "Cada pedido guarda uma referência para o identificador do comprador."
    services.providers.chat_json.return_value = (
        '{"tags": ["seguranca-e-privacidade", "bancos-de-dados", "bancos-de-dados"]}'
    )
    assert classifier.classify(text) == ["bancos-de-dados", "seguranca-e-privacidade"]
    args, kwargs = services.providers.chat_json.call_args
    system, user = args
    assert "persistência e integridade de dados" in system
    assert "chave estrangeira" in system
    assert json.loads(user) == {"trecho": text}
    assert "bancos-de-dados" in kwargs["schema"]["properties"]["tags"]["items"]["enum"]

    services.providers.chat_json.return_value = '{"tags": []}'
    assert classifier.classify("O filme se chama 'SQL'.") == []


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ('{"tags": ["inventado"]}', "fora do catálogo"),
        ('{"tags": ["bancos-de-dados", "inventado"]}', "fora do catálogo"),
        ('{"tags": "bancos-de-dados"}', "deve retornar JSON"),
        ('{"tags": [123]}', "deve retornar JSON"),
        ('{"tags": [], "explicacao": "..."}', "deve retornar JSON"),
        ("sem categoria", "deve retornar JSON"),
    ],
)
def test_invalid_classifier_response_fails(monkeypatch, services, reply, reason):
    monkeypatch.chdir(ROOT)
    services.providers.chat_json.return_value = reply
    with pytest.raises(TagClassificationError, match=reason):
        TagClassifier(load_tag_catalog(), services.providers).classify("Trecho.")


def test_finds_adjacent_catalog_outside_the_lab(tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    adjacent_catalog = tmp_path / "tags-si.yaml"
    adjacent_catalog.write_bytes((ROOT / "config/tags-si.yaml").read_bytes())
    catalog = load_tag_catalog(tmp_path / "lab.yaml")
    assert any(label.id == "bancos-de-dados" for label in catalog.rotulos)


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (None, "Erro ao ler"),
        (b"\xff", "UTF-8"),
        (b"rotulos: [", "YAML inválido"),
        ([VALID_LABEL, VALID_LABEL], "devem ser únicos"),
        ([dict(VALID_LABEL, id=f"tema-{i}") for i in range(11)], "at most 10"),
        ([dict(VALID_LABEL, descricao=" ")], r"rotulos\.0\.descricao"),
        ([dict(VALID_LABEL, id="Tema Inválido")], r"rotulos\.0\.id"),
        ([dict(VALID_LABEL, termos=["!!!"])], "conter letras ou números"),
    ],
)
def test_invalid_catalog_reports_path_and_reason(tmp_path, monkeypatch, data, reason):
    monkeypatch.chdir(tmp_path)
    catalog_path = tmp_path / "tags-si.yaml"
    if isinstance(data, bytes):
        catalog_path.write_bytes(data)
    elif data is not None:
        catalog_path.write_text(yaml.safe_dump({"rotulos": data}), encoding="utf-8")

    with pytest.raises(TagCatalogError, match=reason) as failure:
        load_tag_catalog(tmp_path / "lab.yaml")
    assert str(catalog_path) in str(failure.value)
