import json
import sys

import pytest
from conftest import ROOT, with_modules

from rag_lab import cli


def test_cli_query_serializes_tags(services, monkeypatch, capsys):
    settings = with_modules(services.settings, "bm25")
    monkeypatch.setattr(cli, "load_settings", lambda path: settings)
    monkeypatch.setattr(cli, "PostgresStore", lambda url: services.store)
    monkeypatch.setattr(cli, "Providers", lambda settings: services.providers)
    monkeypatch.setattr(sys, "argv", ["rag-lab", "query", "RAG Embeddings"])

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    expected = {"a": ["dados-e-inteligencia-artificial"], "c": []}
    for items in (payload["retrieval"]["bm25"], payload["evidence"]):
        assert {item["id"]: item["metadata"]["tags"] for item in items} == expected


def test_cli_ingest_uses_catalog_next_to_main_yaml(services, monkeypatch, tmp_path, capsys):
    config_dir = tmp_path / "custom"
    config_dir.mkdir()
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a.md").write_text("SQL consulta tabelas.", encoding="utf-8")
    (config_dir / "lab.yaml").write_bytes((ROOT / "config/lab.yaml").read_bytes())
    tag_path = config_dir / "tags-si.yaml"
    tag_path.write_text("rotulos: [", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_CONFIG", "custom/lab.yaml")
    monkeypatch.setattr(cli, "PostgresStore", lambda url: services.store)
    monkeypatch.setattr(cli, "Providers", lambda settings: services.providers)
    monkeypatch.setattr(sys, "argv", ["rag-lab", "ingest", "docs"])

    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 1
    captured = capsys.readouterr()
    assert "YAML inválido" in captured.err
    assert tag_path.name in captured.err and str(tag_path.parent) in captured.err
    assert not services.store.replaced

    tag_path.write_bytes((ROOT / "config/tags-si.yaml").read_bytes())
    services.providers.chat_json.return_value = '{"tags": ["bancos-de-dados"]}'
    cli.main()
    assert json.loads(capsys.readouterr().out)["total_chunks"] == 1
    chunks, _, _ = services.store.replaced[0]
    assert chunks[0].metadata["tags"] == ["bancos-de-dados"]
