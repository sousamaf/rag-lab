from contextlib import contextmanager

from conftest import with_modules
from fastapi.testclient import TestClient

from rag_lab.api import create_app


def test_api_startup_modules_query_and_error_handling(services, monkeypatch):
    services.settings = with_modules(services.settings, "bm25")

    class Connection:
        def execute(self, sql):
            pass

    @contextmanager
    def connect():
        yield Connection()

    monkeypatch.setattr("rag_lab.api.load_settings", lambda path: services.settings)
    monkeypatch.setattr("rag_lab.api.PostgresStore", lambda url: services.store)
    monkeypatch.setattr("rag_lab.api.Providers", lambda settings: services.providers)
    monkeypatch.setattr(services.store, "connect", connect, raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/modules").json()["bm25"] is True
        response = client.post("/query", json={"question": "RAG", "experiment": "G5"})
        assert response.status_code == 200
        assert response.json()["experiment"] == "G5"
        assert response.json()["evidence"][0]["source"] == "a.txt"
        assert client.post("/query", json={"question": "  "}).status_code == 422

        def fail(system, user):
            raise RuntimeError("SECRET_TOKEN")

        monkeypatch.setattr(services.providers, "chat", fail)
        response = client.post("/query", json={"question": "RAG"})
        assert response.status_code == 503
        assert "SECRET_TOKEN" not in response.text
