"""API de consulta. Ingestão é uma operação local de CLI, não um caminho remoto arbitrário."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rag_lab.config import load_settings
from rag_lab.contracts import Envelope
from rag_lab.modules.base import Services
from rag_lab.pipeline import Pipeline
from rag_lab.providers import Providers
from rag_lab.storage import PostgresStore


def create_app(config_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Não há cache global de configuração: cada inicialização relê o YAML montado.
        settings = load_settings(config_path)
        store = PostgresStore(settings.database_url)
        store.initialize()
        app.state.services = Services(settings, store, Providers(settings))
        yield

    app = FastAPI(title="Laboratório RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        try:
            with app.state.services.store.connect() as connection:
                connection.execute("SELECT 1")
        except Exception as error:
            raise HTTPException(503, "PostgreSQL indisponível.") from error
        return {"status": "ok"}

    @app.get("/modules")
    def modules():
        return {
            name: config.enabled for name, config in app.state.services.settings.modules.items()
        }

    @app.post("/query", response_model=Envelope)
    def query(envelope: Envelope):
        if not envelope.question.strip():
            raise HTTPException(422, "A pergunta deve conter texto.")
        try:
            return Pipeline(app.state.services).run(envelope)
        except Exception as error:
            # URLs e tokens das exceções dos SDKs não são devolvidos ao cliente.
            raise HTTPException(
                503,
                "Falha na consulta. Confira a configuração, reingestão e execute rag-lab doctor.",
            ) from error

    return app
