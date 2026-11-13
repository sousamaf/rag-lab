"""Pontos de entrada pequenos: a mesma implementação é usada dentro e fora do Docker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from rag_lab.config import load_settings
from rag_lab.contracts import Envelope
from rag_lab.doctor import check_environment
from rag_lab.ingestion import ingest_directory
from rag_lab.modules.base import Services
from rag_lab.pipeline import Pipeline
from rag_lab.providers import Providers
from rag_lab.storage import PostgresStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratório RAG didático")
    parser.add_argument("--config", help="YAML (padrão: RAG_CONFIG ou config/lab.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Valida YAML e mostra os módulos ativos")
    doctor = commands.add_parser(
        "doctor", help="Valida requisitos; faz uma chamada curta aos modelos"
    )
    doctor.add_argument("--offline", action="store_true", help="Não consulta serviços externos")
    doctor.add_argument("--port", type=int, help="Verifica disponibilidade da porta da API")
    commands.add_parser("init-db", help="Cria extensão e tabelas sem apagar dados")
    ingest = commands.add_parser("ingest", help="Indexa .txt e .md UTF-8 de uma pasta")
    ingest.add_argument("directory", type=Path)
    query = commands.add_parser("query", help="Consulta e retorna envelope JSON com fontes e trace")
    query.add_argument("question")
    serve = commands.add_parser("serve", help="Inicia a API HTTP e documentação /docs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        services = Services(settings, PostgresStore(settings.database_url), Providers(settings))
        if args.command == "validate":
            result = {
                "valid": True,
                "modules": {name: switch.enabled for name, switch in settings.modules.items()},
            }
        elif args.command == "doctor":
            result = check_environment(services, offline=args.offline, port=args.port)
        elif args.command == "init-db":
            services.store.initialize()
            result = {"initialized": True}
        elif args.command == "ingest":
            result = ingest_directory(args.directory, settings, services.store, services.providers)
        elif args.command == "query":
            if not args.question.strip():
                parser.error("A pergunta deve conter texto.")
            result = (
                Pipeline(services).run(Envelope(question=args.question)).model_dump(mode="json")
            )
        else:
            import uvicorn

            from rag_lab.api import create_app

            uvicorn.run(create_app(args.config), host=args.host, port=args.port)
            return
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "doctor" and not result["ok"]:
            sys.exit(1)
    except ValidationError as error:
        # Exibe a localização e a regra violada sem imprimir o valor de entrada (pode ser segredo).
        for issue in error.errors(include_input=False, include_url=False):
            print(
                f"Configuração inválida: {'.'.join(map(str, issue['loc']))}: {issue['msg']}",
                file=sys.stderr,
            )
        sys.exit(1)
    except Exception as error:
        print(
            f"Falha ({type(error).__name__}). Revise YAML, serviços e execute rag-lab doctor.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
