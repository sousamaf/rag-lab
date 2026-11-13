"""Preflight compartilhado por CLI e PowerShell, com falhas que impedem a inicialização."""

from __future__ import annotations

import importlib.metadata
import socket
import sys

from rag_lab.modules.base import Services


def check_environment(services: Services, *, offline: bool = False, port: int | None = None):
    checks = []

    def check(name, action, hint):
        try:
            detail = action()
            checks.append({"check": name, "ok": True, "detail": detail})
        except Exception as error:
            # Exceções de SDK podem carregar URLs e credenciais; reporte a classe, não seu texto.
            checks.append({"check": name, "ok": False, "detail": f"{type(error).__name__}: {hint}"})

    def python_version():
        if not (3, 11) <= sys.version_info[:2] < (3, 15):
            raise RuntimeError("Python incompatível")
        return sys.version.split()[0]

    def dependencies():
        from packaging.requirements import Requirement

        for entry in importlib.metadata.requires("rag-lab") or []:
            requirement = Requirement(entry)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            version = importlib.metadata.version(requirement.name)
            if version not in requirement.specifier:
                raise RuntimeError("Dependência incompatível")
        return "Dependências instaladas e versões compatíveis"

    def database():
        with services.store.connect() as connection:
            version = int(
                connection.execute("SHOW server_version_num").fetchone()["server_version_num"]
            )
            if version < 160000:
                raise RuntimeError("Use PostgreSQL 16 ou superior")
            extension = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
            ).fetchone()
            if not extension:
                raise RuntimeError("Habilite CREATE EXTENSION vector no banco configurado")
            if not connection.execute(
                "SELECT has_schema_privilege(current_user, 'public', 'CREATE') AS allowed",
            ).fetchone()["allowed"]:
                raise RuntimeError("Usuário precisa criar as tabelas do laboratório")
        return f"PostgreSQL {version // 10000}; pgvector {extension['extversion']}"

    def llm():
        services.providers.chat("Responda somente OK.", "Teste de disponibilidade do laboratório.")
        return "Endpoint, credencial e modelo responderam"

    def embeddings():
        services.providers.embed(["Teste de dimensão do laboratório."])
        return "Endpoint, modelo e dimensão validados"

    def port_available():
        with socket.socket() as listener:
            listener.bind(("0.0.0.0", port))
        return f"Porta {port} disponível"

    check("python", python_version, "Use Python 3.11 a 3.14; recomendado 3.12.")
    check("dependencies", dependencies, "Execute uv sync --frozen (Windows: start.ps1 -Install).")
    if port is not None:
        check("api_port", port_available, "Libere a porta ou escolha outra com --port/-Port.")
    if not offline:
        check(
            "postgresql_pgvector",
            database,
            "Confira DATABASE_URL, PostgreSQL >=16, CREATE EXTENSION vector e permissão CREATE.",
        )
        check(
            "llm", llm, "Confira llm no YAML, credencial, serviço e modelo instalado no endpoint."
        )
        if services.settings.modules["embedding"].enabled:
            check(
                "embeddings",
                embeddings,
                "Confira embeddings no YAML, credencial, modelo instalado e dimensions.",
            )
    return {"ok": all(item["ok"] for item in checks), "checks": checks}
