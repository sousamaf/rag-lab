"""G7 (Fase F9): Coletor experimental controlado com instrumentação de chamadas e manifesto."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_lab.config import Settings, configuration_path, load_settings  # noqa: E402
from rag_lab.contracts import Envelope  # noqa: E402
from rag_lab.modules.base import Services  # noqa: E402
from rag_lab.pipeline import Pipeline  # noqa: E402
from rag_lab.providers import Providers  # noqa: E402
from rag_lab.storage import PostgresStore  # noqa: E402


class InstrumentedProviders:
    """Wrapper fino sobre Providers para contabilizar chamadas e textos por consulta."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.chat_calls = 0
        self.embed_calls = 0
        self.embed_texts = 0

    def reset_counts(self) -> None:
        self.chat_calls = 0
        self.embed_calls = 0
        self.embed_texts = 0

    def snapshot_counts(self) -> dict[str, int]:
        return {
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "embed_texts": self.embed_texts,
        }

    def chat(self, system: str, user: str) -> str:
        self.chat_calls += 1
        return self._inner.chat(system, user)

    def chat_json(self, system: str, user: str) -> str:
        self.chat_calls += 1
        return self._inner.chat_json(system, user)

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch = list(texts)
        self.embed_calls += 1
        self.embed_texts += len(batch)
        return self._inner.embed(batch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def file_sha256(path: Path, *, normalize_lf: bool = True) -> str:
    """
    Calcula o SHA-256 determinístico do arquivo.
    Normaliza CRLF -> LF (sha256-lf) para produzir o mesmo hash em Windows e Linux/Docker.
    """
    raw = path.read_bytes()
    if normalize_lf:
        raw = raw.replace(b"\r\n", b"\n")
    return sha256(raw).hexdigest()


def corpus_hashes(corpus_dir: Path) -> dict[str, str]:
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise ValueError(f"Diretório de corpus inexistente: {corpus_dir}")
    files = sorted(
        p for p in corpus_dir.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        raise ValueError(f"Nenhum arquivo .md ou .txt encontrado em: {corpus_dir}")
    return {p.name: file_sha256(p) for p in files}


def load_gabarito(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Arquivo de gabarito não encontrado: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("perguntas"), list):
        raise ValueError(f"Estrutura inválida no gabarito: {path}")
    for item in data["perguntas"]:
        if not isinstance(item, dict) or "id" not in item or "pergunta" not in item:
            raise ValueError("Item de pergunta inválido no gabarito.")
    return data


def _redact_url(raw_url: str) -> str:
    try:
        parts = urlsplit(raw_url)
    except Exception:
        return "<redacted>"
    host = parts.hostname or "localhost"
    port = f":{parts.port}" if parts.port else ""
    scheme = parts.scheme or "http"
    return f"{scheme}://{host}{port}{parts.path}"


def sanitize_settings(settings: Settings) -> dict[str, Any]:
    """Serializa Settings removendo database_url e quaisquer credenciais em URLs."""
    dumped = settings.model_dump(mode="json", exclude={"database_url"})
    for section in ("llm", "embeddings"):
        if isinstance(dumped.get(section), dict) and "base_url" in dumped[section]:
            dumped[section]["base_url"] = _redact_url(str(dumped[section]["base_url"]))
    try:
        db_parts = urlsplit(settings.database_url)
        dumped["database_name"] = db_parts.path.lstrip("/") or "rag"
    except Exception:
        dumped["database_name"] = "rag"
    return dumped


def git_metadata(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def _run_git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except Exception:
            return None

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "git_dirty": bool(status),
    }


def query_ollama_model_info(base_url: str, model: str, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {
            "model": model,
            "digest": None,
            "modified_at": None,
            "details": None,
            "reason": "dry_run",
        }
    clean_base = base_url.rstrip("/")
    tags_url = clean_base + "/api/tags"
    show_url = clean_base + "/api/show"
    digest: str | None = None
    modified_at: Any = None
    details: Any = None
    candidate_names = {model, f"{model}:latest"} if ":" not in model else {model}

    try:
        tags_resp = httpx.get(tags_url, timeout=3.0)
        tags_resp.raise_for_status()
        tags_payload = tags_resp.json()
        models_list = tags_payload.get("models") if isinstance(tags_payload, dict) else None
        if isinstance(models_list, list):
            for entry in models_list:
                if not isinstance(entry, dict):
                    continue
                entry_name = str(entry.get("name") or entry.get("model") or "")
                if entry_name in candidate_names:
                    raw_digest = entry.get("digest")
                    if isinstance(raw_digest, str) and raw_digest:
                        digest = raw_digest
                    modified_at = entry.get("modified_at")
                    details = entry.get("details")
                    break
    except Exception as exc:
        return {
            "model": model,
            "digest": None,
            "modified_at": None,
            "details": None,
            "reason": f"offline_or_unreachable: {type(exc).__name__}",
        }

    try:
        show_resp = httpx.post(show_url, json={"name": model}, timeout=3.0)
        if show_resp.status_code == 200:
            show_payload = show_resp.json()
            if isinstance(show_payload, dict):
                if modified_at is None:
                    modified_at = show_payload.get("modified_at")
                if details is None:
                    details = show_payload.get("details")
                if digest is None and isinstance(show_payload.get("digest"), str):
                    digest = show_payload["digest"]
    except Exception:
        pass

    return {
        "model": model,
        "digest": digest,
        "modified_at": modified_at,
        "details": details,
        "reason": None if digest is not None else "model_not_found_in_api_tags",
    }


def _parse_filter_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for raw in values:
        for part in raw.split(","):
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def filter_questions(
    perguntas: list[dict[str, Any]],
    profiles: list[str] | None = None,
    ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    profile_set = set(_parse_filter_list(profiles))
    id_set = set(_parse_filter_list(ids))
    selected: list[dict[str, Any]] = []
    for q in perguntas:
        if profile_set and q.get("perfil") not in profile_set:
            continue
        if id_set and q.get("id") not in id_set:
            continue
        selected.append(q)
    if not selected:
        raise ValueError("Nenhuma pergunta selecionada após aplicar os filtros --profiles/--ids.")
    return selected


def build_manifest(
    *,
    run_id: str,
    config_path: Path,
    settings: Settings,
    gabarito_path: Path,
    gabarito_data: dict[str, Any],
    corpus_dir: Path,
    selected_questions: list[dict[str, Any]],
    reps: int,
    llm_override: str | None = None,
    embedding_override: str | None = None,
    dry_run: bool = False,
    uv_lock_path: Path | None = None,
    ollama_probe: bool = True,
) -> dict[str, Any]:
    all_questions = gabarito_data.get("perguntas") or []
    status_counts: dict[str, int] = {}
    for q in all_questions:
        st = str(q.get("status_revisao") or "indefinido")
        status_counts[st] = status_counts.get(st, 0) + 1
    gabarito_homologado = bool(all_questions) and all(
        str(q.get("status_revisao") or "")
        not in {"", "pendente_validacao_humana", "pendente", "rascunho"}
        for q in all_questions
    )
    lock_file = uv_lock_path or (REPO_ROOT / "uv.lock")
    uv_sha = file_sha256(lock_file) if lock_file.exists() else None

    if ollama_probe:
        llm_info = query_ollama_model_info(settings.llm.base_url, settings.llm.model, dry_run)
        emb_info = query_ollama_model_info(
            settings.embeddings.base_url, settings.embeddings.model, dry_run
        )
    else:
        llm_info = {"model": settings.llm.model, "digest": None, "reason": "probe_disabled"}
        emb_info = {
            "model": settings.embeddings.model,
            "digest": None,
            "reason": "probe_disabled",
        }

    return {
        "schema": "g7-run-manifest/1",
        "hash_mode": "sha256-lf",
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "git": git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "config_path": config_path.as_posix(),
        "config_name": config_path.stem,
        "config_sha256": file_sha256(config_path),
        "config_effective": sanitize_settings(settings),
        "overrides": {
            "llm_model": llm_override,
            "embedding_model": embedding_override,
        },
        "uv_lock_sha256": uv_sha,
        "gabarito_path": gabarito_path.as_posix(),
        "gabarito_sha256": file_sha256(gabarito_path),
        "gabarito_homologado": gabarito_homologado,
        "gabarito_status_counts": status_counts,
        "corpus_dir": corpus_dir.as_posix(),
        "corpus_sha256": corpus_hashes(corpus_dir),
        "ollama_models": {
            "llm": llm_info,
            "embeddings": emb_info,
        },
        "execution_plan": {
            "selected_questions": len(selected_questions),
            "reps": reps,
            "total_runs": len(selected_questions) * reps,
            "question_ids": [str(q["id"]) for q in selected_questions],
            "profiles": sorted({str(q.get("perfil", "")) for q in selected_questions}),
        },
    }


def run_collection(
    args: argparse.Namespace,
    *,
    services_override: Services | None = None,
) -> dict[str, Any]:
    if args.reps < 1:
        raise ValueError("--reps deve ser maior ou igual a 1.")
    config_path = configuration_path(args.config)
    if not config_path.exists():
        raise ValueError(f"Arquivo de configuração não encontrado: {config_path}")

    settings = load_settings(config_path)
    if args.llm_model:
        settings.llm.model = args.llm_model
    if args.embedding_model:
        settings.embeddings.model = args.embedding_model

    gabarito_data = load_gabarito(args.gabarito)
    selected_questions = filter_questions(
        gabarito_data["perguntas"],
        profiles=args.profiles,
        ids=args.ids,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}_{config_path.stem}"

    manifest = build_manifest(
        run_id=run_id,
        config_path=config_path,
        settings=settings,
        gabarito_path=args.gabarito,
        gabarito_data=gabarito_data,
        corpus_dir=args.corpus_dir,
        selected_questions=selected_questions,
        reps=args.reps,
        llm_override=args.llm_model,
        embedding_override=args.embedding_model,
        dry_run=bool(args.dry_run),
        ollama_probe=services_override is None,
    )

    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        return manifest

    if services_override is not None:
        base_services = services_override
        base_services.settings = settings
    else:
        base_services = Services(
            settings,
            PostgresStore(settings.database_url),
            Providers(settings),
        )

    instrumented = InstrumentedProviders(base_services.providers)
    services = Services(settings, base_services.store, instrumented)
    pipeline = Pipeline(services)

    envelopes_dir = run_dir / "envelopes"
    envelopes_dir.mkdir(parents=True, exist_ok=True)

    total_wall_ms = 0.0
    total_calls = {"chat_calls": 0, "embed_calls": 0, "embed_texts": 0}
    completed = 0
    resumed = 0
    executed = 0
    resume_enabled = bool(getattr(args, "resume", False))

    for rep in range(1, args.reps + 1):
        for q in selected_questions:
            qid = str(q["id"])
            perfil = str(q.get("perfil", ""))
            out_file = envelopes_dir / f"{qid}__rep{rep}.json"

            if resume_enabled and out_file.exists():
                try:
                    existing = json.loads(out_file.read_text(encoding="utf-8"))
                    if isinstance(existing, dict) and isinstance(existing.get("envelope"), dict):
                        Envelope.model_validate(existing["envelope"])
                        completed += 1
                        resumed += 1
                        total_wall_ms = round(
                            total_wall_ms + float(existing.get("wall_time_ms", 0.0)), 3
                        )
                        existing_calls = existing.get("calls") or {}
                        for k in total_calls:
                            total_calls[k] += int(existing_calls.get(k, 0))
                        continue
                except Exception:
                    pass

            instrumented.reset_counts()
            start = perf_counter()
            envelope = pipeline.run(
                Envelope(
                    question=str(q["pergunta"]),
                    attributes={
                        "grupo": "G7",
                        "run_id": run_id,
                        "config": config_path.stem,
                        "pergunta_id": qid,
                        "perfil": perfil,
                        "rep": rep,
                    },
                )
            )
            wall_ms = round((perf_counter() - start) * 1000.0, 3)
            calls = instrumented.snapshot_counts()
            total_wall_ms = round(total_wall_ms + wall_ms, 3)
            for k in total_calls:
                total_calls[k] += calls[k]
            completed += 1
            executed += 1

            record = {
                "run_id": run_id,
                "config_name": config_path.stem,
                "question_id": qid,
                "perfil": perfil,
                "rep": rep,
                "wall_time_ms": wall_ms,
                "calls": calls,
                "envelope": envelope.model_dump(mode="json"),
            }
            out_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    manifest["summary"] = {
        "completed_runs": completed,
        "executed_runs": executed,
        "resumed_runs": resumed,
        "total_wall_time_ms": total_wall_ms,
        "total_calls": total_calls,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G7: Coletor experimental controlado com instrumentação e manifesto."
    )
    parser.add_argument("--config", type=Path, required=True, help="Caminho do YAML experimental")
    parser.add_argument(
        "--gabarito",
        type=Path,
        default=Path("experiments/g7/perguntas_gabarito.yaml"),
        help="Caminho do gabarito YAML (padrão: experiments/g7/perguntas_gabarito.yaml)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("experiments/g7/corpus"),
        help="Diretório do corpus autoral .md (padrão: experiments/g7/corpus)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/results/g7"),
        help="Diretório raiz de saída (padrão: data/results/g7)",
    )
    parser.add_argument("--run-id", help="Identificador da execução (padrão: <timestamp>_<config>)")
    parser.add_argument("--reps", type=int, default=1, help="Repetições por pergunta (padrão: 1)")
    parser.add_argument(
        "--profiles",
        nargs="*",
        help="Filtro opcional por perfis (ex.: literal semantica)",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Filtro opcional por IDs de pergunta (ex.: L02 S02)",
    )
    parser.add_argument("--llm-model", help="Override opcional em memória para llm.model")
    parser.add_argument(
        "--embedding-model", help="Override opcional em memória para embeddings.model"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula (question_id, rep) cujo envelope JSON já exista e seja válido no diretório",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida YAML, gabarito e hashes SHA-256 sem conectar ao banco ou ao Ollama",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_collection(args)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"Erro em rodar.py ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
