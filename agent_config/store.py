from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agent_config.schema import AgentConfig


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent
DB_AGENT_TYPE = "livekit"


def get_config_dir() -> Path:
    raw = os.getenv("AGENT_CONFIG_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_CONFIG_DIR


def use_db_backend() -> bool:
    return bool(os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL"))


def _sanitize_db_url(database_url: str) -> str:
    parts = urlsplit(database_url)
    if not parts.scheme:
        return database_url

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in {"pgbouncer", "connection_limit", "pool_timeout", "schema"}
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(filtered_query), parts.fragment)
    )


def _get_db_connection():
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "DATABASE_URL is set but psycopg is not installed. Install project dependencies first."
        ) from exc

    database_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DIRECT_URL or DATABASE_URL is required for DB-backed agent config storage.")
    return psycopg.connect(_sanitize_db_url(database_url))


def ensure_config_dir() -> Path:
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def config_path(agent_name: str) -> Path:
    return ensure_config_dir() / f"{agent_name}.json"


def _save_agent_config_file(config: AgentConfig) -> Path:
    path = config_path(config.name)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def _save_agent_config_db(
    config: AgentConfig,
    *,
    phone_number: str,
    agent_type: str = DB_AGENT_TYPE,
) -> str:
    payload = json.loads(config.model_dump_json())
    with _get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO voice_virtual_agents (phone_number, config, name, agent_type)
                VALUES (%s, %s::jsonb, %s, %s)
                ON CONFLICT (phone_number)
                DO UPDATE SET
                    config = EXCLUDED.config,
                    name = EXCLUDED.name,
                    agent_type = EXCLUDED.agent_type
                RETURNING id
                """,
                (phone_number, json.dumps(payload), config.name, agent_type),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to persist voice virtual agent config.")
            return str(row[0])


def save_agent_config(
    config: AgentConfig,
    *,
    phone_number: str | None = None,
    agent_type: str = DB_AGENT_TYPE,
) -> Path | str:
    if use_db_backend():
        if not phone_number:
            raise ValueError("phone_number is required when storing agent config in the database.")
        return _save_agent_config_db(config, phone_number=phone_number, agent_type=agent_type)
    return _save_agent_config_file(config)


def _load_agent_config_file(agent_name: str) -> AgentConfig:
    path = config_path(agent_name)
    if not path.exists():
        raise FileNotFoundError(f"No agent config found for '{agent_name}' at {path}")
    return AgentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _load_agent_config_db(agent_name: str) -> AgentConfig:
    with _get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM voice_virtual_agents
                WHERE name = %s AND agent_type = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_name, DB_AGENT_TYPE),
            )
            row = cur.fetchone()
            if row is None:
                raise FileNotFoundError(f"No DB-backed agent config found for '{agent_name}'.")
            payload = row[0]
            return AgentConfig.model_validate(payload)


def load_agent_config(agent_name: str) -> AgentConfig:
    if use_db_backend():
        return _load_agent_config_db(agent_name)
    return _load_agent_config_file(agent_name)


def _list_agent_configs_file() -> list[AgentConfig]:
    config_dir = ensure_config_dir()
    configs: list[AgentConfig] = []
    for path in sorted(config_dir.glob("*.json")):
        configs.append(AgentConfig.model_validate_json(path.read_text(encoding="utf-8")))
    return configs


def _list_agent_configs_db() -> list[AgentConfig]:
    with _get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM voice_virtual_agents
                WHERE agent_type = %s
                ORDER BY created_at DESC
                """,
                (DB_AGENT_TYPE,),
            )
            rows = cur.fetchall()
    return [AgentConfig.model_validate(row[0]) for row in rows]


def list_agent_configs() -> list[AgentConfig]:
    if use_db_backend():
        return _list_agent_configs_db()
    return _list_agent_configs_file()


def load_agent_config_from_metadata(metadata: str | None) -> AgentConfig:
    if not metadata:
        raise ValueError("Dispatch metadata is missing; cannot determine target agent.")

    payload = json.loads(metadata)
    agent_name = payload.get("config_name") or payload.get("target_agent")
    if not agent_name:
        raise ValueError("Dispatch metadata did not include config_name or target_agent.")
    return load_agent_config(agent_name)
