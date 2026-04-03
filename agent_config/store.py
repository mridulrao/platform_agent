from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
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


def _get_dict_cursor(conn):
    import psycopg

    return conn.cursor(row_factory=psycopg.rows.dict_row)


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


def _get_agent_record(agent_name: str) -> tuple[str, str | None]:
    with _get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, agent_type
                FROM voice_virtual_agents
                WHERE name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_name,),
            )
            row = cur.fetchone()
            if row is None:
                raise FileNotFoundError(f"No DB-backed agent record found for '{agent_name}'.")
            return str(row[0]), row[1]


def _take_agent_lock(cur, agent_name: str) -> None:
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (agent_name,))


def _latest_deployment_row(cur, agent_id: str, namespace: str, deployment_name: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT *
        FROM voice_virtual_agent_deployments
        WHERE voice_virtual_agent_id = %s
          AND namespace = %s
          AND deployment_name = %s
        ORDER BY updated_at DESC, created_at DESC
        LIMIT 1
        """,
        (agent_id, namespace, deployment_name),
    )
    return cur.fetchone()


def get_latest_agent_deployment(agent_name: str, namespace: str, deployment_name: str) -> dict[str, Any] | None:
    if not use_db_backend():
        return None

    agent_id, _ = _get_agent_record(agent_name)
    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            return _latest_deployment_row(cur, agent_id, namespace, deployment_name)


def enqueue_agent_deployment_request(
    *,
    agent_name: str,
    namespace: str,
    deployment_name: str,
    service_name: str | None,
    image: str | None,
    image_tag: str | None,
    desired_state: str,
    agent_version: str,
    requested_by: str | None = None,
) -> dict[str, Any] | None:
    if not use_db_backend():
        return None

    agent_id, agent_type = _get_agent_record(agent_name)
    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            _take_agent_lock(cur, agent_name)
            existing = _latest_deployment_row(cur, agent_id, namespace, deployment_name)

            if existing and existing["desired_state"] == desired_state and existing["workflow_status"] in {
                "pending",
                "applying",
                "deleting",
                "reconciling",
            }:
                cur.execute(
                    """
                    INSERT INTO voice_virtual_agent_deployment_events (
                        deployment_id,
                        event_type,
                        workflow_status,
                        runtime_status,
                        health_status,
                        message,
                        payload
                    ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        existing["id"],
                        "request_deduplicated",
                        existing["workflow_status"],
                        existing["runtime_status"],
                        existing["health_status"],
                        f"Duplicate {desired_state} request deduplicated.",
                        json.dumps({"requested_by": requested_by}),
                    ),
                )
                return existing

            if existing:
                cur.execute(
                    """
                    UPDATE voice_virtual_agent_deployments
                    SET
                        updated_at = now(),
                        agent_version = %s,
                        service_name = %s,
                        image = %s,
                        image_tag = %s,
                        desired_state = %s,
                        workflow_status = 'pending',
                        runtime_status = 'unknown',
                        health_status = 'unknown',
                        requested_by = COALESCE(%s, requested_by),
                        last_error = NULL,
                        status_reason = 'request_enqueued',
                        status_details = %s::jsonb
                    WHERE id = %s::uuid
                    RETURNING *
                    """,
                    (
                        agent_version,
                        service_name,
                        image,
                        image_tag,
                        desired_state,
                        requested_by,
                        json.dumps({"requested_by": requested_by, "desired_state": desired_state}),
                        existing["id"],
                    ),
                )
                row = cur.fetchone()
            else:
                cur.execute(
                    """
                    INSERT INTO voice_virtual_agent_deployments (
                        voice_virtual_agent_id,
                        updated_at,
                        agent_name,
                        agent_type,
                        agent_version,
                        namespace,
                        deployment_name,
                        service_name,
                        image,
                        image_tag,
                        desired_state,
                        workflow_status,
                        runtime_status,
                        health_status,
                        desired_replicas,
                        ready_replicas,
                        available_replicas,
                        restart_count,
                        requested_by,
                        status_reason,
                        status_details,
                        last_observed_at
                    ) VALUES (
                        %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', 'unknown', 'unknown', 1, 0, 0, 0, %s, 'request_enqueued', %s::jsonb, now()
                    )
                    RETURNING *
                    """,
                    (
                        agent_id,
                        agent_name,
                        agent_type,
                        agent_version,
                        namespace,
                        deployment_name,
                        service_name,
                        image,
                        image_tag,
                        desired_state,
                        requested_by,
                        json.dumps({"requested_by": requested_by, "desired_state": desired_state}),
                    ),
                )
                row = cur.fetchone()

            cur.execute(
                """
                INSERT INTO voice_virtual_agent_deployment_events (
                    deployment_id,
                    event_type,
                    workflow_status,
                    runtime_status,
                    health_status,
                    message,
                    payload
                ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    row["id"],
                    "request_enqueued",
                    row["workflow_status"],
                    row["runtime_status"],
                    row["health_status"],
                    f"{desired_state} request enqueued for {agent_name}.",
                    json.dumps({"requested_by": requested_by, "desired_state": desired_state}),
                ),
            )
            return row


def claim_next_deployment_request(worker_id: str) -> dict[str, Any] | None:
    if not use_db_backend():
        return None

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT *
                FROM voice_virtual_agent_deployments
                WHERE workflow_status = 'pending'
                ORDER BY updated_at ASC, created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row is None:
                return None

            next_status = "deleting" if row["desired_state"] == "deleted" else "applying"
            cur.execute(
                """
                UPDATE voice_virtual_agent_deployments
                SET
                    updated_at = now(),
                    workflow_status = %s,
                    status_reason = 'claimed',
                    status_details = (COALESCE(status_details::jsonb, '{}'::jsonb) || %s::jsonb)::json
                WHERE id = %s::uuid
                RETURNING *
                """,
                (
                    next_status,
                    json.dumps({"worker_id": worker_id}),
                    row["id"],
                ),
            )
            claimed = cur.fetchone()
            cur.execute(
                """
                INSERT INTO voice_virtual_agent_deployment_events (
                    deployment_id,
                    event_type,
                    workflow_status,
                    runtime_status,
                    health_status,
                    message,
                    payload
                ) VALUES (%s::uuid, 'request_claimed', %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    claimed["id"],
                    claimed["workflow_status"],
                    claimed["runtime_status"],
                    claimed["health_status"],
                    f"Deployment request claimed by {worker_id}.",
                    json.dumps({"worker_id": worker_id}),
                ),
            )
            return claimed


def list_reconcilable_deployments() -> list[dict[str, Any]]:
    if not use_db_backend():
        return []

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT *
                FROM voice_virtual_agent_deployments
                WHERE workflow_status IN ('applying', 'applied', 'reconciling', 'ready', 'deleting')
                ORDER BY updated_at ASC
                """
            )
            return list(cur.fetchall())


def get_sip_binding_by_phone_number(phone_number: str) -> dict[str, Any] | None:
    if not use_db_backend():
        return None

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT *
                FROM voice_virtual_agent_sip_bindings
                WHERE phone_number = %s
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (phone_number,),
            )
            return cur.fetchone()


def get_sip_binding_by_agent_name(agent_name: str) -> dict[str, Any] | None:
    if not use_db_backend():
        return None

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT *
                FROM voice_virtual_agent_sip_bindings
                WHERE agent_name = %s
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (agent_name,),
            )
            return cur.fetchone()


def save_agent_sip_binding_status(
    *,
    agent_name: str,
    phone_number: str,
    trunk_name: str,
    dispatch_rule_name: str,
    room_prefix: str,
    desired_state: str,
    workflow_status: str,
    health_status: str | None,
    trunk_id: str | None = None,
    dispatch_rule_id: str | None = None,
    agent_version: str | None = None,
    last_error: str | None = None,
    status_reason: str | None = None,
    status_details: dict | None = None,
    event_type: str | None = None,
    event_message: str | None = None,
) -> None:
    if not use_db_backend():
        return

    agent_id, _ = _get_agent_record(agent_name)
    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            _take_agent_lock(cur, agent_name)
            cur.execute(
                """
                SELECT *
                FROM voice_virtual_agent_sip_bindings
                WHERE agent_name = %s OR phone_number = %s
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (agent_name, phone_number),
            )
            row = cur.fetchone()

            if row is None:
                cur.execute(
                    """
                    INSERT INTO voice_virtual_agent_sip_bindings (
                        voice_virtual_agent_id,
                        updated_at,
                        provider,
                        agent_name,
                        agent_version,
                        phone_number,
                        trunk_id,
                        trunk_name,
                        dispatch_rule_id,
                        dispatch_rule_name,
                        room_prefix,
                        desired_state,
                        workflow_status,
                        health_status,
                        is_active,
                        last_checked_at,
                        provisioned_at,
                        last_error,
                        last_error_at,
                        status_reason,
                        status_details
                    ) VALUES (
                        %s, now(), 'livekit', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, now(),
                        CASE WHEN %s::text = 'provisioned' THEN now() ELSE NULL END,
                        %s,
                        CASE WHEN %s::text IS NOT NULL THEN now() ELSE NULL END,
                        %s,
                        %s::jsonb
                    )
                    RETURNING id
                    """,
                    (
                        agent_id,
                        agent_name,
                        agent_version,
                        phone_number,
                        trunk_id,
                        trunk_name,
                        dispatch_rule_id,
                        dispatch_rule_name,
                        room_prefix,
                        desired_state,
                        workflow_status,
                        health_status,
                        workflow_status,
                        last_error,
                        last_error,
                        status_reason,
                        json.dumps(status_details or {}),
                    ),
                )
                binding_id = str(cur.fetchone()["id"])
            else:
                binding_id = str(row["id"])
                cur.execute(
                    """
                    UPDATE voice_virtual_agent_sip_bindings
                    SET
                        updated_at = now(),
                        voice_virtual_agent_id = %s::uuid,
                        agent_version = %s,
                        phone_number = %s,
                        trunk_id = %s,
                        trunk_name = %s,
                        dispatch_rule_id = %s,
                        dispatch_rule_name = %s,
                        room_prefix = %s,
                        desired_state = %s,
                        workflow_status = %s,
                        health_status = %s,
                        is_active = CASE WHEN %s::text = 'deleted' THEN false ELSE true END,
                        last_checked_at = now(),
                        provisioned_at = CASE
                            WHEN %s::text = 'provisioned' THEN COALESCE(provisioned_at, now())
                            ELSE provisioned_at
                        END,
                        last_error = %s,
                        last_error_at = CASE WHEN %s::text IS NOT NULL THEN now() ELSE last_error_at END,
                        status_reason = %s,
                        status_details = %s::jsonb
                    WHERE id = %s::uuid
                    """,
                    (
                        agent_id,
                        agent_version,
                        phone_number,
                        trunk_id,
                        trunk_name,
                        dispatch_rule_id,
                        dispatch_rule_name,
                        room_prefix,
                        desired_state,
                        workflow_status,
                        health_status,
                        desired_state,
                        workflow_status,
                        last_error,
                        last_error,
                        status_reason,
                        json.dumps(status_details or {}),
                        binding_id,
                    ),
                )

            if event_type:
                cur.execute(
                    """
                    INSERT INTO voice_virtual_agent_sip_binding_events (
                        sip_binding_id,
                        event_type,
                        workflow_status,
                        health_status,
                        message,
                        payload
                    ) VALUES (%s::uuid, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        binding_id,
                        event_type,
                        workflow_status,
                        health_status,
                        event_message,
                        json.dumps(status_details or {}),
                    ),
                )


def save_agent_deployment_status(
    *,
    agent_name: str,
    agent_version: str,
    namespace: str,
    deployment_name: str,
    service_name: str | None,
    image: str | None,
    image_tag: str | None,
    desired_state: str,
    workflow_status: str,
    runtime_status: str | None,
    health_status: str | None,
    desired_replicas: int = 1,
    ready_replicas: int = 0,
    available_replicas: int = 0,
    restart_count: int = 0,
    generation: int | None = None,
    observed_generation: int | None = None,
    status_reason: str | None = None,
    status_details: dict | None = None,
    last_error: str | None = None,
    event_type: str | None = None,
    event_message: str | None = None,
) -> None:
    if not use_db_backend():
        return

    agent_id, agent_type = _get_agent_record(agent_name)
    with _get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM voice_virtual_agent_deployments
                WHERE voice_virtual_agent_id = %s
                  AND namespace = %s
                  AND deployment_name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_id, namespace, deployment_name),
            )
            row = cur.fetchone()

            if row is None:
                cur.execute(
                    """
                    INSERT INTO voice_virtual_agent_deployments (
                        voice_virtual_agent_id,
                        updated_at,
                        agent_name,
                        agent_type,
                        agent_version,
                        namespace,
                        deployment_name,
                        service_name,
                        image,
                        image_tag,
                        desired_state,
                        workflow_status,
                        runtime_status,
                        health_status,
                        desired_replicas,
                        ready_replicas,
                        available_replicas,
                        restart_count,
                        generation,
                        observed_generation,
                        status_reason,
                        status_details,
                        last_error,
                        last_error_at,
                        last_observed_at,
                        deployed_at,
                        ready_at,
                        deleted_at
                    ) VALUES (
                        %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s,
                        CASE WHEN %s IS NOT NULL THEN now() ELSE NULL END,
                        now(),
                        CASE WHEN %s IN ('applied', 'deployed', 'ready') THEN now() ELSE NULL END,
                        CASE WHEN %s = 'ready' THEN now() ELSE NULL END,
                        CASE WHEN %s = 'deleted' THEN now() ELSE NULL END
                    )
                    RETURNING id
                    """,
                    (
                        agent_id,
                        agent_name,
                        agent_type,
                        agent_version,
                        namespace,
                        deployment_name,
                        service_name,
                        image,
                        image_tag,
                        desired_state,
                        workflow_status,
                        runtime_status,
                        health_status,
                        desired_replicas,
                        ready_replicas,
                        available_replicas,
                        restart_count,
                        generation,
                        observed_generation,
                        status_reason,
                        json.dumps(status_details or {}),
                        last_error,
                        last_error,
                        workflow_status,
                        runtime_status,
                        workflow_status,
                    ),
                )
                deployment_id = str(cur.fetchone()[0])
            else:
                deployment_id = str(row[0])
                cur.execute(
                    """
                    UPDATE voice_virtual_agent_deployments
                    SET
                        updated_at = now(),
                        agent_version = %s,
                        service_name = %s,
                        image = %s,
                        image_tag = %s,
                        desired_state = %s,
                        workflow_status = %s,
                        runtime_status = %s,
                        health_status = %s,
                        desired_replicas = %s,
                        ready_replicas = %s,
                        available_replicas = %s,
                        restart_count = %s,
                        generation = %s,
                        observed_generation = %s,
                        status_reason = %s,
                        status_details = %s::jsonb,
                        last_error = %s,
                        last_error_at = CASE WHEN %s IS NOT NULL THEN now() ELSE last_error_at END,
                        last_observed_at = now(),
                        deployed_at = CASE
                            WHEN deployed_at IS NULL AND %s IN ('applied', 'deployed', 'ready') THEN now()
                            ELSE deployed_at
                        END,
                        ready_at = CASE
                            WHEN %s = 'ready' THEN now()
                            ELSE ready_at
                        END,
                        deleted_at = CASE
                            WHEN %s = 'deleted' THEN now()
                            ELSE deleted_at
                        END
                    WHERE id = %s::uuid
                    """,
                    (
                        agent_version,
                        service_name,
                        image,
                        image_tag,
                        desired_state,
                        workflow_status,
                        runtime_status,
                        health_status,
                        desired_replicas,
                        ready_replicas,
                        available_replicas,
                        restart_count,
                        generation,
                        observed_generation,
                        status_reason,
                        json.dumps(status_details or {}),
                        last_error,
                        last_error,
                        workflow_status,
                        runtime_status,
                        workflow_status,
                        deployment_id,
                    ),
                )

            if event_type:
                cur.execute(
                    """
                    INSERT INTO voice_virtual_agent_deployment_events (
                        deployment_id,
                        event_type,
                        workflow_status,
                        runtime_status,
                        health_status,
                        message,
                        payload
                    ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        deployment_id,
                        event_type,
                        workflow_status,
                        runtime_status,
                        health_status,
                        event_message,
                        json.dumps(status_details or {}),
                    ),
                )
