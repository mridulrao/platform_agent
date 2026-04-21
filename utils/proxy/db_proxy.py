from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from datetime import datetime
from typing import Any
from uuid import UUID

import aiohttp
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from agent_config.store import _get_db_connection, _get_dict_cursor, use_db_backend


load_dotenv()


router = APIRouter(prefix="/db-proxy", tags=["db-proxy"])
logger = logging.getLogger("voice-agent-db-proxy")


class CallUpsertRequest(BaseModel):
    id: str
    caller: str | None = None
    channel: str | None = None
    status: str | None = None
    ended_reason: str | None = None
    details_url: str | None = None
    ended_at: datetime | None = None
    called_at: datetime | None = None
    recording_url: str | None = None
    interaction_id: str | None = None
    voice_virtual_agent_id: str | None = None
    call_data: dict[str, Any] | None = None


class CallPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caller: str | None = None
    channel: str | None = None
    status: str | None = None
    ended_reason: str | None = None
    details_url: str | None = None
    ended_at: datetime | None = None
    called_at: datetime | None = None
    recording_url: str | None = None
    interaction_id: str | None = None
    voice_virtual_agent_id: str | None = None
    call_data: dict[str, Any] | None = None


class CallEventCreateRequest(BaseModel):
    sequence_number: int | None = None
    event_type: str = "message"
    speaker: str | None = None
    content_text: str | None = None
    content_json: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SessionObservabilityUpsertRequest(BaseModel):
    call_id: str
    room_name: str | None = None
    agent_name: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    disconnect_reason: str | None = None
    duration_seconds: float | None = None
    user_turn_count: int | None = None
    assistant_turn_count: int | None = None
    usage_update_count: int | None = None
    avg_e2e_latency_seconds: float | None = None
    max_e2e_latency_seconds: float | None = None
    avg_llm_ttft_seconds: float | None = None
    avg_tts_ttfb_seconds: float | None = None
    usage_summary: dict[str, Any] | None = None
    llm_token_count: dict[str, Any] | None = None
    tts_token_count: dict[str, Any] | None = None
    latency_summary: dict[str, Any] | None = None
    numeric_usage_totals: dict[str, Any] | None = None
    metrics_json: dict[str, Any] | None = None


class DBProxyClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float = 10.0) -> None:
        resolved_base_url = base_url or os.getenv("DB_PROXY_URL") or "http://127.0.0.1:8000"
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        logger.info("DB proxy client configured with base_url=%s", self.base_url)

    async def create_call(
        self,
        *,
        call_id: str,
        caller: str | None = None,
        channel: str | None = None,
        status: str | None = None,
        ended_reason: str | None = None,
        details_url: str | None = None,
        ended_at: datetime | None = None,
        called_at: datetime | None = None,
        recording_url: str | None = None,
        interaction_id: str | None = None,
        voice_virtual_agent_id: str | None = None,
        call_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = CallUpsertRequest(
            id=call_id,
            caller=caller,
            channel=channel,
            status=status,
            ended_reason=ended_reason,
            details_url=details_url,
            ended_at=ended_at,
            called_at=called_at,
            recording_url=recording_url,
            interaction_id=interaction_id,
            voice_virtual_agent_id=voice_virtual_agent_id,
            call_data=call_data,
        ).model_dump(mode="json")
        return await self._request("POST", "/db-proxy/calls", payload)

    async def update_call(self, call_id: str, **fields: Any) -> dict[str, Any]:
        payload = CallPatchRequest(**fields).model_dump(mode="json", exclude_none=True)
        return await self._request("PATCH", f"/db-proxy/calls/{call_id}", payload)

    async def save_call_event(
        self,
        *,
        call_id: str,
        sequence_number: int | None = None,
        event_type: str = "message",
        speaker: str | None = None,
        content_text: str | None = None,
        content_json: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = CallEventCreateRequest(
            sequence_number=sequence_number,
            event_type=event_type,
            speaker=speaker,
            content_text=content_text,
            content_json=content_json,
            metadata=metadata,
        ).model_dump(mode="json", exclude_none=True)
        return await self._request("POST", f"/db-proxy/calls/{call_id}/events", payload)

    async def save_transcript_entry(
        self,
        *,
        call_id: str,
        role: str,
        content: str,
        content_type: str | None = None,
        sequence_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged_metadata = dict(metadata or {})
        if content_type:
            merged_metadata["content_type"] = content_type
        return await self.save_call_event(
            call_id=call_id,
            sequence_number=sequence_number,
            event_type="message",
            speaker=role,
            content_text=content,
            metadata=merged_metadata or None,
        )

    async def save_session_observability(
        self,
        *,
        call_id: str,
        room_name: str | None = None,
        agent_name: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        disconnect_reason: str | None = None,
        duration_seconds: float | None = None,
        user_turn_count: int | None = None,
        assistant_turn_count: int | None = None,
        usage_update_count: int | None = None,
        avg_e2e_latency_seconds: float | None = None,
        max_e2e_latency_seconds: float | None = None,
        avg_llm_ttft_seconds: float | None = None,
        avg_tts_ttfb_seconds: float | None = None,
        usage_summary: dict[str, Any] | None = None,
        llm_token_count: dict[str, Any] | None = None,
        tts_token_count: dict[str, Any] | None = None,
        latency_summary: dict[str, Any] | None = None,
        numeric_usage_totals: dict[str, Any] | None = None,
        metrics_json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = SessionObservabilityUpsertRequest(
            call_id=call_id,
            room_name=room_name,
            agent_name=agent_name,
            started_at=started_at,
            ended_at=ended_at,
            disconnect_reason=disconnect_reason,
            duration_seconds=duration_seconds,
            user_turn_count=user_turn_count,
            assistant_turn_count=assistant_turn_count,
            usage_update_count=usage_update_count,
            avg_e2e_latency_seconds=avg_e2e_latency_seconds,
            max_e2e_latency_seconds=max_e2e_latency_seconds,
            avg_llm_ttft_seconds=avg_llm_ttft_seconds,
            avg_tts_ttfb_seconds=avg_tts_ttfb_seconds,
            usage_summary=usage_summary,
            llm_token_count=llm_token_count,
            tts_token_count=tts_token_count,
            latency_summary=latency_summary,
            numeric_usage_totals=numeric_usage_totals,
            metrics_json=metrics_json,
        ).model_dump(mode="json", exclude_none=True)
        return await self._request("POST", f"/db-proxy/calls/{call_id}/observability", payload)

    async def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.request(method, url, json=payload) as response:
                try:
                    body = await response.json(content_type=None)
                except Exception:
                    body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"DB proxy request failed ({response.status}): {body}")
                return body


def _require_db_backend() -> None:
    if not use_db_backend():
        raise HTTPException(status_code=503, detail="DATABASE_URL or DIRECT_URL is required.")


def _merge_json_sql(existing_column: str, excluded_column: str) -> str:
    return f"""
        CASE
            WHEN {excluded_column} IS NULL THEN {existing_column}
            WHEN {existing_column} IS NULL THEN {excluded_column}
            ELSE {existing_column} || {excluded_column}
        END
    """


def _is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        UUID(value)
        return True
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=16)
def _get_table_columns(table_name: str) -> tuple[str, ...]:
    with _get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                """,
                (table_name,),
            )
            rows = cur.fetchall()
    return tuple(row[0] for row in rows)


def _prepare_call_data(call_identifier: str, call_data: dict[str, Any] | None) -> dict[str, Any] | None:
    merged = dict(call_data or {})
    if not _is_uuid(call_identifier):
        merged.setdefault("external_call_id", call_identifier)
    return merged or None


def _find_call_row(cur, call_identifier: str, columns: set[str]) -> dict[str, Any] | None:
    clauses: list[str] = []
    params: list[Any] = []

    if _is_uuid(call_identifier):
        clauses.append("id = %s")
        params.append(call_identifier)

    if "interaction_id" in columns:
        clauses.append("interaction_id = %s")
        params.append(call_identifier)

    if "call_data" in columns:
        clauses.append("call_data->>'external_call_id' = %s")
        params.append(call_identifier)

    if not clauses:
        return None

    cur.execute(
        f"""
        SELECT *
        FROM voice_virtual_agent_calls
        WHERE {" OR ".join(clauses)}
        ORDER BY created_at DESC
        LIMIT 1
        """,
        params,
    )
    return cur.fetchone()


def _create_or_update_call(payload: CallUpsertRequest) -> dict[str, Any]:
    _require_db_backend()

    columns = set(_get_table_columns("voice_virtual_agent_calls"))
    call_data = _prepare_call_data(payload.id, payload.call_data)

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            existing_row = _find_call_row(cur, payload.id, columns)

            if existing_row is not None:
                assignments: list[str] = []
                params: list[Any] = []

                for field_name, value in (
                    ("caller", payload.caller),
                    ("channel", payload.channel),
                    ("status", payload.status),
                    ("ended_reason", payload.ended_reason),
                    ("details_url", payload.details_url),
                    ("ended_at", payload.ended_at),
                    ("called_at", payload.called_at),
                    ("recording_url", payload.recording_url),
                    ("interaction_id", payload.interaction_id),
                    ("voice_virtual_agent_id", payload.voice_virtual_agent_id),
                ):
                    if field_name in columns and value is not None:
                        assignments.append(f"{field_name} = %s")
                        params.append(value)

                if "call_data" in columns and call_data is not None:
                    assignments.append(
                        "call_data = "
                        + _merge_json_sql("voice_virtual_agent_calls.call_data", "%s::jsonb").strip()
                    )
                    params.append(json.dumps(call_data))

                if not assignments:
                    return existing_row

                if "updated_at" in columns:
                    assignments.append("updated_at = now()")

                params.append(existing_row["id"])
                cur.execute(
                    f"""
                    UPDATE voice_virtual_agent_calls
                    SET {", ".join(assignments)}
                    WHERE id = %s
                    RETURNING *
                    """,
                    params,
                )
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=500, detail="Failed to persist call record.")
                return row

            insert_columns: list[str] = []
            insert_placeholders: list[str] = []
            insert_params: list[Any] = []

            if "id" in columns and _is_uuid(payload.id):
                insert_columns.append("id")
                insert_placeholders.append("%s")
                insert_params.append(payload.id)

            for field_name, value in (
                ("caller", payload.caller),
                ("channel", payload.channel),
                ("status", payload.status),
                ("ended_reason", payload.ended_reason),
                ("details_url", payload.details_url),
                ("ended_at", payload.ended_at),
                ("called_at", payload.called_at),
                ("recording_url", payload.recording_url),
                ("voice_virtual_agent_id", payload.voice_virtual_agent_id),
            ):
                if field_name in columns and value is not None:
                    insert_columns.append(field_name)
                    insert_placeholders.append("%s")
                    insert_params.append(value)

            interaction_id = payload.interaction_id
            if "interaction_id" in columns and interaction_id is None and not _is_uuid(payload.id):
                interaction_id = payload.id
            if "interaction_id" in columns and interaction_id is not None:
                insert_columns.append("interaction_id")
                insert_placeholders.append("%s")
                insert_params.append(interaction_id)

            if "call_data" in columns and call_data is not None:
                insert_columns.append("call_data")
                insert_placeholders.append("%s::jsonb")
                insert_params.append(json.dumps(call_data))

            if "updated_at" in columns:
                insert_columns.append("updated_at")
                insert_placeholders.append("now()")

            if not insert_columns:
                raise HTTPException(status_code=500, detail="No compatible call columns were found.")

            cur.execute(
                f"""
                INSERT INTO voice_virtual_agent_calls ({", ".join(insert_columns)})
                VALUES ({", ".join(insert_placeholders)})
                RETURNING *
                """,
                insert_params,
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to persist call record.")
            return row


def _patch_call(call_id: str, payload: CallPatchRequest) -> dict[str, Any]:
    _require_db_backend()

    columns = set(_get_table_columns("voice_virtual_agent_calls"))
    assignments: list[str] = []
    params: list[Any] = []

    for field_name in (
        "caller",
        "channel",
        "status",
        "ended_reason",
        "details_url",
        "ended_at",
        "called_at",
        "recording_url",
        "interaction_id",
        "voice_virtual_agent_id",
    ):
        value = getattr(payload, field_name)
        if field_name in columns and value is not None:
            assignments.append(f"{field_name} = %s")
            params.append(value)

    call_data = _prepare_call_data(call_id, payload.call_data)

    if "call_data" in columns and call_data is not None:
        assignments.append(
            "call_data = "
            + _merge_json_sql("voice_virtual_agent_calls.call_data", "%s::jsonb").strip()
        )
        encoded_call_data = json.dumps(call_data)
        params.extend([encoded_call_data, encoded_call_data, encoded_call_data])

    if not assignments:
        raise HTTPException(status_code=400, detail="No call fields were provided to update.")

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            existing_row = _find_call_row(cur, call_id, columns)
            if existing_row is None:
                raise HTTPException(status_code=404, detail=f"Call '{call_id}' was not found.")

            if "updated_at" in columns:
                assignments.append("updated_at = now()")

            params.append(existing_row["id"])
            cur.execute(
                f"""
                UPDATE voice_virtual_agent_calls
                SET {", ".join(assignments)}
                WHERE id = %s
                RETURNING *
                """,
                params,
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to update call record.")
            return row


def _create_call_event(call_id: str, payload: CallEventCreateRequest) -> dict[str, Any]:
    _require_db_backend()

    columns = set(_get_table_columns("voice_virtual_agent_calls"))

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            call_row = _find_call_row(cur, call_id, columns)
            if call_row is None:
                raise HTTPException(status_code=404, detail=f"Call '{call_id}' was not found.")
            internal_call_id = call_row["id"]

            cur.execute(
                """
                SELECT id
                FROM voice_virtual_agent_calls
                WHERE id = %s
                FOR UPDATE
                """,
                (internal_call_id,),
            )

            cur.execute(
                """
                WITH next_sequence AS (
                    SELECT COALESCE(MAX(sequence_number), 0) + 1 AS value
                    FROM voice_virtual_agent_call_events
                    WHERE call_id = %s
                )
                INSERT INTO voice_virtual_agent_call_events (
                    call_id,
                    sequence_number,
                    event_type,
                    speaker,
                    content_text,
                    content_json,
                    metadata
                )
                SELECT
                    %s,
                    COALESCE(%s, next_sequence.value),
                    %s,
                    %s,
                    %s,
                    %s::jsonb,
                    %s::jsonb
                FROM next_sequence
                RETURNING *
                """,
                (
                    internal_call_id,
                    internal_call_id,
                    payload.sequence_number,
                    payload.event_type,
                    payload.speaker,
                    payload.content_text,
                    json.dumps(payload.content_json) if payload.content_json is not None else None,
                    json.dumps(payload.metadata) if payload.metadata is not None else None,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to persist call event.")
            return row


def _ensure_session_observability_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS voice_virtual_agent_session_observability (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at TIMESTAMPTZ(6) NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ(6) NOT NULL DEFAULT now(),
            call_id UUID NOT NULL UNIQUE REFERENCES voice_virtual_agent_calls(id) ON DELETE CASCADE,
            room_name VARCHAR,
            agent_name VARCHAR,
            started_at TIMESTAMPTZ(6),
            ended_at TIMESTAMPTZ(6),
            disconnect_reason VARCHAR,
            duration_seconds DOUBLE PRECISION,
            user_turn_count INTEGER NOT NULL DEFAULT 0,
            assistant_turn_count INTEGER NOT NULL DEFAULT 0,
            usage_update_count INTEGER NOT NULL DEFAULT 0,
            avg_e2e_latency_seconds DOUBLE PRECISION,
            max_e2e_latency_seconds DOUBLE PRECISION,
            avg_llm_ttft_seconds DOUBLE PRECISION,
            avg_tts_ttfb_seconds DOUBLE PRECISION,
            usage_summary JSONB,
            llm_token_count JSONB,
            tts_token_count JSONB,
            latency_summary JSONB,
            numeric_usage_totals JSONB,
            metrics_json JSONB
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vva_session_observability_created_at
        ON voice_virtual_agent_session_observability (created_at)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vva_session_observability_agent_name_created_at
        ON voice_virtual_agent_session_observability (agent_name, created_at)
        """
    )
    cur.execute(
        """
        ALTER TABLE voice_virtual_agent_session_observability
        ADD COLUMN IF NOT EXISTS llm_token_count JSONB
        """
    )
    cur.execute(
        """
        ALTER TABLE voice_virtual_agent_session_observability
        ADD COLUMN IF NOT EXISTS tts_token_count JSONB
        """
    )


def _upsert_session_observability(call_id: str, payload: SessionObservabilityUpsertRequest) -> dict[str, Any]:
    _require_db_backend()

    columns = set(_get_table_columns("voice_virtual_agent_calls"))

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            call_row = _find_call_row(cur, call_id, columns)
            if call_row is None:
                raise HTTPException(status_code=404, detail=f"Call '{call_id}' was not found.")

            _ensure_session_observability_table(cur)

            cur.execute(
                """
                INSERT INTO voice_virtual_agent_session_observability (
                    call_id,
                    room_name,
                    agent_name,
                    started_at,
                    ended_at,
                    disconnect_reason,
                    duration_seconds,
                    user_turn_count,
                    assistant_turn_count,
                    usage_update_count,
                    avg_e2e_latency_seconds,
                    max_e2e_latency_seconds,
                    avg_llm_ttft_seconds,
                    avg_tts_ttfb_seconds,
                    usage_summary,
                    llm_token_count,
                    tts_token_count,
                    latency_summary,
                    numeric_usage_totals,
                    metrics_json
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
                )
                ON CONFLICT (call_id)
                DO UPDATE SET
                    updated_at = now(),
                    room_name = EXCLUDED.room_name,
                    agent_name = EXCLUDED.agent_name,
                    started_at = EXCLUDED.started_at,
                    ended_at = EXCLUDED.ended_at,
                    disconnect_reason = EXCLUDED.disconnect_reason,
                    duration_seconds = EXCLUDED.duration_seconds,
                    user_turn_count = EXCLUDED.user_turn_count,
                    assistant_turn_count = EXCLUDED.assistant_turn_count,
                    usage_update_count = EXCLUDED.usage_update_count,
                    avg_e2e_latency_seconds = EXCLUDED.avg_e2e_latency_seconds,
                    max_e2e_latency_seconds = EXCLUDED.max_e2e_latency_seconds,
                    avg_llm_ttft_seconds = EXCLUDED.avg_llm_ttft_seconds,
                    avg_tts_ttfb_seconds = EXCLUDED.avg_tts_ttfb_seconds,
                    usage_summary = EXCLUDED.usage_summary,
                    llm_token_count = EXCLUDED.llm_token_count,
                    tts_token_count = EXCLUDED.tts_token_count,
                    latency_summary = EXCLUDED.latency_summary,
                    numeric_usage_totals = EXCLUDED.numeric_usage_totals,
                    metrics_json = EXCLUDED.metrics_json
                RETURNING *
                """,
                (
                    call_row["id"],
                    payload.room_name,
                    payload.agent_name,
                    payload.started_at,
                    payload.ended_at,
                    payload.disconnect_reason,
                    payload.duration_seconds,
                    payload.user_turn_count or 0,
                    payload.assistant_turn_count or 0,
                    payload.usage_update_count or 0,
                    payload.avg_e2e_latency_seconds,
                    payload.max_e2e_latency_seconds,
                    payload.avg_llm_ttft_seconds,
                    payload.avg_tts_ttfb_seconds,
                    json.dumps(payload.usage_summary) if payload.usage_summary is not None else None,
                    json.dumps(payload.llm_token_count) if payload.llm_token_count is not None else None,
                    json.dumps(payload.tts_token_count) if payload.tts_token_count is not None else None,
                    json.dumps(payload.latency_summary) if payload.latency_summary is not None else None,
                    json.dumps(payload.numeric_usage_totals) if payload.numeric_usage_totals is not None else None,
                    json.dumps(payload.metrics_json) if payload.metrics_json is not None else None,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to persist session observability.")
            return row


@router.get("/health")
def db_proxy_health() -> dict[str, str]:
    _require_db_backend()
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database health check failed: {exc!r}") from exc
    return {"status": "ok"}


@router.post("/calls")
def create_or_update_call(payload: CallUpsertRequest) -> dict[str, Any]:
    return _create_or_update_call(payload)


@router.patch("/calls/{call_id}")
def patch_call(call_id: str, payload: CallPatchRequest) -> dict[str, Any]:
    return _patch_call(call_id, payload)


@router.post("/calls/{call_id}/events")
def create_call_event(call_id: str, payload: CallEventCreateRequest) -> dict[str, Any]:
    return _create_call_event(call_id, payload)


@router.post("/calls/{call_id}/observability")
def upsert_session_observability(
    call_id: str,
    payload: SessionObservabilityUpsertRequest,
) -> dict[str, Any]:
    return _upsert_session_observability(call_id, payload)


def build_app() -> FastAPI:
    app = FastAPI(title="Voice Agent DB Proxy")
    app.include_router(router)
    return app
