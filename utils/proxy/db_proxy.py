from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import aiohttp
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from agent_config.store import _get_db_connection, _get_dict_cursor, use_db_backend


router = APIRouter(prefix="/db-proxy", tags=["db-proxy"])


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


class DBProxyClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float = 10.0) -> None:
        resolved_base_url = base_url or os.getenv("DB_PROXY_URL") or "http://127.0.0.1:8000"
        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

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

    async def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.request(method, url, json=payload) as response:
                body = await response.json(content_type=None)
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


def _create_or_update_call(payload: CallUpsertRequest) -> dict[str, Any]:
    _require_db_backend()

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            cur.execute(
                f"""
                INSERT INTO voice_virtual_agent_calls (
                    id,
                    caller,
                    channel,
                    status,
                    ended_reason,
                    details_url,
                    ended_at,
                    called_at,
                    recording_url,
                    interaction_id,
                    voice_virtual_agent_id,
                    call_data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id)
                DO UPDATE SET
                    caller = COALESCE(EXCLUDED.caller, voice_virtual_agent_calls.caller),
                    channel = COALESCE(EXCLUDED.channel, voice_virtual_agent_calls.channel),
                    status = COALESCE(EXCLUDED.status, voice_virtual_agent_calls.status),
                    ended_reason = COALESCE(EXCLUDED.ended_reason, voice_virtual_agent_calls.ended_reason),
                    details_url = COALESCE(EXCLUDED.details_url, voice_virtual_agent_calls.details_url),
                    ended_at = COALESCE(EXCLUDED.ended_at, voice_virtual_agent_calls.ended_at),
                    called_at = COALESCE(EXCLUDED.called_at, voice_virtual_agent_calls.called_at),
                    recording_url = COALESCE(EXCLUDED.recording_url, voice_virtual_agent_calls.recording_url),
                    interaction_id = COALESCE(EXCLUDED.interaction_id, voice_virtual_agent_calls.interaction_id),
                    voice_virtual_agent_id = COALESCE(
                        EXCLUDED.voice_virtual_agent_id,
                        voice_virtual_agent_calls.voice_virtual_agent_id
                    ),
                    call_data = {_merge_json_sql("voice_virtual_agent_calls.call_data", "EXCLUDED.call_data")},
                    updated_at = now()
                RETURNING *
                """,
                (
                    payload.id,
                    payload.caller,
                    payload.channel,
                    payload.status,
                    payload.ended_reason,
                    payload.details_url,
                    payload.ended_at,
                    payload.called_at,
                    payload.recording_url,
                    payload.interaction_id,
                    payload.voice_virtual_agent_id,
                    json.dumps(payload.call_data) if payload.call_data is not None else None,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to persist call record.")
            return row


def _patch_call(call_id: str, payload: CallPatchRequest) -> dict[str, Any]:
    _require_db_backend()

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
        if value is not None:
            assignments.append(f"{field_name} = %s")
            params.append(value)

    if payload.call_data is not None:
        assignments.append(
            "call_data = "
            + _merge_json_sql("voice_virtual_agent_calls.call_data", "%s::jsonb").strip()
        )
        params.append(json.dumps(payload.call_data))

    if not assignments:
        raise HTTPException(status_code=400, detail="No call fields were provided to update.")

    assignments.append("updated_at = now()")
    params.append(call_id)

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
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
                raise HTTPException(status_code=404, detail=f"Call '{call_id}' was not found.")
            return row


def _create_call_event(call_id: str, payload: CallEventCreateRequest) -> dict[str, Any]:
    _require_db_backend()

    with _get_db_connection() as conn:
        with _get_dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT id
                FROM voice_virtual_agent_calls
                WHERE id = %s
                FOR UPDATE
                """,
                (call_id,),
            )
            call_row = cur.fetchone()
            if call_row is None:
                raise HTTPException(status_code=404, detail=f"Call '{call_id}' was not found.")

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
                    call_id,
                    call_id,
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


@router.get("/health")
def db_proxy_health() -> dict[str, str]:
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


def build_app() -> FastAPI:
    app = FastAPI(title="Voice Agent DB Proxy")
    app.include_router(router)
    return app
