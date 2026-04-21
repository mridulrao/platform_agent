from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from livekit.agents.voice.events import ConversationItemAddedEvent


logger = logging.getLogger("voice-agent-observability")


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_json_compatible(model_dump())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_json_compatible(to_dict())

    if hasattr(value, "__dict__"):
        return _to_json_compatible(vars(value))

    return str(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_numeric_usage(value: Any, totals: dict[str, float], prefix: str = "") -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        key = prefix or "value"
        totals[key] = totals.get(key, 0.0) + float(value)
        return
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            child_prefix = f"{prefix}.{sub_key}" if prefix else str(sub_key)
            _collect_numeric_usage(sub_value, totals, child_prefix)
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            _collect_numeric_usage(item, totals, child_prefix)


@dataclass
class SessionObservability:
    call_id: str | None = None
    room_name: str | None = None
    agent_name: str | None = None
    db_proxy: Any | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    disconnect_reason: str | None = None
    user_turn_count: int = 0
    assistant_turn_count: int = 0
    usage_update_count: int = 0
    usage_summary: dict[str, Any] = field(default_factory=dict)
    llm_token_count: dict[str, Any] = field(default_factory=dict)
    tts_token_count: dict[str, Any] = field(default_factory=dict)
    _numeric_usage_totals: dict[str, float] = field(default_factory=dict)
    _latency_buckets: dict[str, list[float]] = field(
        default_factory=lambda: {
            "e2e_latency": [],
            "llm_node_ttft": [],
            "tts_node_ttfb": [],
            "transcription_delay": [],
            "end_of_turn_delay": [],
            "on_user_turn_completed_delay": [],
        }
    )
    _listeners_registered: bool = False

    def register(self, session) -> None:
        if self._listeners_registered:
            return

        try:
            session.on("conversation_item_added", self.on_conversation_item_added)
            session.on("session_usage_updated", self.on_session_usage_updated)
            self._listeners_registered = True
        except Exception as e:
            logger.warning("Failed to register observability listeners: %s", e)

    def on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        try:
            item = getattr(event, "item", None)
            if item is None:
                return

            role = getattr(item, "role", None) or "unknown"
            if role == "user":
                self.user_turn_count += 1
            elif role == "assistant":
                self.assistant_turn_count += 1

            metrics = _to_json_compatible(getattr(item, "metrics", None) or {})
            if not isinstance(metrics, dict):
                return

            for key in self._latency_buckets:
                value = _maybe_float(metrics.get(key))
                if value is not None:
                    self._latency_buckets[key].append(value)
        except Exception as e:
            logger.warning("Error collecting conversation metrics: %s", e, exc_info=False)

    def on_session_usage_updated(self, event: Any) -> None:
        try:
            usage = _to_json_compatible(getattr(event, "usage", None))
            if not isinstance(usage, dict):
                return

            self.usage_update_count += 1
            self.usage_summary = usage
            self._numeric_usage_totals.clear()
            _collect_numeric_usage(usage, self._numeric_usage_totals)
            self.llm_token_count = self._extract_token_usage(usage, usage_kind="llm")
            self.tts_token_count = self._extract_token_usage(usage, usage_kind="tts")
        except Exception as e:
            logger.warning("Error collecting usage metrics: %s", e, exc_info=False)

    def _extract_token_usage(self, usage: dict[str, Any], *, usage_kind: str) -> dict[str, Any]:
        raw_items = usage.get("model_usage") or usage.get("modelUsage") or []
        if not isinstance(raw_items, list):
            return {}

        entries: dict[str, Any] = {}
        totals: dict[str, float] = {}

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type") or item.get("usage_type") or item.get("kind") or "").lower()
            provider = str(item.get("provider") or "unknown")
            model = str(item.get("model") or "unknown")
            has_tts_markers = any(key in item for key in ("characters_count", "audio_duration"))

            if usage_kind == "llm":
                matches_kind = "llm" in item_type or (
                    "tts" not in item_type
                    and not has_tts_markers
                    and any(
                    key in item
                    for key in ("input_tokens", "output_tokens", "input_cached_tokens")
                    )
                )
            else:
                matches_kind = "tts" in item_type or has_tts_markers or (
                    "llm" not in item_type
                    and any(
                    key in item
                    for key in ("input_audio_tokens", "output_audio_tokens")
                    )
                )

            if not matches_kind:
                continue

            token_payload = {
                key: item.get(key)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "input_cached_tokens",
                    "input_text_tokens",
                    "output_text_tokens",
                    "input_audio_tokens",
                    "output_audio_tokens",
                    "characters_count",
                    "audio_duration",
                )
                if item.get(key) is not None
            }
            if not token_payload:
                continue

            entry_key = f"{provider}:{model}"
            entries[entry_key] = token_payload
            _collect_numeric_usage(token_payload, totals)

        if not entries:
            return {}

        return {
            "by_model": entries,
            "totals": totals,
        }

    def finalize(self, disconnect_reason: str | None = None) -> None:
        self.disconnect_reason = disconnect_reason
        self.ended_at = datetime.now(timezone.utc)

    def build_summary(self) -> dict[str, Any]:
        ended_at = self.ended_at or datetime.now(timezone.utc)
        duration_seconds = max((ended_at - self.started_at).total_seconds(), 0.0)

        latency_summary: dict[str, Any] = {}
        for key, values in self._latency_buckets.items():
            if not values:
                continue
            latency_summary[key] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
            }

        return {
            "call_id": self.call_id,
            "room_name": self.room_name,
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": duration_seconds,
            "disconnect_reason": self.disconnect_reason,
            "user_turn_count": self.user_turn_count,
            "assistant_turn_count": self.assistant_turn_count,
            "usage_update_count": self.usage_update_count,
            "usage_summary": self.usage_summary,
            "llm_token_count": self.llm_token_count,
            "tts_token_count": self.tts_token_count,
            "numeric_usage_totals": self._numeric_usage_totals,
            "latency_summary": latency_summary,
        }

    async def flush(self) -> None:
        if not self.db_proxy or not self.call_id:
            logger.debug(
                "Skipping observability flush: db_proxy=%s call_id=%s",
                bool(self.db_proxy),
                bool(self.call_id),
            )
            return

        if self.ended_at is None:
            self.finalize()

        summary = self.build_summary()
        latency_summary = summary.get("latency_summary", {})

        try:
            await self.db_proxy.save_session_observability(
                call_id=self.call_id,
                room_name=self.room_name,
                agent_name=self.agent_name,
                started_at=self.started_at,
                ended_at=self.ended_at,
                disconnect_reason=self.disconnect_reason,
                user_turn_count=self.user_turn_count,
                assistant_turn_count=self.assistant_turn_count,
                usage_update_count=self.usage_update_count,
                duration_seconds=summary.get("duration_seconds"),
                avg_e2e_latency_seconds=((latency_summary.get("e2e_latency") or {}).get("avg")),
                max_e2e_latency_seconds=((latency_summary.get("e2e_latency") or {}).get("max")),
                avg_llm_ttft_seconds=((latency_summary.get("llm_node_ttft") or {}).get("avg")),
                avg_tts_ttfb_seconds=((latency_summary.get("tts_node_ttfb") or {}).get("avg")),
                usage_summary=self.usage_summary,
                llm_token_count=self.llm_token_count,
                tts_token_count=self.tts_token_count,
                latency_summary=latency_summary,
                numeric_usage_totals=self._numeric_usage_totals,
                metrics_json=summary,
            )
        except Exception as e:
            logger.warning("Failed to flush session observability: %s", e, exc_info=False)
