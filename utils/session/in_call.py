from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from livekit.agents.voice.events import ConversationItemAddedEvent


logger = logging.getLogger("voice-agent-session")


def _coerce_content_to_text(raw_content: Any) -> str:
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        chunks: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if text_value:
                    chunks.append(str(text_value))
                    continue
            text_attr = getattr(item, "text", None) or getattr(item, "content", None)
            if text_attr:
                chunks.append(str(text_attr))
        return " ".join(chunk.strip() for chunk in chunks if chunk).strip()
    return str(raw_content)


@dataclass
class VVASessionInfo:
    # sip details
    org_id: str | None = None
    phone_number: str | None = None
    session_type: str | None = None # can be sip / webrtc
    channel: str | None = None

    # user details
    user_sys_id: str | None = None
    full_name: str | None = None

    # conversation tracking
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_transcripts: List[Dict[str, Any]] = field(default_factory=list)
    agent_transcripts: List[Dict[str, Any]] = field(default_factory=list)

    # user call state
    disconnect_reason: str | None = None
    call_id: str | None = None
    db_proxy: Any | None = None
    _listeners_registered: bool = False
    _transcript_sequence_number: int = 0


    # -------------------------------------------------------------------------
    # Conversation listeners
    # -------------------------------------------------------------------------
    def register_conversation_listeners(self, session) -> None:
        """Register event listeners for conversation tracking."""
        if self._listeners_registered:
            return

        try:
            session.on("conversation_item_added", self._on_conversation_item_added)
            self._listeners_registered = True
        except Exception as e:
            logger.warning("Failed to register conversation listeners: %s", e)


    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        """Process and store new conversation items."""
        try:
            item = getattr(event, "item", None)
            if item is None:
                return

            role = getattr(item, "role", None) or "unknown"
            raw_content = getattr(item, "content", None)
            content = _coerce_content_to_text(raw_content)

            entry = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }

            self.conversation_history.append(entry)

            if role == "user":
                self.user_transcripts.append(entry)
                self._save_conversation_turn_to_db(role, content)
            elif role == "assistant":
                self.agent_transcripts.append(entry)
                self._save_conversation_turn_to_db(role, content)

        except Exception as e:
            logger.warning("Error in _on_conversation_item_added: %s", e, exc_info=False)

    def _save_conversation_turn_to_db(self, role: str, content: str) -> None:
        """Save a conversation turn to the DB proxy in real-time (non-blocking)."""
        if not self.db_proxy or not self.call_id:
            logger.debug(
                "Skipping transcript save: db_proxy=%s call_id=%s",
                bool(self.db_proxy),
                bool(self.call_id),
            )
            return

        self._transcript_sequence_number += 1
        sequence_number = self._transcript_sequence_number

        async def _do_save() -> None:
            try:
                await self.db_proxy.save_transcript_entry(
                    call_id=self.call_id,
                    role=role,
                    content=content,
                    content_type="FinalTranscript",
                    sequence_number=sequence_number,
                )
                logger.debug("Saved transcript via proxy: %s - %s...", role, (content or "")[:50])
            except Exception as e:
                # keep quiet; don't break conversation flow
                logger.debug("Error saving transcript via proxy: %s", e, exc_info=False)

        # Only schedule if we're in an active loop (worker always is)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        loop.create_task(_do_save())

    async def create_call_record(
        self,
        *,
        room_name: str,
        agent_name: str | None = None,
        voice_virtual_agent_id: str | None = None,
    ) -> None:
        """Create or upsert the call row when the participant first connects."""
        if not self.db_proxy or not self.call_id:
            logger.debug(
                "Skipping call creation: db_proxy=%s call_id=%s",
                bool(self.db_proxy),
                bool(self.call_id),
            )
            return

        call_data = {
            "room_name": room_name,
            "channel": self.channel,
            "session_type": self.session_type,
            "phone_number": self.phone_number if self.channel == "sip" else None,
        }
        if agent_name:
            call_data["agent_name"] = agent_name

        try:
            await self.db_proxy.create_call(
                call_id=self.call_id,
                caller=self.phone_number if self.channel == "sip" else None,
                channel=self.channel,
                status="started",
                called_at=datetime.now().astimezone(),
                voice_virtual_agent_id=voice_virtual_agent_id,
                call_data=call_data,
            )
            logger.debug("Created call record for call_id=%s", self.call_id)
        except Exception as e:
            logger.warning("Failed to create call record: %s", e, exc_info=False)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    def get_conversation_history(self):
        return self.conversation_history

    def get_user_transcripts(self):
        return self.user_transcripts

    def get_agent_transcripts(self):
        return self.agent_transcripts

    def get_formatted_history_for_tools(self):
        return [{"role": item["role"], "content": item["content"]} for item in self.conversation_history]

    def get_call_id_and_phone_number(self, room_name: str) -> tuple[str, str]:
        parts = room_name.split("_")
        phone_number = parts[1] if len(parts) > 1 else ""
        call_id = parts[2] if len(parts) > 2 else ""
        return call_id, phone_number

    def resolve_session_type(
        self,
        *,
        participant_kind: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        attributes = attributes or {}

        if participant_kind == "sip":
            return "sip"

        return (
            attributes.get("channel")
            or "webrtc"
        )

    def set_connection_details(
        self,
        *,
        room_name: str,
        participant_kind: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        resolved_session_type = self.resolve_session_type(
            participant_kind=participant_kind,
            attributes=attributes,
        )
        self.session_type = resolved_session_type
        self.channel = resolved_session_type
        self.phone_number = self.resolve_phone_number(
            room_name=room_name,
            attributes=attributes,
        )
        self.call_id = self.resolve_call_id(
            room_name=room_name,
            attributes=attributes,
        )

    def resolve_phone_number(
        self,
        *,
        room_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> str | None:
        attributes = attributes or {}
        _, room_phone_number = self.get_call_id_and_phone_number(room_name)

        return (
            attributes.get("sip.phoneNumber")
            or attributes.get("phone_number")
            or room_phone_number
            or None
        )

    def resolve_call_id(
        self,
        *,
        room_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> str | None:
        attributes = attributes or {}
        room_call_id, _ = self.get_call_id_and_phone_number(room_name)

        return (
            attributes.get("sip.callID")
            or room_call_id
            or None
        )
