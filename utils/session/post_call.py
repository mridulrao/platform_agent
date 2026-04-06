from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any


logger = logging.getLogger("voice-agent-post-call")


@dataclass
class VVAPostCallInfo:
    call_id: str | None = None
    channel: str | None = None
    phone_number: str | None = None
    disconnect_reason: str | None = None
    db_proxy: Any | None = None

    async def update_call_record_on_end(self) -> None:
        """Update the call row when the session shuts down."""
        if not self.db_proxy or not self.call_id:
            logger.debug(
                "Skipping post-call update: db_proxy=%s call_id=%s",
                bool(self.db_proxy),
                bool(self.call_id),
            )
            return

        call_data = {
            "channel": self.channel,
            "phone_number": self.phone_number if self.channel == "sip" else None,
        }

        try:
            await self.db_proxy.update_call(
                self.call_id,
                caller=self.phone_number if self.channel == "sip" else None,
                channel=self.channel,
                status="ended",
                ended_reason=self.disconnect_reason,
                ended_at=datetime.now().astimezone(),
                call_data=call_data,
            )
            logger.debug("Updated call record on shutdown for call_id=%s", self.call_id)
        except Exception as e:
            logger.warning("Failed to update call record on shutdown: %s", e, exc_info=False)
