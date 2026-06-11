from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 30


class LiveKitSIPAPIError(RuntimeError):
    pass


class LiveKitSIPClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = (base_url or os.getenv("TELEPHONY_BASE_URL", "")).strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        if not self.base_url:
            raise ValueError(
                "Telephony management API base URL is missing. Set TELEPHONY_BASE_URL or enter it in the UI."
            )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def config(self) -> dict[str, Any]:
        return self._request("GET", "/config/livekit")

    def list_inbound_trunks(self) -> dict[str, Any]:
        return self._request("GET", "/sip/trunks/inbound")

    def list_outbound_trunks(self) -> dict[str, Any]:
        return self._request("GET", "/sip/trunks/outbound")

    def list_dispatch_rules(self) -> dict[str, Any]:
        return self._request("GET", "/sip/dispatch-rules")

    def list_rooms(self) -> dict[str, Any]:
        return self._request("GET", "/livekit/rooms")

    def list_room_participants(self, room_name: str) -> dict[str, Any]:
        return self._request("GET", f"/livekit/rooms/{room_name}/participants")

    def list_room_dispatches(self, room_name: str) -> dict[str, Any]:
        return self._request("GET", f"/livekit/rooms/{room_name}/dispatches")

    def agents_summary(self) -> dict[str, Any]:
        return self._request("GET", "/livekit/agents/summary")

    def delete_trunk(self, trunk_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/sip/trunks/{trunk_id}")

    def delete_dispatch_rule(self, rule_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/sip/dispatch-rules/{rule_id}")

    def delete_provisioning(self, trunk_id: str, dispatch_rule_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"trunk_id": trunk_id}
        if dispatch_rule_id:
            payload["dispatch_rule_id"] = dispatch_rule_id
        return self._request("POST", "/sip/provision/delete", json=payload)

    def update_inbound_trunk(self, trunk_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/sip/trunks/inbound/{trunk_id}", json=payload)

    def update_outbound_trunk(self, trunk_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/sip/trunks/outbound/{trunk_id}", json=payload)

    def update_dispatch_rule(self, rule_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/sip/dispatch-rules/{rule_id}", json=payload)

    def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            json=json,
            timeout=self.timeout_seconds,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LiveKitSIPAPIError(
                f"LiveKit SIP API returned non-JSON response for {path}: HTTP {response.status_code}."
            ) from exc

        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else payload
            raise LiveKitSIPAPIError(f"LiveKit SIP API request failed for {path}: {detail}")

        if not isinstance(payload, dict):
            raise LiveKitSIPAPIError(f"LiveKit SIP API returned unexpected response for {path}.")

        return payload
