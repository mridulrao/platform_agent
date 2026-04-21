from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from dotenv import load_dotenv
from livekit import api
from livekit.protocol.sip import ListSIPDispatchRuleRequest, ListSIPInboundTrunkRequest

from agent_config.store import list_active_sip_bindings, use_db_backend


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("list-sip-trunks")


def _safe_json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_bound_agent_name(dispatch_rule: Any, binding_by_rule_id: dict[str, dict[str, Any]]) -> str:
    binding = binding_by_rule_id.get(getattr(dispatch_rule, "sip_dispatch_rule_id", ""))
    if binding and binding.get("agent_name"):
        return str(binding["agent_name"])

    room_config = getattr(dispatch_rule, "room_config", None)
    agents = list(getattr(room_config, "agents", []) or [])
    if not agents:
        return ""

    first_agent = agents[0]
    metadata = _safe_json_loads(getattr(first_agent, "metadata", ""))
    return (
        str(metadata.get("config_name") or "")
        or str(metadata.get("target_agent") or "")
        or str(getattr(first_agent, "agent_name", "") or "")
    )


def _format_row(row: dict[str, str], widths: dict[str, int]) -> str:
    return " | ".join(row[key].ljust(widths[key]) for key in widths)


def _resolve_trunk_name(trunk: Any, binding: dict[str, Any] | None) -> str:
    return str(
        getattr(trunk, "name", "") or (binding.get("trunk_name", "") if binding else "") or "-"
    )


async def main() -> None:
    livekit_api = api.LiveKitAPI()

    try:
        trunks_response = await livekit_api.sip.list_inbound_trunk(ListSIPInboundTrunkRequest())
        rules_response = await livekit_api.sip.list_dispatch_rule(ListSIPDispatchRuleRequest())
    finally:
        await livekit_api.aclose()

    bindings = list_active_sip_bindings() if use_db_backend() else []
    binding_by_trunk_id = {
        str(binding.get("trunk_id")): binding for binding in bindings if binding.get("trunk_id")
    }
    binding_by_rule_id = {
        str(binding.get("dispatch_rule_id")): binding
        for binding in bindings
        if binding.get("dispatch_rule_id")
    }

    rules_by_trunk_id: dict[str, list[Any]] = {}
    for rule in rules_response.items:
        for trunk_id in list(getattr(rule, "trunk_ids", []) or []):
            rules_by_trunk_id.setdefault(str(trunk_id), []).append(rule)

    rows: list[dict[str, str]] = []
    for trunk in trunks_response.items:
        trunk_id = str(getattr(trunk, "sip_trunk_id", "") or "")
        trunk_numbers = ", ".join(list(getattr(trunk, "numbers", []) or [])) or "-"
        dispatch_rules = rules_by_trunk_id.get(trunk_id, [])
        binding = binding_by_trunk_id.get(trunk_id)

        if not dispatch_rules:
            rows.append(
                {
                    "trunk_id": trunk_id or "-",
                    "trunk_name": _resolve_trunk_name(trunk, binding),
                    "phone_numbers": trunk_numbers,
                    "dispatch_rule": str(binding.get("dispatch_rule_name", "")) if binding else "-",
                    "agent_name": str(binding.get("agent_name", "")) if binding else "-",
                }
            )
            continue

        for rule in dispatch_rules:
            rows.append(
                {
                    "trunk_id": trunk_id or "-",
                    "trunk_name": _resolve_trunk_name(trunk, binding),
                    "phone_numbers": trunk_numbers,
                    "dispatch_rule": str(getattr(rule, "name", "") or "-"),
                    "agent_name": _extract_bound_agent_name(rule, binding_by_rule_id) or "-",
                }
            )

    rows.sort(key=lambda item: (item["trunk_name"], item["phone_numbers"], item["dispatch_rule"]))

    if not rows:
        logger.info("No inbound SIP trunks found.")
        return

    headers = {
        "trunk_id": "TRUNK ID",
        "trunk_name": "TRUNK NAME",
        "phone_numbers": "PHONE NUMBER(S)",
        "dispatch_rule": "DISPATCH RULE",
        "agent_name": "AGENT NAME",
    }
    widths = {
        key: max(len(headers[key]), max(len(row[key]) for row in rows))
        for key in headers
    }

    print(_format_row(headers, widths))
    print("-+-".join("-" * widths[key] for key in headers))
    for row in rows:
        print(_format_row(row, widths))


if __name__ == "__main__":
    asyncio.run(main())
