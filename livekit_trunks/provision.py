from __future__ import annotations

import json
import logging

from livekit import api
from livekit.protocol.sip import (
    CreateSIPDispatchRuleRequest,
    CreateSIPInboundTrunkRequest,
    DeleteSIPTrunkRequest,
    ListSIPInboundTrunkRequest,
    SIPDispatchRule,
    SIPDispatchRuleIndividual,
    SIPDispatchRuleInfo,
    SIPInboundTrunkInfo,
)

from agent_config.schema import SIPProvisionConfig


logger = logging.getLogger("livekit-sip-provision")


async def provision_inbound_sip_for_agent(
    agent_name: str, sip_config: SIPProvisionConfig, dispatch_agent_name: str | None = None
) -> dict[str, str]:
    livekit_api = api.LiveKitAPI()
    resolved_dispatch_agent_name = dispatch_agent_name or agent_name

    try:
        # Clean up any existing trunks for this phone number to avoid conflicts.
        # Stale trunks can be left behind from failed previous provisioning attempts.
        try:
            existing_trunks = await livekit_api.sip.list_sip_inbound_trunk(ListSIPInboundTrunkRequest())
            for trunk in existing_trunks.items:
                if sip_config.phone_number in (trunk.numbers or []):
                    logger.info("Deleting stale inbound trunk %s for number %s", trunk.sip_trunk_id, sip_config.phone_number)
                    await livekit_api.sip.delete_sip_trunk(DeleteSIPTrunkRequest(sip_trunk_id=trunk.sip_trunk_id))
        except Exception as cleanup_err:
            logger.warning("Failed to clean up existing trunks: %s", cleanup_err)

        trunk_request = CreateSIPInboundTrunkRequest(
            trunk=SIPInboundTrunkInfo(
                name=sip_config.trunk_friendly_name,
                numbers=[sip_config.phone_number],
            )
        )
        trunk_response = await livekit_api.sip.create_sip_inbound_trunk(trunk_request)

        dispatch_request = CreateSIPDispatchRuleRequest(
            dispatch_rule=SIPDispatchRuleInfo(
                name=sip_config.dispatch_rule_name,
                rule=SIPDispatchRule(
                    dispatch_rule_individual=SIPDispatchRuleIndividual(
                        room_prefix=f"{sip_config.room_prefix}_"
                    )
                ),
                room_config=api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name=resolved_dispatch_agent_name,
                            metadata=json.dumps(
                                {
                                    "target_agent": resolved_dispatch_agent_name,
                                    "config_name": agent_name,
                                }
                            ),
                        )
                    ]
                ),
                trunk_ids=[trunk_response.sip_trunk_id],
                hide_phone_number=sip_config.hide_phone_number,
            )
        )
        dispatch_response = await livekit_api.sip.create_sip_dispatch_rule(dispatch_request)

        return {
            "agent_name": agent_name,
            "dispatch_agent_name": resolved_dispatch_agent_name,
            "trunk_id": trunk_response.sip_trunk_id,
            "dispatch_rule_id": getattr(dispatch_response, "sip_dispatch_rule_id", ""),
        }
    finally:
        await livekit_api.aclose()
