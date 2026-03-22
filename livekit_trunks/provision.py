from __future__ import annotations

import json
import logging

from livekit import api
from livekit.protocol.sip import (
    CreateSIPDispatchRuleRequest,
    CreateSIPInboundTrunkRequest,
    SIPDispatchRule,
    SIPDispatchRuleIndividual,
    SIPDispatchRuleInfo,
    SIPInboundTrunkInfo,
)

from agent_config.schema import SIPProvisionConfig


logger = logging.getLogger("livekit-sip-provision")


async def provision_inbound_sip_for_agent(
    agent_name: str, sip_config: SIPProvisionConfig
) -> dict[str, str]:
    livekit_api = api.LiveKitAPI()

    try:
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
                            agent_name=agent_name,
                            metadata=json.dumps(
                                {
                                    "target_agent": agent_name,
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
            "trunk_id": trunk_response.sip_trunk_id,
            "dispatch_rule_id": getattr(dispatch_response, "sip_dispatch_rule_id", ""),
        }
    finally:
        await livekit_api.aclose()
