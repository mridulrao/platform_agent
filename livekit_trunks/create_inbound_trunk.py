from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from agent_config.schema import SIPProvisionConfig
from livekit_trunks.provision import provision_inbound_sip_for_agent


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create-inbound-trunk")


async def main() -> None:
    agent_name = os.getenv("TARGET_AGENT_NAME")
    if not agent_name:
        raise RuntimeError("Set TARGET_AGENT_NAME to the agent you want to provision.")

    sip_config = SIPProvisionConfig(
        phone_number=os.getenv("INBOUND_PHONE_NUMBER", ""),
        trunk_friendly_name=os.getenv("LIVEKIT_INBOUND_TRUNK_FRIENDLY_NAME", ""),
        dispatch_rule_name=os.getenv("LIVEKIT_INBOUND_DISPATCH_RULE_FRIENDLY_NAME", ""),
        room_prefix=os.getenv("LIVEKIT_ROOM_PREFIX", "inbound"),
        hide_phone_number=os.getenv("LIVEKIT_HIDE_PHONE_NUMBER", "false").lower() == "true",
    )
    result = await provision_inbound_sip_for_agent(agent_name, sip_config)
    logger.info("Provisioned inbound SIP resources: %s", result)


if __name__ == "__main__":
    asyncio.run(main())
