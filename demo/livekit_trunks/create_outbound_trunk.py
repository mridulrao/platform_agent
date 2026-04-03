from livekit import api  
from livekit.protocol.sip import (  
    CreateSIPOutboundTrunkRequest, 
    CreateSIPDispatchRuleRequest,  
    SIPDispatchRule,  
    SIPDispatchRuleIndividual,  
    SIPOutboundTrunkInfo,           
    SIPDispatchRuleInfo,  
)  
import asyncio  
import os  
from dotenv import load_dotenv  
import logging  
  
# Configure logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
load_dotenv()  
  
  
def get_required_env(var_name: str, default: str = "") -> str:  
    value = os.getenv(var_name)  
    if not value and not default:  
        raise RuntimeError(f"Missing required environment variable: {var_name}")  
    return value or default  
  
  
async def setup_livekit_trunk_dynamic():  
    livekit_api = api.LiveKitAPI()  
    try:  
        # Step 1: Create outbound trunk  
        logger.info("Creating outbound trunk...")  
        friendly_name = get_required_env("LIVEKIT_OUTBOUND_TRUNK_FRIENDLY_NAME")  
          
        # Use SIPOutboundTrunkInfo for outbound trunks  
        trunk = SIPOutboundTrunkInfo(  
            name=friendly_name,  
            address=get_required_env("LIVEKIT_SIP_ADDRESS"),  # Required for outbound  
            numbers=[  
                get_required_env("OUTBOUND_PHONE_NUMBER"),  
            ],  
        )  
          
        # Use CreateSIPOutboundTrunkRequest for outbound trunks  
        request = CreateSIPOutboundTrunkRequest(trunk=trunk)  
        trunk_response = await livekit_api.sip.create_sip_outbound_trunk(request)  # Changed method  
        logger.info(f"Outbound Trunk Created: {trunk_response.sip_trunk_id}")  
  
    except Exception as e:  
        logger.error(f"Error creating outbound trunk: {str(e)}")  
        logger.error(  
            "Outbound trunk creation failed. For self-hosted LiveKit, SIP requires Redis "  
            "and a SIP-enabled server configuration before this API will work."  
        )  
        return  
  
    try:  
        logger.info("\nCreating dynamic dispatch rule")  
        friendly_name_dispatch = get_required_env(  
            "LIVEKIT_OUTBOUND_DISPATCH_RULE_FRIENDLY_NAME"  
        )  
        dispatch_request = CreateSIPDispatchRuleRequest(  
            dispatch_rule=SIPDispatchRuleInfo(  
                name=friendly_name_dispatch,  
                rule=SIPDispatchRule(  
                    dispatch_rule_individual=SIPDispatchRuleIndividual(  
                        room_prefix="outbound_"  
                    )  
                ),  
                room_config=api.RoomConfiguration(  
                    agents=[  
                        api.RoomAgentDispatch(  
                            agent_name="outbound_agent",  
                            metadata='{"target_agent": "outbound_agent"}'  
                        )  
                    ]  
                ),  
                trunk_ids=[trunk_response.sip_trunk_id],  # Use outbound trunk ID  
                hide_phone_number=False,  
            )  
        )  
  
        dispatch_response = await livekit_api.sip.create_sip_dispatch_rule(dispatch_request)  
        logger.info(f"Successfully created dynamic dispatch rule: {dispatch_response}")  
    except Exception as e:  
        logger.error(f"Error creating dispatch rule: {str(e)}")  
    finally:  
        await livekit_api.aclose()  
  
  
async def main():  
    logger.info("Setting up LiveKit SIP outbound trunk with dynamic room dispatch...")  
    await setup_livekit_trunk_dynamic()  
    logger.info("\nSetup complete!")  
  
  
if __name__ == "__main__":  
    asyncio.run(main())