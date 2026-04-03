from livekit import api    
from livekit.protocol.sip import (    
    CreateSIPInboundTrunkRequest,  # Changed from outbound  
    CreateSIPDispatchRuleRequest,    
    SIPDispatchRule,    
    SIPDispatchRuleIndividual,    
    SIPInboundTrunkInfo,           # Changed from outbound  
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
        # Step 1: Create inbound trunk  
        logger.info("Creating inbound trunk...")    
        friendly_name = get_required_env("LIVEKIT_INBOUND_TRUNK_FRIENDLY_NAME")  # Updated env var  
            
        # Use SIPInboundTrunkInfo for inbound trunks  
        trunk = SIPInboundTrunkInfo(    
            name=friendly_name,    
            numbers=[    
                get_required_env("INBOUND_PHONE_NUMBER"),  # Updated env var  
            ],   
        )    
            
        # Use CreateSIPInboundTrunkRequest for inbound trunks  
        request = CreateSIPInboundTrunkRequest(trunk=trunk)    
        trunk_response = await livekit_api.sip.create_sip_inbound_trunk(request)  # Changed method  
        logger.info(f"Inbound Trunk Created: {trunk_response.sip_trunk_id}")    
    
    except Exception as e:    
        logger.error(f"Error creating inbound trunk: {str(e)}")    
        logger.error(    
            "Inbound trunk creation failed. For self-hosted LiveKit, SIP requires Redis "    
            "and a SIP-enabled server configuration before this API will work."    
        )    
        return    
    
    try:    
        logger.info("\nCreating dynamic dispatch rule")    
        friendly_name_dispatch = get_required_env(    
            "LIVEKIT_INBOUND_DISPATCH_RULE_FRIENDLY_NAME"  # Updated env var  
        )    
        dispatch_request = CreateSIPDispatchRuleRequest(    
            dispatch_rule=SIPDispatchRuleInfo(    
                name=friendly_name_dispatch,    
                rule=SIPDispatchRule(    
                    dispatch_rule_individual=SIPDispatchRuleIndividual(    
                        room_prefix="inbound_"  # Updated prefix  
                    )    
                ),    
                room_config=api.RoomConfiguration(    
                    agents=[    
                        api.RoomAgentDispatch(    
                            agent_name="inbound_agent",  # Updated agent name  
                            metadata='{"target_agent": "inbound_agent"}'    
                        )    
                    ]    
                ),    
                trunk_ids=[trunk_response.sip_trunk_id],  # Use inbound trunk ID    
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
    logger.info("Setting up LiveKit SIP inbound trunk with dynamic room dispatch...")  # Updated message  
    await setup_livekit_trunk_dynamic()    
    logger.info("\nSetup complete!")    
    
    
if __name__ == "__main__":    
    asyncio.run(main())