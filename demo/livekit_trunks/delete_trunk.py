from livekit import api
from livekit.protocol import sip as proto_sip
import asyncio
from dotenv import load_dotenv
import os

from livekit.protocol.sip import (
    CreateSIPInboundTrunkRequest,
    CreateSIPDispatchRuleRequest,
    SIPDispatchRule,
    SIPDispatchRuleIndividual,
    SIPInboundTrunkInfo,
    DeleteSIPTrunkRequest
)

load_dotenv()

async def delete_sip_trunk(trunk_id: str):
    """
    Delete a SIP trunk using its trunk ID.
    
    Args:
        trunk_id (str): The ID of the trunk to delete
    """
    try:
        livekit_api = api.LiveKitAPI()

        request = DeleteSIPTrunkRequest(sip_trunk_id = trunk_id)

        await livekit_api.sip.delete_sip_trunk(request)

        print(f"Successfully deleted trunk: {trunk_id}")
        
    except Exception as e:
        print(f"Error deleting trunk: {str(e)}")
    finally:
        await livekit_api.aclose()

# Example usage
async def main():
    await delete_sip_trunk("ST_uTK2xCSNpCFw")

if __name__ == "__main__":
    asyncio.run(main())
