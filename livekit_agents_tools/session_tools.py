import re
import json
import time 
import logging
import requests
import aiohttp
import traceback
import asyncio
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List
from livekit import api
from livekit.agents import function_tool, RunContext
from livekit.agents import ChatMessage, ChatContext


@function_tool    
async def end_call(context: RunContext):    
    """    
    Use this function to end the call
    """   
    # Wait for the agent's speech to complete before ending the call  
    await context.wait_for_playout()  
      
    room_name = context.session.userdata.ctx.room.name
  
    async with api.LiveKitAPI() as livekit_api:  
        await livekit_api.room.delete_room(  
            api.DeleteRoomRequest(room=room_name))  
        print("Room deleted successfully")  
  
    return True

@function_tool    
async def transfer_call(context: RunContext):    
    """    
    Use this function to transfer the call
    """   
    # Wait for the agent's speech to complete before ending the call  
    await context.wait_for_playout()  
      
    room_name = context.session.userdata.ctx.room.name
  
    async with api.LiveKitAPI() as livekit_api:  
        await livekit_api.room.delete_room(  
            api.DeleteRoomRequest(room=room_name))  
        print("Room deleted successfully")  
  
    return True