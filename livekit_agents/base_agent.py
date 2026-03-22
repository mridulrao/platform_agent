import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import AsyncIterable, Dict, List, Optional

from livekit import rtc
from livekit.agents import (
    ChatContext,
    ChatMessage,
    FunctionTool,
    JobProcess,
    llm,
    stt,
    tts,
)
from livekit.agents.voice import Agent, ModelSettings, RunContext

logger = logging.getLogger("BaseAgent")


class BaseAgent(Agent):
    def __init__(self, agent_name: str, *args, **kwargs):
        kwargs.setdefault('allow_interruptions', True) 
        super().__init__(*args, **kwargs) 

        self.filler_words = [
            "Okay, ",
            "All right, ",
            "Got it, ",
            "Understood, ",
            "Makes sense, ",
            "Sure, ",
            "Right, ",
            "Just a moment, ",
        ]

    async def play_filler_words(self, duration: float = 1.0, interval: float = 0.5):
        """
        Play filler words during processing delays to maintain engagement
        """
        # Check if we should interrupt filler words  
        if hasattr(self.session, '_current_speech') and self.session._current_speech is not None:  
            return  
          
        filler = random.choice(self.filler_words)  
        # Use allow_interruptions=True for filler words so they can be stopped  
        self.session.say(filler, add_to_chat_ctx=False, allow_interruptions=True)


    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        # Only play filler words if no current speech is active  
        if not hasattr(self.session, '_current_speech') or self.session._current_speech is None:  
            await self.play_filler_words()