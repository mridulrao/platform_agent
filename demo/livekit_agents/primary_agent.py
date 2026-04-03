from __future__ import annotations  
import asyncio  
import logging  
import json  
from dataclasses import dataclass  
from dotenv import load_dotenv  
  
from livekit.agents import (  
    AgentSession,  
    AutoSubscribe,  
    JobContext,  
    JobProcess,  
    RoomInputOptions,  # Added import  
    WorkerOptions,  
    cli,  
)  
from livekit.plugins import openai, deepgram, google, silero, noise_cancellation, cartesia 
from livekit.agents.voice import Agent, ModelSettings, RunContext

load_dotenv()  
  
logger = logging.getLogger("demo-agent")  
logger.setLevel(logging.INFO)   
  
  
def prewarm(proc: JobProcess):  
    proc.userdata["vad"] = silero.VAD.load(  
        min_speech_duration=0.07,  
        min_silence_duration=0.7,  
        activation_threshold=0.5,  
        force_cpu=True,  
        sample_rate=16000,  
    )  

# Define language-specific agent  
class EnglishAgent(Agent):  
    """English language support agent."""  
  
    def __init__(self, instructions: str, vad_instance) -> None:  
        super().__init__(  
            instructions=instructions,  
            stt=deepgram.STT(
                model="nova-3",
                language="en-IN",
                detect_language=False,
                interim_results=True,
            ),  
            llm=openai.LLM(model="gpt-4.1-mini"),  
            # tts=google.TTS(  
            #     gender="female",  
            #     language="en-IN",  
            #     voice_name="en-IN-Chirp3-HD-Aoede",  # male - en-IN-Chirp3-HD-Achird
            #     credentials_file="google_creds.json",  
            # ),  
            tts=cartesia.TTS(
                model="sonic-3",
                voice="7ea5e9c2-b719-4dc3-b870-5ba5f14d31d8",
            ), 
            vad=vad_instance,  
        )  
  
    async def on_enter(self):  
        logger.info("Agent activated")  
        await self.session.say(f"Hi, how can I help you today?")
  
  
async def entrypoint(ctx: JobContext):  
    logger.info(f"Worker triggered for room: {ctx.room.name}")  
  
    # Connect to the room and wait for the caller to join  
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)  

  
    # Initialize session & agent  
    session = AgentSession()  
    agent = EnglishAgent(  
        instructions="You are a helpful assistant. Greet the caller and ask how you can help.",  
        vad_instance=ctx.proc.userdata["vad"],  
    )  
  
    # Start the agent session in this room with telephone noise cancellation  
    await session.start(  
        agent=agent,   
        room=ctx.room
    )  
  
  
def main():  
    cli.run_app(  
        WorkerOptions(  
            entrypoint_fnc=entrypoint,  
            prewarm_fnc=prewarm,  
            agent_name="inbound_agent",  
            job_memory_warn_mb=2000,  
            shutdown_process_timeout=80.0,  
            initialize_process_timeout=540,  
            num_idle_processes=1,
            port=8082,
        )  
    )  
  
  
if __name__ == "__main__":  
    main()
