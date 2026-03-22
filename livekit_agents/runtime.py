from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from livekit.agents import AgentSession, AutoSubscribe, JobContext, JobProcess, RoomInputOptions
from livekit.plugins import noise_cancellation, silero

from agent_config.schema import AgentConfig
from livekit_agents.base_agent import BaseAgent
from livekit_agents.factory import build_llm, build_stt, build_tts


logger = logging.getLogger("configurable-livekit-agent")


@dataclass
class SessionUserData:
    ctx: JobContext


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.07,
        min_silence_duration=0.5,
        activation_threshold=0.5,
        force_cpu=True,
        sample_rate=16000,
    )


def load_tools(tool_paths: list[str]) -> list[Any]:
    loaded_tools: list[Any] = []
    for tool_path in tool_paths:
        module_path, sep, attr_name = tool_path.rpartition(".")
        if not sep:
            raise ValueError(
                f"Invalid tool path '{tool_path}'. Use a full import path like "
                "'livekit_agents_tools.session_tools.end_call'."
            )
        module = importlib.import_module(module_path)
        loaded_tools.append(getattr(module, attr_name))
    return loaded_tools


class ConfigurableVoiceAgent(BaseAgent):
    def __init__(self, config: AgentConfig, vad_instance) -> None:
        super().__init__(
            agent_name=config.name,
            instructions=config.system_prompt,
            stt=build_stt(config),
            llm=build_llm(config),
            tts=build_tts(config),
            vad=vad_instance,
            tools=load_tools(config.tools),
        )
        self.config = config

    async def on_enter(self) -> None:
        await self.session.say(f"Hello, this is {self.config.name.replace('_', ' ')}.")


async def run_agent_session(ctx: JobContext, config: AgentConfig) -> None:
    logger.info("Starting agent '%s' for room '%s'", config.name, ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(userdata=SessionUserData(ctx=ctx))
    agent = ConfigurableVoiceAgent(
        config=config,
        vad_instance=ctx.proc.userdata["vad"],
    )

    room_input_options = None
    if config.session.enable_telephony_noise_cancellation:
        room_input_options = RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony()
        )

    start_kwargs = dict(config.session.kwargs)
    if room_input_options is not None:
        start_kwargs["room_input_options"] = room_input_options

    await session.start(
        agent=agent,
        room=ctx.room,
        **start_kwargs,
    )
