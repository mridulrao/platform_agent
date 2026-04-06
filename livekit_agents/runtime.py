from __future__ import annotations

import importlib
import logging
from typing import Any

from livekit import rtc
from livekit.agents import AgentSession, AutoSubscribe, JobContext, JobProcess, RoomInputOptions
from livekit.plugins import noise_cancellation

from agent_config.schema import AgentConfig
from livekit_agents.base_agent import BaseAgent
from livekit_agents.factory import build_llm, build_stt, build_tts, build_vad
from utils.noise_cancelation.webrtc_ns_module import build_webrtc_noise_canceller
from utils.proxy.db_proxy import DBProxyClient
from utils.session.in_call import VVASessionInfo

logger = logging.getLogger("configurable-livekit-agent")


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad_ready"] = True


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


def _participant_attributes(participant: rtc.RemoteParticipant | None) -> dict[str, str]:
    attributes = getattr(participant, "attributes", None) if participant else None
    if not isinstance(attributes, dict):
        return {}
    return {str(key): str(value) for key, value in attributes.items()}


def _is_sip_participant(participant: rtc.RemoteParticipant | None) -> bool:
    if participant is None:
        return False
    return getattr(participant, "kind", None) == rtc.ParticipantKind.PARTICIPANT_KIND_SIP


def _build_session_info(
    ctx: JobContext,
    participant: rtc.RemoteParticipant | None,
) -> VVASessionInfo:
    attributes = _participant_attributes(participant)
    session_info = VVASessionInfo()
    participant_kind = "sip" if _is_sip_participant(participant) else "webrtc"

    session_info.set_connection_details(
        room_name=ctx.room.name,
        participant_kind=participant_kind,
        attributes=attributes,
    )
    session_info.db_proxy = DBProxyClient()
    return session_info


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

    participant = await ctx.wait_for_participant()
    session_info = _build_session_info(ctx, participant)
    session = AgentSession(userdata=session_info)
    session_info.register_conversation_listeners(session)
    agent = ConfigurableVoiceAgent(
        config=config,
        vad_instance=build_vad(config),
    )

    room_input_options = None
    if config.session.enable_telephony_noise_cancellation:
        provider = config.session.noise_cancellation_provider.strip().lower()
        if provider in {"bvc", "bvc_telephony", "bvctelephony", "livekit_bvc"}:
            processor = noise_cancellation.BVCTelephony()
        elif provider in {"webrtc", "webrtc_noise_gain", "custom_webrtc_noise_gain"}:
            processor = build_webrtc_noise_canceller(config.session.noise_cancellation_kwargs)
        else:
            raise ValueError(
                "Unsupported noise_cancellation_provider "
                f"'{config.session.noise_cancellation_provider}'."
            )
        room_input_options = RoomInputOptions(
            noise_cancellation=processor
        )

    start_kwargs = dict(config.session.kwargs)
    if room_input_options is not None:
        start_kwargs["room_input_options"] = room_input_options

    await session.start(
        agent=agent,
        room=ctx.room,
        **start_kwargs,
    )
