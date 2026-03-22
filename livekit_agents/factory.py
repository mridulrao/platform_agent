from __future__ import annotations

from livekit.plugins import deepgram, google, openai

from agent_config.schema import AgentConfig


def build_llm(config: AgentConfig):
    kwargs = dict(config.llm.kwargs)
    if config.llm.model is not None:
        kwargs.setdefault("model", config.llm.model)

    if config.llm.provider == "openai":
        return openai.LLM(**kwargs)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")


def build_stt(config: AgentConfig):
    kwargs = dict(config.stt.kwargs)
    if config.stt.model is not None:
        kwargs.setdefault("model", config.stt.model)

    if config.stt.provider == "deepgram":
        return deepgram.STT(**kwargs)
    raise ValueError(f"Unsupported STT provider: {config.stt.provider}")


def build_tts(config: AgentConfig):
    kwargs = dict(config.tts.kwargs)
    if config.tts.model is not None:
        kwargs.setdefault("model", config.tts.model)

    if config.tts.provider == "google":
        return google.TTS(**kwargs)
    if config.tts.provider == "openai":
        return openai.TTS(**kwargs)
    raise ValueError(f"Unsupported TTS provider: {config.tts.provider}")
