from __future__ import annotations

from livekit.plugins import deepgram, google, openai, silero

try:
    from livekit.plugins import elevenlabs
except ImportError:  # pragma: no cover - optional dependency
    elevenlabs = None

from agent_config.schema import AgentConfig


def _require_elevenlabs():
    if elevenlabs is None:
        raise ValueError(
            "ElevenLabs provider selected, but the livekit ElevenLabs plugin is not installed. "
            "Install `livekit-plugins-elevenlabs` first."
        )
    return elevenlabs


def build_llm(config: AgentConfig):
    kwargs = dict(config.llm.kwargs)
    if config.llm.model is not None:
        kwargs.setdefault("model", config.llm.model)

    if config.llm.provider == "azure_openai":
        return openai.LLM.with_azure(**kwargs)
    if config.llm.provider == "openai":
        return openai.LLM(**kwargs)
    raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")


def build_stt(config: AgentConfig):
    kwargs = dict(config.stt.kwargs)
    if config.stt.model is not None:
        kwargs.setdefault("model", config.stt.model)

    if config.stt.provider == "elevenlabs":
        return _require_elevenlabs().STT(**kwargs)
    if config.stt.provider == "azure_openai":
        return openai.STT.with_azure(**kwargs)
    if config.stt.provider == "deepgram":
        return deepgram.STT(**kwargs)
    raise ValueError(f"Unsupported STT provider: {config.stt.provider}")


def build_tts(config: AgentConfig):
    kwargs = dict(config.tts.kwargs)
    if config.tts.model is not None:
        kwargs.setdefault("model", config.tts.model)

    if config.tts.provider == "elevenlabs":
        return _require_elevenlabs().TTS(**kwargs)
    if config.tts.provider == "google":
        return google.TTS(**kwargs)
    if config.tts.provider == "azure_openai":
        return openai.TTS.with_azure(**kwargs)
    if config.tts.provider == "openai":
        return openai.TTS(**kwargs)
    raise ValueError(f"Unsupported TTS provider: {config.tts.provider}")


def build_vad(config: AgentConfig):
    kwargs = {
        "min_speech_duration": 0.07,
        "min_silence_duration": 0.5,
        "activation_threshold": 0.5,
        "force_cpu": True,
        "sample_rate": 16000,
    }
    kwargs.update(config.vad.kwargs)

    if config.vad.provider == "silero":
        return silero.VAD.load(**kwargs)
    raise ValueError(f"Unsupported VAD provider: {config.vad.provider}")
