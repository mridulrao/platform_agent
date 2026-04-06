from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProviderConfig(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkerConfig(BaseModel):
    agent_name: str | None = None
    job_memory_warn_mb: int = 2000
    shutdown_process_timeout: float = 80.0
    initialize_process_timeout: int = 540
    num_idle_processes: int = 1
    port: int = 8082


class SessionConfig(BaseModel):
    enable_telephony_noise_cancellation: bool = True
    noise_cancellation_provider: str = "bvc_telephony"
    noise_cancellation_kwargs: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class VADConfig(BaseModel):
    provider: str = "silero"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    name: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    llm: ProviderConfig
    stt: ProviderConfig
    tts: ProviderConfig
    vad: VADConfig = Field(default_factory=VADConfig)
    tools: list[str] = Field(default_factory=list)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        if not normalized.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Agent name must contain only letters, numbers, hyphens, underscores, or spaces."
            )
        return normalized

    @field_validator("worker")
    @classmethod
    def fill_worker_name(cls, value: WorkerConfig, info) -> WorkerConfig:
        if not value.agent_name and info.data.get("name"):
            value.agent_name = info.data["name"]
        return value


class SIPProvisionConfig(BaseModel):
    phone_number: str = Field(..., min_length=3)
    trunk_friendly_name: str = Field(..., min_length=1)
    dispatch_rule_name: str = Field(..., min_length=1)
    room_prefix: str = "inbound"
    hide_phone_number: bool = False
