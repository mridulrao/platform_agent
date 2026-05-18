from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProviderConfig(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkerConfig(BaseModel):
    agent_name: str | None = None
    db_proxy_url: str | None = None
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
    transfer_phone_number: str | None = None
    greeting_message: str | None = None

    @field_validator("transfer_phone_number")
    @classmethod
    def validate_transfer_phone_number(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if not re.fullmatch(r"\+[1-9]\d{1,14}", normalized):
            raise ValueError("Transfer phone number must be E.164, for example +14155550123.")
        return normalized


class VADConfig(BaseModel):
    provider: str = "silero"
    kwargs: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    dataset_id: str
    name: str = ""
    rag_config: dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True, "top_k": 5, "min_similarity": 0.5,
    })


class MCPServerConfig(BaseModel):
    mcp_server_id: str
    name: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)


class VvaSkillConfig(BaseModel):
    skill_id: str
    name: str = ""
    has_script: bool = False
    script_name: str = ""
    trigger_hint: str = ""
    instructions: str = ""
    mcp_bindings: dict[str, str] = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)


class VvaAutomationConfig(BaseModel):
    agent_id: str
    name: str = ""
    trigger_hint: str = ""


class AgentConfig(BaseModel):
    name: str = Field(..., min_length=1)
    system_prompt: str = Field(..., min_length=1)
    llm: ProviderConfig
    stt: ProviderConfig
    tts: ProviderConfig
    vad: VADConfig = Field(default_factory=VADConfig)
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    vva_skills: list[VvaSkillConfig] = Field(default_factory=list)
    vva_automations: list[VvaAutomationConfig] = Field(default_factory=list)

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
