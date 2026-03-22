from __future__ import annotations

import json
from pathlib import Path

from agent_config.schema import AgentConfig


CONFIG_DIR = Path(__file__).resolve().parent


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def config_path(agent_name: str) -> Path:
    return ensure_config_dir() / f"{agent_name}.json"


def save_agent_config(config: AgentConfig) -> Path:
    path = config_path(config.name)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_agent_config(agent_name: str) -> AgentConfig:
    path = config_path(agent_name)
    if not path.exists():
        raise FileNotFoundError(f"No agent config found for '{agent_name}' at {path}")
    return AgentConfig.model_validate_json(path.read_text(encoding="utf-8"))


def list_agent_configs() -> list[AgentConfig]:
    ensure_config_dir()
    configs: list[AgentConfig] = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        configs.append(AgentConfig.model_validate_json(path.read_text(encoding="utf-8")))
    return configs


def load_agent_config_from_metadata(metadata: str | None) -> AgentConfig:
    if not metadata:
        raise ValueError("Dispatch metadata is missing; cannot determine target agent.")

    payload = json.loads(metadata)
    agent_name = payload.get("config_name") or payload.get("target_agent")
    if not agent_name:
        raise ValueError("Dispatch metadata did not include config_name or target_agent.")
    return load_agent_config(agent_name)
