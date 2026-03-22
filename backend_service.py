from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agent_config.schema import AgentConfig, SIPProvisionConfig
from agent_config.store import save_agent_config
from livekit_trunks.provision import provision_inbound_sip_for_agent


ROOT = Path(__file__).resolve().parent


@dataclass
class CreateAgentResult:
    config_path: str
    worker_command: list[str]


def build_worker_command(agent_name: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "livekit_agents" / "create_agent.py"),
        "--agent-name",
        agent_name,
    ]


def create_agent_backend(config: AgentConfig) -> CreateAgentResult:
    config_path = save_agent_config(config)

    return CreateAgentResult(
        config_path=str(config_path),
        worker_command=build_worker_command(config.name),
    )


def provision_sip_backend(agent_name: str, sip_config: SIPProvisionConfig) -> dict[str, str]:
    import asyncio

    return asyncio.run(provision_inbound_sip_for_agent(agent_name, sip_config))


def launch_agent_worker(agent_name: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["TARGET_AGENT_NAME"] = agent_name
    return subprocess.Popen(
        build_worker_command(agent_name),
        cwd=ROOT,
        env=env,
        text=True,
    )


def serialize_result(result: CreateAgentResult) -> str:
    return json.dumps(
        {
            "config_path": result.config_path,
            "worker_command": result.worker_command,
        },
        indent=2,
    )
