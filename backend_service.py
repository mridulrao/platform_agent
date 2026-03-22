from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent_config.schema import AgentConfig, SIPProvisionConfig
from agent_config.store import save_agent_config
from livekit_trunks.provision import provision_inbound_sip_for_agent


ROOT = Path(__file__).resolve().parent
RUN_LOG_DIR = ROOT / ".run_logs"


@dataclass
class CreateAgentResult:
    config_path: str
    worker_command: list[str]


@dataclass
class StartAgentResult:
    pid: int | None
    command: list[str]
    log_path: str
    started: bool
    message: str


def build_worker_command(agent_name: str, mode: str = "dev") -> list[str]:
    return [
        sys.executable,
        str(ROOT / "livekit_agents" / "create_agent.py"),
        mode,
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


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start_agent_backend(agent_name: str, mode: str) -> StartAgentResult:
    if mode not in {"dev", "start"}:
        raise ValueError("Mode must be either 'dev' or 'start'.")

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"{agent_name}-{mode}.log"
    env = os.environ.copy()
    env["TARGET_AGENT_NAME"] = agent_name
    command = build_worker_command(agent_name, mode=mode)

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    time.sleep(2)
    if process.poll() is None:
        return StartAgentResult(
            pid=process.pid,
            command=command,
            log_path=str(log_path),
            started=True,
            message=f"Agent '{agent_name}' is running in {mode} mode and appears ready.",
        )

    last_output = ""
    if log_path.exists():
        last_output = log_path.read_text(encoding="utf-8")[-1000:]

    return StartAgentResult(
        pid=process.pid,
        command=command,
        log_path=str(log_path),
        started=False,
        message=(
            f"Agent '{agent_name}' exited during startup."
            + (f"\n\nRecent logs:\n{last_output}" if last_output else "")
        ),
    )


def serialize_result(result: CreateAgentResult) -> str:
    return json.dumps(
        {
            "config_path": result.config_path,
            "worker_command": result.worker_command,
        },
        indent=2,
    )
