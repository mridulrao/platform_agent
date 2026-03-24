from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".worker_state"
LOG_DIR = ROOT / ".run_logs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_config.store import load_agent_config


load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=False)


@dataclass
class WorkerState:
    agent_name: str
    mode: str
    pid: int
    port: int
    log_path: str
    command: list[str]
    started_at: float


def state_path(agent_name: str) -> Path:
    return STATE_DIR / f"{agent_name}.json"


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_state(agent_name: str) -> WorkerState | None:
    path = state_path(agent_name)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return WorkerState(**payload)


def write_state(state: WorkerState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(state.agent_name).write_text(
        json.dumps(state.__dict__, indent=2),
        encoding="utf-8",
    )


def remove_state(agent_name: str) -> None:
    path = state_path(agent_name)
    if path.exists():
        path.unlink()


def healthcheck(port: int, timeout_seconds: float = 2.0) -> tuple[bool, str]:
    url = f"http://127.0.0.1:{port}/"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
            return response.status == 200, body
    except URLError as exc:
        return False, str(exc.reason)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return False, str(exc)


def build_worker_command(mode: str) -> list[str]:
    python_bin = ROOT / ".venv" / "bin" / "python"
    if python_bin.exists():
        return [str(python_bin), "-m", "livekit_agents.create_agent", mode]
    return [sys.executable, "-m", "livekit_agents.create_agent", mode]


def deploy_worker(agent_name: str, mode: str, wait_seconds: float) -> int:
    if mode not in {"dev", "start"}:
        raise ValueError("Mode must be either 'dev' or 'start'.")

    existing = read_state(agent_name)
    if existing and is_process_running(existing.pid):
        print(
            json.dumps(
                {
                    "agent_name": agent_name,
                    "status": "already_running",
                    "pid": existing.pid,
                    "port": existing.port,
                    "log_path": existing.log_path,
                },
                indent=2,
            )
        )
        return 0

    config = load_agent_config(agent_name)
    port = config.worker.port
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{agent_name}-{mode}.log"
    command = build_worker_command(mode)
    env = os.environ.copy()
    env["TARGET_AGENT_NAME"] = agent_name

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    state = WorkerState(
        agent_name=agent_name,
        mode=mode,
        pid=process.pid,
        port=port,
        log_path=str(log_path),
        command=[f"TARGET_AGENT_NAME={agent_name}", *command],
        started_at=time.time(),
    )
    write_state(state)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            last_output = log_path.read_text(encoding="utf-8")[-2000:] if log_path.exists() else ""
            print(
                json.dumps(
                    {
                        "agent_name": agent_name,
                        "status": "failed",
                        "pid": process.pid,
                        "port": port,
                        "log_path": str(log_path),
                        "recent_logs": last_output,
                    },
                    indent=2,
                )
            )
            return 1

        healthy, detail = healthcheck(port)
        if healthy:
            print(
                json.dumps(
                    {
                        "agent_name": agent_name,
                        "status": "running",
                        "pid": process.pid,
                        "port": port,
                        "health_url": f"http://127.0.0.1:{port}/",
                        "worker_url": f"http://127.0.0.1:{port}/worker",
                        "log_path": str(log_path),
                        "command": state.command,
                    },
                    indent=2,
                )
            )
            return 0

        time.sleep(1)

    print(
        json.dumps(
            {
                "agent_name": agent_name,
                "status": "starting",
                "pid": process.pid,
                "port": port,
                "healthcheck": f"http://127.0.0.1:{port}/",
                "last_health_error": detail,
                "log_path": str(log_path),
            },
            indent=2,
        )
    )
    return 0


def status_worker(agent_name: str) -> int:
    state = read_state(agent_name)
    if state is None:
        print(json.dumps({"agent_name": agent_name, "status": "not_deployed"}, indent=2))
        return 1

    running = is_process_running(state.pid)
    healthy = False
    detail = "process not running"
    if running:
        healthy, detail = healthcheck(state.port)

    print(
        json.dumps(
            {
                "agent_name": agent_name,
                "status": "running" if running else "stopped",
                "healthy": healthy,
                "health_detail": detail,
                "pid": state.pid,
                "port": state.port,
                "log_path": state.log_path,
                "command": state.command,
            },
            indent=2,
        )
    )
    return 0 if running else 1


def stop_worker(agent_name: str, wait_seconds: float) -> int:
    state = read_state(agent_name)
    if state is None:
        print(json.dumps({"agent_name": agent_name, "status": "not_deployed"}, indent=2))
        return 1

    if not is_process_running(state.pid):
        remove_state(agent_name)
        print(json.dumps({"agent_name": agent_name, "status": "already_stopped"}, indent=2))
        return 0

    os.kill(state.pid, signal.SIGTERM)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not is_process_running(state.pid):
            remove_state(agent_name)
            print(json.dumps({"agent_name": agent_name, "status": "stopped"}, indent=2))
            return 0
        time.sleep(1)

    print(
        json.dumps(
            {
                "agent_name": agent_name,
                "status": "stop_timeout",
                "pid": state.pid,
            },
            indent=2,
        )
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy or manage a LiveKit agent worker as an independent process."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Start a detached worker")
    deploy_parser.add_argument("agent_name")
    deploy_parser.add_argument("--mode", default="start", choices=["dev", "start"])
    deploy_parser.add_argument("--wait-seconds", type=float, default=30.0)

    status_parser = subparsers.add_parser("status", help="Show worker status")
    status_parser.add_argument("agent_name")

    stop_parser = subparsers.add_parser("stop", help="Stop a detached worker")
    stop_parser.add_argument("agent_name")
    stop_parser.add_argument("--wait-seconds", type=float, default=20.0)

    args = parser.parse_args()

    if args.command == "deploy":
        return deploy_worker(args.agent_name, args.mode, args.wait_seconds)
    if args.command == "status":
        return status_worker(args.agent_name)
    if args.command == "stop":
        return stop_worker(args.agent_name, args.wait_seconds)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
