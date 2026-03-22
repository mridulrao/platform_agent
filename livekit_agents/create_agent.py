from __future__ import annotations

import argparse
import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli

from agent_config.store import load_agent_config
from livekit_agents.runtime import prewarm, run_agent_session


load_dotenv()

logger = logging.getLogger("configurable-livekit-worker")
logger.setLevel(logging.INFO)


def resolve_target_agent_name(ctx: JobContext) -> str:
    explicit = os.getenv("TARGET_AGENT_NAME")
    if explicit:
        return explicit

    for participant in ctx.room.remote_participants.values():
        metadata = getattr(participant, "metadata", None)
        if not metadata:
            continue
        try:
            payload = json.loads(metadata)
        except json.JSONDecodeError:
            continue
        candidate = payload.get("config_name") or payload.get("target_agent")
        if candidate:
            return candidate

    raise RuntimeError(
        "Could not determine target agent name from TARGET_AGENT_NAME or participant metadata."
    )


async def entrypoint(ctx: JobContext) -> None:
    agent_name = resolve_target_agent_name(ctx)
    config = load_agent_config(agent_name)
    await run_agent_session(ctx, config)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent-name", dest="agent_name")
    args, _ = parser.parse_known_args()
    if args.agent_name:
        os.environ["TARGET_AGENT_NAME"] = args.agent_name

    target_agent_name = os.getenv("TARGET_AGENT_NAME", "configurable_agent")
    config = load_agent_config(target_agent_name)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=config.worker.agent_name or config.name,
            job_memory_warn_mb=config.worker.job_memory_warn_mb,
            shutdown_process_timeout=config.worker.shutdown_process_timeout,
            initialize_process_timeout=config.worker.initialize_process_timeout,
            num_idle_processes=config.worker.num_idle_processes,
            port=config.worker.port,
        )
    )


if __name__ == "__main__":
    main()
