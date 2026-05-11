from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from livekit.agents import ToolError, function_tool

from skill_jobs.models import SkillJob
from skill_jobs.store import SkillJobStore


logger = logging.getLogger("livekit-skill-dispatch")
ROOT = Path(__file__).resolve().parents[1]
WORKER_MODULE = "skill_jobs.worker"

try:
    from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
except ImportError:  # pragma: no cover - depends on upgraded livekit-agents
    AsyncRunContext = Any  # type: ignore[assignment]
    AsyncToolset = None  # type: ignore[assignment]

AsyncToolsetBase = AsyncToolset if AsyncToolset is not None else object


class SkillDispatchToolset(AsyncToolsetBase):  # type: ignore[misc, valid-type]
    def __init__(self, *, poll_interval_seconds: float = 2.0) -> None:
        self.store = SkillJobStore()
        self.poll_interval_seconds = poll_interval_seconds
        super().__init__(
            id="skill_dispatch",
            on_duplicate_call="reject",
        )

    @function_tool
    async def generate_sample_agreement(
        self,
        ctx: AsyncRunContext,
        user_name: str,
    ) -> dict[str, Any]:
        """Use this when the user asks to create, draft, write, or generate a sample agreement for a person. This launches a dedicated background worker, monitors progress, and saves the completed agreement locally."""
        normalized_name = user_name.strip()
        if not normalized_name:
            raise ToolError("The user's name is required to generate the agreement.")

        job = SkillJob(
            job_id=f"skill_job_{uuid.uuid4().hex[:12]}",
            skill_name="agreement_generator",
            status="queued",
            payload={"user_name": normalized_name},
            progress_message="The document request has been queued.",
        )
        self.store.create(job)
        self._spawn_worker(job.job_id)

        await ctx.update(
            f"I created a dedicated agreement worker for {normalized_name} and it has started processing."
        )

        last_progress = None
        while True:
            current = self.store.get(job.job_id)

            if current.progress_message and current.progress_message != last_progress:
                last_progress = current.progress_message
                await ctx.update(current.progress_message)

            if current.status == "completed":
                summary = current.metadata.get("summary") if isinstance(current.metadata, dict) else None
                return {
                    "job_id": current.job_id,
                    "status": current.status,
                    "artifact_path": current.artifact_path,
                    "summary": summary,
                }

            if current.status == "needs_input":
                missing = ", ".join(current.required_input) or "additional input"
                await ctx.update(f"The worker needs more input before it can continue: {missing}.")
                return {
                    "job_id": current.job_id,
                    "status": current.status,
                    "required_input": current.required_input,
                }

            if current.status == "failed":
                raise ToolError(current.error_message or "The agreement worker failed.")

            if current.status == "cancelled":
                raise ToolError("The agreement worker was cancelled.")

            await asyncio.sleep(self.poll_interval_seconds)

    def _spawn_worker(self, job_id: str) -> None:
        env = os.environ.copy()
        subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE, "--job-id", job_id],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def build_skill_dispatch_toolset() -> SkillDispatchToolset | None:
    if AsyncToolset is None:
        logger.warning(
            "AsyncToolset is not available in the installed livekit-agents package. "
            "Upgrade to livekit-agents>=1.5.2 to enable background skill dispatch."
        )
        return None
    return SkillDispatchToolset(
        poll_interval_seconds=float(os.getenv("SKILL_JOB_POLL_INTERVAL_SECONDS", "2.0"))
    )
