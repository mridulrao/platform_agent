from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from skill_jobs.store import SkillJobStore
from skill_jobs.models import utc_now_iso


logger = logging.getLogger("skill-job-worker")
ROOT = Path(__file__).resolve().parents[1]
SANDBOX_ROOT = ROOT / "sandbox_env"
RUN_SKILL_SCRIPT = SANDBOX_ROOT / "run_skill.sh"

SKILL_DEFINITIONS = {
    "agreement_generator": {
        "script_path": SANDBOX_ROOT / "skills" / "agreement_generator" / "generate_agreement.sh",
        "required_fields": ["user_name"],
    },
}


def _run_root_for_job(job_id: str) -> Path:
    return SANDBOX_ROOT / "runs" / job_id


def _find_artifact(run_root: Path) -> Path | None:
    workspace = run_root / "workspace"
    if not workspace.exists():
        return None
    candidates = [
        path for path in workspace.iterdir()
        if path.is_file() and path.name != "input.json"
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[0]


async def process_job(job_id: str) -> int:
    store = SkillJobStore()
    job = store.get(job_id)
    skill_def = SKILL_DEFINITIONS.get(job.skill_name)
    if skill_def is None:
        store.update(
            job_id,
            status="failed",
            error_message=f"Unknown skill: {job.skill_name}",
            completed_at=utc_now_iso(),
        )
        return 1

    missing_fields = [
        field for field in skill_def["required_fields"]
        if not str(job.payload.get(field, "")).strip()
    ]
    if missing_fields:
        store.update(
            job_id,
            status="needs_input",
            progress_message="The worker is waiting for additional input.",
            required_input=missing_fields,
        )
        return 0

    run_root = _run_root_for_job(job_id)
    input_path = run_root / "worker_input.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(job.payload, indent=2), encoding="utf-8")

    store.update(
        job_id,
        status="running",
        progress_message="The dedicated document worker is running the skill.",
        started_at=utc_now_iso(),
    )

    env = os.environ.copy()
    env["SANDBOX_RUN_ID"] = job_id

    process = await asyncio.create_subprocess_exec(
        str(RUN_SKILL_SCRIPT),
        str(skill_def["script_path"]),
        str(input_path),
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    store.update(job_id, worker_pid=process.pid)
    stdout_raw, stderr_raw = await process.communicate()
    stdout_text = stdout_raw.decode("utf-8", errors="replace").strip()
    stderr_text = stderr_raw.decode("utf-8", errors="replace").strip()

    if process.returncode != 0:
        store.update(
            job_id,
            status="failed",
            error_message=stderr_text or stdout_text or "Skill worker failed.",
            completed_at=utc_now_iso(),
        )
        return process.returncode

    result = json.loads(stdout_text)
    if result.get("status") == "timeout":
        store.update(
            job_id,
            status="failed",
            error_message="Skill execution timed out.",
            completed_at=utc_now_iso(),
            metadata={"result": result},
        )
        return 124

    artifact = _find_artifact(run_root)
    artifact_path = str(artifact) if artifact is not None else None
    summary = result.get("stdout", "").strip() or "Skill completed successfully."
    store.update(
        job_id,
        status="completed",
        progress_message="The document worker finished successfully.",
        artifact_path=artifact_path,
        completed_at=utc_now_iso(),
        metadata={"result": result, "summary": summary},
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(process_job(args.job_id)))


if __name__ == "__main__":
    main()
