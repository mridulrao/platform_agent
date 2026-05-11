from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from skill_jobs.models import SkillJob, utc_now_iso


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_DIR = ROOT / "sandbox_env" / "job_state"


class SkillJobStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        env_root = os.getenv("SKILL_JOB_STORE_DIR")
        self.root_dir = Path(root_dir or env_root or DEFAULT_STORE_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def job_path(self, job_id: str) -> Path:
        return self.root_dir / f"{job_id}.json"

    def create(self, job: SkillJob) -> SkillJob:
        path = self.job_path(job.job_id)
        if path.exists():
            raise FileExistsError(f"Job already exists: {job.job_id}")
        self._write(path, job.to_dict())
        return job

    def get(self, job_id: str) -> SkillJob:
        path = self.job_path(job_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SkillJob.from_dict(payload)

    def update(self, job_id: str, **changes: Any) -> SkillJob:
        job = self.get(job_id)
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = utc_now_iso()
        self._write(self.job_path(job_id), job.to_dict())
        return job

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        temp_path.replace(path)

