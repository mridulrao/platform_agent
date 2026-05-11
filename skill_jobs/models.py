from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


JobStatus = Literal[
    "queued",
    "running",
    "needs_input",
    "completed",
    "failed",
    "cancelled",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SkillJob:
    job_id: str
    skill_name: str
    status: JobStatus
    payload: dict[str, Any] = field(default_factory=dict)
    progress_message: str | None = None
    error_message: str | None = None
    artifact_path: str | None = None
    required_input: list[str] = field(default_factory=list)
    worker_pid: int | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillJob":
        return cls(**payload)

