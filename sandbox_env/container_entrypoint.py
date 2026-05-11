from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    skill_path = os.environ["SKILL_PATH"]
    work_dir = os.environ["WORK_DIR"]
    timeout_seconds = int(os.environ.get("SKILL_TIMEOUT_SECONDS", "15"))
    output_path = Path(os.environ["RESULT_PATH"])

    env = {
        "HOME": "/home/sandboxuser",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "WORK_DIR": work_dir,
        "DOCUMENT_DATE": os.environ.get("DOCUMENT_DATE", ""),
    }

    started_at = time.monotonic()
    try:
        completed = subprocess.run(
            ["/bin/bash", skill_path],
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        payload = {
            "status": "completed",
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }
    except subprocess.TimeoutExpired as exc:
        payload = {
            "status": "timeout",
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_seconds": round(time.monotonic() - started_at, 3),
        }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
