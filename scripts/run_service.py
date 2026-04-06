from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


DB_PROXY_PORT = int(os.getenv("DB_PROXY_PORT", "8000"))
DB_PROXY_HOST = os.getenv("DB_PROXY_HOST", "0.0.0.0")
STREAMLIT_PORT = os.getenv("STREAMLIT_SERVER_PORT", "8501")
STREAMLIT_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
DB_PROXY_HEALTH_URL = f"http://127.0.0.1:{DB_PROXY_PORT}/db-proxy/health"


def _wait_for_db_proxy(process: subprocess.Popen, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "not checked"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"DB proxy exited before becoming healthy with code {process.returncode}.")
        try:
            with urlopen(DB_PROXY_HEALTH_URL, timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"unexpected status {response.status}"
        except URLError as exc:
            last_error = str(exc.reason)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"DB proxy did not become healthy at {DB_PROXY_HEALTH_URL}: {last_error}")


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    processes: list[subprocess.Popen] = []

    db_proxy_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:build_app",
            "--factory",
            "--host",
            DB_PROXY_HOST,
            "--port",
            str(DB_PROXY_PORT),
        ]
    )
    processes.append(db_proxy_process)

    def handle_signal(signum, _frame) -> None:
        _terminate(processes)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    _wait_for_db_proxy(db_proxy_process)

    streamlit_process = subprocess.Popen(
        [
            "streamlit",
            "run",
            "streamlit_app.py",
            f"--server.port={STREAMLIT_PORT}",
            f"--server.address={STREAMLIT_ADDRESS}",
        ]
    )
    processes.append(streamlit_process)

    while True:
        for process in processes:
            return_code = process.poll()
            if return_code is not None:
                _terminate([p for p in processes if p is not process])
                return return_code
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
