from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]

SECRET_KEYS = [
    "DATABASE_URL",
    "DIRECT_URL",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_VERSION",
    "ELEVEN_API_KEY",
    "DEEPGRAM_API_KEY",
]

CONFIGMAP_KEYS = [
    "AGENT_WORKER_IMAGE",
    "K8S_NAMESPACE",
    "KUBECTL_BIN",
    "LIVEKIT_ROOM_PREFIX",
    "LIVEKIT_HIDE_PHONE_NUMBER",
]


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_runtime_config(env_file: Path) -> str:
    values = {k: v for k, v in dotenv_values(env_file).items() if v is not None}
    namespace = values.get("K8S_NAMESPACE", "platform-agent")
    secret_lines = []
    for key in SECRET_KEYS:
        value = values.get(key)
        if value:
            secret_lines.append(f'  {key}: "{_quote(value)}"')

    configmap_lines = []
    for key in CONFIGMAP_KEYS:
        value = values.get(key)
        if value:
            configmap_lines.append(f'  {key}: "{_quote(value)}"')

    secret_body = "\n".join(secret_lines) if secret_lines else "  {}"
    configmap_body = "\n".join(configmap_lines) if configmap_lines else "  {}"

    return f"""apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: platform-agent-config
  namespace: {namespace}
data:
{configmap_body}
---
apiVersion: v1
kind: Secret
metadata:
  name: platform-agent-secrets
  namespace: {namespace}
type: Opaque
stringData:
{secret_body}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Kubernetes namespace/configmap/secret from a .env file.")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    args = parser.parse_args()
    print(render_runtime_config(Path(args.env_file)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
