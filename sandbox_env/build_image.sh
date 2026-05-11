#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${SANDBOX_IMAGE_NAME:-voice-agent-skill-sandbox}"

docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"
