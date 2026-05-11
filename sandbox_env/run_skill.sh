#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${ROOT_DIR}/sandbox.env"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

IMAGE_NAME="${SANDBOX_IMAGE_NAME:-voice-agent-skill-sandbox}"
TIMEOUT_SECONDS="${SANDBOX_TIMEOUT_SECONDS:-15}"
MEMORY_LIMIT="${SANDBOX_MEMORY_LIMIT:-256m}"
CPU_LIMIT="${SANDBOX_CPU_LIMIT:-0.50}"
PIDS_LIMIT="${SANDBOX_PIDS_LIMIT:-64}"

SKILL_ARG="${1:-}"
if [[ -z "${SKILL_ARG}" ]]; then
  echo "Usage: ./sandbox_env/run_skill.sh path/to/skill.sh [input.json]" >&2
  exit 2
fi

HOST_SKILL_PATH="$(cd "$(dirname "${SKILL_ARG}")" && pwd)/$(basename "${SKILL_ARG}")"
HOST_INPUT_PATH="${2:-}"

if [[ ! -f "${HOST_SKILL_PATH}" ]]; then
  echo "Skill script not found: ${HOST_SKILL_PATH}" >&2
  exit 2
fi

RUN_ID="${SANDBOX_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${ROOT_DIR}/runs/${RUN_ID}"
mkdir -p "${RUN_ROOT}/workspace"

RESULT_PATH="${RUN_ROOT}/result.json"
LOG_PATH="${RUN_ROOT}/docker.log"

if [[ -n "${HOST_INPUT_PATH}" ]]; then
  cp "${HOST_INPUT_PATH}" "${RUN_ROOT}/workspace/input.json"
fi

docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --tmpfs /home/sandboxuser:rw,nosuid,size=32m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit "${PIDS_LIMIT}" \
  --memory "${MEMORY_LIMIT}" \
  --cpus "${CPU_LIMIT}" \
  -e SKILL_PATH=/workspace/skill.sh \
  -e WORK_DIR=/workspace/run \
  -e RESULT_PATH=/results/result.json \
  -e SKILL_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
  -e DOCUMENT_DATE="$(date '+%Y-%m-%d')" \
  -v "${HOST_SKILL_PATH}:/workspace/skill.sh:ro" \
  -v "${RUN_ROOT}/workspace:/workspace/run:rw" \
  -v "${RUN_ROOT}:/results:rw" \
  "${IMAGE_NAME}" >"${LOG_PATH}" 2>&1

cat "${RESULT_PATH}"
