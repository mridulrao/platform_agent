#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="${1:-}"

if [[ -z "${USER_NAME}" ]]; then
  echo "Usage: ./sandbox_env/run_agreement_skill.sh \"User Name\"" >&2
  exit 2
fi

TMP_INPUT="$(mktemp "${ROOT_DIR}/agreement_input.XXXXXX.json")"
trap 'rm -f "${TMP_INPUT}"' EXIT

python - <<'PY' "${TMP_INPUT}" "${USER_NAME}"
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
user_name = sys.argv[2]
output_path.write_text(
    json.dumps({"user_name": user_name}, indent=2),
    encoding="utf-8",
)
PY

"${ROOT_DIR}/run_skill.sh" \
  "${ROOT_DIR}/skills/agreement_generator/generate_agreement.sh" \
  "${TMP_INPUT}"
