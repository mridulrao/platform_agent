#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${WORK_DIR}/input.json"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Missing input file at ${INPUT_FILE}" >&2
  exit 2
fi

USER_NAME="$(python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/workspace/run/input.json").read_text(encoding="utf-8"))
name = str(payload.get("user_name", "")).strip()
print(name)
PY
)"

if [[ -z "${USER_NAME}" ]]; then
  echo "Input is missing required field: user_name" >&2
  exit 2
fi

SAFE_NAME="$(printf '%s' "${USER_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
DOCUMENT_DATE="${DOCUMENT_DATE:-$(date '+%Y-%m-%d')}"
OUTPUT_FILE="${WORK_DIR}/sample_agreement_${SAFE_NAME}.md"

cat > "${OUTPUT_FILE}" <<EOF
# Sample Service Agreement

Prepared on: ${DOCUMENT_DATE}

This Sample Service Agreement ("Agreement") is made between **Example Company** and **${USER_NAME}**.

## 1. Purpose

This Agreement is intended as a simple sample document that can be shared with ${USER_NAME} for review.

## 2. Services

Example Company agrees to provide standard support, onboarding assistance, and agreed project deliverables to ${USER_NAME}.

## 3. Term

This Agreement begins on ${DOCUMENT_DATE} and continues until the sample engagement is completed or either party chooses to end it with written notice.

## 4. Fees

Any commercial terms, pricing, or payment schedules should be added before this Agreement is used in a production setting.

## 5. Confidentiality

Both parties should treat shared business, technical, and personal information as confidential unless disclosure is required by law.

## 6. Limitation

This is a sample agreement generated for demonstration purposes only and is not legal advice.

## 7. Acceptance

Accepted by:

- Example Company Representative: ____________________
- ${USER_NAME}: ____________________

EOF

printf 'Agreement created for %s\n' "${USER_NAME}"
printf 'Saved file: %s\n' "${OUTPUT_FILE}"
