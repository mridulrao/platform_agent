---
name: agreement-generator
description: Generate a short sample agreement document for a named user, save it locally, and return the file path. Use when a worker needs to create a simple shareable agreement from a user name.
---

# Agreement Generator

Use this skill when the task is to create a lightweight agreement document for a user and save it as a local file.

## Inputs

- Required: `user_name`
- Source: `${WORK_DIR}/input.json`

Expected input shape:

```json
{
  "user_name": "Jane Doe"
}
```

## Output Requirements

- Create exactly one agreement document in `${WORK_DIR}`.
- Name the file `sample_agreement_<normalized_user_name>.md`.
- Mention the user's name in:
  - the opening paragraph
  - the services section
  - the signature block
- Include the current date.
- Make it clear the document is a sample and not legal advice.

## Agreement Structure

Generate a concise agreement with these sections in order:

1. Title
2. Prepared date
3. Introductory paragraph naming both parties
4. Purpose
5. Services
6. Term
7. Fees
8. Confidentiality
9. Limitation / sample disclaimer
10. Acceptance / signature lines

## Writing Guidance

- Keep the tone professional and neutral.
- Keep the agreement short enough to review quickly.
- Do not invent pricing, legal jurisdiction, or detailed obligations beyond a safe sample template.
- If the user name is missing or blank, fail clearly instead of generating a broken document.

## Execution Notes

- The bash entrypoint for this skill is `generate_agreement.sh`.
- The script should print a short success message and the saved file path to `stdout`.
- Write only inside `${WORK_DIR}`.
