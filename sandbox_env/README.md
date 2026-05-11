# Skill Sandbox Environment

This directory is intentionally separate from the LiveKit runtime. Its only job is to run bash-based skills inside a constrained container and return structured results.

## Design Goals

- Keep skill execution independent from LiveKit, SIP, and worker deployment code.
- Make skill runs reproducible with a dedicated image and fixed resource limits.
- Treat each skill run like a short-lived job that returns `stdout`, `stderr`, exit code, and duration.
- Default to least privilege: no network, read-only container filesystem, dropped Linux capabilities, non-root user, and tight CPU/memory/PID limits.

## Directory Layout

```text
sandbox_env/
├── Dockerfile
├── build_image.sh
├── container_entrypoint.py
├── run_agreement_skill.sh
├── run_skill.sh
├── sandbox.env.example
├── skills/
│   └── agreement_generator/
│       ├── agents/openai.yaml
│       ├── generate_agreement.sh
│       └── SKILL.md
└── runs/
```

## Execution Model

1. A skill is a plain bash script.
2. `run_skill.sh` launches a fresh Docker container for each execution.
3. The skill script is mounted read-only at `/workspace/skill.sh`.
4. A per-run writable directory is mounted at `/workspace/run`.
5. The container writes a JSON result to `runs/<timestamp>/result.json`.

Example result:

```json
{
  "status": "completed",
  "exit_code": 0,
  "stdout": "Agreement created for Jane Doe\nSaved file: /workspace/run/sample_agreement_jane_doe_.md\n",
  "stderr": "",
  "duration_seconds": 0.014
}
```

The generated agreement file is saved into the same per-run workspace directory as the copied input file.

## Setup

Copy the example config:

```bash
cp sandbox_env/sandbox.env.example sandbox_env/sandbox.env
```

Build the sandbox image:

```bash
./sandbox_env/build_image.sh
```

Run the agreement skill manually:

```bash
./sandbox_env/run_agreement_skill.sh "Jane Doe"
```

Run the same skill through the generic runner:

```bash
./sandbox_env/run_skill.sh \
  sandbox_env/skills/agreement_generator/generate_agreement.sh \
  sandbox_env/sample_input.json
```

## Recommended Skill Contract

Use a simple contract so your voice agent can call skills consistently later:

- Input arrives as files in `${WORK_DIR}`.
- Primary request payload lives at `${WORK_DIR}/input.json`.
- Skills write generated artifacts back into `${WORK_DIR}`.
- Skills send user-visible text to `stdout`.
- Skills send debug or failure detail to `stderr`.
- Success and failure are represented by the process exit code.

## How To Design This Well

Prefer this separation of responsibilities:

- `sandbox_env/`: execution isolation, limits, image, run logs, and result capture.
- `skills/` or future skill packages: business logic only.
- voice agent runtime: decides *when* to call a skill, but not *how* the sandbox works.

That gives you a clean future integration point:

- the LiveKit agent can eventually call `sandbox_env/run_skill.sh`
- or a Python wrapper can invoke the same container contract
- or this folder can become a small internal service later without changing skill scripts

## Security Notes

This is a practical local sandbox, not a perfect security boundary.

- It blocks common accidental damage with no network, read-only root filesystem, and low privileges.
- It is good for trusted or semi-trusted internal bash skills.
- If you need to run truly untrusted code, move to a stronger isolation layer such as Firecracker microVMs, gVisor, Kata Containers, or a remote execution worker pool.
