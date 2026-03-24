Worker deployment helper

Start a worker as an independent background process:
`./.venv/bin/python scripts/deploy_agent_worker.py deploy <agent-name>`

Check worker status:
`./.venv/bin/python scripts/deploy_agent_worker.py status <agent-name>`

Stop a worker:
`./.venv/bin/python scripts/deploy_agent_worker.py stop <agent-name>`

Notes
- The script reads the agent config from `agent_config/<agent-name>.json`.
- It writes state into `.worker_state/<agent-name>.json`.
- It writes logs into `.run_logs/<agent-name>-<mode>.log`.
- Health is checked using the worker's built-in endpoint on `http://127.0.0.1:<worker.port>/`.
