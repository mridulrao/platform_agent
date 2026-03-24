Kubernetes deployment

1) Build and push images:
`docker build -f Dockerfile.service -t your-registry/platform-agent-service:latest .`
`docker build -f Dockerfile.worker -t your-registry/platform-agent-worker:latest .`

2) Create the namespace, configmap, and secrets from `.env`:
`./.venv/bin/python scripts/render_k8s_runtime_config.py > k8s/runtime-config.generated.yaml`
`kubectl apply -f k8s/runtime-config.generated.yaml`

3) Apply RBAC for the Streamlit service:
`kubectl apply -f k8s/service-rbac.yaml`

4) Deploy the Streamlit service:
`kubectl apply -f k8s/service-deployment.yaml`

5) Render and apply a worker manifest for a saved agent config:
`./.venv/bin/python scripts/render_k8s_worker_manifest.py livekit-agent-test --image your-registry/platform-agent-worker:latest > k8s/livekit-agent-test-worker.yaml`
`kubectl apply -f k8s/livekit-agent-test-worker.yaml`

Notes
- Both service and worker load their shared runtime environment via `envFrom` from `platform-agent-config` and `platform-agent-secrets`.
- `scripts/render_k8s_runtime_config.py` reads values from `.env`.
- Both service and worker read agent configs from the database through `DATABASE_URL`.
- The `voice_virtual_agents` table stores the config JSON and phone number for each LiveKit agent.
- The worker exposes `/` for health and `/worker` for status on the configured worker port.
