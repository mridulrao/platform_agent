from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_config.store import load_agent_config


def render_manifest(agent_name: str, image: str, replicas: int, namespace: str) -> str:
    config = load_agent_config(agent_name)
    port = config.worker.port
    app_name = f"agent-worker-{agent_name}"
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
        - name: worker
          image: {image}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: {port}
              name: health
          env:
            - name: TARGET_AGENT_NAME
              value: {agent_name}
          envFrom:
            - configMapRef:
                name: platform-agent-config
            - secretRef:
                name: platform-agent-secrets
          readinessProbe:
            httpGet:
              path: /
              port: {port}
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /
              port: {port}
            initialDelaySeconds: 20
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  selector:
    app: {app_name}
  ports:
    - name: health
      port: {port}
      targetPort: {port}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Kubernetes YAML for an agent worker.")
    parser.add_argument("agent_name")
    parser.add_argument("--image", required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--namespace", default="platform-agent")
    args = parser.parse_args()
    print(render_manifest(args.agent_name, args.image, args.replicas, args.namespace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
