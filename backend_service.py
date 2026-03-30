from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agent_config.schema import AgentConfig, SIPProvisionConfig
from agent_config.store import save_agent_config
from livekit_trunks.provision import provision_inbound_sip_for_agent
from scripts.render_k8s_worker_manifest import render_manifest


ROOT = Path(__file__).resolve().parent
RUN_LOG_DIR = ROOT / ".run_logs"
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_agent_worker.py"
GENERATED_K8S_DIR = ROOT / "k8s" / "generated"
DEFAULT_WORKER_IMAGE = "fpaiopsstaging.azurecr.io/platform-agent-worker:latest"
DEFAULT_K8S_NAMESPACE = "platform-agent"

# Load local environment for Streamlit/UI-driven actions such as SIP provisioning.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=False)


@dataclass
class CreateAgentResult:
    config_path: str
    worker_command: list[str]
    manifest_path: str | None = None
    deployment_name: str | None = None
    service_name: str | None = None
    image: str | None = None
    namespace: str | None = None
    applied: bool = False
    kubectl_command: list[str] | None = None


@dataclass
class StartAgentResult:
    pid: int | None
    port: int | None
    command: list[str]
    log_path: str
    started: bool
    message: str
    status: str
    healthy: bool | None = None
    health_url: str | None = None
    worker_url: str | None = None


@dataclass
class KubernetesAgentStatusResult:
    agent_name: str
    namespace: str
    deployment_name: str
    service_name: str
    deployed: bool
    desired_replicas: int
    ready_replicas: int
    running_pods: int
    total_running_workers: int
    message: str


def build_worker_command(agent_name: str, mode: str = "dev") -> list[str]:
    return [
        sys.executable,
        "-m",
        "livekit_agents.create_agent",
        mode,
    ]


def build_kubectl_apply_command(manifest_path: Path) -> list[str]:
    return [
        os.getenv("KUBECTL_BIN", "kubectl"),
        "apply",
        "-f",
        str(manifest_path),
    ]


def build_kubectl_delete_command(agent_name: str, namespace: str) -> list[str]:
    deployment_name = _deployment_name(agent_name)
    return [
        os.getenv("KUBECTL_BIN", "kubectl"),
        "delete",
        "deployment",
        deployment_name,
        "service",
        deployment_name,
        "-n",
        namespace,
        "--ignore-not-found=true",
    ]


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _resolve_worker_image() -> str:
    return os.getenv("AGENT_WORKER_IMAGE", DEFAULT_WORKER_IMAGE)


def _resolve_k8s_namespace() -> str:
    return os.getenv("K8S_NAMESPACE", DEFAULT_K8S_NAMESPACE)


def _generated_manifest_path(agent_name: str) -> Path:
    GENERATED_K8S_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_K8S_DIR / f"{agent_name}.yaml"


def _deployment_name(agent_name: str) -> str:
    return f"agent-worker-{agent_name}"


def deploy_agent_to_kubernetes_backend(agent_name: str) -> CreateAgentResult:
    image = _resolve_worker_image()
    namespace = _resolve_k8s_namespace()
    manifest_path = _generated_manifest_path(agent_name)
    manifest_path.write_text(
        render_manifest(agent_name, image=image, replicas=1, namespace=namespace),
        encoding="utf-8",
    )

    kubectl_command = build_kubectl_apply_command(manifest_path)
    result = _run_command(kubectl_command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "kubectl apply failed."
        raise RuntimeError(detail)

    return CreateAgentResult(
        config_path="database",
        worker_command=build_worker_command(agent_name),
        manifest_path=str(manifest_path),
        deployment_name=_deployment_name(agent_name),
        service_name=_deployment_name(agent_name),
        image=image,
        namespace=namespace,
        applied=True,
        kubectl_command=kubectl_command,
    )


def create_agent_backend(config: AgentConfig, phone_number: str) -> CreateAgentResult:
    config_path = save_agent_config(config, phone_number=phone_number)

    return CreateAgentResult(
        config_path=str(config_path),
        worker_command=build_worker_command(config.name),
    )


def provision_sip_backend(agent_name: str, sip_config: SIPProvisionConfig) -> dict[str, str]:
    import asyncio

    return asyncio.run(provision_inbound_sip_for_agent(agent_name, sip_config))


def get_kubernetes_agent_status_backend(agent_name: str) -> KubernetesAgentStatusResult:
    namespace = _resolve_k8s_namespace()
    deployment_name = _deployment_name(agent_name)
    service_name = deployment_name

    deployment_result = _run_command(
        [os.getenv("KUBECTL_BIN", "kubectl"), "get", "deployment", deployment_name, "-n", namespace, "-o", "json"]
    )
    deployed = deployment_result.returncode == 0
    desired_replicas = 0
    ready_replicas = 0
    running_pods = 0

    if deployed:
        payload = json.loads(deployment_result.stdout)
        desired_replicas = int(payload.get("spec", {}).get("replicas", 0) or 0)
        ready_replicas = int(payload.get("status", {}).get("readyReplicas", 0) or 0)

        pods_result = _run_command(
            [
                os.getenv("KUBECTL_BIN", "kubectl"),
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"app={deployment_name}",
                "-o",
                "json",
            ]
        )
        if pods_result.returncode == 0:
            pods_payload = json.loads(pods_result.stdout)
            running_pods = sum(
                1 for item in pods_payload.get("items", []) if item.get("status", {}).get("phase") == "Running"
            )

    all_workers_result = _run_command(
        [os.getenv("KUBECTL_BIN", "kubectl"), "get", "pods", "-n", namespace, "-o", "json"]
    )
    total_running_workers = 0
    if all_workers_result.returncode == 0:
        pods_payload = json.loads(all_workers_result.stdout)
        total_running_workers = sum(
            1
            for item in pods_payload.get("items", [])
            if item.get("metadata", {}).get("labels", {}).get("app", "").startswith("agent-worker-")
            and item.get("status", {}).get("phase") == "Running"
        )

    if deployed:
        message = f"{running_pods} running pod(s) for {agent_name}; {total_running_workers} worker pod(s) running total."
    else:
        message = f"{agent_name} is not deployed; {total_running_workers} worker pod(s) running total."

    return KubernetesAgentStatusResult(
        agent_name=agent_name,
        namespace=namespace,
        deployment_name=deployment_name,
        service_name=service_name,
        deployed=deployed,
        desired_replicas=desired_replicas,
        ready_replicas=ready_replicas,
        running_pods=running_pods,
        total_running_workers=total_running_workers,
        message=message,
    )


def delete_agent_deployment_backend(agent_name: str) -> StartAgentResult:
    namespace = _resolve_k8s_namespace()
    result = _run_command(build_kubectl_delete_command(agent_name, namespace))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "kubectl delete failed."
        return StartAgentResult(
            pid=None,
            port=None,
            command=[],
            log_path="",
            started=False,
            message=detail,
            status="delete_failed",
            healthy=False,
        )

    return StartAgentResult(
        pid=None,
        port=None,
        command=[],
        log_path="",
        started=True,
        message=f"Agent '{agent_name}' deployment deleted from Kubernetes namespace '{namespace}'.",
        status="deleted",
        healthy=False,
    )


def serialize_result(result: CreateAgentResult) -> str:
    return json.dumps(
        {
            "config_path": result.config_path,
            "worker_command": result.worker_command,
            "manifest_path": result.manifest_path,
            "deployment_name": result.deployment_name,
            "service_name": result.service_name,
            "image": result.image,
            "namespace": result.namespace,
            "applied": result.applied,
            "kubectl_command": result.kubectl_command,
        },
        indent=2,
    )
