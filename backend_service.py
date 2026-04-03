from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException

from agent_config.schema import AgentConfig, SIPProvisionConfig
from agent_config.store import (
    claim_next_deployment_request,
    enqueue_agent_deployment_request,
    get_latest_agent_deployment,
    get_sip_binding_by_agent_name,
    get_sip_binding_by_phone_number,
    list_reconcilable_deployments,
    load_agent_config,
    save_agent_config,
    save_agent_deployment_status,
    save_agent_sip_binding_status,
    use_db_backend,
)
from livekit_trunks.provision import provision_inbound_sip_for_agent
from scripts.render_k8s_worker_manifest import render_manifest


ROOT = Path(__file__).resolve().parent
RUN_LOG_DIR = ROOT / ".run_logs"
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_agent_worker.py"
GENERATED_K8S_DIR = ROOT / "k8s" / "generated"
DEFAULT_WORKER_IMAGE = "fpaiopsstaging.azurecr.io/platform-agent-worker:latest"
DEFAULT_K8S_NAMESPACE = "platform-agent"
DEPLOYMENT_WORKER_POLL_SECONDS = 2.0
DEPLOYMENT_RECONCILE_SECONDS = 10.0

# Load local environment for Streamlit/UI-driven actions such as SIP provisioning.
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=False)

logger = logging.getLogger("platform-agent-backend")


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
        "--agent-name",
        agent_name,
    ]


_K8S_CONFIG_LOADED = False
_DEPLOYMENT_WORKER_THREAD: threading.Thread | None = None
_DEPLOYMENT_WORKER_LOCK = threading.Lock()
_DEPLOYMENT_WORKER_ID = f"deployment-worker-{uuid.uuid4()}"


def _load_kubernetes_config() -> None:
    global _K8S_CONFIG_LOADED
    if _K8S_CONFIG_LOADED:
        return

    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()

    _K8S_CONFIG_LOADED = True


def _apps_api() -> client.AppsV1Api:
    _load_kubernetes_config()
    return client.AppsV1Api()


def _core_api() -> client.CoreV1Api:
    _load_kubernetes_config()
    return client.CoreV1Api()


def _resolve_worker_image() -> str:
    return os.getenv("AGENT_WORKER_IMAGE", DEFAULT_WORKER_IMAGE)


def _resolve_k8s_namespace() -> str:
    return os.getenv("K8S_NAMESPACE", DEFAULT_K8S_NAMESPACE)


def _generated_manifest_path(agent_name: str) -> Path:
    GENERATED_K8S_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_K8S_DIR / f"{agent_name}.yaml"


def _deployment_name(agent_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", agent_name.lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    return f"agent-worker-{normalized}"


def _split_image_tag(image: str) -> str | None:
    if "@" in image:
        return image.split("@", 1)[1]
    if ":" in image.rsplit("/", 1)[-1]:
        return image.rsplit(":", 1)[1]
    return None


def _build_worker_deployment_spec(agent_name: str, namespace: str, image: str) -> client.V1Deployment:
    config_payload = load_agent_config(agent_name)
    port = config_payload.worker.port
    app_name = _deployment_name(agent_name)

    return client.V1Deployment(
        metadata=client.V1ObjectMeta(name=app_name, namespace=namespace, labels={"app": app_name}),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels={"app": app_name}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": app_name}),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="worker",
                            image=image,
                            image_pull_policy="IfNotPresent",
                            ports=[client.V1ContainerPort(container_port=port, name="health")],
                            env=[client.V1EnvVar(name="TARGET_AGENT_NAME", value=agent_name)],
                            env_from=[
                                client.V1EnvFromSource(
                                    config_map_ref=client.V1ConfigMapEnvSource(name="platform-agent-config")
                                ),
                                client.V1EnvFromSource(
                                    secret_ref=client.V1SecretEnvSource(name="platform-agent-secrets")
                                ),
                            ],
                            readiness_probe=client.V1Probe(
                                http_get=client.V1HTTPGetAction(path="/", port=port),
                                initial_delay_seconds=10,
                                period_seconds=10,
                            ),
                            liveness_probe=client.V1Probe(
                                http_get=client.V1HTTPGetAction(path="/", port=port),
                                initial_delay_seconds=20,
                                period_seconds=20,
                            ),
                        )
                    ]
                ),
            ),
        ),
    )


def _build_worker_service_spec(agent_name: str, namespace: str) -> client.V1Service:
    config_payload = load_agent_config(agent_name)
    port = config_payload.worker.port
    app_name = _deployment_name(agent_name)

    return client.V1Service(
        metadata=client.V1ObjectMeta(name=app_name, namespace=namespace, labels={"app": app_name}),
        spec=client.V1ServiceSpec(
            selector={"app": app_name},
            ports=[client.V1ServicePort(name="health", port=port, target_port=port)],
        ),
    )


def _upsert_deployment(agent_name: str, namespace: str, image: str) -> None:
    apps_api = _apps_api()
    app_name = _deployment_name(agent_name)
    body = _build_worker_deployment_spec(agent_name, namespace, image)

    try:
        apps_api.read_namespaced_deployment(name=app_name, namespace=namespace)
        apps_api.patch_namespaced_deployment(name=app_name, namespace=namespace, body=body)
    except ApiException as exc:
        if exc.status != 404:
            raise
        apps_api.create_namespaced_deployment(namespace=namespace, body=body)


def _upsert_service(agent_name: str, namespace: str) -> None:
    core_api = _core_api()
    app_name = _deployment_name(agent_name)
    body = _build_worker_service_spec(agent_name, namespace)

    try:
        core_api.read_namespaced_service(name=app_name, namespace=namespace)
        core_api.patch_namespaced_service(name=app_name, namespace=namespace, body=body)
    except ApiException as exc:
        if exc.status != 404:
            raise
        core_api.create_namespaced_service(namespace=namespace, body=body)


def _list_worker_pods(namespace: str, label_selector: str | None = None) -> list[client.V1Pod]:
    core_api = _core_api()
    pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    return [
        item
        for item in pods.items
        if item.metadata
        and item.metadata.labels
        and item.metadata.labels.get("app", "").startswith("agent-worker-")
    ]


def _count_running_worker_pods(namespace: str, label_selector: str | None = None) -> int:
    return sum(1 for item in _list_worker_pods(namespace, label_selector) if item.status and item.status.phase == "Running")


def _deployment_condition_reason(deployment: client.V1Deployment | None) -> str | None:
    if not deployment or not deployment.status or not deployment.status.conditions:
        return None
    for condition in deployment.status.conditions:
        if condition.status == "True":
            return condition.reason or condition.type
    return deployment.status.conditions[-1].reason or deployment.status.conditions[-1].type


def _runtime_status_for_deployment(
    deployment: client.V1Deployment | None,
    running_pods: int,
    ready_replicas: int,
    desired_replicas: int,
) -> str:
    if deployment is None:
        return "not_found"
    if ready_replicas >= max(1, desired_replicas):
        return "ready"
    if running_pods > 0:
        return "progressing"
    return "unknown"


def _health_status_for_runtime(runtime_status: str) -> str:
    if runtime_status == "ready":
        return "healthy"
    if runtime_status in {"progressing", "unknown"}:
        return "unknown"
    return "unhealthy"


def _record_deployment_status(**kwargs) -> None:
    try:
        save_agent_deployment_status(**kwargs)
    except Exception:
        # Deployment tracking should not block cluster operations while schema/migrations catch up.
        return


def _reconcile_single_deployment(agent_name: str, namespace: str, deployment_name: str, service_name: str | None) -> None:
    apps_api = _apps_api()

    deployed = False
    desired_replicas = 0
    ready_replicas = 0
    available_replicas = 0
    running_pods = 0
    deployment = None

    try:
        deployment = apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        deployed = True
        desired_replicas = int(deployment.spec.replicas or 0)
        ready_replicas = int(deployment.status.ready_replicas or 0)
        available_replicas = int(deployment.status.available_replicas or 0)
        running_pods = _count_running_worker_pods(namespace, label_selector=f"app={deployment_name}")
    except ApiException as exc:
        if exc.status != 404:
            raise

    runtime_status = _runtime_status_for_deployment(deployment, running_pods, ready_replicas, desired_replicas)
    desired_state = "deployed" if deployed else "deleted"
    workflow_status = "ready" if runtime_status == "ready" else ("deleted" if runtime_status == "not_found" else "reconciling")

    _record_deployment_status(
        agent_name=agent_name,
        agent_version=_split_image_tag(_resolve_worker_image()) or "latest",
        namespace=namespace,
        deployment_name=deployment_name,
        service_name=service_name,
        image=_resolve_worker_image() if deployed else None,
        image_tag=_split_image_tag(_resolve_worker_image()) if deployed else None,
        desired_state=desired_state,
        workflow_status=workflow_status,
        runtime_status=runtime_status,
        health_status=_health_status_for_runtime(runtime_status),
        desired_replicas=desired_replicas,
        ready_replicas=ready_replicas,
        available_replicas=available_replicas,
        generation=deployment.metadata.generation if deployment and deployment.metadata else None,
        observed_generation=deployment.status.observed_generation if deployment and deployment.status else None,
        status_reason=_deployment_condition_reason(deployment) if deployment else "not_found",
        status_details={
            "running_pods": running_pods,
            "conditions": [
                {
                    "type": condition.type,
                    "status": condition.status,
                    "reason": condition.reason,
                    "message": condition.message,
                }
                for condition in ((deployment.status.conditions or []) if deployment and deployment.status else [])
            ],
        },
        event_type="status_reconciled",
        event_message=f"Reconciled deployment state for {agent_name}: {runtime_status}.",
    )


def _process_claimed_request(request: dict) -> None:
    agent_name = request["agent_name"]
    namespace = request["namespace"]
    deployment_name = request["deployment_name"]
    image = request.get("image") or _resolve_worker_image()

    try:
        if request["desired_state"] == "deleted":
            try:
                _apps_api().delete_namespaced_deployment(name=deployment_name, namespace=namespace)
            except ApiException as exc:
                if exc.status != 404:
                    raise
            try:
                _core_api().delete_namespaced_service(name=deployment_name, namespace=namespace)
            except ApiException as exc:
                if exc.status != 404:
                    raise

            _record_deployment_status(
                agent_name=agent_name,
                agent_version=request.get("agent_version") or _split_image_tag(image) or "latest",
                namespace=namespace,
                deployment_name=deployment_name,
                service_name=request.get("service_name"),
                image=image,
                image_tag=_split_image_tag(image),
                desired_state="deleted",
                workflow_status="deleted",
                runtime_status="not_found",
                health_status="unknown",
                status_reason="deleted",
                status_details={"resource": "deployment_and_service"},
                event_type="deployment_deleted",
                event_message=f"Deployment deleted for {agent_name}.",
            )
            return

        _upsert_deployment(agent_name, namespace, image)
        _upsert_service(agent_name, namespace)
        _record_deployment_status(
            agent_name=agent_name,
            agent_version=request.get("agent_version") or _split_image_tag(image) or "latest",
            namespace=namespace,
            deployment_name=deployment_name,
            service_name=request.get("service_name"),
            image=image,
            image_tag=_split_image_tag(image),
            desired_state="deployed",
            workflow_status="applied",
            runtime_status="progressing",
            health_status="unknown",
            status_reason="apply_succeeded",
            status_details={"phase": "apply"},
            event_type="deployment_applied",
            event_message=f"Deployment applied for {agent_name}.",
        )
        _reconcile_single_deployment(agent_name, namespace, deployment_name, request.get("service_name"))
    except Exception as exc:
        detail = (
            exc.body or exc.reason
            if isinstance(exc, ApiException)
            else str(exc)
        ) or "Kubernetes deployment operation failed."
        logger.exception("Deployment worker failed for agent '%s'", agent_name)
        _record_deployment_status(
            agent_name=agent_name,
            agent_version=request.get("agent_version") or _split_image_tag(image) or "latest",
            namespace=namespace,
            deployment_name=deployment_name,
            service_name=request.get("service_name"),
            image=image,
            image_tag=_split_image_tag(image),
            desired_state=request["desired_state"],
            workflow_status="failed",
            runtime_status="unknown",
            health_status="unhealthy",
            last_error=detail,
            status_reason="operation_failed",
            status_details={"error": detail},
            event_type="deployment_failed",
            event_message=detail,
        )


def _deployment_worker_loop() -> None:
    last_reconcile = 0.0
    while True:
        request = None
        try:
            request = claim_next_deployment_request(_DEPLOYMENT_WORKER_ID) if use_db_backend() else None
            if request:
                _process_claimed_request(request)

            now = time.time()
            if use_db_backend() and now - last_reconcile >= DEPLOYMENT_RECONCILE_SECONDS:
                for deployment in list_reconcilable_deployments():
                    _reconcile_single_deployment(
                        deployment["agent_name"],
                        deployment["namespace"],
                        deployment["deployment_name"],
                        deployment.get("service_name"),
                    )
                last_reconcile = now
        except Exception as exc:
            logger.exception("Background deployment worker loop failed")
            if request:
                _record_deployment_status(
                    agent_name=request["agent_name"],
                    agent_version=request.get("agent_version") or "latest",
                    namespace=request["namespace"],
                    deployment_name=request["deployment_name"],
                    service_name=request.get("service_name"),
                    image=request.get("image"),
                    image_tag=request.get("image_tag"),
                    desired_state=request["desired_state"],
                    workflow_status="failed",
                    runtime_status="unknown",
                    health_status="unhealthy",
                    last_error=str(exc),
                    status_reason="worker_loop_failed",
                    status_details={"worker_id": _DEPLOYMENT_WORKER_ID, "error": str(exc)},
                    event_type="deployment_failed",
                    event_message=str(exc),
                )

        time.sleep(DEPLOYMENT_WORKER_POLL_SECONDS)


def ensure_deployment_worker_running() -> None:
    global _DEPLOYMENT_WORKER_THREAD
    if not use_db_backend():
        return

    with _DEPLOYMENT_WORKER_LOCK:
        if _DEPLOYMENT_WORKER_THREAD and _DEPLOYMENT_WORKER_THREAD.is_alive():
            return
        _DEPLOYMENT_WORKER_THREAD = threading.Thread(
            target=_deployment_worker_loop,
            name="agent-deployment-worker",
            daemon=True,
        )
        _DEPLOYMENT_WORKER_THREAD.start()


def deploy_agent_to_kubernetes_backend(agent_name: str) -> CreateAgentResult:
    image = _resolve_worker_image()
    namespace = _resolve_k8s_namespace()
    deployment_name = _deployment_name(agent_name)
    manifest_path = _generated_manifest_path(agent_name)
    manifest_path.write_text(
        render_manifest(agent_name, image=image, replicas=1, namespace=namespace),
        encoding="utf-8",
    )

    ensure_deployment_worker_running()
    if use_db_backend():
        enqueue_agent_deployment_request(
            agent_name=agent_name,
            namespace=namespace,
            deployment_name=deployment_name,
            service_name=deployment_name,
            image=image,
            image_tag=_split_image_tag(image),
            desired_state="deployed",
            agent_version=_split_image_tag(image) or "latest",
        )
    else:
        _upsert_deployment(agent_name, namespace, image)
        _upsert_service(agent_name, namespace)

    return CreateAgentResult(
        config_path="database",
        worker_command=build_worker_command(agent_name),
        manifest_path=str(manifest_path),
        deployment_name=deployment_name,
        service_name=deployment_name,
        image=image,
        namespace=namespace,
        applied=not use_db_backend(),
        kubectl_command=None,
    )


def create_agent_backend(config: AgentConfig, phone_number: str) -> CreateAgentResult:
    config_path = save_agent_config(config, phone_number=phone_number)

    return CreateAgentResult(
        config_path=str(config_path),
        worker_command=build_worker_command(config.name),
    )


def provision_sip_backend(agent_name: str, sip_config: SIPProvisionConfig) -> dict[str, str]:
    import asyncio

    existing_phone_binding = get_sip_binding_by_phone_number(sip_config.phone_number) if use_db_backend() else None
    if existing_phone_binding:
        raise ValueError(
            f"Phone number {sip_config.phone_number} is already assigned to agent "
            f"{existing_phone_binding.get('agent_name')}."
        )

    existing_agent_binding = get_sip_binding_by_agent_name(agent_name) if use_db_backend() else None
    if existing_agent_binding and existing_agent_binding.get("phone_number") != sip_config.phone_number:
        raise ValueError(
            f"Agent {agent_name} already has SIP binding for phone number "
            f"{existing_agent_binding.get('phone_number')}."
        )

    if use_db_backend():
        save_agent_sip_binding_status(
            agent_name=agent_name,
            phone_number=sip_config.phone_number,
            trunk_name=sip_config.trunk_friendly_name,
            dispatch_rule_name=sip_config.dispatch_rule_name,
            room_prefix=sip_config.room_prefix,
            desired_state="provisioned",
            workflow_status="provisioning",
            health_status="unknown",
            status_reason="provisioning_started",
            status_details={"hide_phone_number": sip_config.hide_phone_number},
            event_type="sip_provisioning_started",
            event_message=f"Started SIP provisioning for {agent_name}.",
        )

    try:
        result = asyncio.run(provision_inbound_sip_for_agent(agent_name, sip_config))
    except Exception as exc:
        if use_db_backend():
            save_agent_sip_binding_status(
                agent_name=agent_name,
                phone_number=sip_config.phone_number,
                trunk_name=sip_config.trunk_friendly_name,
                dispatch_rule_name=sip_config.dispatch_rule_name,
                room_prefix=sip_config.room_prefix,
                desired_state="provisioned",
                workflow_status="failed",
                health_status="unhealthy",
                last_error=str(exc),
                status_reason="provisioning_failed",
                status_details={"hide_phone_number": sip_config.hide_phone_number},
                event_type="sip_provisioning_failed",
                event_message=str(exc),
            )
        raise

    if use_db_backend():
        save_agent_sip_binding_status(
            agent_name=agent_name,
            phone_number=sip_config.phone_number,
            trunk_name=sip_config.trunk_friendly_name,
            dispatch_rule_name=sip_config.dispatch_rule_name,
            room_prefix=sip_config.room_prefix,
            desired_state="provisioned",
            workflow_status="provisioned",
            health_status="healthy",
            trunk_id=result.get("trunk_id"),
            dispatch_rule_id=result.get("dispatch_rule_id"),
            status_reason="provisioned",
            status_details={"hide_phone_number": sip_config.hide_phone_number},
            event_type="sip_provisioned",
            event_message=f"SIP trunk and dispatch rule created for {agent_name}.",
        )

    return result


def get_kubernetes_agent_status_backend(agent_name: str) -> KubernetesAgentStatusResult:
    namespace = _resolve_k8s_namespace()
    deployment_name = _deployment_name(agent_name)
    service_name = deployment_name
    apps_api = _apps_api()

    k8s_deployed = False
    desired_replicas = 0
    ready_replicas = 0
    running_pods = 0
    total_running_workers = _count_running_worker_pods(namespace)
    app_label = f"app={deployment_name}"

    try:
        deployment = apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        k8s_deployed = True
        desired_replicas = int(deployment.spec.replicas or 0)
        ready_replicas = int(deployment.status.ready_replicas or 0)
        running_pods = _count_running_worker_pods(namespace, label_selector=app_label)
    except ApiException as exc:
        if exc.status != 404:
            detail = exc.body or exc.reason or "Failed to read Kubernetes deployment status."
            _record_deployment_status(
                agent_name=agent_name,
                agent_version="unknown",
                namespace=namespace,
                deployment_name=deployment_name,
                service_name=service_name,
                image=None,
                image_tag=None,
                desired_state="deployed",
                workflow_status="failed",
                runtime_status="unknown",
                health_status="unhealthy",
                last_error=detail,
                status_reason="status_read_failed",
                status_details={"phase": "status", "error": detail},
                event_type="status_failed",
                event_message=detail,
            )
            raise RuntimeError(detail) from exc

    if use_db_backend():
        ensure_deployment_worker_running()
        latest = get_latest_agent_deployment(agent_name, namespace, deployment_name)
        if latest:
            workflow_status = latest.get("workflow_status") or "unknown"
            runtime_status = latest.get("runtime_status") or ("ready" if k8s_deployed else "not_found")
            if k8s_deployed:
                message = f"{workflow_status} / {runtime_status} for {agent_name}."
            else:
                message = f"{workflow_status} / {runtime_status} for {agent_name}; deployment not yet present in Kubernetes."
            return KubernetesAgentStatusResult(
                agent_name=agent_name,
                namespace=namespace,
                deployment_name=deployment_name,
                service_name=service_name,
                deployed=k8s_deployed,
                desired_replicas=desired_replicas or int(latest.get("desired_replicas") or 0),
                ready_replicas=ready_replicas or int(latest.get("ready_replicas") or 0),
                running_pods=running_pods,
                total_running_workers=total_running_workers,
                message=message,
            )

    if k8s_deployed:
        message = f"{running_pods} running pod(s) for {agent_name}; {total_running_workers} worker pod(s) running total."
    else:
        message = f"{agent_name} is not deployed; {total_running_workers} worker pod(s) running total."

    return KubernetesAgentStatusResult(
        agent_name=agent_name,
        namespace=namespace,
        deployment_name=deployment_name,
        service_name=service_name,
        deployed=k8s_deployed,
        desired_replicas=desired_replicas,
        ready_replicas=ready_replicas,
        running_pods=running_pods,
        total_running_workers=total_running_workers,
        message=message,
    )


def delete_agent_deployment_backend(agent_name: str) -> StartAgentResult:
    namespace = _resolve_k8s_namespace()
    deployment_name = _deployment_name(agent_name)
    ensure_deployment_worker_running()

    if use_db_backend():
        enqueue_agent_deployment_request(
            agent_name=agent_name,
            namespace=namespace,
            deployment_name=deployment_name,
            service_name=deployment_name,
            image=_resolve_worker_image(),
            image_tag=_split_image_tag(_resolve_worker_image()),
            desired_state="deleted",
            agent_version=_split_image_tag(_resolve_worker_image()) or "latest",
        )
        return StartAgentResult(
            pid=None,
            port=None,
            command=[],
            log_path="",
            started=True,
            message=f"Deletion requested for '{agent_name}' in Kubernetes namespace '{namespace}'.",
            status="delete_queued",
            healthy=False,
        )

    try:
        _apps_api().delete_namespaced_deployment(name=deployment_name, namespace=namespace)
    except ApiException as exc:
        if exc.status != 404:
            detail = exc.body or exc.reason or "Kubernetes deployment delete failed."
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

    try:
        _core_api().delete_namespaced_service(name=deployment_name, namespace=namespace)
    except ApiException as exc:
        if exc.status != 404:
            detail = exc.body or exc.reason or "Kubernetes service delete failed."
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
