from __future__ import annotations

import json
import os
import re

import streamlit as st

from agent_config.schema import AgentConfig, ProviderConfig, SIPProvisionConfig, VADConfig, WorkerConfig
from agent_config.store import list_agent_configs, use_db_backend
from backend_service import (
    create_agent_backend,
    delete_agent_deployment_backend,
    deploy_agent_to_kubernetes_backend,
    get_kubernetes_agent_status_backend,
    provision_sip_backend,
    serialize_result,
)


LLM_OPTIONS = {
    "azure_openai": {
        "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
        "default_model": "gpt-4.1-mini",
        "default_kwargs": {
            "azure_deployment": "gpt-4.1-mini",
            "api_version": "2024-10-01-preview",
        },
    },
    "openai": {
        "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        "default_model": "gpt-4.1-mini",
        "default_kwargs": {},
    },
}

STT_OPTIONS = {
    "deepgram": {
        "models": ["nova-3", "nova-2"],
        "default_model": "nova-3",
        "default_kwargs": {
            "language": "en-IN",
            "detect_language": False,
            "interim_results": True,
        },
    },
    "azure_openai": {
        "models": ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        "default_model": "gpt-4o-mini-transcribe",
        "default_kwargs": {
            "azure_deployment": "gpt-4o-mini-transcribe",
            "api_version": "2024-10-01-preview",
            "language": "en",
            "detect_language": False,
        },
    },
    "elevenlabs": {
        "models": ["scribe_v2", "scribe_v2_realtime"],
        "default_model": "scribe_v2_realtime",
        "default_kwargs": {
            "language": "en",
            "detect_language": False,
        },
    },
}

TTS_OPTIONS = {
    "google": {
        "models": [""],
        "default_model": "",
        "default_kwargs": {
            "language": "en-IN",
            "voice_name": "en-IN-Chirp3-HD-Achernar",
            "gender": "female",
            "credentials_file": "google_creds.json",
        },
    },
    "azure_openai": {
        "models": ["gpt-4o-mini-tts"],
        "default_model": "gpt-4o-mini-tts",
        "default_kwargs": {
            "azure_deployment": "gpt-4o-mini-tts",
            "api_version": "2024-10-01-preview",
            "voice": "coral",
        },
    },
    "openai": {
        "models": ["gpt-4o-mini-tts"],
        "default_model": "gpt-4o-mini-tts",
        "default_kwargs": {
            "voice": "alloy",
        },
    },
    "elevenlabs": {
        "models": ["eleven_flash_v2_5", "eleven_multilingual_v2"],
        "default_model": "eleven_flash_v2_5",
        "default_kwargs": {
            "voice_id": "",
        },
    },
}

VAD_OPTIONS = {
    "silero": {
        "default_kwargs": {
            "min_speech_duration": 0.07,
            "min_silence_duration": 0.5,
            "activation_threshold": 0.5,
            "force_cpu": True,
            "sample_rate": 16000,
        }
    }
}


def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2)


def _provider_index(options: dict[str, dict], selected: str) -> int:
    keys = list(options.keys())
    return keys.index(selected) if selected in keys else 0


def _sync_model_and_kwargs(prefix: str, options: dict[str, dict]) -> None:
    provider = st.session_state[f"{prefix}_provider"]
    spec = options[provider]
    st.session_state[f"{prefix}_model"] = spec["default_model"]
    kwargs = dict(spec["default_kwargs"])
    if prefix in {"llm", "stt", "tts"} and st.session_state[f"{prefix}_model"]:
        if "azure_deployment" in kwargs:
            kwargs["azure_deployment"] = st.session_state[f"{prefix}_model"]
    st.session_state[f"{prefix}_kwargs"] = _json_text(kwargs)


def _sync_kwargs_for_selected_model(prefix: str) -> None:
    raw = st.session_state.get(f"{prefix}_kwargs", "{}")
    try:
        kwargs = parse_json_dict(raw, f"{prefix.upper()} kwargs")
    except Exception:
        kwargs = {}
    model = st.session_state.get(f"{prefix}_model", "")
    if "azure_deployment" in kwargs:
        kwargs["azure_deployment"] = model
        st.session_state[f"{prefix}_kwargs"] = _json_text(kwargs)


def _sync_vad_kwargs() -> None:
    provider = st.session_state["vad_provider"]
    st.session_state["vad_kwargs"] = _json_text(VAD_OPTIONS[provider]["default_kwargs"])


def parse_json_dict(raw: str, label: str) -> dict:
    if not raw.strip():
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def validate_e164(phone_number: str) -> str:
    value = phone_number.strip()
    if not re.fullmatch(r"\+[1-9]\d{1,14}", value):
        raise ValueError("Phone number must be in E.164 format, for example +14155550123.")
    return value


st.set_page_config(page_title="LiveKit Agent Builder", layout="centered")
st.title("LiveKit Agent Builder")
st.caption("Create an agent config and deploy the worker to Kubernetes with the shared base image.")
if "DATABASE_URL" in os.environ:
    st.info("Agent configs will be written directly to the `voice_virtual_agents` table.")

if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = "azure_openai"
    _sync_model_and_kwargs("llm", LLM_OPTIONS)
if "stt_provider" not in st.session_state:
    st.session_state["stt_provider"] = "deepgram"
    _sync_model_and_kwargs("stt", STT_OPTIONS)
if "tts_provider" not in st.session_state:
    st.session_state["tts_provider"] = "google"
    _sync_model_and_kwargs("tts", TTS_OPTIONS)
if "vad_provider" not in st.session_state:
    st.session_state["vad_provider"] = "silero"
    _sync_vad_kwargs()
if "last_worker_command" not in st.session_state:
    st.session_state["last_worker_command"] = ""
if "kubernetes_agent_status" not in st.session_state:
    st.session_state["kubernetes_agent_status"] = {}

name = st.text_input("Agent name", placeholder="sales_agent_india")
agent_phone_number = st.text_input(
    "Agent phone number (E.164)",
    placeholder="+14155550123",
    help="Stored with the voice virtual agent record used by the worker deployment.",
)
system_prompt = st.text_area("System prompt", height=180)
tools_raw = st.text_area(
    "Tool import paths (one per line)",
    value="livekit_agents_tools.session_tools.end_call",
    height=80,
)

st.subheader("LLM")
llm_provider = st.selectbox(
    "LLM provider",
    options=list(LLM_OPTIONS.keys()),
    index=_provider_index(LLM_OPTIONS, st.session_state["llm_provider"]),
    key="llm_provider",
    on_change=_sync_model_and_kwargs,
    args=("llm", LLM_OPTIONS),
)
llm_model = st.selectbox(
    "LLM model",
    options=LLM_OPTIONS[llm_provider]["models"],
    index=LLM_OPTIONS[llm_provider]["models"].index(st.session_state["llm_model"])
    if st.session_state["llm_model"] in LLM_OPTIONS[llm_provider]["models"]
    else 0,
    key="llm_model",
    on_change=_sync_kwargs_for_selected_model,
    args=("llm",),
)
llm_kwargs = st.text_area("LLM kwargs (JSON)", key="llm_kwargs", height=140)

st.subheader("STT")
stt_provider = st.selectbox(
    "STT provider",
    options=list(STT_OPTIONS.keys()),
    index=_provider_index(STT_OPTIONS, st.session_state["stt_provider"]),
    key="stt_provider",
    on_change=_sync_model_and_kwargs,
    args=("stt", STT_OPTIONS),
)
stt_model = st.selectbox(
    "STT model",
    options=STT_OPTIONS[stt_provider]["models"],
    index=STT_OPTIONS[stt_provider]["models"].index(st.session_state["stt_model"])
    if st.session_state["stt_model"] in STT_OPTIONS[stt_provider]["models"]
    else 0,
    key="stt_model",
    on_change=_sync_kwargs_for_selected_model,
    args=("stt",),
)
stt_kwargs = st.text_area("STT kwargs (JSON)", key="stt_kwargs", height=140)

st.subheader("TTS")
tts_provider = st.selectbox(
    "TTS provider",
    options=list(TTS_OPTIONS.keys()),
    index=_provider_index(TTS_OPTIONS, st.session_state["tts_provider"]),
    key="tts_provider",
    on_change=_sync_model_and_kwargs,
    args=("tts", TTS_OPTIONS),
)
tts_model = st.selectbox(
    "TTS model",
    options=TTS_OPTIONS[tts_provider]["models"],
    index=TTS_OPTIONS[tts_provider]["models"].index(st.session_state["tts_model"])
    if st.session_state["tts_model"] in TTS_OPTIONS[tts_provider]["models"]
    else 0,
    key="tts_model",
    on_change=_sync_kwargs_for_selected_model,
    args=("tts",),
)
tts_kwargs = st.text_area("TTS kwargs (JSON)", key="tts_kwargs", height=160)

st.subheader("VAD")
vad_provider = st.selectbox(
    "VAD provider",
    options=list(VAD_OPTIONS.keys()),
    index=_provider_index(VAD_OPTIONS, st.session_state["vad_provider"]),
    key="vad_provider",
    on_change=_sync_vad_kwargs,
)
vad_kwargs = st.text_area("VAD kwargs (JSON)", key="vad_kwargs", height=140)

st.subheader("Worker options")
worker_agent_name = st.text_input("Worker agent_name", value="")
worker_port = st.number_input("Port", min_value=1, max_value=65535, value=8082)
worker_job_memory_warn_mb = st.number_input("job_memory_warn_mb", min_value=1, value=2000)
worker_shutdown_timeout = st.number_input("shutdown_process_timeout", min_value=1.0, value=80.0)
worker_initialize_timeout = st.number_input("initialize_process_timeout", min_value=1, value=540)
worker_num_idle_processes = st.number_input("num_idle_processes", min_value=0, value=1)

st.subheader("Session options")
enable_noise_cancellation = st.checkbox("Enable telephony noise cancellation", value=True)
noise_cancellation_provider = st.selectbox(
    "Noise cancellation provider",
    options=["bvc_telephony", "webrtc_noise_gain"],
    index=0,
)
noise_cancellation_kwargs = st.text_area(
    "Noise cancellation kwargs (JSON)",
    value='{"noise_suppression_level": 2, "auto_gain_dbfs": 0}',
    height=110,
)
session_kwargs = st.text_area("session.start kwargs (JSON)", value="{}", height=110)

submitted = st.button("Create agent")

if submitted:
    try:
        config = AgentConfig(
            name=name,
            system_prompt=system_prompt,
            tools=[line.strip() for line in tools_raw.splitlines() if line.strip()],
            llm=ProviderConfig(
                provider=llm_provider,
                model=llm_model or None,
                kwargs=parse_json_dict(llm_kwargs, "LLM kwargs"),
            ),
            stt=ProviderConfig(
                provider=stt_provider,
                model=stt_model or None,
                kwargs=parse_json_dict(stt_kwargs, "STT kwargs"),
            ),
            tts=ProviderConfig(
                provider=tts_provider,
                model=tts_model or None,
                kwargs=parse_json_dict(tts_kwargs, "TTS kwargs"),
            ),
            vad=VADConfig(
                provider=vad_provider,
                kwargs=parse_json_dict(vad_kwargs, "VAD kwargs"),
            ),
            worker=WorkerConfig(
                agent_name=worker_agent_name or None,
                port=int(worker_port),
                job_memory_warn_mb=int(worker_job_memory_warn_mb),
                shutdown_process_timeout=float(worker_shutdown_timeout),
                initialize_process_timeout=int(worker_initialize_timeout),
                num_idle_processes=int(worker_num_idle_processes),
            ),
            session={
                "enable_telephony_noise_cancellation": enable_noise_cancellation,
                "noise_cancellation_provider": noise_cancellation_provider,
                "noise_cancellation_kwargs": parse_json_dict(
                    noise_cancellation_kwargs, "Noise cancellation kwargs"
                ),
                "kwargs": parse_json_dict(session_kwargs, "Session kwargs"),
            },
        )
        result = create_agent_backend(config, validate_e164(agent_phone_number))
        st.session_state["last_worker_command"] = (
            f"TARGET_AGENT_NAME={config.name} " + " ".join(result.worker_command)
        )
        st.success("Agent config saved successfully.")
        st.code(serialize_result(result), language="json")
        st.caption("Use the Agent Deployment section below when you want to deploy this agent.")
        st.code(st.session_state["last_worker_command"], language="bash")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Create SIP Trunk")
st.caption(
    "This uses your LiveKit credentials from environment variables. "
    "Enter the agent name, E.164 phone number, trunk name, and dispatch rule name."
)
with st.form("sip_form"):
    sip_agent_name = st.text_input("Agent name", placeholder="sales_agent_india")
    phone_number = st.text_input("Agent phone number (E.164)", placeholder="+14155550123")
    trunk_friendly_name = st.text_input("SIP trunk name", value="sales-agent-trunk")
    dispatch_rule_name = st.text_input("Dispatch rule name", value="sales-agent-dispatch")
    room_prefix = st.text_input("Room prefix", value="inbound")
    hide_phone_number = st.checkbox("Hide phone number", value=False)
    sip_submitted = st.form_submit_button("Create SIP trunk")

if sip_submitted:
    try:
        result = provision_sip_backend(
            sip_agent_name.strip(),
            SIPProvisionConfig(
                phone_number=validate_e164(phone_number),
                trunk_friendly_name=trunk_friendly_name.strip(),
                dispatch_rule_name=dispatch_rule_name.strip(),
                room_prefix=room_prefix.strip() or "inbound",
                hide_phone_number=hide_phone_number,
            ),
        )
        st.success("SIP trunk and dispatch rule created successfully.")
        st.code(json.dumps(result, indent=2), language="json")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Agent Deployment")
st.caption(
    "Deploy saved agents to Kubernetes here."
)
saved_agent_names = [saved.name for saved in list_agent_configs()]
if saved_agent_names:
    run_agent_name = st.selectbox(
        "Saved agent",
        options=saved_agent_names,
        index=0,
    )
    if st.button("Deploy"):
        try:
            result = deploy_agent_to_kubernetes_backend(run_agent_name)
            st.session_state["kubernetes_agent_status"][run_agent_name] = get_kubernetes_agent_status_backend(
                run_agent_name
            ).__dict__
            if use_db_backend():
                st.success(f"{run_agent_name} deployment request queued.")
            else:
                st.success(f"{run_agent_name} deployed to Kubernetes.")
            st.code(serialize_result(result), language="json")
        except Exception as exc:
            st.error(str(exc))
    if st.button("Status"):
        try:
            result = get_kubernetes_agent_status_backend(run_agent_name)
            st.session_state["kubernetes_agent_status"][run_agent_name] = result.__dict__
            if result.deployed:
                st.success(result.message)
            else:
                st.warning(result.message)
        except Exception as exc:
            st.error(str(exc))
    if st.button("Stop"):
        try:
            result = delete_agent_deployment_backend(run_agent_name)
            st.session_state["kubernetes_agent_status"][run_agent_name] = get_kubernetes_agent_status_backend(
                run_agent_name
            ).__dict__
            if result.started:
                st.success(result.message)
            else:
                st.error(result.message)
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("No saved agents yet. Create an agent config first.")

if st.session_state["kubernetes_agent_status"]:
    st.write("Agent worker status")
    for agent_name, details in st.session_state["kubernetes_agent_status"].items():
        if details.get("deployed"):
            st.success(details.get("message", f"{agent_name} deployed."))
        else:
            st.warning(details.get("message", f"{agent_name} is not deployed."))
        st.code(
            json.dumps(
                {
                    "agent_name": agent_name,
                    "namespace": details.get("namespace"),
                    "deployment_name": details.get("deployment_name"),
                    "service_name": details.get("service_name"),
                    "deployed": details.get("deployed"),
                    "desired_replicas": details.get("desired_replicas"),
                    "ready_replicas": details.get("ready_replicas"),
                    "running_pods": details.get("running_pods"),
                    "total_running_workers": details.get("total_running_workers"),
                },
                indent=2,
            ),
            language="json",
        )

st.divider()
st.subheader("Saved agents")
for saved in list_agent_configs():
    st.write(f"- {saved.name}")
