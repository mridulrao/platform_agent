from __future__ import annotations

import json
import os
import re

import streamlit as st

from agent_config.schema import AgentConfig, ProviderConfig, SIPProvisionConfig, VADConfig, WorkerConfig
from agent_config.store import list_agent_configs, use_db_backend
from backend_service import (
    buy_and_bind_vonage_number_backend,
    buy_vonage_number_backend,
    bind_vonage_number_to_agent_backend,
    create_agent_backend,
    delete_livekit_dispatch_rule_backend,
    delete_livekit_sip_trunk_backend,
    delete_agent_deployment_backend,
    deploy_agent_to_kubernetes_backend,
    get_kubernetes_agent_status_backend,
    livekit_sip_config_backend,
    livekit_sip_health_backend,
    livekit_sip_inventory_backend,
    list_vonage_numbers_backend,
    provision_sip_backend,
    search_vonage_numbers_backend,
    serialize_result,
    update_livekit_dispatch_rule_backend,
    update_livekit_inbound_trunk_backend,
    update_livekit_outbound_trunk_backend,
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


def normalize_msisdn(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("+"):
        stripped = stripped[1:]
    if not stripped.isdigit():
        raise ValueError("Vonage number must contain digits only, optionally prefixed with '+'.")
    return stripped


def build_default_trunk_name(agent_name: str) -> str:
    return f"{agent_name.strip() or 'agent'}-trunk"


def build_default_dispatch_rule_name(agent_name: str) -> str:
    return f"{agent_name.strip() or 'agent'}-dispatch"


def display_number_option(number: dict) -> str:
    msisdn = number.get("e164") or f"+{number.get('msisdn', '')}"
    country = number.get("country") or "?"
    number_type = number.get("number_type") or number.get("type") or "unknown"
    features = ",".join(number.get("features") or [])
    return f"{msisdn} | {country} | {number_type} | {features or 'no-features'}"


def display_trunk_option(trunk: dict) -> str:
    trunk_id = trunk.get("sip_trunk_id") or "unknown-trunk"
    name = trunk.get("name") or "unnamed"
    numbers = ",".join(trunk.get("numbers") or [])
    return f"{name} | {trunk_id} | {numbers or 'no-numbers'}"


def display_dispatch_rule_option(rule: dict) -> str:
    rule_id = rule.get("sip_dispatch_rule_id") or "unknown-rule"
    name = rule.get("name") or "unnamed"
    trunk_ids = ",".join(rule.get("trunk_ids") or [])
    return f"{name} | {rule_id} | {trunk_ids or 'no-trunks'}"


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
if "vonage_search_results" not in st.session_state:
    st.session_state["vonage_search_results"] = []
if "vonage_owned_numbers" not in st.session_state:
    st.session_state["vonage_owned_numbers"] = []
if "livekit_sip_health" not in st.session_state:
    st.session_state["livekit_sip_health"] = None
if "livekit_sip_config" not in st.session_state:
    st.session_state["livekit_sip_config"] = None
if "livekit_sip_inventory" not in st.session_state:
    st.session_state["livekit_sip_inventory"] = None
if "last_worker_command" not in st.session_state:
    st.session_state["last_worker_command"] = ""
if "kubernetes_agent_status" not in st.session_state:
    st.session_state["kubernetes_agent_status"] = {}

agent_tab, telephony_tab = st.tabs(["Agent Configuration", "Telephony"])

with agent_tab:
    st.subheader("Create Agent")
    name = st.text_input("Agent name", placeholder="sales_agent_india")
    agent_phone_number = st.text_input(
        "Agent phone number (E.164)",
        placeholder="+14155550123",
        help="Stored with the voice virtual agent record used by the worker deployment.",
    )
    system_prompt = st.text_area("System prompt", height=180)
    tools_raw = st.text_area(
        "Tool import paths (one per line)",
        value=(
            "livekit_agents_tools.session_tools.end_call\n"
            "livekit_agents_tools.session_tools.transfer_call"
        ),
        height=80,
    )
    skills_raw = st.text_area(
        "Skill import paths (one per line)",
        value="livekit_agents_tools.skill_dispatch_toolset.build_skill_dispatch_toolset",
        height=80,
        help="Use builder paths for background skills that should be attached at the AgentSession level.",
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
    worker_db_proxy_url = st.text_input(
        "DB proxy URL",
        value=os.getenv("DB_PROXY_URL", "http://platform-agent-service:8000"),
    )
    worker_port = st.number_input("Port", min_value=1, max_value=65535, value=8082)
    worker_job_memory_warn_mb = st.number_input("job_memory_warn_mb", min_value=1, value=2000)
    worker_shutdown_timeout = st.number_input("shutdown_process_timeout", min_value=1.0, value=80.0)
    worker_initialize_timeout = st.number_input("initialize_process_timeout", min_value=1, value=540)
    worker_num_idle_processes = st.number_input("num_idle_processes", min_value=0, value=1)

    st.subheader("Session options")
    enable_noise_cancellation = st.checkbox("Enable telephony noise cancellation", value=True)
    transfer_phone_number = st.text_input(
        "Transfer target phone number (E.164)",
        value="",
        help="Optional default destination used by livekit_agents_tools.session_tools.transfer_call.",
    )
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
                skills=[line.strip() for line in skills_raw.splitlines() if line.strip()],
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
                    db_proxy_url=worker_db_proxy_url.strip() or None,
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
                    "transfer_phone_number": validate_e164(transfer_phone_number)
                    if transfer_phone_number.strip()
                    else None,
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
    st.subheader("Agent Deployment")
    st.caption("Deploy saved agents to Kubernetes here.")
    saved_agent_names = [saved.name for saved in list_agent_configs()]
    if saved_agent_names:
        run_agent_name = st.selectbox(
            "Saved agent",
            options=saved_agent_names,
            index=0,
            key="deploy_agent_name",
        )
        deploy_col, status_col, stop_col = st.columns(3)
        with deploy_col:
            if st.button("Deploy", use_container_width=True):
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
        with status_col:
            if st.button("Status", use_container_width=True):
                try:
                    result = get_kubernetes_agent_status_backend(run_agent_name)
                    st.session_state["kubernetes_agent_status"][run_agent_name] = result.__dict__
                    if result.deployed:
                        st.success(result.message)
                    else:
                        st.warning(result.message)
                except Exception as exc:
                    st.error(str(exc))
        with stop_col:
            if st.button("Stop", use_container_width=True):
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

with telephony_tab:
    st.subheader("Telephony")
    st.caption(
        "Manage number procurement, SIP trunk creation and deletion, and agent attachment through dispatch rules."
    )

    saved_agent_names_for_telephony = [saved.name for saved in list_agent_configs()]
    telephony_base_url = st.text_input(
        "Telephony API base URL",
        value=os.getenv("TELEPHONY_BASE_URL", ""),
        placeholder="https://telephony.demo.futurepath.ai",
    )

    telephony_control_col1, telephony_control_col2, telephony_control_col3 = st.columns(3)
    with telephony_control_col1:
        if st.button("Check health", use_container_width=True):
            try:
                st.session_state["livekit_sip_health"] = livekit_sip_health_backend(telephony_base_url.strip() or None)
                st.success("Telephony API is reachable.")
            except Exception as exc:
                st.error(str(exc))
    with telephony_control_col2:
        if st.button("Load config", use_container_width=True):
            try:
                st.session_state["livekit_sip_config"] = livekit_sip_config_backend(telephony_base_url.strip() or None)
                st.success("Loaded telephony config.")
            except Exception as exc:
                st.error(str(exc))
    with telephony_control_col3:
        if st.button("Refresh telephony inventory", use_container_width=True):
            try:
                st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                st.success("Loaded telephony inventory.")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state["livekit_sip_health"] is not None:
        with st.expander("Health response", expanded=False):
            st.code(json.dumps(st.session_state["livekit_sip_health"], indent=2), language="json")
    if st.session_state["livekit_sip_config"] is not None:
        with st.expander("Config response", expanded=False):
            st.code(json.dumps(st.session_state["livekit_sip_config"], indent=2), language="json")

    procurement_tab, trunking_tab, attachment_tab = st.tabs(
        ["Phone Procurement", "Trunking", "Attach Agent"]
    )

    with procurement_tab:
        st.caption(
            "Search available Vonage numbers, buy them, and review numbers you already own."
        )
        with st.form("vonage_search_form"):
            search_country = st.text_input("Country code", value="US", help="Two-letter ISO code, for example `US` or `GB`.")
            search_number_type = st.selectbox(
                "Number type",
                options=["", "landline", "mobile-lvn", "landline-toll-free"],
                format_func=lambda value: value or "Any supported type",
            )
            search_features = st.multiselect(
                "Features",
                options=["VOICE", "SMS"],
                default=["VOICE"],
            )
            search_pattern = st.text_input("Number pattern", placeholder="555")
            search_pattern_mode = st.selectbox(
                "Pattern match",
                options=[1, 0, 2],
                format_func=lambda value: {
                    0: "Starts with",
                    1: "Contains",
                    2: "Ends with",
                }[value],
            )
            search_size = st.number_input("Result limit", min_value=1, max_value=100, value=20, step=1)
            run_number_search = st.form_submit_button("Search available numbers")

        if run_number_search:
            try:
                st.session_state["vonage_search_results"] = search_vonage_numbers_backend(
                    country=search_country.strip().upper(),
                    number_type=search_number_type or None,
                    features=search_features or None,
                    pattern=search_pattern.strip() or None,
                    search_pattern=search_pattern_mode,
                    size=int(search_size),
                )
                if st.session_state["vonage_search_results"]:
                    st.success(f"Found {len(st.session_state['vonage_search_results'])} available numbers.")
                else:
                    st.info("No available numbers matched that search.")
            except Exception as exc:
                st.error(str(exc))

        if st.session_state["vonage_search_results"]:
            st.dataframe(st.session_state["vonage_search_results"], use_container_width=True)
            selected_search_number = st.selectbox(
                "Available number",
                options=st.session_state["vonage_search_results"],
                format_func=display_number_option,
                key="vonage_search_selected_number",
            )
            if st.button("Buy selected number", use_container_width=True):
                try:
                    result = buy_vonage_number_backend(
                        country=str(selected_search_number["country"]),
                        msisdn=str(selected_search_number["msisdn"]),
                    )
                    st.success("Vonage number purchased successfully.")
                    st.code(json.dumps(result, indent=2), language="json")
                except Exception as exc:
                    st.error(str(exc))

        if st.button("Refresh owned Vonage numbers", use_container_width=True):
            try:
                st.session_state["vonage_owned_numbers"] = list_vonage_numbers_backend()
            except Exception as exc:
                st.error(str(exc))

        if st.session_state["vonage_owned_numbers"]:
            st.dataframe(st.session_state["vonage_owned_numbers"], use_container_width=True)
        else:
            st.info("Use `Refresh owned Vonage numbers` after you have credentials set and at least one number purchased.")

    with trunking_tab:
        st.caption(
            "Create trunks and dispatch rules together, and manage existing inbound and outbound trunks."
        )

        with st.form("sip_form"):
            sip_agent_name = st.text_input("Saved config name", placeholder="sales_agent_india")
            phone_number = st.text_input("Agent phone number (E.164)", placeholder="+14155550123")
            trunk_friendly_name = st.text_input("SIP trunk name", value="sales-agent-trunk")
            dispatch_rule_name = st.text_input("Dispatch rule name", value="sales-agent-dispatch")
            room_prefix = st.text_input("Room prefix", value="inbound")
            hide_phone_number = st.checkbox("Hide phone number", value=False)
            sip_submitted = st.form_submit_button("Create trunk and dispatch rule")

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

        inventory = st.session_state["livekit_sip_inventory"]
        if inventory is not None:
            inbound_tab, outbound_tab = st.tabs(["Inbound Trunks", "Outbound Trunks"])
            bindings = inventory.get("bindings") or []

            with inbound_tab:
                inbound_items = (inventory.get("inbound_trunks") or {}).get("items") or []
                if inbound_items:
                    st.dataframe(inbound_items, use_container_width=True)
                    selected_inbound_trunk = st.selectbox(
                        "Inbound trunk",
                        options=inbound_items,
                        format_func=display_trunk_option,
                        key="selected_inbound_trunk",
                    )
                    inbound_trunk_id = str(selected_inbound_trunk.get("sip_trunk_id") or "")
                    inbound_payload_default = {
                        "name": selected_inbound_trunk.get("name"),
                        "numbers": selected_inbound_trunk.get("numbers") or [],
                    }
                    inbound_payload_text = st.text_area(
                        "Update inbound trunk JSON",
                        value=json.dumps(inbound_payload_default, indent=2),
                        height=180,
                        key=f"inbound_payload_{inbound_trunk_id}",
                    )
                    inbound_delete_with_rule = st.checkbox(
                        "Delete linked dispatch rule too",
                        value=True,
                        key=f"inbound_delete_with_rule_{inbound_trunk_id}",
                    )
                    matching_rule_id = ""
                    for binding in bindings:
                        if str(binding.get("trunk_id") or "") == inbound_trunk_id and str(binding.get("dispatch_rule_id") or ""):
                            matching_rule_id = str(binding.get("dispatch_rule_id"))
                            break
                    inbound_action_col1, inbound_action_col2 = st.columns(2)
                    with inbound_action_col1:
                        if st.button("Update inbound trunk", key=f"update_inbound_{inbound_trunk_id}", use_container_width=True):
                            try:
                                payload = parse_json_dict(inbound_payload_text, "Inbound trunk JSON")
                                result = update_livekit_inbound_trunk_backend(
                                    inbound_trunk_id,
                                    payload,
                                    base_url=telephony_base_url.strip() or None,
                                )
                                st.success("Inbound trunk updated.")
                                st.code(json.dumps(result, indent=2), language="json")
                                st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                            except Exception as exc:
                                st.error(str(exc))
                    with inbound_action_col2:
                        if st.button("Delete inbound trunk", key=f"delete_inbound_{inbound_trunk_id}", use_container_width=True):
                            try:
                                result = delete_livekit_sip_trunk_backend(
                                    inbound_trunk_id,
                                    matching_rule_id or None,
                                    base_url=telephony_base_url.strip() or None,
                                    delete_linked_dispatch_rule=inbound_delete_with_rule,
                                )
                                st.success("Inbound trunk deleted.")
                                st.code(json.dumps(result, indent=2), language="json")
                                st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                            except Exception as exc:
                                st.error(str(exc))
                else:
                    st.info("No inbound trunks found.")

            with outbound_tab:
                outbound_items = (inventory.get("outbound_trunks") or {}).get("items") or []
                if outbound_items:
                    st.dataframe(outbound_items, use_container_width=True)
                    selected_outbound_trunk = st.selectbox(
                        "Outbound trunk",
                        options=outbound_items,
                        format_func=display_trunk_option,
                        key="selected_outbound_trunk",
                    )
                    outbound_trunk_id = str(selected_outbound_trunk.get("sip_trunk_id") or "")
                    outbound_payload_default = {
                        "name": selected_outbound_trunk.get("name"),
                        "address": selected_outbound_trunk.get("address"),
                        "numbers": selected_outbound_trunk.get("numbers") or [],
                    }
                    outbound_payload_text = st.text_area(
                        "Update outbound trunk JSON",
                        value=json.dumps(outbound_payload_default, indent=2),
                        height=180,
                        key=f"outbound_payload_{outbound_trunk_id}",
                    )
                    outbound_delete_with_rule = st.checkbox(
                        "Delete linked dispatch rule too",
                        value=True,
                        key=f"outbound_delete_with_rule_{outbound_trunk_id}",
                    )
                    outbound_matching_rule_id = ""
                    for binding in bindings:
                        if str(binding.get("trunk_id") or "") == outbound_trunk_id and str(binding.get("dispatch_rule_id") or ""):
                            outbound_matching_rule_id = str(binding.get("dispatch_rule_id"))
                            break
                    outbound_action_col1, outbound_action_col2 = st.columns(2)
                    with outbound_action_col1:
                        if st.button("Update outbound trunk", key=f"update_outbound_{outbound_trunk_id}", use_container_width=True):
                            try:
                                payload = parse_json_dict(outbound_payload_text, "Outbound trunk JSON")
                                result = update_livekit_outbound_trunk_backend(
                                    outbound_trunk_id,
                                    payload,
                                    base_url=telephony_base_url.strip() or None,
                                )
                                st.success("Outbound trunk updated.")
                                st.code(json.dumps(result, indent=2), language="json")
                                st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                            except Exception as exc:
                                st.error(str(exc))
                    with outbound_action_col2:
                        if st.button("Delete outbound trunk", key=f"delete_outbound_{outbound_trunk_id}", use_container_width=True):
                            try:
                                result = delete_livekit_sip_trunk_backend(
                                    outbound_trunk_id,
                                    outbound_matching_rule_id or None,
                                    base_url=telephony_base_url.strip() or None,
                                    delete_linked_dispatch_rule=outbound_delete_with_rule,
                                )
                                st.success("Outbound trunk deleted.")
                                st.code(json.dumps(result, indent=2), language="json")
                                st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                            except Exception as exc:
                                st.error(str(exc))
                else:
                    st.info("No outbound trunks found.")
        else:
            st.info("Refresh telephony inventory to manage existing trunks.")

    with attachment_tab:
        st.caption(
            "Attach a purchased number to an agent by creating the dispatch rule and binding the LiveKit SIP URI."
        )
        if saved_agent_names_for_telephony and st.session_state["vonage_owned_numbers"]:
            selected_owned_number = st.selectbox(
                "Owned number",
                options=st.session_state["vonage_owned_numbers"],
                format_func=display_number_option,
                key="vonage_owned_selected_number",
            )
            owned_agent_name = st.selectbox(
                "Agent to bind",
                options=saved_agent_names_for_telephony,
                index=0,
                key="vonage_owned_selected_agent",
            )
            owned_livekit_sip_uri = st.text_input(
                "LiveKit SIP URI",
                value=os.getenv("LIVEKIT_SIP_URI", os.getenv("LIVEKIT_SIP_ADDRESS", "")),
                placeholder="sip:your-project.sip.livekit.cloud",
                key="vonage_owned_sip_uri",
            )
            owned_trunk_name = st.text_input(
                "SIP trunk name",
                value=build_default_trunk_name(owned_agent_name),
                key="vonage_owned_trunk_name",
            )
            owned_dispatch_rule_name = st.text_input(
                "Dispatch rule name",
                value=build_default_dispatch_rule_name(owned_agent_name),
                key="vonage_owned_dispatch_name",
            )
            owned_room_prefix = st.text_input("Room prefix", value="inbound", key="vonage_owned_room_prefix")
            owned_hide_phone_number = st.checkbox("Hide phone number in LiveKit room", value=False, key="vonage_owned_hide")

            if st.button("Attach owned number to agent", use_container_width=True):
                try:
                    result = bind_vonage_number_to_agent_backend(
                        agent_name=owned_agent_name,
                        country=str(selected_owned_number["country"]),
                        msisdn=normalize_msisdn(str(selected_owned_number["msisdn"])),
                        livekit_sip_uri=owned_livekit_sip_uri.strip() or None,
                        sip_config=SIPProvisionConfig(
                            phone_number=validate_e164(str(selected_owned_number["e164"])),
                            trunk_friendly_name=owned_trunk_name.strip(),
                            dispatch_rule_name=owned_dispatch_rule_name.strip(),
                            room_prefix=owned_room_prefix.strip() or "inbound",
                            hide_phone_number=owned_hide_phone_number,
                        ),
                    )
                    st.success("Vonage number attached to the selected agent.")
                    st.code(json.dumps(result, indent=2), language="json")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("Load owned Vonage numbers and create at least one saved agent before attaching a number.")

        inventory = st.session_state["livekit_sip_inventory"]
        if inventory is not None:
            bindings = inventory.get("bindings") or []
            if bindings:
                st.write("Current bindings")
                st.dataframe(bindings, use_container_width=True)

            rule_items = (inventory.get("dispatch_rules") or {}).get("items") or []
            if rule_items:
                st.write("Dispatch rules")
                st.dataframe(rule_items, use_container_width=True)
                selected_rule = st.selectbox(
                    "Dispatch rule",
                    options=rule_items,
                    format_func=display_dispatch_rule_option,
                    key="selected_dispatch_rule",
                )
                dispatch_rule_id = str(selected_rule.get("sip_dispatch_rule_id") or "")
                dispatch_payload_default = {
                    "name": selected_rule.get("name"),
                    "rule": selected_rule.get("rule") or {},
                    "room_config": selected_rule.get("room_config") or {},
                    "trunk_ids": selected_rule.get("trunk_ids") or [],
                    "hide_phone_number": selected_rule.get("hide_phone_number", False),
                }
                dispatch_payload_text = st.text_area(
                    "Update dispatch rule JSON",
                    value=json.dumps(dispatch_payload_default, indent=2),
                    height=240,
                    key=f"dispatch_payload_{dispatch_rule_id}",
                )
                rule_action_col1, rule_action_col2 = st.columns(2)
                with rule_action_col1:
                    if st.button("Update dispatch rule", key=f"update_rule_{dispatch_rule_id}", use_container_width=True):
                        try:
                            payload = parse_json_dict(dispatch_payload_text, "Dispatch rule JSON")
                            result = update_livekit_dispatch_rule_backend(
                                dispatch_rule_id,
                                payload,
                                base_url=telephony_base_url.strip() or None,
                            )
                            st.success("Dispatch rule updated.")
                            st.code(json.dumps(result, indent=2), language="json")
                            st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                        except Exception as exc:
                            st.error(str(exc))
                with rule_action_col2:
                    if st.button("Delete dispatch rule", key=f"delete_rule_{dispatch_rule_id}", use_container_width=True):
                        try:
                            result = delete_livekit_dispatch_rule_backend(
                                dispatch_rule_id,
                                base_url=telephony_base_url.strip() or None,
                            )
                            st.success("Dispatch rule deleted.")
                            st.code(json.dumps(result, indent=2), language="json")
                            st.session_state["livekit_sip_inventory"] = livekit_sip_inventory_backend(telephony_base_url.strip() or None)
                        except Exception as exc:
                            st.error(str(exc))
            else:
                st.info("No dispatch rules found yet.")
        else:
            st.info("Refresh telephony inventory to review and edit current bindings.")
