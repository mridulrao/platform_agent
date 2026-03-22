from __future__ import annotations

import json
import re

import streamlit as st

from agent_config.schema import AgentConfig, ProviderConfig, SIPProvisionConfig, WorkerConfig
from agent_config.store import list_agent_configs
from backend_service import create_agent_backend, provision_sip_backend, serialize_result


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
st.caption("Create a generic agent config, then provision SIP separately if needed.")

with st.form("agent_form"):
    name = st.text_input("Agent name", placeholder="sales_agent_india")
    system_prompt = st.text_area("System prompt", height=180)
    tools_raw = st.text_area(
        "Tool import paths (one per line)",
        value="livekit_agents_tools.session_tools.end_call",
        height=80,
    )

    st.subheader("LLM")
    llm_provider = st.text_input("LLM provider", value="openai")
    llm_model = st.text_input("LLM model", value="gpt-4.1-mini")
    llm_kwargs = st.text_area("LLM kwargs (JSON)", value="{}", height=110)

    st.subheader("STT")
    stt_provider = st.text_input("STT provider", value="deepgram")
    stt_model = st.text_input("STT model", value="nova-3")
    stt_kwargs = st.text_area(
        "STT kwargs (JSON)",
        value='{\n  "language": "en-IN",\n  "detect_language": false,\n  "interim_results": true\n}',
        height=140,
    )

    st.subheader("TTS")
    tts_provider = st.text_input("TTS provider", value="google")
    tts_model = st.text_input("TTS model", value="")
    tts_kwargs = st.text_area(
        "TTS kwargs (JSON)",
        value='{\n  "language": "en-IN",\n  "voice_name": "en-IN-Chirp3-HD-Achernar",\n  "gender": "female",\n  "credentials_file": "google_creds.json"\n}',
        height=160,
    )

    st.subheader("Worker options")
    worker_agent_name = st.text_input("Worker agent_name", value="")
    worker_port = st.number_input("Port", min_value=1, max_value=65535, value=8082)
    worker_job_memory_warn_mb = st.number_input("job_memory_warn_mb", min_value=1, value=2000)
    worker_shutdown_timeout = st.number_input("shutdown_process_timeout", min_value=1.0, value=80.0)
    worker_initialize_timeout = st.number_input("initialize_process_timeout", min_value=1, value=540)
    worker_num_idle_processes = st.number_input("num_idle_processes", min_value=0, value=1)

    st.subheader("Session options")
    enable_noise_cancellation = st.checkbox("Enable telephony noise cancellation", value=True)
    session_kwargs = st.text_area("session.start kwargs (JSON)", value="{}", height=110)

    submitted = st.form_submit_button("Create agent")

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
                "kwargs": parse_json_dict(session_kwargs, "Session kwargs"),
            },
        )
        result = create_agent_backend(config)
        st.success("Agent config created successfully.")
        st.code(serialize_result(result), language="json")
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
st.subheader("Saved agents")
for saved in list_agent_configs():
    st.write(f"- {saved.name}")
