Config-driven LiveKit voice agent builder

Flow
1) Create an agent config from the UI or Python service.
2) Save the config into `agent_config/<agent-name>.json`.
3) Start a LiveKit worker for that config:
   `python3 livekit_agents/create_agent.py --agent-name <agent-name>`
4) Provision SIP separately when needed:
   `TARGET_AGENT_NAME=<agent-name> python3 livekit_trunks/create_inbound_trunk.py`
5) The dispatch rule points to the agent name you provision.

Main files
- `streamlit_app.py`: simple UI for creating configs
- `backend_service.py`: backend entrypoint used by the UI
- `agent_config/schema.py`: validated config model
- `agent_config/store.py`: config persistence
- `livekit_agents/factory.py`: provider-specific LLM/STT/TTS builders
- `livekit_agents/runtime.py`: generic config-driven agent session
- `livekit_trunks/provision.py`: SIP trunk + dispatch provisioning
