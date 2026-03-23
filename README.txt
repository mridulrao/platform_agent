Config-driven LiveKit voice agent builder

Flow
1) Create an agent config from the UI or Python service.
2) Save the config into `agent_config/<agent-name>.json`.
3) Start a LiveKit worker for that config:
   `TARGET_AGENT_NAME=<agent-name> python3 -m livekit_agents.create_agent dev`
4) Provision SIP separately when needed:
   `TARGET_AGENT_NAME=<agent-name> python3 livekit_trunks/create_inbound_trunk.py`
5) The dispatch rule points to the agent name you provision.

Azure OpenAI
- LiveKit's Python OpenAI plugin supports Azure OpenAI via `openai.LLM.with_azure(...)` for LLMs and `openai.TTS.with_azure(...)` for TTS.
- Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `OPENAI_API_VERSION` in your environment before starting agents that use the `azure_openai` provider.
- In the UI, the default LLM provider is `azure_openai`. Set `llm.kwargs.azure_deployment` to your Azure deployment name.

ElevenLabs
- LiveKit supports ElevenLabs as both `tts.provider="elevenlabs"` and `stt.provider="elevenlabs"`.
- Set `ELEVEN_API_KEY` in your environment before starting agents that use ElevenLabs.
- Typical TTS config uses `tts.kwargs.voice_id` and optionally `tts.model`, for example `eleven_flash_v2_5`.
- Typical STT config uses `stt.model`, for example `scribe_v2_realtime`.

Main files
- `streamlit_app.py`: simple UI for creating configs
- `backend_service.py`: backend entrypoint used by the UI
- `agent_config/schema.py`: validated config model
- `agent_config/store.py`: config persistence
- `livekit_agents/factory.py`: provider-specific LLM/STT/TTS builders
- `livekit_agents/runtime.py`: generic config-driven agent session
- `livekit_trunks/provision.py`: SIP trunk + dispatch provisioning
