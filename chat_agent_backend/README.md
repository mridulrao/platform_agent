# Example Chat Agent Backend

This package is a minimal OpenAI-compatible facade for an existing chat agent.
It demonstrates mixed tool handling:

- backend-owned tools are executed inside this FastAPI service
- LiveKit-owned tools are passed through to the model and returned to LiveKit

LiveKit can use it through the existing OpenAI plugin by setting:

```json
{
  "provider": "openai",
  "model": "existing-chat-agent",
  "kwargs": {
    "base_url": "http://127.0.0.1:9000/v1",
    "api_key": "dev-key"
  }
}
```

Run locally:

```bash
EXAMPLE_CHAT_AGENT_API_KEY=dev-key ./.venv/bin/uvicorn chat_agent_backend.app:app --host 127.0.0.1 --port 9000
```

The important endpoint is:

```text
POST /v1/chat/completions
```

By default, the backend forwards the request to Azure OpenAI Chat Completions.
It reads:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `OPENAI_API_VERSION`

You can optionally set `CHAT_AGENT_AZURE_DEPLOYMENT` to override the deployment
used by the backend. If you do not set it, the backend tries to extract the
deployment name from `AZURE_OPENAI_ENDPOINT`.

The model value from LiveKit can stay as a logical name such as
`existing-chat-agent`.

The sample backend tools are:

- `get_current_time`
- `lookup_order_status`

The typical flow is:

1. LiveKit sends messages plus LiveKit tool schemas like `end_call` and `transfer_call`.
2. This backend adds its own backend tool schemas.
3. Azure OpenAI may request backend tools first.
4. This backend executes those backend tools and calls Azure OpenAI again.
5. Once the model is done, it may still return a LiveKit tool call like `transfer_call`.
6. LiveKit receives that final response and executes the phone action.

Try prompts like:

- "What time is it in Los Angeles?"
- "Check order ABC-123 for me."
- "Transfer me to a human agent."
- "Check order ABC-123 and then transfer me to a human."

If you later add your own RAG or orchestration, put it in the backend tool
execution path or before the OpenAI call. Keep the outer OpenAI-compatible
request and response shape stable for LiveKit.

Voice-only actions such as `end_call` and `transfer_call` should remain LiveKit
tools. The backend should decide when to call them by returning OpenAI-style
`tool_calls`; LiveKit should execute the actual phone-side action.
