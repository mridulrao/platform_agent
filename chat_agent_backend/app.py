from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError, AsyncAzureOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None


app = FastAPI(title="Example OpenAI-Compatible Chat Agent Backend")


BACKEND_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date and time for the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name like America/Los_Angeles.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order_status",
            "description": "Look up a fake order status for demo and testing flows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Customer order ID.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _require_api_key(authorization: str | None) -> None:
    expected_api_key = os.getenv("EXAMPLE_CHAT_AGENT_API_KEY")
    if not expected_api_key:
        return

    expected_header = f"Bearer {expected_api_key}"
    if authorization != expected_header:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _azure_endpoint_config() -> tuple[str, str | None, str]:
    raw_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not raw_endpoint:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_ENDPOINT is not configured")

    parsed = urlparse(raw_endpoint)
    base_endpoint = raw_endpoint.rstrip("/")
    deployment = os.getenv("CHAT_AGENT_AZURE_DEPLOYMENT")
    api_version = os.getenv("OPENAI_API_VERSION") or os.getenv("CHAT_AGENT_AZURE_API_VERSION")

    if parsed.scheme and parsed.netloc:
        base_endpoint = f"{parsed.scheme}://{parsed.netloc}"
        path_parts = [part for part in parsed.path.split("/") if part]
        if "deployments" in path_parts:
            index = path_parts.index("deployments")
            if index + 1 < len(path_parts):
                deployment = deployment or path_parts[index + 1]
        query = parse_qs(parsed.query)
        if query.get("api-version"):
            api_version = api_version or query["api-version"][0]

    if not api_version:
        raise HTTPException(status_code=500, detail="OPENAI_API_VERSION is not configured")

    return base_endpoint, deployment, api_version


def _openai_client() -> AsyncAzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_API_KEY is not configured")

    base_endpoint, _, api_version = _azure_endpoint_config()
    return AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=base_endpoint,
        api_version=api_version,
    )


def _azure_chat_model() -> str:
    _, deployment, _ = _azure_endpoint_config()
    model = os.getenv("CHAT_AGENT_AZURE_DEPLOYMENT") or deployment
    if not model:
        raise HTTPException(status_code=500, detail="Azure deployment is not configured")
    return model


def _backend_tool_map() -> dict[str, dict[str, Any]]:
    tool_map: dict[str, dict[str, Any]] = {}
    for tool in BACKEND_TOOLS:
        function = tool.get("function") or {}
        name = function.get("name")
        if name:
            tool_map[name] = tool
    return tool_map


def _merged_tools(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [*request.tools, *BACKEND_TOOLS]


def _tool_name(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function") or {}
    return function.get("name")


def _tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw_arguments = (tool_call.get("function") or {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    return ""


async def _run_backend_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_current_time":
        timezone = arguments.get("timezone") or "America/Los_Angeles"
        try:
            current_time = datetime.now(ZoneInfo(timezone)).isoformat()
        except ZoneInfoNotFoundError:
            current_time = datetime.now().astimezone().isoformat()
            timezone = "system-local"
        return json.dumps(
            {
                "timezone": timezone,
                "current_time": current_time,
                "note": "Sample backend tool result.",
            }
        )

    if name == "lookup_order_status":
        order_id = str(arguments.get("order_id") or "").strip()
        if not order_id:
            return json.dumps({"error": "order_id is required"})
        return json.dumps(
            {
                "order_id": order_id,
                "status": "processing",
                "estimated_delivery": "2026-04-10",
                "note": "Sample backend tool result.",
            }
        )

    return json.dumps({"error": f"Unknown backend tool: {name}"})


def _base_messages(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    messages = [message.model_dump(mode="json", exclude_none=True) for message in request.messages]
    messages.insert(
        0,
        {
            "role": "system",
            "content": (
                "You are a voice-enabled assistant. "
                "You may use backend tools for business logic such as time lookup or order lookup. "
                "Use LiveKit tools like end_call and transfer_call only when you want the phone layer to act. "
                "Do not call a LiveKit tool until you are ready for that action to happen."
            ),
        },
    )
    return messages


def _chat_payload(
    *,
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stream: bool,
) -> dict[str, Any]:
    payload = request.model_dump(mode="json", exclude_none=True)
    payload["model"] = _azure_chat_model()
    payload["messages"] = messages
    payload["tools"] = _merged_tools(request)
    payload["stream"] = stream
    if not stream:
        payload.pop("stream_options", None)
    return payload


async def _call_openai_chat(
    *,
    request: ChatCompletionRequest,
    messages: list[dict[str, Any]],
    stream: bool,
):
    try:
        return await _openai_client().chat.completions.create(
            **_chat_payload(request=request, messages=messages, stream=stream)
        )
    except APIError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _resolve_backend_tool_calls(
    request: ChatCompletionRequest,
) -> dict[str, Any]:
    backend_tool_names = set(_backend_tool_map())
    messages = _base_messages(request)

    for _ in range(8):
        completion = await _call_openai_chat(request=request, messages=messages, stream=False)
        completion_dict = completion.model_dump(mode="json", exclude_none=True)
        choice = (completion_dict.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        backend_calls = [
            tool_call for tool_call in tool_calls if (_tool_name(tool_call) in backend_tool_names)
        ]

        if not backend_calls:
            return completion_dict

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": backend_calls,
            }
        )

        for tool_call in backend_calls:
            tool_name = _tool_name(tool_call)
            if not tool_name:
                continue
            tool_result = await _run_backend_tool(tool_name, _tool_arguments(tool_call))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": tool_result,
                }
            )

    raise HTTPException(status_code=500, detail="Exceeded backend tool resolution limit")


def _build_chunk(
    *,
    completion: dict[str, Any],
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    chunk = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "created": completion.get("created"),
        "model": completion.get("model"),
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"


async def _stream_from_completion(completion: dict[str, Any]) -> AsyncIterator[str]:
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    yield _build_chunk(completion=completion, delta={"role": "assistant"})

    content = _extract_text_content(message)
    if content:
        for word in content.split(" "):
            if word:
                yield _build_chunk(completion=completion, delta={"content": f"{word} "})

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") or {}
            yield _build_chunk(
                completion=completion,
                delta={
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_call.get("id"),
                            "type": "function",
                            "function": {
                                "name": function.get("name"),
                                "arguments": function.get("arguments", ""),
                            },
                        }
                    ]
                },
            )

    yield _build_chunk(completion=completion, delta={}, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend_tools": [tool["function"]["name"] for tool in BACKEND_TOOLS],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
):
    _require_api_key(authorization)
    completion = await _resolve_backend_tool_calls(request)

    if request.stream:
        return StreamingResponse(_stream_from_completion(completion), media_type="text/event-stream")

    return completion
