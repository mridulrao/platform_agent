"""
MCP tool proxy for LiveKit voice agents.

Dynamically registers TechOps MCP server tools as LiveKit @function_tool
functions. Tool calls are proxied to TechOps via HTTP — no MCP infrastructure
needed in Platform Agent.

Usage in agent config:
  "mcp_servers": [
    {
      "mcp_server_id": "uuid",
      "name": "ServiceNow",
      "allowed_tools": ["create_incident", "get_incident"],
      "tools": [
        {"name": "create_incident", "description": "Create an incident", "inputSchema": {...}},
        ...
      ]
    }
  ]

At worker startup, this module generates a @function_tool for each allowed
MCP tool. The LLM sees proper tool names, descriptions, and parameter schemas.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp
from livekit.agents import RunContext, function_tool


logger = logging.getLogger("voice-agent-mcp-proxy")


def _get_session_info(context: Any) -> Any:
    # Try different ways to get session userdata
    if hasattr(context, "session") and hasattr(context.session, "userdata"):
        return context.session.userdata
    if hasattr(context, "userdata"):
        return context.userdata
    return None


def _get_mcp_api_url() -> str:
    url = (
        os.getenv("TECHOPS_MCP_API_URL")
        or os.getenv("TECHOPS_RAG_API_URL")
        or os.getenv("PLATFORM_AGENT_PROXY_URL")
        or ""
    )
    if not url:
        raise RuntimeError("TechOps API URL not configured for MCP proxy.")
    return url.rstrip("/")


def _get_vva_token() -> str:
    return os.getenv("VVA_RAG_INTERNAL_TOKEN", "")


async def _save_tool_event(context: RunContext, tool_name: str, arguments: dict, result: dict) -> None:
    """Save tool call as a transcript event via DB Proxy."""
    try:
        session_info = _get_session_info(context)
        db_proxy = getattr(session_info, "db_proxy", None)
        call_id = getattr(session_info, "call_id", None)
        if not db_proxy or not call_id:
            return

        import json as _json
        content = f"Tool: {tool_name}"
        if result.get("ok") and result.get("result"):
            r = result["result"]
            if isinstance(r, dict):
                # Extract key info for display
                num = r.get("number") or r.get("task_effective_number") or ""
                desc = r.get("short_description") or r.get("message") or ""
                content = f"Tool: {tool_name} → {num} {desc}".strip()
            else:
                content = f"Tool: {tool_name} → {str(r)[:200]}"
        elif result.get("error"):
            content = f"Tool: {tool_name} → Error: {result['error'][:200]}"

        await db_proxy.save_call_event(
            call_id=call_id,
            event_type="tool_call",
            speaker="tool",
            content_text=content,
            content_json={"tool_name": tool_name, "arguments": arguments, "result_ok": result.get("ok", False)},
        )
    except Exception as exc:
        logger.warning("Failed to save tool event for %s: %s", tool_name, exc)


async def _call_mcp_proxy(context: RunContext, mcp_server_id: str, tool_name: str, arguments: dict) -> dict:
    """Call TechOps MCP proxy endpoint."""
    api_url = _get_mcp_api_url()
    token = _get_vva_token()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["X-VVA-Token"] = token

    payload = {
        "mcp_server_id": mcp_server_id,
        "tool_name": tool_name,
        "arguments": arguments,
    }

    logger.info("MCP proxy call: server=%s tool=%s", mcp_server_id, tool_name)

    try:
        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{api_url}/api/v1/vva/mcp/call",
                json=payload,
                headers=headers,
            ) as response:
                data = await response.json()
                if not data.get("ok"):
                    logger.error("MCP proxy error: %s", data.get("error"))
                    await _save_tool_event(context, tool_name, arguments, data)
                    return {
                        "ok": False,
                        "error": data.get("error", "Unknown error"),
                    }
                await _save_tool_event(context, tool_name, arguments, data)
                return data
    except Exception as exc:
        logger.error("MCP proxy request failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def build_mcp_proxy_tools(mcp_servers: list[dict[str, Any]]) -> list[Any]:
    """Build LiveKit @function_tool functions for each MCP tool.

    Called at worker startup. Returns a list of tool functions that can be
    passed to the agent's tools list.
    """
    tools = []

    for server in mcp_servers:
        server_id = server.get("mcp_server_id", "")
        server_name = server.get("name", "MCP Server")
        allowed = set(server.get("allowed_tools") or [])
        server_tools = server.get("tools") or []

        for tool_def in server_tools:
            tool_name = tool_def.get("name", "")
            if not tool_name:
                continue

            # Apply allowlist if specified
            if allowed and tool_name not in allowed:
                continue

            description = tool_def.get("description", f"Tool from {server_name}")

            # Build the function_tool dynamically — pass schema for typed params
            input_schema = tool_def.get("inputSchema") or tool_def.get("input_schema")
            tool_fn = _make_proxy_tool(server_id, tool_name, description, input_schema)
            tools.append(tool_fn)

            logger.info("Registered MCP proxy tool: %s (from %s)", tool_name, server_name)

    return tools


def _make_proxy_tool(mcp_server_id: str, tool_name: str, description: str, input_schema: dict | None = None):
    """Create a @function_tool that proxies to TechOps MCP endpoint.

    If input_schema is provided, we build typed parameters so the LLM sees
    proper argument names and types. Otherwise falls back to a generic
    arguments dict.
    """
    if input_schema and input_schema.get("properties"):
        # Build a function with explicit typed parameters from the MCP schema
        return _make_typed_proxy_tool(mcp_server_id, tool_name, description, input_schema)

    # Fallback: generic arguments dict
    @function_tool(name=tool_name, description=description)
    async def _proxy_tool(context: RunContext, arguments: dict) -> dict:
        """Pass arguments as a JSON object matching the tool's expected input."""
        return await _call_mcp_proxy(context, mcp_server_id, tool_name, arguments)

    return _proxy_tool


def _make_typed_proxy_tool(mcp_server_id: str, tool_name: str, description: str, input_schema: dict):
    """Build a proxy tool with explicit parameters derived from inputSchema.

    This gives the LLM proper parameter names, types, and descriptions
    instead of a generic 'arguments' dict.
    """
    import inspect

    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    # Map JSON schema types to Python type hints
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # Build parameter list for the function signature
    # Required params MUST come before optional params
    params = [inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=RunContext)]

    # Sort: required first, then optional
    sorted_props = sorted(
        properties.items(),
        key=lambda item: 0 if item[0] in required else 1,
    )

    for prop_name, prop_schema in sorted_props:
        prop_type = type_map.get(prop_schema.get("type", "string"), str)
        is_required = prop_name in required

        if is_required:
            param = inspect.Parameter(
                prop_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=prop_type,
            )
        else:
            param = inspect.Parameter(
                prop_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=prop_type | None,
            )
        params.append(param)

    sig = inspect.Signature(params, return_annotation=dict)

    # Capture param names for argument extraction
    param_names = [p.name for p in params if p.name != "context"]

    # Build the actual async function that accepts positional args
    # LiveKit SDK passes args positionally based on the signature
    async def _impl(*args, **kwargs) -> dict:
        # First arg is always context
        context = args[0] if args else kwargs.get("context")
        # Remaining args are the tool parameters in signature order
        all_args = {}

        # Map positional args to param names (skip context at index 0)
        for i, val in enumerate(args[1:]):  # skip context
            if i < len(param_names):
                all_args[param_names[i]] = val

        # Merge keyword args
        all_args.update(kwargs)

        # Remove context if it leaked in
        all_args.pop("context", None)

        # Filter out None values (optional params not provided)
        arguments = {k: v for k, v in all_args.items() if v is not None}
        return await _call_mcp_proxy(context, mcp_server_id, tool_name, arguments)

    # Set the signature so LiveKit SDK can introspect it
    _impl.__signature__ = sig
    _impl.__name__ = tool_name
    _impl.__qualname__ = tool_name
    _impl.__doc__ = description
    _impl.__annotations__ = {p.name: p.annotation for p in params if p.annotation != inspect.Parameter.empty}
    _impl.__annotations__["return"] = dict

    return function_tool(name=tool_name, description=description)(_impl)
