"""
VVA Skill and Automation Toolset for LiveKit voice agents.

Provides two tools:
- execute_skill: runs a skill (script or LLM-guided) via TechOps
- trigger_automation: triggers a pre-built automation agent via TechOps

Both support background execution with voice progress updates via ctx.update().

Usage:
    In agent config, skills and automations are configured.
    At worker startup, this toolset is built from the config and added to the session.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp

from livekit.agents import function_tool

try:
    from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
except ImportError:
    AsyncRunContext = Any  # type: ignore[assignment]
    AsyncToolset = None  # type: ignore[assignment]

AsyncToolsetBase = AsyncToolset if AsyncToolset is not None else object


logger = logging.getLogger("voice-agent-vva-skills")


def _get_api_url() -> str:
    return (
        os.getenv("TECHOPS_RAG_API_URL")
        or os.getenv("TECHOPS_API_URL")
        or os.getenv("PLATFORM_AGENT_PROXY_URL")
        or ""
    ).rstrip("/")


def _get_vva_token() -> str:
    return os.getenv("VVA_RAG_INTERNAL_TOKEN", "")


async def _http_post(url: str, payload: dict, timeout_sec: float = 120.0) -> dict:
    token = _get_vva_token()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["X-VVA-Token"] = token

    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            return await resp.json()


async def _http_get(url: str) -> dict:
    token = _get_vva_token()
    headers: dict[str, str] = {}
    if token:
        headers["X-VVA-Token"] = token

    timeout = aiohttp.ClientTimeout(total=30.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            return await resp.json()


async def _save_tool_event(ctx: Any, tool_name: str, result: dict) -> None:
    """Save tool call event to call transcript."""
    try:
        session_info = getattr(ctx, "session", ctx)
        if hasattr(session_info, "userdata"):
            session_info = session_info.userdata
        db_proxy = getattr(session_info, "db_proxy", None)
        call_id = getattr(session_info, "call_id", None)
        if not db_proxy or not call_id:
            return

        ok = result.get("ok", result.get("status") == "COMPLETED")
        content = f"Skill: {tool_name}"
        if ok and result.get("result"):
            r = result["result"]
            if isinstance(r, dict):
                summary = r.get("summary", r.get("status", ""))
                content = f"Skill: {tool_name} → {str(summary)[:200]}"
            else:
                content = f"Skill: {tool_name} → {str(r)[:200]}"
        elif result.get("error"):
            content = f"Skill: {tool_name} → Error: {result['error'][:200]}"

        import json as _json
        await db_proxy.save_call_event(
            call_id=call_id,
            event_type="tool_call",
            speaker="tool",
            content_text=content,
            content_json={"tool_name": tool_name, "type": "skill", "ok": ok},
        )
    except Exception as exc:
        logger.debug("Failed to save skill tool event: %s", exc)


class VvaSkillToolset(AsyncToolsetBase):
    """AsyncToolset for VVA skill and automation execution."""

    def __init__(
        self,
        skills: list[dict[str, Any]],
        automations: list[dict[str, Any]],
        *,
        poll_interval_seconds: float = 2.0,
    ):
        self.skills = skills or []
        self.automations = automations or []
        self.api_url = _get_api_url()
        self.poll_interval_seconds = poll_interval_seconds
        super().__init__(id="vva_skills", on_duplicate_call="allow")

    def _find_skill(self, name: str) -> dict | None:
        for s in self.skills:
            if s.get("name", "").lower() == name.lower() or s.get("skill_id") == name:
                return s
        return None

    def _find_automation(self, name: str) -> dict | None:
        for a in self.automations:
            if a.get("name", "").lower() == name.lower() or a.get("agent_id") == name:
                return a
        return None

    @function_tool
    async def execute_skill(
        self,
        ctx: AsyncRunContext,
        skill_name: str,
        arguments: dict | None = None,
    ) -> dict[str, Any]:
        """Execute a skill for a specific procedure.

        Use this when the caller needs a multi-step procedure like password reset,
        system reboot, health check, or investigation. The skill runs server-side.

        For scripted skills, results return in 2-3 seconds.
        For investigation skills, results may take 30-60 seconds — inform the caller.

        Args:
            skill_name: Name of the skill to execute.
            arguments: Optional arguments for the skill (varies per skill).
        """
        arguments = arguments or {}
        skill = self._find_skill(skill_name)
        if not skill:
            return {"ok": False, "error": f"Skill '{skill_name}' not found"}

        # Call TechOps — it auto-detects script vs instructions
        logger.info("Executing skill: %s", skill_name)

        try:
            result = await _http_post(f"{self.api_url}/api/v1/vva/skill/execute", {
                "skill_id": skill["skill_id"],
                "script_name": skill.get("script_name"),
                "arguments": arguments,
                "mcp_bindings": skill.get("mcp_bindings", {}),
            })
        except Exception as exc:
            logger.error("Skill execution failed: %s", exc)
            return {"ok": False, "error": str(exc)}

        await _save_tool_event(ctx, f"skill:{skill_name}", result)

        # If mode=instructions, the LLM should follow the returned instructions
        # using its available MCP tools
        if result.get("mode") == "instructions":
            return {
                "ok": True,
                "mode": "instructions",
                "instructions": result.get("result", {}).get("instructions", ""),
                "message": "Follow these instructions to complete the task using your available tools.",
            }

        return result

    @function_tool
    async def trigger_automation(
        self,
        ctx: AsyncRunContext,
        automation_name: str,
        task: str,
    ) -> dict[str, Any]:
        """Trigger a pre-built automation agent for complex tasks.

        Use this for multi-skill tasks like full IT triage, infrastructure
        diagnostics, or complex investigations that span multiple systems.

        The automation runs in background. Continue talking to the caller
        while it processes.

        Args:
            automation_name: Name of the automation to trigger.
            task: Description of what needs to be done.
        """
        automation = self._find_automation(automation_name)
        if not automation:
            return {"ok": False, "error": f"Automation '{automation_name}' not found"}

        logger.info("Triggering automation: %s", automation_name)
        await ctx.update(f"Starting {automation_name}. This may take a minute or two.")

        try:
            resp = await _http_post(
                f"{self.api_url}/api/v1/vva/executions?agent_id={automation['agent_id']}&task={task}",
                {},
            )
        except Exception as exc:
            logger.error("Failed to trigger automation: %s", exc)
            return {"ok": False, "error": str(exc)}

        execution_id = resp.get("execution_id")
        if not execution_id:
            return {"ok": False, "error": resp.get("detail", "Failed to start automation")}

        return await self._poll_execution(ctx, execution_id, f"automation:{automation_name}")

    async def _poll_execution(self, ctx: AsyncRunContext, execution_id: str, label: str) -> dict:
        """Poll an execution for completion, sending progress updates via ctx.update()."""
        last_step = ""

        while True:
            try:
                status = await _http_get(
                    f"{self.api_url}/api/v1/vva/executions/{execution_id}"
                )
            except Exception as exc:
                logger.error("Failed to poll execution %s: %s", execution_id, exc)
                return {"ok": False, "error": f"Polling failed: {exc}"}

            # Send progress updates
            current_step = status.get("current_step", "")
            if current_step and current_step != last_step:
                last_step = current_step
                try:
                    await ctx.update(f"Progress: {current_step}")
                except Exception:
                    pass

            exec_status = status.get("status", "")

            if exec_status in ("COMPLETED", "completed"):
                try:
                    await ctx.update("Complete. Let me share the findings.")
                except Exception:
                    pass
                result = {"ok": True, "result": status.get("result"), "execution_id": execution_id}
                await _save_tool_event(ctx, label, result)
                return result

            if exec_status in ("FAILED", "failed", "CANCELLED", "cancelled"):
                result = {"ok": False, "error": status.get("error", "Execution failed"), "execution_id": execution_id}
                await _save_tool_event(ctx, label, result)
                return result

            await asyncio.sleep(self.poll_interval_seconds)


def build_vva_skill_toolset(
    skills: list[dict[str, Any]] | None = None,
    automations: list[dict[str, Any]] | None = None,
) -> VvaSkillToolset | None:
    """Build the VVA skill toolset. Returns None if AsyncToolset not available."""
    if AsyncToolset is None:
        logger.warning("AsyncToolset not available. Upgrade livekit-agents>=1.5.2.")
        return None

    if not skills and not automations:
        return None

    return VvaSkillToolset(
        skills=skills or [],
        automations=automations or [],
        poll_interval_seconds=float(os.getenv("VVA_SKILL_POLL_INTERVAL_SECONDS", "2.0")),
    )
