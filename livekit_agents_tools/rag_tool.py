"""
RAG knowledge base search tool for LiveKit voice agents.

Searches TechOps platform datasets via the RAG API. Dataset IDs and RAG config
are loaded from the agent's config at session start and stored in session userdata.

Usage in agent config:
  "tools": ["livekit_agents_tools.rag_tool.search_knowledge_base"],
  "datasets": [
    {"dataset_id": "uuid-1", "name": "IT KB", "rag_config": {"enabled": true, "top_k": 5}}
  ]
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp
from livekit.agents import RunContext, function_tool


logger = logging.getLogger("voice-agent-rag-tool")


def _get_session_info(context: RunContext) -> Any:
    return getattr(context.session, "userdata", None)


async def _save_tool_event(context: RunContext, query: str, result: dict) -> None:
    """Save RAG search as a transcript event via DB Proxy."""
    try:
        session_info = _get_session_info(context)
        db_proxy = getattr(session_info, "db_proxy", None)
        call_id = getattr(session_info, "call_id", None)
        if not db_proxy or not call_id:
            return

        result_count = len(result.get("results", []))
        ok = result.get("ok", False)
        # Extract unique source documents from results
        sources = list({r.get("source", "") for r in result.get("results", []) if r.get("source")})
        content = f"Tool: search_knowledge_base"
        if ok:
            content = f"Tool: search_knowledge_base({query[:60]}) -> {result_count} results"
            if sources:
                content += f" from {', '.join(s for s in sources[:3])}"
        else:
            content = f"Tool: search_knowledge_base({query[:60]}) -> {result.get('message', 'failed')}"

        await db_proxy.save_call_event(
            call_id=call_id,
            event_type="tool_call",
            speaker="tool",
            content_text=content,
            content_json={"tool_name": "search_knowledge_base", "arguments": {"query": query}, "result_ok": ok, "result_count": result_count, "sources": sources},
        )
    except Exception as exc:
        logger.warning("Failed to save RAG tool event: %s", exc)


def _get_rag_api_url(session_info: Any) -> str:
    url = (
        getattr(session_info, "rag_api_url", None)
        or os.getenv("TECHOPS_RAG_API_URL")
        or os.getenv("PLATFORM_AGENT_PROXY_URL")
        or ""
    )
    if not url:
        raise RuntimeError("RAG API URL not configured. Set TECHOPS_RAG_API_URL.")
    return url.rstrip("/")


def _get_datasets(session_info: Any) -> list[dict[str, Any]]:
    datasets = getattr(session_info, "datasets", None) or []
    return [ds for ds in datasets if ds.get("rag_config", {}).get("enabled", True)]


@function_tool
async def search_knowledge_base(
    context: RunContext,
    query: str,
) -> dict[str, Any]:
    """Search the knowledge base for information relevant to the caller's question.

    Use this tool when the caller asks a question that might be answered by
    documentation, articles, or knowledge base content. Returns the most
    relevant passages from linked knowledge bases.

    Args:
        query: The search query based on the caller's question.
    """
    session_info = _get_session_info(context)
    datasets = _get_datasets(session_info)

    if not datasets:
        err = {"ok": False, "message": "No knowledge base datasets configured for this agent.", "results": []}
        await _save_tool_event(context, query, err)
        return err

    rag_api_url = _get_rag_api_url(session_info)
    dataset_ids = [ds["dataset_id"] for ds in datasets]

    # Use the highest top_k and lowest min_similarity across all datasets
    top_k = max(ds.get("rag_config", {}).get("top_k", 5) for ds in datasets)
    min_similarity = min(ds.get("rag_config", {}).get("min_similarity", 0.5) for ds in datasets)

    payload = {
        "query": query,
        "dataset_ids": dataset_ids,
        "config": {
            "top_k": top_k,
            "min_similarity": min_similarity,
        },
    }

    logger.info(
        "RAG search: query=%r, datasets=%s, top_k=%d, min_similarity=%.2f",
        query[:80], dataset_ids, top_k, min_similarity,
    )

    try:
        vva_token = os.getenv("VVA_RAG_INTERNAL_TOKEN", "")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if vva_token:
            headers["X-VVA-Token"] = vva_token

        timeout = aiohttp.ClientTimeout(total=15.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{rag_api_url}/api/v1/rag/vva/query",
                json=payload,
                headers=headers,
            ) as response:
                if response.status >= 400:
                    error_body = await response.text()
                    logger.error("RAG API error %d: %s", response.status, error_body[:200])
                    return {
                        "ok": False,
                        "message": f"Knowledge base search failed (status {response.status})",
                        "results": [],
                    }

                data = await response.json()
    except asyncio.TimeoutError:
        logger.error("RAG search timed out for query: %s", query[:80])
        return {
            "ok": False,
            "message": "Knowledge base search timed out.",
            "results": [],
        }
    except Exception as exc:
        logger.error("RAG search failed: %s", exc)
        return {
            "ok": False,
            "message": f"Knowledge base search error: {exc}",
            "results": [],
        }

    # Format results for voice-friendly consumption
    raw_results = data.get("results", [])
    results = []
    for r in raw_results[:top_k]:
        results.append({
            "text": r.get("chunk_text", r.get("text", "")),
            "score": round(r.get("similarity", r.get("score", 0)), 3),
            "source": r.get("document_filename", r.get("document_name", r.get("source", ""))),
            "page": r.get("start_page"),
        })

    # Also include the LLM-generated answer if available
    answer = data.get("answer")

    logger.info("RAG search returned %d results for query: %s", len(results), query[:80])

    result = {
        "ok": True,
        "message": f"Found {len(results)} relevant results.",
        "results": results,
        "answer": answer,
        "query": query,
    }
    await _save_tool_event(context, query, result)
    return result
