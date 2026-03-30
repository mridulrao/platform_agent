import asyncio
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder  # type: ignore

from indexing_config import indexing_config


logger = logging.getLogger(__name__)

RERANK_TOP_N = int(os.getenv("PGVECTOR_RERANK_TOP_N", str(indexing_config.rerank_top_n)))
USE_LLM_RERANK = bool(int(os.getenv("PGVECTOR_USE_LLM_RERANK", str(int(indexing_config.use_llm_rerank)))))
LLM_GROUP_SIZE = int(os.getenv("PGVECTOR_LLM_GROUP_SIZE", str(indexing_config.llm_group_size)))
LLM_RERANK_TIMEOUT = float(os.getenv("PGVECTOR_LLM_RERANK_TIMEOUT", str(indexing_config.llm_rerank_timeout)))
LLM_RERANK_MODEL = str(
    os.getenv(
        "PGVECTOR_LLM_RERANK_MODEL",
        indexing_config.llm_rerank_model,
    )
)
LLM_SMALL_POOL_THRESHOLD = int(
    os.getenv("PGVECTOR_LLM_SMALL_POOL_THRESHOLD", str(indexing_config.llm_small_pool_threshold))
)
CONTENT_SNIPPET_IN_PROMPT = int(
    os.getenv("PGVECTOR_CONTENT_SNIPPET_IN_PROMPT", str(indexing_config.content_snippet_in_prompt))
)
USE_CE_RERANK = bool(int(os.getenv("PGVECTOR_USE_CE_RERANK", str(int(indexing_config.use_ce_rerank)))))
CE_MODEL_NAME = str(
    os.getenv("PGVECTOR_CE_MODEL_NAME", indexing_config.ce_model_name)
)
CE_BATCH_SIZE = int(os.getenv("PGVECTOR_CE_BATCH_SIZE", str(indexing_config.ce_batch_size)))
CE_TIE_DEDUP_ENABLED = bool(int(os.getenv("PGVECTOR_CE_TIE_DEDUP_ENABLED", str(int(indexing_config.ce_tie_dedup_enabled)))))
CE_TIE_MARGIN = float(os.getenv("PGVECTOR_CE_TIE_MARGIN", str(indexing_config.ce_tie_margin)))
CE_TIE_TOPK = int(os.getenv("PGVECTOR_CE_TIE_TOPK", str(indexing_config.ce_tie_topk)))
CE_INTRA_FAMILY_CE = bool(int(os.getenv("PGVECTOR_CE_INTRA_FAMILY_CE", str(int(indexing_config.ce_intra_family_ce)))))
RERANK_STRATEGY = str(os.getenv("PGVECTOR_RERANK_STRATEGY", indexing_config.rerank_strategy)).strip().lower()
AUTO_PICK_RATIO = 999
AUTO_PICK_MIN_GAP = 999
TOP2_VS_REST_RATIO = float(os.getenv("KB_TOP2_VS_REST_RATIO", str(indexing_config.top2_vs_rest_ratio)))
KEEP_SCORE_RATIO = float(os.getenv("KB_KEEP_SCORE_RATIO", str(indexing_config.keep_score_ratio)))
MIN_CANDIDATES = int(os.getenv("KB_MIN_CANDIDATES", str(indexing_config.min_candidates)))
MAX_CANDIDATES = int(os.getenv("KB_MAX_CANDIDATES", str(indexing_config.max_candidates)))
DEBUG_RERANK = bool(int(os.getenv("PGVECTOR_DEBUG_RERANK", str(int(indexing_config.debug_rerank)))))
DEBUG_RERANK_TOPM = int(os.getenv("PGVECTOR_DEBUG_RERANK_TOPM", str(indexing_config.debug_rerank_topm)))

_CE_PREDICT_LOCK = threading.Lock()
_CE_MODEL = None
_CE_LOCK = threading.Lock()
_llm_rerank_client = None
_llm_rerank_client_lock = threading.Lock()
_SCENARIO_RE = re.compile(r"\s*-\s*scenario\s*\d+\s*$", re.IGNORECASE)


def _tokenize(text_in: str) -> List[str]:
    if not text_in:
        return []
    return [t for t in re.sub(r"[^a-zA-Z0-9]+", " ", text_in.lower()).strip().split() if len(t) > 1]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _get_ce_model():
    global _CE_MODEL
    if not USE_CE_RERANK or CrossEncoder is None:
        return None
    if _CE_MODEL is None:
        with _CE_LOCK:
            if _CE_MODEL is None:
                logger.info("Loading CrossEncoder reranker: %s", CE_MODEL_NAME)
                _CE_MODEL = CrossEncoder(CE_MODEL_NAME)
    return _CE_MODEL


def _candidate_text(kb: Dict[str, Any]) -> str:
    def _clean(x: Any) -> str:
        return str(x or "").strip()

    title = _clean(kb.get("article_title"))
    summary = _clean(kb.get("summary"))
    triggers = _clean(kb.get("query_triggers"))
    content = _clean(kb.get("content"))
    acronyms = _clean(kb.get("acronyms"))
    entities = _clean(kb.get("secondary_entities"))

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if triggers:
        parts.append(f"Triggers: {triggers}")
    if content:
        parts.append(f"Relevant content: {content}")
    meta = " ".join(x for x in [acronyms, entities] if x)
    if meta and len(meta) < 200:
        parts.append(f"Keywords: {meta}")
    return "\n".join(parts)


def _build_ce_query(q_raw: str, user_query_json: Dict[str, Any]) -> str:
    if not user_query_json:
        return q_raw
    parts: List[str] = [q_raw]
    kw = user_query_json.get("keywords") or {}
    specific = [k for k in (kw.get("specific") or []) if isinstance(k, str) and k.strip()]
    if specific:
        parts.append(" ".join(specific))
    issues = [i for i in (user_query_json.get("issues") or []) if isinstance(i, str) and i.strip()]
    if issues:
        parts.append(" ".join(issues))
    usecases = [u for u in (user_query_json.get("usecase") or []) if isinstance(u, str) and u.strip()]
    if usecases:
        parts.append(" ".join(usecases))
    search_phrases = [p for p in (user_query_json.get("search_phrases") or []) if isinstance(p, str) and p.strip()]
    if search_phrases:
        parts.append(" ".join(search_phrases))
    generic = [k for k in (kw.get("generic") or []) if isinstance(k, str) and k.strip()]
    if generic:
        parts.append(" ".join(generic))
    acronyms = [a for a in (user_query_json.get("acronyms") or []) if isinstance(a, str) and a.strip()]
    if acronyms:
        parts.append(" ".join(acronyms))
    article_type = (user_query_json.get("article_type") or "").strip()
    if article_type:
        parts.append(article_type)
    return " ".join(parts)


def _canonical_key(kb: Dict[str, Any]) -> str:
    topic = str(kb.get("primary_topic") or "").strip().lower()
    if topic:
        return f"topic:{topic}"
    title = str(kb.get("article_title") or "").strip().lower()
    title = _SCENARIO_RE.sub("", title)
    title = re.sub(r"\s+", " ", title).strip()
    if title:
        return f"title:{title}"
    return f"id:{str(kb.get('id') or '').strip()}"


def retrieval_score_for_tiebreak(kb: Dict[str, Any]) -> float:
    retrieval = kb.get("retrieval") or {}
    if isinstance(retrieval, dict) and retrieval.get("rrf_score") is not None:
        try:
            return float(retrieval.get("rrf_score") or 0.0)
        except Exception:
            return 0.0
    if isinstance(retrieval, dict) and retrieval.get("vector_distance") is not None:
        try:
            return -float(retrieval.get("vector_distance") or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _short(s: Any, n: int = 56) -> str:
    s = str(s or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _fmt_float(x: Any, default: str = "") -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return default


def debug_print_rerank_table(
    *,
    label: str,
    q_raw: str,
    window: List[Dict[str, Any]],
    ce_scores_by_id: Optional[Dict[str, float]] = None,
    topm: int = 10,
) -> None:
    if not DEBUG_RERANK:
        return
    ce_scores_by_id = ce_scores_by_id or {}
    by_ret = sorted(window, key=retrieval_score_for_tiebreak, reverse=True)
    ret_rank = {str(k.get("id") or "").strip(): i + 1 for i, k in enumerate(by_ret)}
    by_ce = sorted(window, key=lambda k: ce_scores_by_id.get(str(k.get("id") or "").strip(), float("-inf")), reverse=True)
    ce_rank = {
        str(k.get("id") or "").strip(): i + 1
        for i, k in enumerate(by_ce)
        if str(k.get("id") or "").strip() in ce_scores_by_id
    }
    tag = f"[{label}] " if label else ""
    print("\n" + "=" * 110)
    print(f"{tag}RERANK DEBUG  |  q={_short(q_raw, 90)}")
    print("-" * 110)
    print(f"{tag}{'RRF#':>4}  {'RRF':>8}  {'CE#':>4}  {'CE':>8}  {'ID':<18}  {'TITLE'}")
    print("-" * 110)
    for kb in by_ret[: max(1, min(topm, len(by_ret)))]:
        kid = str(kb.get("id") or "").strip()
        title = _short(kb.get("article_title"), 60)
        rrf = retrieval_score_for_tiebreak(kb)
        ce = ce_scores_by_id.get(kid)
        print(
            f"{tag}{ret_rank.get(kid, 999):>4}  {_fmt_float(rrf):>8}  "
            f"{ce_rank.get(kid, 0):>4}  {_fmt_float(ce, ''):>8}  "
            f"{kid:<18}  {title}"
        )
    print("=" * 110 + "\n")


async def _ce_score_all(
    q_raw: str,
    kb_list: List[Dict[str, Any]],
    user_query_json: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    model = _get_ce_model()
    if model is None or not kb_list or not q_raw:
        return []
    ce_query = _build_ce_query(q_raw, user_query_json or {})
    pairs = [(ce_query, _candidate_text(kb)) for kb in kb_list]
    loop = asyncio.get_running_loop()

    def _predict():
        with _CE_PREDICT_LOCK:
            return model.predict(pairs, batch_size=CE_BATCH_SIZE)

    scores = await loop.run_in_executor(None, _predict)
    ranked = list(zip(kb_list, [float(s) for s in scores]))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _dedup_window(kb_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    families: Dict[str, List[Dict[str, Any]]] = {}
    for kb in kb_list:
        families.setdefault(_canonical_key(kb), []).append(kb)
    for items in families.values():
        items.sort(key=retrieval_score_for_tiebreak, reverse=True)
    reps = [items[0] for items in families.values()]
    reps.sort(key=retrieval_score_for_tiebreak, reverse=True)
    return reps, families


async def ce_pick_best_id_with_tie_dedup(
    q_raw: str,
    kb_list: List[Dict[str, Any]],
    user_query_json: Optional[Dict[str, Any]] = None,
    label: str = "",
) -> Tuple[str, Dict[str, float]]:
    ranked = await _ce_score_all(q_raw, kb_list, user_query_json)
    if not ranked:
        return "", {}
    scores_by_id = {str(kb.get("id") or "").strip(): float(score) for kb, score in ranked if str(kb.get("id") or "").strip()}
    debug_print_rerank_table(label=label or "ce", q_raw=q_raw, window=kb_list, ce_scores_by_id=scores_by_id, topm=DEBUG_RERANK_TOPM)
    best_kb, best_sc = ranked[0]
    best_id = str(best_kb.get("id") or "").strip()
    if (not CE_TIE_DEDUP_ENABLED) or len(ranked) < max(2, CE_TIE_TOPK):
        return best_id, scores_by_id
    kth_idx = min(len(ranked) - 1, max(1, CE_TIE_TOPK - 1))
    kth_sc = ranked[kth_idx][1]
    margin = float(best_sc) - float(kth_sc)
    if margin >= CE_TIE_MARGIN:
        return best_id, scores_by_id
    reps, families = _dedup_window(kb_list)
    if len(reps) >= len(kb_list):
        return best_id, scores_by_id
    ranked2 = await _ce_score_all(q_raw, reps, user_query_json)
    if not ranked2:
        return best_id, scores_by_id
    rep_scores = {str(k.get("id") or "").strip(): float(sc) for k, sc in ranked2}
    debug_print_rerank_table(label=(label or "ce") + ":dedup_reps", q_raw=q_raw, window=reps, ce_scores_by_id=rep_scores, topm=DEBUG_RERANK_TOPM)
    best_rep = ranked2[0][0]
    fam_key = _canonical_key(best_rep)
    members = families.get(fam_key, [best_rep])
    chosen = members[0]
    if CE_INTRA_FAMILY_CE and len(members) > 1:
        intra_ranked = await _ce_score_all(q_raw, members, user_query_json)
        if intra_ranked:
            chosen = intra_ranked[0][0]
    return str(chosen.get("id") or "").strip(), scores_by_id


async def ce_pick_best_id(
    q_raw: str,
    kb_list: List[Dict[str, Any]],
    user_query_json: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, float]]:
    ranked = await _ce_score_all(q_raw, kb_list, user_query_json)
    if not ranked:
        return "", {}
    scores_by_id = {str(kb.get("id") or "").strip(): float(score) for kb, score in ranked if str(kb.get("id") or "").strip()}
    best_id = str(ranked[0][0].get("id") or "").strip()
    return best_id, scores_by_id


def _get_llm_rerank_client():
    global _llm_rerank_client
    if _llm_rerank_client is not None:
        return _llm_rerank_client
    with _llm_rerank_client_lock:
        if _llm_rerank_client is not None:
            return _llm_rerank_client
        try:
            _llm_rerank_client = indexing_config.get_llm_client()
            logger.info("LLM reranker client initialised (model=%s)", LLM_RERANK_MODEL)
        except Exception as exc:
            logger.warning("LLM reranker client init failed: %s", exc)
            _llm_rerank_client = None
    return _llm_rerank_client


def _extract_chat_text(resp) -> str:
    try:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None)
        return (content or "").strip()
    except Exception:
        return ""


def _llm_call_sync(client, prompt: str, max_tokens: int = 64) -> str:
    resp = client.chat.completions.create(
        model=LLM_RERANK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise reranking assistant. "
                    "Return exactly one JSON object and nothing else. "
                    "The JSON must have exactly one key: `choice`, whose value is a single integer."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max(16, int(max_tokens)),
        response_format={"type": "json_object"},
    )
    return _extract_chat_text(resp)


async def _llm_call(client, prompt: str, max_tokens: int = 64) -> str:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _llm_call_sync, client, prompt, max_tokens),
        timeout=LLM_RERANK_TIMEOUT,
    )


def _build_group_prompt(
    q_raw: str,
    candidates: List[Dict[str, Any]],
    user_query_json: Optional[Dict[str, Any]] = None,
) -> str:
    def _fmt(raw: str) -> str:
        parts = re.split(r"[\n;]+", raw or "")
        return " / ".join(p.strip() for p in parts if p.strip())

    def _join_list(lst: Any) -> str:
        if not lst or not isinstance(lst, list):
            return ""
        return ", ".join(str(x).strip() for x in lst if str(x).strip())

    uqj = user_query_json or {}
    kw = uqj.get("keywords") or {}
    ctx_parts: List[str] = []
    if specific := _join_list(kw.get("specific")):
        ctx_parts.append(f"Specific keywords : {specific}")
    if issues := _join_list(uqj.get("issues")):
        ctx_parts.append(f"Issues described  : {issues}")
    if usecases := _join_list(uqj.get("usecase")):
        ctx_parts.append(f"Use-case          : {usecases}")
    if phrases := _join_list(uqj.get("search_phrases")):
        ctx_parts.append(f"Search phrases    : {phrases}")
    if generic := _join_list(kw.get("generic")):
        ctx_parts.append(f"Generic keywords  : {generic}")
    if acronyms := _join_list(uqj.get("acronyms")):
        ctx_parts.append(f"Acronyms          : {acronyms}")
    if atype := (uqj.get("article_type") or "").strip():
        ctx_parts.append(f"Article type      : {atype}")
    if cat := (uqj.get("category") or "").strip():
        ctx_parts.append(f"Category          : {cat}")

    ctx_block = ""
    if ctx_parts:
        ctx_block = "Extracted query context:\n" + "\n".join(f"  {p}" for p in ctx_parts) + "\n\n"

    header = (
        f"User query: {q_raw}\n\n"
        f"{ctx_block}"
        "Pick the single article that MOST SPECIFICALLY and DIRECTLY addresses this query.\n"
        "Use the extracted context above to guide your decision — prefer articles whose\n"
        "keywords, triggers, or content match the specific issues and use-case described.\n"
        "If two articles seem similar, choose the one with the most targeted, precise match.\n"
        "Return exactly one JSON object with this shape: {\"choice\": 3}\n"
        "- `choice` must be a single integer from the candidate numbers shown below.\n"
        "- Do not include any other keys.\n"
        "- Do not include explanation, markdown, prose, or code fences.\n"
        "- Never return null, a string, or an out-of-range number.\n\n"
    )

    entries = []
    for i, kb in enumerate(candidates, 1):
        title = (kb.get("article_title", "") or "").strip()
        intent = (kb.get("primary_intent", "") or "").strip()
        summary = (kb.get("summary", "") or "").strip()
        triggers = _fmt(kb.get("query_triggers", "") or "")
        keywords = _fmt(kb.get("primary_specific_keywords", "") or "")
        entities = _fmt(kb.get("secondary_entities", "") or "")
        acronyms = _fmt(kb.get("acronyms", "") or "")
        content = (kb.get("content", "") or "")[:CONTENT_SNIPPET_IN_PROMPT].strip()
        block = f"[{i}] {title}"
        if intent:
            block += f"\n    Intent   : {intent}"
        if summary:
            block += f"\n    Summary  : {summary}"
        if triggers:
            block += f"\n    Triggers : {triggers}"
        kw_parts = " / ".join(p for p in [keywords, entities, acronyms] if p)
        if kw_parts:
            block += f"\n    Keywords : {kw_parts}"
        if content:
            block += f"\n    Content  : {content}"
        entries.append(block)
    return header + "\n\n".join(entries) + "\n\nBest article number:"


async def llm_pick_best_id(
    q_raw: str,
    candidates: List[Dict[str, Any]],
    user_query_json: Optional[Dict[str, Any]] = None,
) -> str:
    if not USE_LLM_RERANK or not q_raw or not candidates:
        return ""
    client = _get_llm_rerank_client()
    if client is None:
        return ""
    if len(candidates) == 1:
        return str(candidates[0].get("id", "")).strip()

    async def _run_group(group: List[Dict[str, Any]], label: str) -> Optional[Dict[str, Any]]:
        if len(group) == 1:
            return group[0]
        id_map = {str(i + 1): str(kb.get("id", "")).strip() for i, kb in enumerate(group)}
        prompt = _build_group_prompt(q_raw, group, user_query_json)
        try:
            raw = await _llm_call(client, prompt, max_tokens=16)
            parsed_choice = ""
            try:
                parsed = json.loads(raw or "")
                parsed_choice = str(int(parsed.get("choice"))) if isinstance(parsed, dict) and parsed.get("choice") is not None else ""
            except Exception:
                parsed_choice = ""
            if not parsed_choice:
                match = re.search(r"\d+", raw or "")
                parsed_choice = match.group(0) if match else ""
            if parsed_choice not in id_map:
                logger.warning("LLM group %s unparseable: %r; using group top-1", label, raw)
                return group[0]
            kb_id = id_map[parsed_choice]
            return next((kb for kb in group if str(kb.get("id", "")).strip() == kb_id), group[0])
        except Exception as exc:
            logger.warning("LLM group %s error: %s; using group top-1", label, exc)
            return group[0]

    try:
        if len(candidates) <= LLM_SMALL_POOL_THRESHOLD:
            final = await _run_group(candidates, "direct")
            return str(final.get("id", "")).strip() if final else ""
        n_groups = max(1, (len(candidates) + LLM_GROUP_SIZE - 1) // LLM_GROUP_SIZE)
        groups = [g for g in [candidates[i::n_groups] for i in range(n_groups)] if g]
        winners_raw = await asyncio.gather(*[_run_group(g, str(i + 1)) for i, g in enumerate(groups)])
        finalists = [w for w in winners_raw if w is not None]
        if not finalists:
            return ""
        if len(finalists) == 1:
            return str(finalists[0].get("id", "")).strip()
        final = await _run_group(finalists, "final")
        return str(final.get("id", "")).strip() if final else ""
    except asyncio.TimeoutError:
        logger.warning("LLM tournament timed out; falling back to top-1")
        return str(candidates[0].get("id", "")).strip() if candidates else ""
    except Exception as exc:
        logger.warning("LLM tournament error: %s; falling back to top-1", exc)
        return str(candidates[0].get("id", "")).strip() if candidates else ""


def lexical_pick(user_query_json: Dict, kb_list: List[Dict[str, Any]]) -> str:
    q_raw = (user_query_json or {}).get("_raw_query") or ""
    q_tokens = _tokenize(q_raw)
    if not q_raw or not kb_list:
        return kb_list[0].get("id", "") if kb_list else ""

    ret_raw = [float((kb.get("retrieval") or {}).get("rrf_score", 0.0)) for kb in kb_list]
    r_min, r_max = min(ret_raw), max(ret_raw)
    r_range = (r_max - r_min) if r_max > r_min else 1.0

    best_id, best_score = "", -1.0
    for kb, raw_ret in zip(kb_list, ret_raw):
        kid = str(kb.get("id", "")).strip()
        if not kid:
            continue
        norm_ret = (raw_ret - r_min) / r_range
        short = " ".join(
            filter(
                None,
                [
                    str(kb.get("article_title", "") or ""),
                    str(kb.get("primary_intent", "") or ""),
                    str(kb.get("query_triggers", "") or "")[:200],
                ],
            )
        )
        lex = 0.6 * _jaccard(q_tokens, _tokenize(short)) + 0.4 * _fuzzy_ratio(q_raw, short)
        score = 0.70 * norm_ret + 0.30 * lex
        if score > best_score:
            best_score, best_id = score, kid
    return best_id


async def rerank_candidates(
    user_query_json: Dict,
    q_raw: str,
    candidates_summary_list: List[Dict[str, Any]],
    label: str = "",
    strategy: Optional[str] = None,
) -> str:
    if not candidates_summary_list:
        return ""
    strat = (strategy or RERANK_STRATEGY or "ce").strip().lower()
    window = candidates_summary_list[: max(1, RERANK_TOP_N)]
    if DEBUG_RERANK:
        debug_print_rerank_table(
            label=(label or "rerank") + ":retrieval_only",
            q_raw=q_raw,
            window=window,
            ce_scores_by_id=None,
            topm=DEBUG_RERANK_TOPM,
        )
    if strat == "ce":
        if not (USE_CE_RERANK and CrossEncoder is not None):
            logger.warning("[%s] CE requested but unavailable; falling back to lexical", label)
            return lexical_pick(user_query_json, window)
        best_id, _ = await ce_pick_best_id_with_tie_dedup(q_raw=q_raw, kb_list=window, user_query_json=user_query_json, label=label or "")
        return best_id or lexical_pick(user_query_json, window)
    if strat == "llm":
        if not USE_LLM_RERANK:
            logger.warning("[%s] LLM requested but disabled; falling back to lexical", label)
            return lexical_pick(user_query_json, window)
        best_id = await llm_pick_best_id(q_raw, window, user_query_json)
        return best_id or lexical_pick(user_query_json, window)
    if strat == "lexical":
        return lexical_pick(user_query_json, window)
    logger.warning("[%s] Unknown strategy '%s'; falling back to lexical", label, strat)
    return lexical_pick(user_query_json, window)


def get_score(obj: Dict[str, Any]) -> float:
    if obj is None:
        return 0.0
    if "rrf_score" in obj and obj["rrf_score"] is not None:
        try:
            return float(obj["rrf_score"])
        except Exception:
            return 0.0
    if "vector_distance" in obj and obj["vector_distance"] is not None:
        try:
            return 1.0 / (1.0 + max(float(obj["vector_distance"]), 0.0))
        except Exception:
            return 0.0
    return 0.0


def pick_kbs_for_rerank(
    retrieved: List[Dict[str, Any]],
    *,
    auto_pick_ratio: float = AUTO_PICK_RATIO,
    auto_pick_min_gap: float = AUTO_PICK_MIN_GAP,
    top2_vs_rest_ratio: float = TOP2_VS_REST_RATIO,
    keep_score_ratio: float = KEEP_SCORE_RATIO,
    min_candidates: int = MIN_CANDIDATES,
    max_candidates: int = MAX_CANDIDATES,
) -> Tuple[Optional[str], List[str], Dict[str, Dict[str, Any]], str]:
    if not retrieved:
        return None, [], {}, "rerank"
    scores_by_id: Dict[str, Dict[str, Any]] = {}
    items: List[Tuple[str, float, Dict[str, Any]]] = []
    for row in retrieved:
        kbid = str(row.get("KBID") or row.get("kbid") or "").strip()
        if not kbid:
            continue
        score_obj = {k: v for k, v in row.items() if k not in ("KBID", "kbid")}
        scores_by_id[kbid] = score_obj
        items.append((kbid, get_score(score_obj), score_obj))
    if not items:
        return None, [], {}, "rerank"
    items.sort(key=lambda x: x[1], reverse=True)
    top1_id, s1, _ = items[0]
    s2 = items[1][1] if len(items) > 1 else 0.0
    s3 = items[2][1] if len(items) > 2 else 0.0
    if len(items) == 1:
        return top1_id, [top1_id], scores_by_id, "direct"
    if (s2 > 0 and s1 >= s2 * auto_pick_ratio) or ((s1 - s2) >= auto_pick_min_gap):
        return top1_id, [top1_id], scores_by_id, "direct"
    if len(items) >= 3 and s3 > 0 and s2 >= s3 * top2_vs_rest_ratio:
        return None, [items[0][0], items[1][0]], scores_by_id, "rerank"
    thresh = s1 * keep_score_ratio
    candidate_ids = [kbid for (kbid, score, _) in items if score >= thresh]
    if len(candidate_ids) < min_candidates:
        candidate_ids = [kbid for (kbid, _, _) in items[:min_candidates]]
    if len(candidate_ids) > max_candidates:
        candidate_ids = candidate_ids[:max_candidates]
    return None, candidate_ids, scores_by_id, "rerank"
