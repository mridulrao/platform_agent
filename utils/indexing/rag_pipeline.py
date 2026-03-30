"""
KB Retrieval Pipeline — two-level shard-parallel reranking
===========================================================
"""

from openai import AzureOpenAI
import os
import json
import asyncio
from sqlalchemy import text
import pandas as pd
import time
import logging
import threading
import contextvars
import random
import re
from collections import Counter

from sentence_transformers import CrossEncoder  # type: ignore

from functools import wraps
from typing import Dict, List, Any, Tuple, Optional, Callable, TypeVar
from pathlib import Path

from search_pgvec import (
    PgVectorFTSHybridKnowledgeBaseSearcher,
    search_knowledge_base_pgvec_fts,
)
from pipeline_utils.rerank import (
    DEBUG_RERANK,
    DEBUG_RERANK_TOPM,
    RERANK_STRATEGY,
    RERANK_TOP_N,
    debug_print_rerank_table as _debug_print_rerank_table,
    get_score as _get_score,
    pick_kbs_for_rerank,
    rerank_candidates as _rerank_candidates,
)
from pgvec_client import get_shared_pgvector_client
from instructions.get_useful_kb_json_instructions import instructions as get_useful_kb_instructions
from agents.background_user_query_worker import UserQueryExtractor
from indexing_config import indexing_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config-driven knobs
# ─────────────────────────────────────────────────────────────────────────────

# ── Retrieval knobs ──────────────────────────────────────────────────────────
PER_SHARD_TOP_K = indexing_config.per_shard_top_k
RETRIEVAL_TOP_K = indexing_config.retrieval_top_k

CONTENT_SNIPPET_CHARS = indexing_config.content_snippet_chars

# ── Validation knobs ──────────────────────────────────────────────────────────
VALIDATION_CONCURRENCY      = 1
VALIDATION_THROTTLE_SECONDS = float(
    os.getenv("KB_VALIDATION_THROTTLE_SECONDS", str(indexing_config.kb_validation_throttle_seconds))
)

# ── OpenAI retry knobs ────────────────────────────────────────────────────────
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", str(indexing_config.openai_max_retries)))
OPENAI_BACKOFF_INITIAL = float(os.getenv("OPENAI_BACKOFF_INITIAL", str(indexing_config.openai_backoff_initial)))
OPENAI_BACKOFF_MAX = float(os.getenv("OPENAI_BACKOFF_MAX", str(indexing_config.openai_backoff_max)))
OPENAI_BACKOFF_MULTIPLIER = float(
    os.getenv("OPENAI_BACKOFF_MULTIPLIER", str(indexing_config.openai_backoff_multiplier))
)
OPENAI_BACKOFF_JITTER = float(os.getenv("OPENAI_BACKOFF_JITTER", str(indexing_config.openai_backoff_jitter)))

T = TypeVar("T")

# ─────────────────────────────────────────────────────────────────────────────
# False-positive sentinel tracking
# ─────────────────────────────────────────────────────────────────────────────

_fp_counter: Counter = Counter()
_fp_counter_lock = threading.Lock()


def _record_false_positive(wrong_id: str, query: str) -> None:
    with _fp_counter_lock:
        _fp_counter[wrong_id] += 1
    logger.warning(
        f"FP_SENTINEL  wrong_doc={wrong_id}  "
        f"cumulative_fp_count={_fp_counter[wrong_id]}  "
        f"query={query[:80]!r}"
    )


def log_false_positive_summary(top_n: int = 20) -> None:
    with _fp_counter_lock:
        most_common = _fp_counter.most_common(top_n)
    if not most_common:
        return
    logger.warning("=" * 60)
    logger.warning(f"FP_SENTINEL SUMMARY — top-{top_n} false-positive docs:")
    for rank, (doc_id, count) in enumerate(most_common, 1):
        logger.warning(f"  #{rank:>2}  {doc_id}  (wrong {count}x)")
    logger.warning("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 429 retry
# ─────────────────────────────────────────────────────────────────────────────

def _looks_like_429(exc: Exception) -> bool:
    for attr in ("status_code", "http_status", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and v == 429:
            return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg)


async def _call_with_429_retry_async(fn: Callable[..., Any], *args, **kwargs) -> Any:
    delay = OPENAI_BACKOFF_INITIAL
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if (not _looks_like_429(e)) or (attempt >= OPENAI_MAX_RETRIES):
                raise
            base = min(OPENAI_BACKOFF_MAX, delay)
            sleep_s = base * (1.0 + random.random() * OPENAI_BACKOFF_JITTER)
            logger.warning(f"OpenAI 429. Retry {attempt}/{OPENAI_MAX_RETRIES} in {sleep_s:.2f}s. {e}")
            await asyncio.sleep(sleep_s)
            delay *= OPENAI_BACKOFF_MULTIPLIER
    return await fn(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
def get_cached_pg_client():
    return get_shared_pgvector_client()


_searcher: Optional[PgVectorFTSHybridKnowledgeBaseSearcher] = None
_searcher_lock = threading.Lock()


def get_cached_searcher() -> PgVectorFTSHybridKnowledgeBaseSearcher:
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                _searcher = PgVectorFTSHybridKnowledgeBaseSearcher(
                    kb_collection_id=indexing_config.collection_id,
                )
    return _searcher


# ─────────────────────────────────────────────────────────────────────────────
# Timing / metrics
# ─────────────────────────────────────────────────────────────────────────────

def log_timing(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__}: {time.time() - start_time:.3f}s")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)} ({time.time() - start_time:.3f}s)")
            raise
    return wrapper


class PipelineMetrics:
    def __init__(self):
        self.metrics = {"pipeline_start": None, "pipeline_end": None, "steps": {}}

    def start_pipeline(self):
        self.metrics["pipeline_start"] = time.time()

    def end_pipeline(self):
        self.metrics["pipeline_end"] = time.time()
        total = self.metrics["pipeline_end"] - self.metrics["pipeline_start"]
        logger.info("-" * 60)
        logger.info(f"Pipeline total: {total:.3f}s")
        for name, data in self.metrics["steps"].items():
            dur = data.get("duration", 0)
            logger.info(f"  {name}: {dur:.3f}s ({dur / total * 100:.1f}%)")
        logger.info("-" * 60)
        return self.metrics

    def record_step(self, step_name: str, duration: float, status: str = "success", details: Dict = None):
        self.metrics["steps"][step_name] = {"duration": duration, "status": status, "details": details or {}}


_PIPELINE_METRICS_CV: contextvars.ContextVar[Optional[PipelineMetrics]] = contextvars.ContextVar(
    "pipeline_metrics", default=None
)
_GLOBAL_PIPELINE_METRICS_FALLBACK = PipelineMetrics()


def _pm() -> PipelineMetrics:
    v = _PIPELINE_METRICS_CV.get()
    return v if v is not None else _GLOBAL_PIPELINE_METRICS_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_table_columns(pg_client, full_table_name: str) -> set:
    if "." in full_table_name:
        schema, table = full_table_name.split(".", 1)
        schema = schema.strip('"')
        table  = table.strip('"')
    else:
        schema = indexing_config.pgvector_schema
        table  = full_table_name.strip('"')
    sql = text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = :schema AND table_name = :table"
    )
    with pg_client.get_engine().connect() as conn:
        rows = conn.execute(sql, {"schema": schema, "table": table}).fetchall()
    return {r[0] for r in rows}


def _first_existing(candidates: List[str], existing: set) -> Optional[str]:
    return next((c for c in candidates if c in existing), None)


def _sql_string_agg(col: str, alias: str) -> str:
    return f"NULLIF(STRING_AGG(DISTINCT NULLIF(TRIM({col}), ''), ', '), '') AS {alias}"


def _sql_first_nonempty(col: str, alias: str) -> str:
    return f"MAX(NULLIF(TRIM({col}), '')) AS {alias}"


# ─────────────────────────────────────────────────────────────────────────────
# Query JSON normalization
# ─────────────────────────────────────────────────────────────────────────────

_FILLER_TERMS = {
    "a", "an", "the", "help", "issue", "issues", "problem", "problems", "support",
    "please", "need", "with", "for", "from", "that", "this", "my", "our", "their",
    "system", "computer", "device", "thing", "something",
}
_GENERIC_STOPWORDS = _FILLER_TERMS | {
    "user", "users", "team", "teams", "getting", "having", "facing", "unable",
}


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = value.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _clean_phrase(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value).strip(" ,;:-")
    return value.strip()


def _clean_phrase_list(values: Any, *, lowercase: bool = False, max_items: int = 8) -> List[str]:
    raw_values = values if isinstance(values, list) else ([values] if isinstance(values, str) else [])
    cleaned: List[str] = []
    for value in raw_values:
        phrase = _clean_phrase(value)
        if not phrase:
            continue
        if lowercase:
            phrase = phrase.lower()
        cleaned.append(phrase)
    return _dedupe_preserve_order(cleaned)[:max_items]


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return "both"
    norm = value.strip().lower()
    if norm in {"software", "hardware", "both"}:
        return norm
    return "both"


def _infer_acronyms_from_query(user_query: str) -> List[str]:
    matches = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", user_query or "")
    return _dedupe_preserve_order(matches)[:8]


def _looks_specific_term(term: str) -> bool:
    if not term:
        return False
    if re.search(r"\d", term):
        return True
    if re.search(r"[A-Z]{2,}", term):
        return True
    if len(term.split()) >= 2 and any(tok[:1].isupper() for tok in term.split()):
        return True
    return False


def _normalize_generic_keywords(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        phrase = value.lower()
        if phrase in _GENERIC_STOPWORDS:
            continue
        tokens = [tok for tok in re.split(r"\s+", phrase) if tok and tok not in _GENERIC_STOPWORDS]
        if not tokens:
            continue
        out.append(" ".join(tokens))
    return _dedupe_preserve_order(out)[:10]


def _derive_search_phrases(
    user_query: str,
    specific_keywords: List[str],
    generic_keywords: List[str],
    issues: List[str],
    usecases: List[str],
) -> List[str]:
    phrases: List[str] = []
    if user_query and len(user_query.strip()) <= 120:
        phrases.append(_clean_phrase(user_query))
    phrases.extend(usecases[:3])
    for specific in specific_keywords[:3]:
        for generic in generic_keywords[:2]:
            phrases.append(_clean_phrase(f"{specific} {generic}"))
    for issue in issues[:2]:
        phrases.append(issue)
    return _dedupe_preserve_order([p for p in phrases if p])[:8]


def _fallback_query_json(user_query: str) -> Dict[str, Any]:
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*", user_query or "")]
    generic = _dedupe_preserve_order([tok for tok in tokens if tok not in _GENERIC_STOPWORDS and len(tok) > 2])[:6]
    acronyms = _infer_acronyms_from_query(user_query)
    return {
        "category": "both",
        "article_type": "",
        "keywords": {"specific": [], "generic": generic},
        "issues": [],
        "usecase": [_clean_phrase(user_query)] if _clean_phrase(user_query) else [],
        "acronyms": acronyms,
        "search_phrases": [_clean_phrase(user_query)] if _clean_phrase(user_query) else [],
    }


def _normalize_user_query_json(user_query: str, extracted: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = extracted if isinstance(extracted, dict) else {}
    fallback = _fallback_query_json(user_query)

    keywords = source.get("keywords") if isinstance(source.get("keywords"), dict) else {}
    raw_specific = _clean_phrase_list(keywords.get("specific") or keywords.get("primary_specific"))
    raw_generic = _clean_phrase_list(keywords.get("generic") or keywords.get("primary_generic"), lowercase=True)
    issues = _clean_phrase_list(source.get("issues"), lowercase=True)
    usecases = _clean_phrase_list(source.get("usecase"), lowercase=True)
    acronyms = _clean_phrase_list(source.get("acronyms"))
    search_phrases = _clean_phrase_list(source.get("search_phrases"), lowercase=True)

    specific_keywords: List[str] = []
    generic_keywords: List[str] = []

    for term in raw_specific:
        if _looks_specific_term(term):
            specific_keywords.append(term)
        else:
            generic_keywords.append(term.lower())

    for term in raw_generic:
        if _looks_specific_term(term):
            specific_keywords.append(term)
        else:
            generic_keywords.append(term)

    if not specific_keywords:
        specific_keywords = [term for term in acronyms if _looks_specific_term(term)]

    specific_keywords = _dedupe_preserve_order([_clean_phrase(term) for term in specific_keywords if _clean_phrase(term)])[:8]
    generic_keywords = _normalize_generic_keywords(generic_keywords)
    issues = _normalize_generic_keywords(issues)
    usecases = _dedupe_preserve_order([_clean_phrase(term) for term in usecases if _clean_phrase(term)])[:6]
    acronyms = _dedupe_preserve_order(acronyms + _infer_acronyms_from_query(user_query))[:8]

    if not usecases and _clean_phrase(user_query):
        usecases = [_clean_phrase(user_query).lower()]

    search_phrases = _dedupe_preserve_order(
        search_phrases + _derive_search_phrases(user_query, specific_keywords, generic_keywords, issues, usecases)
    )[:8]

    normalized = {
        "category": _normalize_category(source.get("category")),
        "article_type": _clean_phrase(source.get("article_type")),
        "keywords": {
            "specific": specific_keywords,
            "generic": generic_keywords,
        },
        "issues": issues,
        "usecase": usecases,
        "acronyms": acronyms,
        "search_phrases": search_phrases,
        "_raw_query": user_query,
    }

    if not normalized["keywords"]["specific"] and not normalized["keywords"]["generic"] and not normalized["usecase"]:
        fallback["_raw_query"] = user_query
        return fallback

    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — query parsing
# ─────────────────────────────────────────────────────────────────────────────

extractor = UserQueryExtractor()

@log_timing
async def get_user_query_json(user_query: str) -> Dict:
    step_start = time.time()
    q = await extractor.extract(user_query)
    print("Query extraction result:", q)

    if not q or "error" in q:
        normalized = _normalize_user_query_json(user_query, {})
        logger.warning("Query extraction failed or returned error; using normalized fallback JSON")
        _pm().record_step("Query Extraction", time.time() - step_start, "failed")
        return normalized

    q = _normalize_user_query_json(user_query, q)

    _pm().record_step("Query Extraction", time.time() - step_start, "success")
    return q


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — fetch summaries + keywords
# ─────────────────────────────────────────────────────────────────────────────

@log_timing
async def fetch_summaries_and_keywords_parallel(
    kb_list: List[str],
    kb_collection_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not kb_list:
        return pd.DataFrame(), pd.DataFrame()

    pg_client = get_cached_pg_client()
    kb_set = set(map(str, kb_list))
    collection_id = (kb_collection_id or indexing_config.collection_id or "").strip()

    summaries_table = pg_client.get_table_name("kb_summaries")
    keywords_table  = pg_client.get_table_name("kb_keywords")
    kw_cols  = _get_table_columns(pg_client, keywords_table)
    sum_cols = _get_table_columns(pg_client, summaries_table)
    has_content = "content" in sum_cols

    async def fetch_summaries():
        loop = asyncio.get_running_loop()
        def _fetch():
            base_cols = ["doc_id"]
            opt_cols  = ["article_title", "primary_intent", "query_triggers", "secondary_mentions", "summary"]
            cols = base_cols + [c for c in opt_cols if c in sum_cols]
            if has_content:
                cols.append("content")
            where_parts = ["doc_id = ANY(:kb_ids)"]
            params: Dict[str, Any] = {"kb_ids": list(kb_set)}
            if collection_id:
                where_parts.append("kb_collection_id = :kb_collection_id")
                params["kb_collection_id"] = collection_id
            q = text(
                f"SELECT {', '.join(cols)} FROM {summaries_table} "
                f"WHERE {' AND '.join(where_parts)}"
            )
            with pg_client.get_engine().connect() as conn:
                rows = conn.execute(q, params).fetchall()
            return pd.DataFrame(rows, columns=cols)
        return await loop.run_in_executor(None, _fetch)

    async def fetch_keywords():
        loop = asyncio.get_running_loop()
        def _fetch():
            col_cat      = _first_existing(["category"],           kw_cols)
            col_atype    = _first_existing(["article_type"],        kw_cols)
            col_topic    = _first_existing(["primary_topic"],       kw_cols)
            col_specific = _first_existing(["primary_specific_keyword", "specific_keyword"], kw_cols)
            col_generic  = _first_existing(["primary_generic_keyword",  "generic_keyword"],  kw_cols)
            col_sec      = _first_existing(["secondary_entity", "secondary_entities"], kw_cols)
            col_acr      = _first_existing(["acronym", "acronyms"], kw_cols)
            col_uc       = _first_existing(["usecase"],             kw_cols)

            select_parts = ["kbid"]
            if col_cat:      select_parts.append(_sql_string_agg(col_cat,      "category"))
            if col_atype:    select_parts.append(_sql_string_agg(col_atype,    "article_type"))
            if col_topic:    select_parts.append(_sql_first_nonempty(col_topic,"primary_topic"))
            if col_specific: select_parts.append(_sql_string_agg(col_specific, "primary_specific_keywords"))
            if col_generic:  select_parts.append(_sql_string_agg(col_generic,  "primary_generic_keywords"))
            if col_sec:      select_parts.append(_sql_string_agg(col_sec,      "secondary_entities"))
            if col_acr:      select_parts.append(_sql_string_agg(col_acr,      "acronyms"))
            if col_uc:       select_parts.append(_sql_string_agg(col_uc,       "usecase"))

            if len(select_parts) == 1:
                return pd.DataFrame()

            where_parts = ["kbid = ANY(:kb_ids)"]
            params: Dict[str, Any] = {"kb_ids": list(kb_set)}
            if collection_id:
                where_parts.append("kb_collection_id = :kb_collection_id")
                params["kb_collection_id"] = collection_id
            sql = text(
                f"SELECT {', '.join(select_parts)} FROM {keywords_table} "
                f"WHERE {' AND '.join(where_parts)} GROUP BY kbid"
            )
            with pg_client.get_engine().connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            out_cols = ["kbid"] + [p.split(" AS ")[-1].strip() for p in select_parts[1:]]
            return pd.DataFrame(rows, columns=out_cols)
        return await loop.run_in_executor(None, _fetch)

    df_summary, df_keyword = await asyncio.gather(fetch_summaries(), fetch_keywords())
    return df_summary, df_keyword


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — build candidate objects
# ─────────────────────────────────────────────────────────────────────────────

def _best_snippet(content: str, query: str, max_chars: int = 800) -> str:
    content = (content or "").strip()
    query = (query or "").strip()
    if not content:
        return ""
    if not query:
        return content[:max_chars]

    qtok = _tokenize(query)
    if not qtok:
        return content[:max_chars]

    paras = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    chunks: List[str] = []
    for p in paras:
        if len(p) <= 600:
            chunks.append(p)
        else:
            step = 350
            win = 600
            for i in range(0, len(p), step):
                chunks.append(p[i:i+win])

    best = ""
    best_score = -1.0
    for ch in chunks:
        ctok = _tokenize(ch)
        import difflib
        score = 0.7 * (len(set(qtok) & set(ctok)) / len(set(qtok) | set(ctok)) if qtok and ctok else 0.0) + 0.3 * difflib.SequenceMatcher(None, query.lower(), ch[:300].lower()).ratio()
        if score > best_score:
            best_score = score
            best = ch

    return best[:max_chars]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "very_high"
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.50:
        return "medium"
    if confidence > 0.0:
        return "low"
    return "none"


def _supporting_signal_count(score_obj: Dict[str, Any]) -> int:
    if not score_obj:
        return 0
    return (
        (1 if score_obj.get("vector_distance") is not None else 0)
        + (1 if score_obj.get("fts_keywords_rank") is not None else 0)
        + (1 if score_obj.get("fts_summaries_rank") is not None else 0)
    )


def _build_selection_confidence(
    winner_id: str,
    score_map: Dict[str, Dict[str, Any]],
    *,
    mode: str,
    selected_via: str,
) -> Dict[str, Any]:
    winner_id = str(winner_id or "").strip()
    if not winner_id:
        return {
            "score": 0.0,
            "label": "none",
            "selected_via": selected_via,
            "mode": mode,
            "winner_score": 0.0,
            "runner_up_score": 0.0,
            "score_gap": 0.0,
            "supporting_signals": 0,
            "exact_match": False,
            "phrase_match": False,
        }

    ranked: List[Tuple[str, float, Dict[str, Any]]] = []
    for doc_id, score_obj in (score_map or {}).items():
        ranked.append((str(doc_id).strip(), _get_score(score_obj), score_obj or {}))
    ranked.sort(key=lambda item: item[1], reverse=True)

    winner_score_obj = dict((score_map or {}).get(winner_id, {}) or {})
    winner_score = _get_score(winner_score_obj)

    runner_up_score = 0.0
    for doc_id, score, _ in ranked:
        if doc_id != winner_id:
            runner_up_score = score
            break

    score_gap = max(0.0, winner_score - runner_up_score)
    margin_ratio = score_gap / max(winner_score, 1e-6) if winner_score > 0 else 0.0
    winner_strength = winner_score / (1.0 + winner_score) if winner_score > 0 else 0.0
    ratio_component = 1.0 if runner_up_score <= 0 and winner_score > 0 else (
        (winner_score / max(runner_up_score, 1e-6)) - 1.0
    )
    ratio_component = _clamp01(ratio_component / 1.5)
    signal_component = _supporting_signal_count(winner_score_obj) / 3.0
    exact_match = bool(
        winner_score_obj.get("vector_exact_match")
        or winner_score_obj.get("fts_keywords_exact_match")
        or winner_score_obj.get("fts_summaries_exact_match")
    )
    phrase_match = bool(
        winner_score_obj.get("vector_phrase_match")
        or winner_score_obj.get("fts_keywords_phrase_match")
        or winner_score_obj.get("fts_summaries_phrase_match")
    )

    confidence = (
        0.45 * _clamp01(winner_strength)
        + 0.25 * _clamp01(margin_ratio)
        + 0.15 * ratio_component
        + 0.15 * _clamp01(signal_component)
    )
    if exact_match:
        confidence += 0.08
    elif phrase_match:
        confidence += 0.04
    confidence = _clamp01(confidence)

    return {
        "score": round(confidence, 4),
        "label": _confidence_label(confidence),
        "selected_via": selected_via,
        "mode": mode,
        "winner_score": round(winner_score, 4),
        "runner_up_score": round(runner_up_score, 4),
        "score_gap": round(score_gap, 4),
        "supporting_signals": _supporting_signal_count(winner_score_obj),
        "exact_match": exact_match,
        "phrase_match": phrase_match,
    }


@log_timing
async def generate_summary_useful_kb_list(
    kb_list: List[str],
    q_raw: str,
    retrieval_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    kb_collection_id: Optional[str] = None,
) -> List[Dict]:
    """
    NOTE: q_raw is now required so we can compute a best-matching snippet per doc.
    """
    step_start = time.time()
    if not kb_list:
        return []

    df_summary, df_keyword = await fetch_summaries_and_keywords_parallel(
        kb_list,
        kb_collection_id=kb_collection_id,
    )
    if df_summary.empty:
        _pm().record_step("Summary Generation", time.time() - step_start, "failed")
        return []

    kw_indexed = (
        df_keyword.set_index("kbid")
        if (df_keyword is not None and not df_keyword.empty and "kbid" in df_keyword.columns)
        else None
    )
    retrieval_scores = retrieval_scores or {}

    summary_kb_list: List[Dict] = []
    for _, row in df_summary.iterrows():
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue

        kw = kw_indexed.loc[doc_id] if (kw_indexed is not None and doc_id in kw_indexed.index) else None
        score_obj = retrieval_scores.get(doc_id, {})
        doc_content_full = str(row.get("content", "") or "").strip()

        def _kw(field: str) -> str:
            if kw is None or field not in kw:
                return ""
            return str(kw.get(field) or "")

        entry = {
            "id":                        doc_id,
            "article_title":             str(row.get("article_title",      "") or "").strip(),
            "primary_intent":            str(row.get("primary_intent",     "") or "").strip(),
            "query_triggers":            str(row.get("query_triggers",     "") or "").strip(),
            "secondary_mentions":        str(row.get("secondary_mentions", "") or "").strip(),
            "summary":                   str(row.get("summary",            "") or "").strip(),
            "content":                   _best_snippet(doc_content_full, q_raw, max_chars=800).strip(),
            "primary_topic":             _kw("primary_topic"),
            "category":                  _kw("category"),
            "article_type":              _kw("article_type"),
            "primary_specific_keywords": _kw("primary_specific_keywords"),
            "primary_generic_keywords":  _kw("primary_generic_keywords"),
            "secondary_entities":        _kw("secondary_entities"),
            "acronyms":                  _kw("acronyms"),
            "usecase":                   _kw("usecase"),
            "retrieval":                 score_obj,
        }
        summary_kb_list.append(entry)

    order = {str(k).strip(): i for i, k in enumerate(kb_list)}
    summary_kb_list.sort(key=lambda x: order.get(str(x.get("id", "")).strip(), 10**9))

    _pm().record_step("Summary Generation", time.time() - step_start, "success")
    return summary_kb_list


# ─────────────────────────────────────────────────────────────────────────────
# Level-1 — single shard pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _process_single_shard(
    searcher: PgVectorFTSHybridKnowledgeBaseSearcher,
    user_query_json: Dict,
    q_raw: str,
    shard_id: int,
) -> Optional[Dict[str, Any]]:
    shard_label = f"shard={shard_id}"
    try:
        # ── DIAGNOSTIC 1: confirm what we're sending to retrieval ─────────────
        logger.warning(
            f"[{shard_label}] Starting retrieval. "
            f"PER_SHARD_TOP_K={PER_SHARD_TOP_K}  "
            f"user_query_json keys={list((user_query_json or {}).keys())}  "
            f"q_raw={q_raw!r:.80}"
        )

        retrieved = await _call_with_429_retry_async(
            searcher.process_response_for_shard,
            user_query_json,
            shard_id,
            PER_SHARD_TOP_K,
        )

        # ── DIAGNOSTIC 2: confirm raw retrieval result ────────────────────────
        logger.warning(
            f"[{shard_label}] process_response_for_shard returned "
            f"type={type(retrieved).__name__}  "
            f"len={len(retrieved) if retrieved is not None else 'None'}  "
            f"first_item={str(retrieved[0])[:120] if retrieved else 'N/A'}"
        )

        if not retrieved:
            logger.warning(f"[{shard_label}] Retrieval returned empty — no candidates for this shard.")
            return None

        logger.info(f"[{shard_label}] {len(retrieved)} candidates retrieved.")
        direct_kb, candidates, retrieval_scores, mode = pick_kbs_for_rerank(retrieved)

        if direct_kb:
            logger.info(f"[{shard_label}] Direct pick: {direct_kb}")
            return {
                "shard_candidates": [direct_kb],
                "winner_kb":        direct_kb,
                "shard_id":         shard_id,
                "mode":             "direct",
                "retrieved":        retrieved,
                "retrieval_scores": retrieval_scores,
                "candidates":       [direct_kb],
            }

        if not candidates:
            return None

        shard_summary_list = await generate_summary_useful_kb_list(
            candidates,
            q_raw,
            retrieval_scores,
            kb_collection_id=searcher.kb_collection_id,
        )

        if not shard_summary_list:
            winner = candidates[0]
            logger.warning(f"[{shard_label}] No summaries found; fallback winner={winner}")
            return {
                "shard_candidates": candidates,
                "winner_kb":        winner,
                "shard_id":         shard_id,
                "mode":             f"{mode}_no_summaries_fallback",
                "retrieved":        retrieved,
                "retrieval_scores": retrieval_scores,
                "candidates":       candidates,
            }

        winner = await _rerank_candidates(
            user_query_json=user_query_json,
            q_raw=q_raw,
            candidates_summary_list=shard_summary_list,
            label=shard_label,
            strategy=None,
        )

        if not winner:
            winner = candidates[0]
            logger.warning(f"[{shard_label}] Reranker returned empty; fallback winner={winner}")

        logger.info(
            f"[{shard_label}] Shard rerank winner={winner} "
            f"(candidates={len(candidates)} mode={mode} strategy={RERANK_STRATEGY} window=top-{RERANK_TOP_N})"
        )

        return {
            "shard_candidates": candidates,
            "winner_kb":        winner,
            "shard_id":         shard_id,
            "mode":             f"{mode}_reranked",
            "retrieved":        retrieved,
            "retrieval_scores": retrieval_scores,
            "candidates":       candidates,
        }

    except Exception as e:
        # ── DIAGNOSTIC 3: always surface the real exception ───────────────────
        import traceback
        tb = traceback.format_exc()
        logger.error(
            f"[{shard_label}] Pipeline error (this is why shard returned None): "
            f"{type(e).__name__}: {e}\n{tb}"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Level-2 — final rerank across shard winners
# ─────────────────────────────────────────────────────────────────────────────

async def _final_rerank(
    user_query_json: Dict,
    q_raw: str,
    shard_results: List[Dict[str, Any]],
    kb_collection_id: Optional[str] = None,
    rerank_strategy: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    valid_shards = [r for r in shard_results if r and (r.get("shard_candidates") or r.get("winner_kb"))]
    for sr in valid_shards:
        if not sr.get("shard_candidates"):
            sr["shard_candidates"] = [sr["winner_kb"]]

    if not valid_shards:
        return "", "no_winners", _build_selection_confidence("", {}, mode="no_winners", selected_via="none")

    merged_scores: Dict[str, Dict[str, Any]] = {}
    for sr in valid_shards:
        merged_scores.update(sr.get("retrieval_scores", {}))

    score_by_id: Dict[str, float] = {}
    for sr in valid_shards:
        for kbid in (sr.get("shard_candidates") or []):
            s = _get_score(merged_scores.get(kbid, {}))
            if kbid not in score_by_id or s > score_by_id[kbid]:
                score_by_id[kbid] = s

    all_ids = sorted(score_by_id, key=lambda k: score_by_id[k], reverse=True)

    logger.info(
        f"Final rerank pool: {len(all_ids)} unique candidates from {len(valid_shards)} shard(s) | "
        f"retrieval top-3: {all_ids[:3]} | "
        f"strategy={rerank_strategy or RERANK_STRATEGY} | window=top-{RERANK_TOP_N}"
    )

    if not all_ids:
        return "", "no_winners", _build_selection_confidence("", merged_scores, mode="no_winners", selected_via="none")
    if len(all_ids) == 1:
        mode = "single_candidate"
        return all_ids[0], mode, _build_selection_confidence(
            all_ids[0],
            merged_scores,
            mode=mode,
            selected_via="retrieval",
        )

    final_summary_list = await generate_summary_useful_kb_list(
        all_ids,
        q_raw,
        merged_scores,
        kb_collection_id=kb_collection_id,
    )
    if not final_summary_list:
        logger.warning(f"Final rerank: no summaries; score fallback → {all_ids[0]}")
        mode = "score_fallback"
        return all_ids[0], mode, _build_selection_confidence(
            all_ids[0],
            merged_scores,
            mode=mode,
            selected_via="retrieval",
        )

    final_winner = await _rerank_candidates(
        user_query_json,
        q_raw,
        final_summary_list,
        label="final",
        strategy=rerank_strategy,
    )

    if not final_winner:
        final_winner = all_ids[0]
        logger.warning(f"Final reranker empty; score fallback → {final_winner}")
        mode = "score_fallback"
        return final_winner, mode, _build_selection_confidence(
            final_winner,
            merged_scores,
            mode=mode,
            selected_via="retrieval",
        )

    mode = "single_shard" if len(valid_shards) == 1 else "two_level_rerank"
    return final_winner, mode, _build_selection_confidence(
        final_winner,
        merged_scores,
        mode=mode,
        selected_via="rerank",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@log_timing
async def retrieve_best_kb_id(user_text_query: str, rerank_strategy: Optional[str] = None) -> Dict[str, Any]:
    pm    = PipelineMetrics()
    token = _PIPELINE_METRICS_CV.set(pm)
    pm.start_pipeline()

    resolved_strategy = (rerank_strategy or RERANK_STRATEGY or "ce").strip().lower()

    try:
        user_query_json = await get_user_query_json(user_text_query)
        q_raw = (user_query_json or {}).get("_raw_query", "")

        searcher  = get_cached_searcher()
        shard_ids = searcher.get_shard_ids()

        # ── DIAGNOSTIC 4: confirm searcher state before dispatching shards ────
        logger.warning(
            f"Searcher state — type={type(searcher).__name__}  "
            f"collection_id={getattr(searcher, 'kb_collection_id', 'N/A')}  "
            f"shard_ids={shard_ids}  "
            f"has_process_response_for_shard={hasattr(searcher, 'process_response_for_shard')}"
        )

        logger.info(f"Collection has {len(shard_ids)} shard(s): {shard_ids}")
        logger.info(f"Rerank strategy={resolved_strategy}  window=top-{RERANK_TOP_N}")

        step_start = time.time()
        shard_tasks   = [_process_single_shard(searcher, user_query_json, q_raw, sid) for sid in shard_ids]
        raw_results   = await asyncio.gather(*shard_tasks, return_exceptions=True)

        # ── DIAGNOSTIC 5: show every shard task outcome ───────────────────────
        for i, (sid, r) in enumerate(zip(shard_ids, raw_results)):
            if isinstance(r, Exception):
                logger.error(f"Shard {sid} raised an unhandled exception: {type(r).__name__}: {r}")
            elif r is None:
                logger.warning(f"Shard {sid} returned None (see earlier error logs for reason)")
            else:
                logger.info(f"Shard {sid} returned result with winner={r.get('winner_kb')}  candidates={r.get('candidates')}")

        shard_results = [r for r in raw_results if r is not None and not isinstance(r, Exception)]

        pm.record_step(
            "Shard Pipelines (parallel)",
            time.time() - step_start,
            details={"shards_attempted": len(shard_ids), "shards_with_results": len(shard_results)},
        )
        logger.info(f"Shard pipelines done: {len(shard_results)}/{len(shard_ids)} shards returned results")

        if not shard_results:
            pm.end_pipeline()
            return {
                "status":      "no_results",
                "selected_kb": "",
                "mode":        "no_results",
                "confidence":  _build_selection_confidence("", {}, mode="no_results", selected_via="none"),
                "shard_results": [],
                "num_shards":  len(shard_ids),
            }

        step_start = time.time()
        final_kb, mode, confidence = await _final_rerank(
            user_query_json,
            q_raw,
            shard_results,
            kb_collection_id=searcher.kb_collection_id,
            rerank_strategy=resolved_strategy,
        )
        pm.record_step(
            "Final Rerank",
            time.time() - step_start,
            details={"num_shard_winners": len(shard_results), "mode": mode, "strategy": resolved_strategy},
        )

        logger.info(f"Final answer: {final_kb}  (mode={mode}, strategy={resolved_strategy})")
        pm.end_pipeline()

        return {
            "status":        "success" if final_kb else "no_results",
            "selected_kb":   final_kb,
            "mode":          mode,
            "confidence":    confidence,
            "strategy":      resolved_strategy,
            "shard_results": shard_results,
            "num_shards":    len(shard_ids),
        }

    except Exception as e:
        import traceback
        logger.error(
            f"retrieve_best_kb_id top-level failure: {type(e).__name__}: {e}\n"
            f"{traceback.format_exc()}"
        )
        pm.end_pipeline()
        return {
            "status":      "error",
            "selected_kb": "",
            "mode":        "error",
            "confidence":  _build_selection_confidence("", {}, mode="error", selected_via="none"),
            "error":       str(e),
            "strategy":    resolved_strategy,
            "shard_results": [],
            "num_shards":  0,
        }
    finally:
        _PIPELINE_METRICS_CV.reset(token)


# ─────────────────────────────────────────────────────────────────────────────
# Content fetch
# ─────────────────────────────────────────────────────────────────────────────

@log_timing
async def get_useful_kb_content(useful_kb: str) -> Dict:
    step_start = time.time()
    pg_client  = get_cached_pg_client()
    summaries_table = pg_client.get_table_name("kb_summaries")
    sum_cols    = _get_table_columns(pg_client, summaries_table)
    has_content = "content" in sum_cols
    kb_id       = str(useful_kb).strip()
    loop        = asyncio.get_running_loop()

    def _fetch():
        cols = ["doc_id", "summary"] + (["content"] if has_content else [])
        with pg_client.get_engine().connect() as conn:
            where_parts = ["doc_id = :kb_id"]
            params: Dict[str, Any] = {"kb_id": kb_id}
            if indexing_config.collection_id:
                where_parts.append("kb_collection_id = :kb_collection_id")
                params["kb_collection_id"] = indexing_config.collection_id
            q = text(
                f"SELECT {', '.join(cols)} FROM {summaries_table} "
                f"WHERE {' AND '.join(where_parts)} LIMIT 1"
            )
            row = conn.execute(q, params).fetchone()
        if not row:
            return {}
        return {"id": kb_id, "summary": row[1], "content": row[2] if has_content else ""}

    result = await loop.run_in_executor(None, _fetch)
    _pm().record_step("Content Retrieval", time.time() - step_start, "success" if result else "failed")
    return result


@log_timing
async def get_useful_kb_json(
    useful_kb: str,
    q_raw: str,
    retrieval_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict]:
    return await generate_summary_useful_kb_list(
        [str(useful_kb).strip()],
        q_raw=q_raw,
        retrieval_scores=retrieval_scores or {},
        kb_collection_id=indexing_config.collection_id,
    )


@log_timing
async def get_final_kb_details(
    useful_kb: str,
    q_raw: str,
    retrieval_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict, List[Dict]]:
    if not useful_kb:
        return {}, []
    return await asyncio.gather(
        get_useful_kb_content(useful_kb),
        get_useful_kb_json(useful_kb, q_raw=q_raw, retrieval_scores=retrieval_scores),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

@log_timing
async def test_single_query(
    query: str,
    expected_kb: str,
    query_type: str,
    kb_name: str,
) -> Dict[str, Any]:
    res = await retrieve_best_kb_id(query)

    all_retrieved: List[str] = []
    for sr in (res.get("shard_results") or []):
        for r in (sr.get("retrieved") or []):
            kbid = str(r.get("KBID") or r.get("kbid") or "").strip()
            if kbid and kbid not in all_retrieved:
                all_retrieved.append(kbid)

    top_kb = str(res.get("selected_kb") or "").strip()
    match  = (top_kb == str(expected_kb).strip())
    if not match and top_kb:
        _record_false_positive(top_kb, query)

    return {
        "status":        res.get("status", "error"),
        "query":         query,
        "query_type":    query_type,
        "kb_name":       kb_name,
        "expected_kb":   str(expected_kb),
        "top_kb":        top_kb,
        "match":         match,
        "mode":          res.get("mode"),
        "confidence":    res.get("confidence"),
        "strategy":      res.get("strategy"),
        "num_shards":    res.get("num_shards", 0),
        "shard_winners": [sr.get("winner_kb") for sr in (res.get("shard_results") or [])],
        "retrieved_kbs": all_retrieved,
    }


async def test_all_kb_queries(
    queries_file: str = "kb_out/kb_queries.json",
    batch_size: int = 50,
) -> Dict:
    queries_path = Path(queries_file)
    if not queries_path.exists():
        logger.error(f"Queries file not found: {queries_file}")
        return {"total_tests": 0, "results": [], "statistics": {}, "error": f"File not found: {queries_file}"}

    with open(queries_path, "r", encoding="utf-8") as f:
        kb_queries = json.load(f)

    logger.info(f"Loaded {len(kb_queries)} KB articles with queries")

    test_cases = []
    for kb_entry in kb_queries:
        kb_number = str(kb_entry.get("KB_number"))
        kb_name   = kb_entry.get("name", f"KB_{kb_number}")
        for q in kb_entry.get("generic_queries",  []):
            test_cases.append({"query": q, "expected_kb": kb_number, "query_type": "generic",  "kb_name": kb_name})
        for q in kb_entry.get("specific_queries", []):
            test_cases.append({"query": q, "expected_kb": kb_number, "query_type": "specific", "kb_name": kb_name})

    logger.info(f"Total test cases: {len(test_cases)}")
    logger.info(f"Concurrency={VALIDATION_CONCURRENCY}  throttle={VALIDATION_THROTTLE_SECONDS}s")
    logger.info(
        f"RERANK_STRATEGY={RERANK_STRATEGY}  RERANK_TOP_N={RERANK_TOP_N}  "
        f"USE_CE={USE_CE_RERANK}  USE_LLM={USE_LLM_RERANK}  "
        f"CE_TIE_DEDUP_ENABLED={CE_TIE_DEDUP_ENABLED}  CE_TIE_MARGIN={CE_TIE_MARGIN}  CE_INTRA_FAMILY_CE={CE_INTRA_FAMILY_CE}  "
        f"DEBUG_RERANK={DEBUG_RERANK}  DEBUG_RERANK_TOPM={DEBUG_RERANK_TOPM}"
    )
    logger.info("=" * 80)

    results: List[Dict[str, Any]] = []
    start_time = time.time()
    sem = asyncio.Semaphore(VALIDATION_CONCURRENCY)

    async def _run_case(test_case: Dict, global_idx: int) -> Tuple[int, Dict]:
        async with sem:
            logger.info(f"[{global_idx}/{len(test_cases)}] {test_case['query'][:60]}...")
            result = await test_single_query(
                query=test_case["query"],
                expected_kb=test_case["expected_kb"],
                query_type=test_case["query_type"],
                kb_name=test_case["kb_name"],
            )
            if result["match"]:
                logger.info(
                    f"  ✓ PASS — {result['top_kb']}  "
                    f"(mode={result.get('mode')}  strategy={result.get('strategy')}  shards={result.get('num_shards')})"
                )
            else:
                logger.warning(
                    f"  ✗ FAIL — expected={result['expected_kb']}  got={result['top_kb']}  "
                    f"strategy={result.get('strategy')}  "
                    f"shard_winners={result.get('shard_winners')}  "
                    f"retrieved={result['retrieved_kbs'][:5]}"
                )
            if VALIDATION_THROTTLE_SECONDS > 0:
                await asyncio.sleep(VALIDATION_THROTTLE_SECONDS)
            result["_global_idx"] = global_idx
            return global_idx, result

    num_batches = (len(test_cases) + batch_size - 1) // batch_size
    for batch_num in range(num_batches):
        bs    = batch_num * batch_size
        be    = min(bs + batch_size, len(test_cases))
        batch = test_cases[bs:be]
        logger.info(f"\nBatch {batch_num + 1}/{num_batches} ({len(batch)} queries)")

        tasks = [asyncio.create_task(_run_case(tc, bs + i + 1)) for i, tc in enumerate(batch)]
        for coro in asyncio.as_completed(tasks):
            try:
                _, r = await coro
                results.append(r)
            except Exception as e:
                logger.error(f"Batch task failed: {e}")
                results.append({"status": "error", "match": False, "error": str(e)})

        if batch_num < num_batches - 1:
            import gc; gc.collect()
            await asyncio.sleep(2)

    results.sort(key=lambda x: int(x.get("_global_idx", 10**9)))
    for r in results:
        r.pop("_global_idx", None)

    log_false_positive_summary(top_n=20)

    total_time   = time.time() - start_time
    total_tests  = len(results)
    matched      = sum(1 for r in results if r.get("match"))
    generic_res  = [r for r in results if r.get("query_type") == "generic"]
    specific_res = [r for r in results if r.get("query_type") == "specific"]
    accuracy     = matched / total_tests * 100 if total_tests else 0
    generic_acc  = sum(1 for r in generic_res  if r.get("match")) / len(generic_res)  * 100 if generic_res  else 0
    specific_acc = sum(1 for r in specific_res if r.get("match")) / len(specific_res) * 100 if specific_res else 0

    statistics = {
        "total_tests":      total_tests,
        "matched_tests":    matched,
        "overall_accuracy": round(accuracy, 2),
        "generic_queries":  {
            "total":   len(generic_res),
            "matches": sum(1 for r in generic_res  if r.get("match")),
            "accuracy": round(generic_acc, 2),
        },
        "specific_queries": {
            "total":   len(specific_res),
            "matches": sum(1 for r in specific_res if r.get("match")),
            "accuracy": round(specific_acc, 2),
        },
        "total_time_seconds":     round(total_time, 2),
        "avg_time_per_query":     round(total_time / total_tests, 2) if total_tests else 0,
        "validation_concurrency": VALIDATION_CONCURRENCY,
        "rerank": {
            "strategy":     RERANK_STRATEGY,
            "rerank_top_n": RERANK_TOP_N,
            "use_ce":       USE_CE_RERANK,
            "use_llm":      USE_LLM_RERANK,
            "ce_tie_dedup_enabled": CE_TIE_DEDUP_ENABLED,
            "ce_tie_margin": CE_TIE_MARGIN,
            "ce_tie_topk": CE_TIE_TOPK,
            "ce_intra_family_ce": CE_INTRA_FAMILY_CE,
        },
        "debug": {
            "debug_rerank": DEBUG_RERANK,
            "debug_rerank_topm": DEBUG_RERANK_TOPM,
        },
        "openai_retry": {
            "max_retries":     OPENAI_MAX_RETRIES,
            "backoff_initial": OPENAI_BACKOFF_INITIAL,
            "backoff_max":     OPENAI_BACKOFF_MAX,
            "multiplier":      OPENAI_BACKOFF_MULTIPLIER,
            "jitter":          OPENAI_BACKOFF_JITTER,
        },
        "top_false_positive_docs": [
            {"doc_id": doc_id, "wrong_count": count}
            for doc_id, count in _fp_counter.most_common(20)
        ],
    }

    logger.info(f"\n{'='*80}\nTEST SUMMARY\n{'='*80}")
    logger.info(f"Total: {total_tests}  Matched: {matched}  Accuracy: {accuracy:.2f}%")
    logger.info(f"Generic: {generic_acc:.2f}%  Specific: {specific_acc:.2f}%")
    logger.info(f"Time: {total_time:.2f}s  Avg/query: {total_time/total_tests if total_tests else 0:.2f}s")

    output_file = "kb_validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"statistics": statistics, "results": results}, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_file}")

    return {"statistics": statistics, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

async def main_validation():
    results = await test_all_kb_queries("kb_out/kb_queries.json")
    if results.get("error"):
        logger.error(f"Validation failed: {results['error']}")
        return
    acc   = results.get("statistics", {}).get("overall_accuracy", 0)
    label = "✓ EXCELLENT" if acc >= 90 else ("⚠ GOOD" if acc >= 70 else "✗ NEEDS IMPROVEMENT")
    logger.info(f"{label}: {acc}% accuracy")


async def main_single_query():
    res = await retrieve_best_kb_id("System running slow")
    logger.info(
        f"Selected KB: {res.get('selected_kb')}  "
        f"(mode={res.get('mode')}  confidence={res.get('confidence', {}).get('score')}  "
        f"strategy={res.get('strategy')}  shards={res.get('num_shards')})"
    )


if __name__ == "__main__":
    asyncio.run(main_single_query())
