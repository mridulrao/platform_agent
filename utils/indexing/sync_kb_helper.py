from __future__ import annotations

import json
import random
import re
import time
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from indexing_config import indexing_config

logger = logging.getLogger(__name__)

MAX_RETRIES = indexing_config.indexing_max_retries
BASE_DELAY = indexing_config.indexing_retry_delay_seconds

load_dotenv()

ALLOWED_CATEGORIES = {"Hardware", "Software", "Both"}
ALLOWED_ARTICLE_TYPES = {
    "Setup/Installation",
    "Troubleshooting",
    "Policy/Process",
    "How-To",
    "Reference",
}


# -------------------------
# Text utilities
# -------------------------
def _clean_safelinks(text: str) -> str:
    """
    Remove SafeLinks URLs + base64 blobs and keep structure as much as possible.
    """
    if not text:
        return ""

    # SafeLinks URLs
    safelinks_pattern = r"https://eur01\.safelinks\.protection\.outlook\.com/\?url=[^\s)\"']+"
    text = re.sub(safelinks_pattern, "", text)

    # Remove 'Original URL:' boilerplate
    original_url_pattern = r"Original URL:[^\n]*Click or tap if you trust this link\."
    text = re.sub(original_url_pattern, "", text, flags=re.IGNORECASE)

    # Remove base64 images / huge encoded strings
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE_REMOVED]", text)
    text = re.sub(r"<img[^>]*src=[\"']data:image/[^\"']*[\"'][^>]*>", "[IMAGE_REMOVED]", text, flags=re.IGNORECASE)

    # Very long base64-ish tokens
    text = re.sub(r"\b[A-Za-z0-9+/]{200,}={0,2}\b", "[ENCODED_CONTENT_REMOVED]", text)

    # Normalize newlines first
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Then lightly normalize whitespace on each line (keep paragraph breaks)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()

    return text


def clean_kb_content(text: str, *, max_chars: int = 16000) -> str:
    return _clip_text(_clean_safelinks(text), max_chars=max_chars)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _normalize_generic(items: List[str]) -> List[str]:
    cleaned = []
    for x in items:
        x = (x or "").strip().lower()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^\w\s\-/]", "", x)  # keep wi-fi, vpn, teams
        if len(x) < 2:
            continue
        cleaned.append(x)
    return _dedupe_keep_order(cleaned)


def _clean_phrase(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value).strip(" ,;:-")
    return value.strip()


def _clean_phrase_list(values: Any, *, lowercase: bool = False, max_items: int = 12) -> List[str]:
    raw_values = values if isinstance(values, list) else ([values] if isinstance(values, str) else [])
    cleaned: List[str] = []
    for value in raw_values:
        phrase = _clean_phrase(value)
        if not phrase:
            continue
        if lowercase:
            phrase = phrase.lower()
        cleaned.append(phrase)
    return _dedupe_keep_order(cleaned)[:max_items]


def _looks_specific_term(term: str) -> bool:
    if not term:
        return False
    if re.search(r"\d", term):
        return True
    if re.search(r"[A-Z]{2,}", term):
        return True
    parts = term.split()
    if len(parts) >= 2 and any(part[:1].isupper() for part in parts):
        return True
    return False


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return "Software"
    norm = value.strip().lower()
    mapping = {
        "hardware": "Hardware",
        "software": "Software",
        "both": "Both",
    }
    return mapping.get(norm, "Software")


def _normalize_article_type(value: Any) -> str:
    if not isinstance(value, str):
        return "How-To"
    norm = value.strip().lower()
    mapping = {
        "setup/installation": "Setup/Installation",
        "setup": "Setup/Installation",
        "installation": "Setup/Installation",
        "troubleshooting": "Troubleshooting",
        "policy/process": "Policy/Process",
        "policy": "Policy/Process",
        "process": "Policy/Process",
        "how-to": "How-To",
        "how to": "How-To",
        "reference": "Reference",
    }
    return mapping.get(norm, "How-To")


def _clip_text(text: str, max_chars: int = 14000) -> str:
    text = text or ""
    # keep newlines for structure, but avoid huge payloads
    text = text.strip()
    return text[:max_chars]


def _backoff_sleep(attempt: int) -> None:
    sleep_time = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
    logger.warning(
        "Retrying metadata/summary extraction | attempt=%d/%d sleep=%.2fs",
        attempt,
        MAX_RETRIES,
        sleep_time,
    )
    time.sleep(sleep_time)


def _call_json_chat_completion(
    *,
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Model response was not a JSON object")
            return parsed
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"LLM JSON extraction failed after retries: {exc}")
            _backoff_sleep(attempt)


def _normalize_metadata_result(
    *,
    raw: Dict[str, Any],
    kbid: Optional[str],
    kb_title: Optional[str],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(raw)
    normalized.setdefault("keywords", {})
    kw = normalized["keywords"] if isinstance(normalized["keywords"], dict) else {}

    specific_raw = _clean_phrase_list(kw.get("primary_specific"))
    generic_raw = _clean_phrase_list(kw.get("primary_generic"), lowercase=True)
    secondary_entities = _clean_phrase_list(kw.get("secondary_entities"))
    acronyms = _clean_phrase_list(kw.get("acronyms"))
    usecases = _clean_phrase_list(normalized.get("usecase"), lowercase=True, max_items=8)

    primary_specific: List[str] = []
    primary_generic: List[str] = []

    for term in specific_raw:
        if _looks_specific_term(term):
            primary_specific.append(term)
        else:
            primary_generic.append(term.lower())

    for term in generic_raw:
        if _looks_specific_term(term):
            primary_specific.append(term)
        else:
            primary_generic.append(term)

    primary_specific = _dedupe_keep_order(primary_specific)[:10]
    primary_generic = _normalize_generic(primary_generic)[:12]
    secondary_entities = _dedupe_keep_order(secondary_entities)[:10]
    acronyms = _dedupe_keep_order(acronyms)[:10]
    usecases = _dedupe_keep_order(usecases)[:8]

    if not primary_specific:
        title_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\._]{2,}", kb_title or "")
        primary_specific = _dedupe_keep_order(title_tokens[:5])

    if not primary_generic:
        primary_generic = _normalize_generic(primary_specific)[:8] or ["it support"]

    if not usecases:
        topic = _clean_phrase(normalized.get("primary_topic")) or _clean_phrase(kb_title) or "support article"
        usecases = [f"help with {topic.lower()}"]

    normalized["category"] = _normalize_category(normalized.get("category"))
    normalized["article_type"] = _normalize_article_type(normalized.get("article_type"))
    normalized["primary_topic"] = (
        _clean_phrase(normalized.get("primary_topic"))
        or _clean_phrase(kb_title)
        or _clean_phrase(kbid)
        or "IT support article"
    )
    normalized["keywords"] = {
        "primary_specific": primary_specific,
        "primary_generic": primary_generic,
        "secondary_entities": secondary_entities,
        "acronyms": acronyms,
    }
    normalized["usecase"] = usecases
    return normalized


def _normalize_summary_result(
    *,
    raw: Dict[str, Any],
    document_title: str,
    kbid: Optional[str],
) -> Dict[str, Any]:
    normalized = dict(raw)
    normalized["article_title"] = _clean_phrase(normalized.get("article_title")) or _clean_phrase(document_title)
    normalized["kb_id"] = _clean_phrase(normalized.get("kb_id")) or _clean_phrase(kbid)
    normalized["primary_intent"] = (
        _clean_phrase(normalized.get("primary_intent"))
        or f"Help with {normalized['article_title'] or 'this KB'}"
    )
    normalized["summary"] = (
        _clean_phrase(normalized.get("summary"))
        or f"This KB provides guidance related to: {normalized['article_title'] or 'this article'}."
    )
    normalized["query_triggers"] = _clean_phrase_list(normalized.get("query_triggers"), lowercase=True, max_items=12)
    if not normalized["query_triggers"]:
        base = _clean_phrase(document_title)
        normalized["query_triggers"] = [base.lower()] if base else ["help with this issue"]
    normalized["secondary_mentions"] = _clean_phrase_list(normalized.get("secondary_mentions"), max_items=10)
    return normalized


# -------------------------
# Index-time metadata extraction
# -------------------------
def extract_kb_metadata(kb_content: str, kbid: str | None = None, kb_title: str | None = None) -> dict:
    if not kb_content or not kb_content.strip():
        raise ValueError("KB content is empty")

    client = indexing_config.get_llm_client()

    cleaned_content = clean_kb_content(kb_content, max_chars=14000)

    system_prompt = f"""
You enrich an IT support KB article for retrieval.

CRITICAL: Separate PRIMARY topic vs SECONDARY mentions with high precision.
- PRIMARY = what the article actually helps the user do/fix.
- SECONDARY = ownership teams, escalation groups, referenced platforms, incidental acronyms.
- Do not hallucinate tools, vendors, teams, or policies.
- Preserve exact product/version names when explicitly present.
- Do not put ownership teams or escalation groups in primary_specific unless they are the subject of the article.
- primary_generic must be concise lowercase retrieval terms, not full sentences.
- usecase must be short user-intent phrases focused on the PRIMARY topic.

Return ONE JSON object only.

The output must be valid JSON, not pseudo-JSON.
Use exactly these top-level keys and no others:
- category
- article_type
- primary_topic
- keywords
- usecase

Allowed values:
- category must be exactly one of: "Hardware", "Software", "Both"
- article_type must be exactly one of: "Setup/Installation", "Troubleshooting", "Policy/Process", "How-To", "Reference"

Required JSON shape:
{{
  "category": "Software",
  "article_type": "How-To",
  "primary_topic": "example topic",
  "keywords": {{
    "primary_specific": ["Example Product"],
    "primary_generic": ["example generic term"],
    "secondary_entities": ["Example Team"],
    "acronyms": ["VPN"]
  }},
  "usecase": ["example user intent phrase"]
}}

Rules:
- Do not hallucinate.
- Keep output schema exact. No extra keys.
- All list fields must be JSON arrays of strings, never a single string.
- If a list field has no valid values, return [].
- If primary_topic is unclear, return "".
- primary_specific should contain the exact named systems/products/models that are the article subject.
- secondary_entities should contain named but non-primary tools/teams/orgs/platforms.
- If the article is broad, primary_specific may be empty but primary_generic and usecase must still be useful.
- Prefer 5-15 keywords; 3-8 usecases.
- If ticket routing / assignment rules: article_type="Policy/Process".
Context: kbid="{kbid or ""}" title="{kb_title or ""}"
"""

    user_prompt = f"""
KB Title: {kb_title or ""}
KB ID: {kbid or ""}

ARTICLE:
{cleaned_content}
"""

    raw = _call_json_chat_completion(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=indexing_config.llm_model,
    )
    return _normalize_metadata_result(raw=raw, kbid=kbid, kb_title=kb_title)


# -------------------------
# Retrieval/rerank-friendly summary
# -------------------------
def generate_article_summary(document: str, document_title: str, kbid: str | None = None) -> dict:
    client = indexing_config.get_llm_client()
    document_content = clean_kb_content(document, max_chars=16000)

    system_prompt = """
You summarize an IT support KB for a RAG system.

CRITICAL: Avoid incidental mention bias:
- If teams/org names are only ownership/escalation info, list them as secondary_mentions.
- Focus summary and query_triggers on the PRIMARY purpose.
- Do not invent symptoms, products, or procedures.
- query_triggers should be short realistic user search phrases, not article headings.

Return ONE JSON object only:
{
  "article_title": "...",
  "kb_id": "...",
  "primary_intent": "...",
  "summary": "120-180 words plain text",
  "query_triggers": ["6-12 short user phrases"],
  "secondary_mentions": ["0-10 incidental entities"]
}

Rules:
- No markdown.
- Don’t invent details.
- Keep output schema exact. No extra keys.
- `query_triggers` must be a JSON array of strings, never a single string.
- `secondary_mentions` must be a JSON array of strings, never a single string.
- If there are no triggers or mentions, return [] for that field.
- If `article_title`, `kb_id`, `primary_intent`, or `summary` is unknown, return "" for that field.
"""

    user_prompt = f"""
KB Title: {document_title}
KB ID: {kbid or ""}

DOCUMENT:
{document_content}
"""

    raw = _call_json_chat_completion(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=indexing_config.llm_model,
    )
    return _normalize_summary_result(raw=raw, document_title=document_title, kbid=kbid)


# -------------------------
# Embeddings
# -------------------------
def get_openai_embeddings_batch(texts: list[str]) -> list[Optional[list[float]]]:
    valid_indices = [i for i, t in enumerate(texts) if t and str(t).strip()]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        return [None] * len(texts)

    embedding_client = indexing_config.get_embedding_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = embedding_client.embeddings.create(
                model=indexing_config.embedding_model,
                input=valid_texts,
            )

            out: list[Optional[list[float]]] = [None] * len(texts)
            for j, idx in enumerate(valid_indices):
                out[idx] = resp.data[j].embedding
            return out

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Embedding generation failed after retries: {e}")
            _backoff_sleep(attempt)


if __name__ == "__main__":
    sample_content = """..."""
    sample_title = "Troubleshooting Microsoft Outlook Connection Issues"
    sample_kbid = "KB0012457"

    metadata = extract_kb_metadata(sample_content, kbid=sample_kbid, kb_title=sample_title)
    print(json.dumps(metadata, indent=2))

    summary_obj = generate_article_summary(sample_content, document_title=sample_title, kbid=sample_kbid)
    print(json.dumps(summary_obj, indent=2))
    print("Summary chars:", len(summary_obj["summary"]))

    embeddings = get_openai_embeddings_batch(["Microsoft", "Email", "Outlook"])
    print("Emb dims:", [len(e) if e else None for e in embeddings])
