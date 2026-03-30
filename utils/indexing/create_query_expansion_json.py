"""
Software/hardware query expansion extraction pipeline.
"""

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI
from sqlalchemy import text
from tqdm import tqdm

from indexing_config import indexing_config
from pgvec_client import get_shared_pgvector_client


load_dotenv()

logger = logging.getLogger(__name__)


ANALYST_INSTRUCTIONS = """You are a Software/Hardware/IT Systems Inventory Extractor (HIGH PRECISION).

INPUT:
- The text to analyze will be provided in the user message between <TEXT> ... </TEXT>.
- Treat <TEXT> as the ONLY source of truth.

TASK:
Extract ONLY real, explicitly mentioned software, hardware, or named IT systems/tools/platforms.

STRICT ANTI-HALLUCINATION RULES:
- DO NOT guess, infer, or expand abbreviations into product names unless the full product name appears in <TEXT>.
- DO NOT include generic concepts unless a named product/system is explicitly present.
Examples to EXCLUDE when unnamed: "VPN", "email", "laptop", "shared drive", "ticket", "browser".
- Include internal tools/platforms ONLY if they are named in <TEXT> (e.g., "IT Self-Service Portal", "ServiceNow", "Intune").
- "item" must match the text EXACTLY as written (normalize whitespace only). Do not correct spelling or add vendor names.
- Do NOT extract people, teams, departments, roles, or URLs as items unless they are clearly the name of a system/tool.

ALLOWED ITEM TYPES:
- Software products and services (e.g., "Microsoft Teams", "Okta", "ServiceNow")
- Device classes ONLY if specifically named as a model/brand/product line in <TEXT> (e.g., "Dell Latitude 7420", "iPhone 13")
- Operating systems ONLY if explicitly named (e.g., "Windows 11", "Ubuntu 22.04")
- Network/security/management tools ONLY if explicitly named (e.g., "Intune", "Jamf", "CrowdStrike")

EXTRACTION PROCESS (silent, do not output):
1) Candidate spotting: identify capitalized product names, branded terms, exact tool names, model numbers.
2) Evidence check: verify each candidate appears verbatim in <TEXT>.
3) Purpose/method grounding:
- purpose: ONLY if <TEXT> states what it is used for. Otherwise "".
- method: ONLY if <TEXT> states how it is used/managed/troubleshot. Otherwise "".
- Never invent purpose/method.

DEDUPLICATION:
- Merge duplicates by item name case-insensitively.
- Preserve the first-seen casing exactly as in <TEXT>.
- If two mentions provide different purpose/method, keep the most informative grounded statement(s) (still one short sentence each).

FIELD RULES:
- item: exact name as written (whitespace normalized).
- purpose: one short sentence (max ~20 words), directly supported by <TEXT>.
- method: one short sentence (max ~25 words), directly supported by <TEXT>.
- If purpose/method is not explicitly stated, use "".

OUTPUT (JSON ONLY, no markdown, no extra keys):
{
  "items": [
    {"item": "...", "purpose": "...", "method": "..."}
  ]
}
The top-level object must contain exactly one key: "items".
- Do not use alternate top-level keys like "result", "data", or "output".
- "items" must always be an array, even when empty.
- Each array item must contain exactly these keys: "item", "purpose", "method".
If nothing qualifies, output exactly:
{"items": []}
""".strip()


def get_cached_pg_client():
    return get_shared_pgvector_client()


class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval_sec <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self.min_interval_sec
                    return
                sleep_for = self._next_allowed - now
            time.sleep(min(sleep_for, 0.2))


@dataclass(frozen=True)
class ExtractionScope:
    shard_id: Optional[int] = None
    collection_id: Optional[str] = None

    @property
    def shard_tag(self) -> str:
        return f"shard={self.shard_id}" if self.shard_id is not None else "all"

    @property
    def shard_key(self) -> int:
        return self.shard_id if self.shard_id is not None else -1

    @property
    def collection_key(self) -> str:
        return self.collection_id or ""


@dataclass
class ExtractedItem:
    item: str
    purpose: str = ""
    method: str = ""


@dataclass
class RecordExtractionResult:
    doc_id: Any
    response: str
    json_valid: bool
    parsed_data: Optional[Any]
    status: str

    def as_legacy_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "response": self.response,
            "json_valid": self.json_valid,
            "parsed_data": self.parsed_data,
            "status": self.status,
        }


@dataclass(frozen=True)
class OutputArtifacts:
    text_path: Path
    json_path: Path
    kb_snapshot_path: Path


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def _clean_text_field(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _item_from_raw(raw: Dict[str, Any]) -> Optional[ExtractedItem]:
    name = raw.get("item") or raw.get("name")
    if not isinstance(name, str):
        return None
    normalized = _normalize_name(name)
    if not normalized:
        return None
    return ExtractedItem(
        item=normalized,
        purpose=_clean_text_field(raw.get("purpose")),
        method=_clean_text_field(raw.get("method")),
    )


def _extract_items_from_parsed(parsed: Any) -> List[ExtractedItem]:
    raw_items: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        raw_items = [x for x in parsed if isinstance(x, dict)]
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("items"), list):
            raw_items.extend(x for x in parsed["items"] if isinstance(x, dict))
        if isinstance(parsed.get("result"), list):
            raw_items.extend(x for x in parsed["result"] if isinstance(x, dict))
        if "item" in parsed or "name" in parsed:
            raw_items.append(parsed)
    return [item for raw in raw_items if (item := _item_from_raw(raw)) is not None]


def _items_to_dicts(items: Sequence[ExtractedItem]) -> List[Dict[str, str]]:
    return [asdict(item) for item in items]


def _consolidate_items(items: Sequence[ExtractedItem]) -> List[ExtractedItem]:
    deduped: Dict[str, ExtractedItem] = {}
    for item in items:
        ci_key = item.item.casefold()
        if ci_key not in deduped:
            deduped[ci_key] = ExtractedItem(item=item.item, purpose=item.purpose, method=item.method)
            continue
        prev = deduped[ci_key]
        deduped[ci_key] = ExtractedItem(
            item=prev.item,
            purpose=item.purpose if item.purpose and len(item.purpose) > len(prev.purpose) else prev.purpose,
            method=item.method if item.method and len(item.method) > len(prev.method) else prev.method,
        )
    return list(deduped.values())


def _merge_consolidated_kb(
    current_kb: Dict[str, ExtractedItem],
    new_items: Sequence[ExtractedItem],
) -> Dict[str, ExtractedItem]:
    for item in new_items:
        ci_key = item.item.casefold()
        if ci_key not in current_kb:
            current_kb[ci_key] = item
            continue
        prev = current_kb[ci_key]
        current_kb[ci_key] = ExtractedItem(
            item=prev.item,
            purpose=item.purpose if item.purpose and len(item.purpose) > len(prev.purpose) else prev.purpose,
            method=item.method if item.method and len(item.method) > len(prev.method) else prev.method,
        )
    return current_kb


def _build_user_prompt(summary: str, content: str) -> str:
    article_content = f"Summary: {summary}\n\nContent: {content}"
    return f"<TEXT>\n{article_content}\n</TEXT>"


def _parse_extraction_response(ai_response: str) -> Tuple[bool, Optional[Any], str]:
    try:
        parsed_json = json.loads(ai_response)
    except json.JSONDecodeError:
        return False, None, "invalid_json"

    if not isinstance(parsed_json, (list, dict)):
        return False, parsed_json, "invalid_schema"

    _extract_items_from_parsed(parsed_json)
    return True, parsed_json, "success"


class SwHwExtractor:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        global_rate_limit_sec: float,
        openai_client: Optional[AzureOpenAI] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.rate_limiter = RateLimiter(global_rate_limit_sec)
        self.openai_client = openai_client or AzureOpenAI(
            api_version=indexing_config.llm_settings.api_version,
            azure_endpoint=indexing_config.llm_settings.azure_endpoint,
            api_key=indexing_config.llm_settings.api_key,
        )

    def process_record(self, row: Dict[str, Any]) -> RecordExtractionResult:
        doc_id = row.get("doc_id")
        summary = row.get("summary") or ""
        content = row.get("content") or ""
        analyst_user_prompt = _build_user_prompt(summary, content)
        max_retries = 3
        base_retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": ANALYST_INSTRUCTIONS},
                        {"role": "user", "content": analyst_user_prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                ai_response = (response.choices[0].message.content or "").strip()
                json_valid, parsed_data, status = _parse_extraction_response(ai_response)
                return RecordExtractionResult(
                    doc_id=doc_id,
                    response=ai_response,
                    json_valid=json_valid,
                    parsed_data=parsed_data,
                    status=status,
                )
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < max_retries - 1:
                    time.sleep(base_retry_delay * (2 ** attempt))
                    continue
                return RecordExtractionResult(
                    doc_id=doc_id,
                    response=f"ERROR: {exc}",
                    json_valid=False,
                    parsed_data=None,
                    status="error",
                )

        return RecordExtractionResult(
            doc_id=doc_id,
            response="ERROR: Max retries exceeded",
            json_valid=False,
            parsed_data=None,
            status="error",
        )


class SwHwKnowledgeBaseRepository:
    def __init__(self, pg_client=None):
        self.pg_client = pg_client or get_cached_pg_client()

    def _table_name(self) -> str:
        return self.pg_client.get_table_name("sw_hw_knowledge_base")

    def _legacy_table_name(self, shard_id: Optional[int]) -> str:
        if shard_id is None:
            return self.pg_client.get_table_name("sw_hw_knowledge_base")
        return self.pg_client.get_table_name(f"sw_hw_knowledge_base_shard_{shard_id}")

    def ensure_table(self) -> None:
        table_name = self._table_name()
        with self.pg_client.get_engine().begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        collection_id TEXT NOT NULL DEFAULT '',
                        shard_id INT NOT NULL,
                        data JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (collection_id, shard_id)
                    )
                    """
                )
            )

    def persist_items(self, scope: ExtractionScope, items: Sequence[ExtractedItem]) -> None:
        self.ensure_table()
        table_name = self._table_name()
        payload = json.dumps(_items_to_dicts(items), ensure_ascii=False)
        with self.pg_client.get_engine().begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {table_name} (collection_id, shard_id, data, updated_at)
                    VALUES (:collection_id, :shard_id, CAST(:data AS JSONB), NOW())
                    ON CONFLICT (collection_id, shard_id)
                    DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()
                    """
                ),
                {
                    "collection_id": scope.collection_key,
                    "shard_id": scope.shard_key,
                    "data": payload,
                },
            )
        logger.info(
            "Persisted %s consolidated items for %s to %s",
            len(items),
            scope.shard_tag,
            table_name,
        )

    def load_items_for_shard(
        self,
        shard_id: int,
        collection_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        rows = self._load_from_unified_table(shard_id=shard_id, collection_id=collection_id)
        if rows is not None:
            return rows
        return self._load_from_legacy_table(shard_id)

    def _load_from_unified_table(
        self,
        shard_id: int,
        collection_id: Optional[str],
    ) -> Optional[List[Dict[str, str]]]:
        table_name = self._table_name()
        try:
            with self.pg_client.get_engine().connect() as conn:
                row = conn.execute(
                    text(
                        f"""
                        SELECT data
                        FROM {table_name}
                        WHERE collection_id = :collection_id AND shard_id = :shard_id
                        LIMIT 1
                        """
                    ),
                    {"collection_id": collection_id or "", "shard_id": shard_id},
                ).fetchone()
            if not row:
                return None
            return _coerce_loaded_items(row[0])
        except Exception as exc:
            logger.info("Unified sw/hw table unavailable for shard=%s: %s", shard_id, exc)
            return None

    def _load_from_legacy_table(self, shard_id: int) -> List[Dict[str, str]]:
        table_name = self._legacy_table_name(shard_id)
        try:
            with self.pg_client.get_engine().connect() as conn:
                row = conn.execute(
                    text(f"SELECT data FROM {table_name} WHERE id = 1 LIMIT 1")
                ).fetchone()
            if not row:
                return []
            return _coerce_loaded_items(row[0])
        except Exception as exc:
            logger.info("Legacy sw/hw table unavailable for shard=%s: %s", shard_id, exc)
            return []

    def load_shard_ids(self, collection_id: str) -> List[int]:
        registry_table = self.pg_client.get_table_name("kb_shard_registry")
        try:
            with self.pg_client.get_engine().connect() as conn:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT shard_id
                        FROM {registry_table}
                        WHERE kb_collection_id = :cid
                        ORDER BY shard_id
                        """
                    ),
                    {"cid": collection_id},
                ).fetchall()
        except Exception as exc:
            logger.warning("Failed to read shard registry: %s; defaulting to [0]", exc)
            return [0]

        shard_ids = [int(row[0]) for row in rows if row[0] is not None]
        return shard_ids if shard_ids else [0]


def _coerce_loaded_items(data: Any) -> List[Dict[str, str]]:
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, list):
        return []
    return _items_to_dicts(_extract_items_from_parsed({"items": data}))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate" in msg and "limit" in msg


def _derive_output_artifacts(output_file: str, scope: ExtractionScope) -> OutputArtifacts:
    base_path = Path(output_file)
    if scope.shard_id is not None:
        stem = f"{base_path.stem}_shard_{scope.shard_id}"
        return OutputArtifacts(
            text_path=base_path.with_name(f"{stem}{base_path.suffix}"),
            json_path=base_path.with_name(f"{stem}.json"),
            kb_snapshot_path=base_path.with_name(f"{stem}_kb_snapshot.json"),
        )
    return OutputArtifacts(
        text_path=base_path,
        json_path=base_path.with_suffix(".json"),
        kb_snapshot_path=base_path.with_name(f"{base_path.stem}_kb_snapshot.json"),
    )


def _initialize_output_files(artifacts: OutputArtifacts, scope: ExtractionScope) -> None:
    with artifacts.text_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Software and Hardware Configuration Extraction Results ({scope.shard_tag})\n")
        handle.write("=" * 80 + "\n\n")
    with artifacts.json_path.open("w", encoding="utf-8") as handle:
        json.dump([], handle, ensure_ascii=False, indent=2)


def _append_batch_text_results(
    artifacts: OutputArtifacts,
    batch_number: int,
    num_batches: int,
    start: int,
    end: int,
    batch_results: Sequence[RecordExtractionResult],
) -> None:
    with artifacts.text_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nBATCH {batch_number}/{num_batches} (docs {start}-{end - 1})\n")
        handle.write("-" * 80 + "\n")
        for result in batch_results:
            handle.write(
                f"Document ID: {result.doc_id}\n"
                f"Status: {result.status}\n"
                f"Valid JSON: {result.json_valid}\n"
                f"{'-' * 40}\n"
                f"{result.response}\n"
                f"{'-' * 80}\n"
            )


def _write_json_output(artifacts: OutputArtifacts, valid_results: Sequence[Dict[str, Any]]) -> None:
    with artifacts.json_path.open("w", encoding="utf-8") as handle:
        json.dump(list(valid_results), handle, ensure_ascii=False, indent=2)


def _write_kb_snapshot(artifacts: OutputArtifacts, kb_items: Sequence[ExtractedItem]) -> None:
    with artifacts.kb_snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(_items_to_dicts(kb_items), handle, ensure_ascii=False, indent=2)


def _build_scope_query(scope: ExtractionScope, summaries_table: str) -> Tuple[Any, Dict[str, Any]]:
    where_parts: List[str] = []
    params: Dict[str, Any] = {}

    if scope.shard_id is not None:
        where_parts.append("shard_id = :shard_id")
        params["shard_id"] = scope.shard_id

    if scope.collection_id:
        where_parts.append("kb_collection_id = :collection_id")
        params["collection_id"] = scope.collection_id

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    query = text(
        f"""
        SELECT doc_id, summary, content
        FROM {summaries_table}
        {where_clause}
        ORDER BY doc_id
        """
    )
    return query, params


def _fetch_summary_rows(pg_client, scope: ExtractionScope) -> List[Dict[str, Any]]:
    summaries_table = pg_client.get_table_name("kb_summaries")
    query, params = _build_scope_query(scope, summaries_table)
    logger.info("[%s] Querying %s", scope.shard_tag, summaries_table)
    with pg_client.get_engine().connect() as conn:
        rows = conn.execute(query, params).mappings().all()
    return [dict(row) for row in rows]


def _iter_batches(records: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Tuple[int, int, List[Dict[str, Any]]]]:
    total = len(records)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end, list(records[start:end])


def _process_batch(
    extractor: SwHwExtractor,
    batch_records: Sequence[Dict[str, Any]],
    max_workers: int,
    progress_label: str,
) -> List[RecordExtractionResult]:
    results: List[RecordExtractionResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extractor.process_record, row) for row in batch_records]
        for future in tqdm(as_completed(futures), total=len(futures), desc=progress_label):
            results.append(future.result())
    results.sort(key=lambda result: (result.doc_id is None, result.doc_id))
    return results


def _summarize_results(results_all: Sequence[RecordExtractionResult]) -> Tuple[int, int, int]:
    total_success = sum(1 for result in results_all if result.status == "success")
    total_error = sum(1 for result in results_all if result.status == "error")
    total_valid = sum(1 for result in results_all if result.json_valid)
    return total_success, total_error, total_valid


def _log_final_summary(
    scope: ExtractionScope,
    artifacts: OutputArtifacts,
    results_all: Sequence[RecordExtractionResult],
    kb_final: Sequence[ExtractedItem],
) -> None:
    total_success, total_error, total_valid = _summarize_results(results_all)
    logger.info("=" * 80)
    logger.info("[%s] Results:      %s", scope.shard_tag, artifacts.text_path)
    logger.info("[%s] Per-doc JSON: %s", scope.shard_tag, artifacts.json_path)
    logger.info("[%s] KB snapshot:  %s", scope.shard_tag, artifacts.kb_snapshot_path)
    logger.info("[%s] Total records:   %s", scope.shard_tag, len(results_all))
    logger.info("[%s] Successful:      %s", scope.shard_tag, total_success)
    logger.info("[%s] Failed:          %s", scope.shard_tag, total_error)
    logger.info("[%s] Valid JSON:      %s", scope.shard_tag, total_valid)
    logger.info("[%s] Unique KB items: %s", scope.shard_tag, len(kb_final))
    logger.info("=" * 80)


def extract_software_hardware_config_batched_pgvector(
    output_file: str = "software_hardware_extraction.txt",
    max_workers: int = 5,
    global_rate_limit_sec: float = 1.0,
    batch_size: int = 100,
    persist_to_db: bool = True,
    persist_each_batch: bool = True,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    shard_id: Optional[int] = None,
    collection_id: Optional[str] = None,
) -> Dict[str, Any]:
    scope = ExtractionScope(shard_id=shard_id, collection_id=collection_id)
    artifacts = _derive_output_artifacts(output_file, scope)
    _initialize_output_files(artifacts, scope)

    pg_client = get_cached_pg_client()
    repository = SwHwKnowledgeBaseRepository(pg_client)
    extractor = SwHwExtractor(
        model_name=model_name,
        temperature=temperature,
        global_rate_limit_sec=global_rate_limit_sec,
    )

    kb_dict: Dict[str, ExtractedItem] = {}
    results_all: List[RecordExtractionResult] = []
    valid_results_all: List[Dict[str, Any]] = []

    try:
        records = _fetch_summary_rows(pg_client, scope)
        total = len(records)
        num_batches = (total + batch_size - 1) // batch_size if total else 0

        logger.info(
            "[%s] Retrieved %s records | batch_size=%s max_workers=%s rate_limit=%ss",
            scope.shard_tag,
            total,
            batch_size,
            max_workers,
            global_rate_limit_sec,
        )

        for batch_index, (start, end, batch_records) in enumerate(_iter_batches(records, batch_size), start=1):
            logger.info(
                "[%s] Batch %s/%s: docs %s..%s (count=%s)",
                scope.shard_tag,
                batch_index,
                num_batches,
                start,
                end - 1,
                len(batch_records),
            )

            batch_results = _process_batch(
                extractor=extractor,
                batch_records=batch_records,
                max_workers=max_workers,
                progress_label=f"[{scope.shard_tag}] batch {batch_index}/{num_batches}",
            )
            _append_batch_text_results(artifacts, batch_index, num_batches, start, end, batch_results)

            raw_items_batch: List[ExtractedItem] = []
            for result in batch_results:
                if result.json_valid and result.parsed_data is not None:
                    valid_results_all.append(
                        {"doc_id": result.doc_id, "configurations": result.parsed_data}
                    )
                    raw_items_batch.extend(_extract_items_from_parsed(result.parsed_data))

            batch_consolidated = _consolidate_items(raw_items_batch)
            kb_before = len(kb_dict)
            kb_dict = _merge_consolidated_kb(kb_dict, batch_consolidated)

            logger.info(
                "[%s] Batch raw=%s consolidated=%s KB: %s->%s",
                scope.shard_tag,
                len(raw_items_batch),
                len(batch_consolidated),
                kb_before,
                len(kb_dict),
            )

            kb_snapshot = sorted(kb_dict.values(), key=lambda item: item.item.casefold())
            _write_json_output(artifacts, valid_results_all)
            _write_kb_snapshot(artifacts, kb_snapshot)

            if persist_to_db and persist_each_batch:
                try:
                    repository.persist_items(scope, kb_snapshot)
                except Exception as exc:
                    logger.exception("[%s] Failed to persist batch %s: %s", scope.shard_tag, batch_index, exc)

            results_all.extend(batch_results)

        kb_final = sorted(kb_dict.values(), key=lambda item: item.item.casefold())
        _write_json_output(artifacts, valid_results_all)
        _write_kb_snapshot(artifacts, kb_final)

        if persist_to_db:
            try:
                repository.persist_items(scope, kb_final)
            except Exception as exc:
                logger.exception("[%s] Failed to persist final KB: %s", scope.shard_tag, exc)

        _log_final_summary(scope, artifacts, results_all, kb_final)
        return {
            "results": [result.as_legacy_dict() for result in results_all],
            "knowledge_base": _items_to_dicts(kb_final),
        }
    except Exception as exc:
        logger.exception("[%s] Fatal error: %s", scope.shard_tag, exc)
        return {
            "results": [result.as_legacy_dict() for result in results_all],
            "knowledge_base": [],
        }


def extract_sw_hw_for_shard(
    shard_id: int,
    output_file: str = "software_hardware_extraction.txt",
    max_workers: int = 5,
    global_rate_limit_sec: float = 1.0,
    batch_size: int = 100,
    persist_to_db: bool = True,
    persist_each_batch: bool = True,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    collection_id: Optional[str] = None,
) -> Dict[str, Any]:
    return extract_software_hardware_config_batched_pgvector(
        output_file=output_file,
        max_workers=max_workers,
        global_rate_limit_sec=global_rate_limit_sec,
        batch_size=batch_size,
        persist_to_db=persist_to_db,
        persist_each_batch=persist_each_batch,
        model_name=model_name,
        temperature=temperature,
        shard_id=shard_id,
        collection_id=collection_id,
    )


def _load_shard_ids_from_registry(pg_client, collection_id: str) -> List[int]:
    repository = SwHwKnowledgeBaseRepository(pg_client)
    return repository.load_shard_ids(collection_id)


def load_sw_hw_for_shard(
    shard_id: int,
    pg_client=None,
    collection_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    repository = SwHwKnowledgeBaseRepository(pg_client)
    return repository.load_items_for_shard(shard_id=shard_id, collection_id=collection_id)


async def load_sw_hw_all_shards_parallel(
    shard_ids: List[int],
    pg_client=None,
    collection_id: Optional[str] = None,
) -> Dict[int, List[Dict[str, str]]]:
    repository = SwHwKnowledgeBaseRepository(pg_client)
    loop = asyncio.get_event_loop()
    tasks = {
        sid: loop.run_in_executor(None, repository.load_items_for_shard, sid, collection_id)
        for sid in shard_ids
    }
    return {sid: await task for sid, task in tasks.items()}


def extract_sw_hw_all_shards_parallel(
    collection_id: str,
    shard_ids: Optional[List[int]] = None,
    output_file: str = "software_hardware_extraction.txt",
    max_workers_per_shard: int = 5,
    global_rate_limit_sec: float = 1.0,
    batch_size: int = 100,
    persist_to_db: bool = True,
    persist_each_batch: bool = True,
    model_name: str = "gpt-4o",
    temperature: float = 0.1,
    shard_parallelism: int = 3,
) -> Dict[int, Dict[str, Any]]:
    pg_client = get_cached_pg_client()

    if shard_ids is None:
        shard_ids = _load_shard_ids_from_registry(pg_client, collection_id)

    logger.info(
        "Extracting sw/hw for %s shard(s): %s (shard_parallelism=%s)",
        len(shard_ids),
        shard_ids,
        shard_parallelism,
    )

    results: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=shard_parallelism) as executor:
        future_to_shard = {
            executor.submit(
                extract_sw_hw_for_shard,
                sid,
                output_file,
                max_workers_per_shard,
                global_rate_limit_sec,
                batch_size,
                persist_to_db,
                persist_each_batch,
                model_name,
                temperature,
                collection_id,
            ): sid
            for sid in shard_ids
        }
        for future in as_completed(future_to_shard):
            shard_id = future_to_shard[future]
            try:
                results[shard_id] = future.result()
                logger.info(
                    "[shard=%s] Done - %s unique items",
                    shard_id,
                    len(results[shard_id].get("knowledge_base", [])),
                )
            except Exception as exc:
                logger.exception("[shard=%s] Failed: %s", shard_id, exc)
                results[shard_id] = {"results": [], "knowledge_base": []}
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    collection_id = indexing_config.collection_id
    results = extract_sw_hw_all_shards_parallel(
        collection_id=collection_id,
        output_file="software_hardware_extraction.txt",
        max_workers_per_shard=5,
        global_rate_limit_sec=1.0,
        batch_size=100,
        persist_to_db=True,
        persist_each_batch=True,
        model_name=indexing_config.llm_model,
        temperature=0.1,
        shard_parallelism=3,
    )
    for shard_id, result in sorted(results.items()):
        print(f"Shard {shard_id}: {len(result.get('knowledge_base', []))} unique items")
