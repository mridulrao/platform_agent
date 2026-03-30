"""
Script to compare knowledgebase_document_assist with vva_kb_summaries
and sync new, updated, and deleted documents based on updated_at timestamps.
Only syncs documents where is_deleted = False.

UPDATED (STRICT Scenario Skip - Robust Unicode Handling + PDF filename support):
- Any article title that contains "Scenario <N>" is SKIPPED entirely.
- .pdf extension is stripped before scenario detection.
- KB number is extracted from filename (e.g. KB-0420_Title.pdf → KB0420)
  when row.number is null/invalid.
- Robust to unicode hyphens/dashes (incl Wi-Fi), NBSP, zero-width chars, etc.
"""

import os
import re
import random
import socket
import threading
import asyncio
import logging
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from sqlalchemy import text
from tqdm.auto import tqdm

from pgvec_client import create_pgvector_client
from create_tables_pgvec import KnowledgeBaseInserter
from create_query_expansion_json import (
    extract_software_hardware_config_batched_pgvector,  # kept for backward compat
    extract_sw_hw_all_shards_parallel,                 # new: one table per shard
)
from sync_kb_helper import clean_kb_content
from indexing_config import indexing_config

# Invalidate the query-enrichment LLM's SW/HW KB cache after every sync so it
# picks up the freshly-rebuilt per-shard sw_hw tables on the next request.
try:
    from instructions.background_agent_instructions import reload_kb_cache as _reload_enrichment_kb_cache
except ImportError:
    _reload_enrichment_kb_cache = None  # instructions module may not be installed in this service

try:
    from generate_test_queries_pgvec import main as generate_test_queries_pgvec
except ImportError:
    generate_test_queries_pgvec = None

try:
    from run_test_queries_pgvec import main as run_test_queries_pgvec
except ImportError:
    run_test_queries_pgvec = None

load_dotenv()

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Keep third-party API/client logs from flooding the console while still
    # surfacing warnings and errors from retries or request failures.
    for noisy_logger in (
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
        "azure",
        "azure.core",
        "azure.core.pipeline",
        "azure.core.pipeline.policies.http_logging_policy",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


configure_logging()
logger = logging.getLogger(__name__)

_pg_client = None
_client_lock = threading.Lock()

KB_NUM_RE = re.compile(r"^KB\d+$", re.IGNORECASE)

# Hard limit on how many documents to index. 0 = no limit.
# When set, a random sample of MAX_KB_DOCS is drawn from the full collection.
# Deletion step is automatically skipped when this limit is active to avoid
# purging documents that simply weren't included in the sample.
MAX_KB_DOCS = int(os.getenv("MAX_KB_DOCS", "250"))

# Matches "Scenario <N>" anywhere in the title (case-insensitive).
# Used AFTER .pdf extension has been stripped.
SCENARIO_TITLE_RE = re.compile(r"(?i)\bscenario\s*\d+")

# Extracts KB number from filenames like KB-0420_Title.pdf or KB1003_Title.pdf
KB_FROM_FILENAME_RE = re.compile(r"(?i)^(KB-?\d+)[_\s]")


def _normalize_title_for_scenario_detection(title: str) -> str:
    """
    Normalize unicode dashes/hyphens, whitespace (incl NBSP), and remove zero-width chars
    so suffix matching works reliably.

    This is important because KB titles can contain:
    - "Wi-Fi" (U+2011 non-breaking hyphen) or U+2010 hyphen
    - en dash / em dash separators
    - non-breaking spaces
    - zero-width characters
    """
    if not title:
        return ""

    t = str(title)

    # Remove zero-width chars that break regex matches
    # ZWSP, ZWNJ, ZWJ, BOM
    t = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", t)

    # Normalize all hyphen/dash variants to "-"
    dash_chars = [
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\u2012",  # figure dash
        "\u2013",  # en dash
        "\u2014",  # em dash
        "\u2212",  # minus sign
        "\u00AD",  # soft hyphen
        "\u2043",  # hyphen bullet
    ]
    for ch in dash_chars:
        t = t.replace(ch, "-")

    # Normalize NBSP and other uncommon spaces to regular space
    t = t.replace("\u00A0", " ")  # NBSP

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


def normalize_kb_number(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    s = s.replace(" ", "")
    if not s:
        return None
    # enforce KB prefix
    if not s.startswith("KB"):
        return None
    # keep only digits after KB
    digits = re.sub(r"\D+", "", s[2:])
    if not digits:
        return None
    return f"KB{digits}"


def extract_kb_number_from_name(name: str) -> Optional[str]:
    """
    Extract and normalize KB number from filenames like:
      KB-0420_Data Loss Prevention Alert.pdf  →  KB0420
      KB1003_Setting_Up_MFA.pdf               →  KB1003
      KB-0064_Headset Audio Not Working.pdf   →  KB0064

    Returns None if no KB number pattern is found.
    """
    if not name:
        return None
    # Strip .pdf before matching so KB-0420_Title.pdf still matches
    clean = re.sub(r"(?i)\.pdf\s*$", "", name.strip())
    m = KB_FROM_FILENAME_RE.match(clean)
    if not m:
        return None
    return normalize_kb_number(m.group(1))


def start_healthcheck_server(host: str = "0.0.0.0", port: int = 80) -> None:
    """
    Very small TCP server so k8s tcpSocket probe on port 80 succeeds.
    It just accepts and immediately closes connections.
    """

    def _server():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(5)
            logger.info("Healthcheck TCP server listening on %s:%d", host, port)
            while True:
                conn, _ = s.accept()
                conn.close()

    t = threading.Thread(target=_server, daemon=True)
    t.start()


def get_cached_pg_client():
    """Get or create a cached PgVector client for this module."""
    global _pg_client

    if _pg_client is None:
        with _client_lock:
            if _pg_client is None:
                db2_url = indexing_config.pgvector_database_url
                schema = indexing_config.pgvector_schema
                table_prefix = indexing_config.pgvector_table_prefix

                _pg_client = create_pgvector_client(
                    db2_url,
                    schema=schema,
                    table_prefix=table_prefix,
                )

    return _pg_client


@dataclass
class DocumentInfo:
    """Data class to hold document information"""
    number: str
    updated_at: datetime
    name: str = None
    content: str = None
    is_deleted: bool = False


@dataclass
class ComparisonResult:
    """Data class to hold comparison results"""
    new_documents: List[DocumentInfo]
    updated_documents: List[Tuple[DocumentInfo, DocumentInfo]]  # (source, vector)
    deleted_documents: List[str]

    def summary(self) -> str:
        return f"""
        Comparison Summary:
        -------------------
        New Documents:     {len(self.new_documents)}
        Updated Documents: {len(self.updated_documents)}
        Deleted Documents: {len(self.deleted_documents)}
        """


@dataclass
class SyncResult:
    new_inserted: int = 0
    new_failed: int = 0
    updated_success: int = 0
    updated_failed: int = 0
    deleted_success: int = 0
    deleted_failed: int = 0
    skipped: int = 0

    def total_processed(self) -> int:
        return (
            self.new_inserted + self.new_failed
            + self.updated_success + self.updated_failed
            + self.deleted_success + self.deleted_failed
            + self.skipped
        )

    def summary(self) -> str:
        return f"""
        Sync Summary:
        ─────────────────────────────────────────
        NEW Articles:
        ✓ Inserted:  {self.new_inserted}
        ✗ Failed:    {self.new_failed}

        UPDATED Articles:
        ✓ Updated:   {self.updated_success}
        ✗ Failed:    {self.updated_failed}

        DELETED Articles:
        ✓ Deleted:   {self.deleted_success}
        ✗ Failed:    {self.deleted_failed}

        Skipped:       {self.skipped}
        ─────────────────────────────────────────
        Total Processed: {self.total_processed()}
        """


class KBComparisonService:
    def __init__(self, *, kb_collection_id: str):
        logger.info("Initializing KB Comparison Service...")

        self.pg_client = get_cached_pg_client()
        self.kb_collection_id = kb_collection_id

        database_url = indexing_config.pgvector_database_url
        schema = indexing_config.pgvector_schema
        table_prefix = indexing_config.pgvector_table_prefix

        logger.info("Initializing KB Inserter...")
        self.inserter = KnowledgeBaseInserter(
            database_url=database_url,
            kb_collection_id=self.kb_collection_id,
            schema=schema,
            table_prefix=table_prefix,
            create_tables_if_not_exist=True,
            create_vector_indexes=True,
        )

        logger.info("KB Comparison Service initialized")

    async def fetch_knowledgebase_documents(self, collection_id: str, org_id: str) -> Dict[str, DocumentInfo]:
        """
        Fetch documents from knowledgebase_document_assist table with content.
        Only fetches non-deleted documents (is_deleted = False).

        Processing order:
        1. Strip .pdf extension from title before any detection.
        2. Normalize unicode chars in title.
        3. Skip any title containing "Scenario <N>" (case-insensitive).
        4. Resolve doc key: prefer row.number, then extract from filename, then SRC:{id}.
        """
        logger.info(f"Fetching knowledgebase documents for collection: {collection_id}, org: {org_id}")

        try:
            query = text("""
                SELECT id, number, updated_at, content, name, is_deleted
                FROM public.knowledgebase_document_assist
                WHERE knowledgebase_collection_id = :collection_id
                  AND org_id = :org_id
                  AND is_deleted = FALSE
            """)

            documents: Dict[str, DocumentInfo] = {}
            skipped_scenarios = 0

            with self.pg_client.get_session() as session:
                result = session.execute(query, {"collection_id": collection_id, "org_id": org_id})

                for row in result:
                    raw_title = row.name or ""

                    # ── Step 1: Strip .pdf extension before any title processing ──
                    clean_title = re.sub(r"(?i)\.pdf\s*$", "", raw_title).strip()

                    # ── Step 2: Normalize unicode chars ──
                    norm_title = _normalize_title_for_scenario_detection(clean_title)

                    # ── Step 3: Skip scenario articles ──
                    if SCENARIO_TITLE_RE.search(norm_title):
                        skipped_scenarios += 1
                        logger.info(
                            "Skipping scenario article: raw=%r | norm=%r",
                            raw_title,
                            norm_title,
                        )
                        continue

                    # ── Step 4: Resolve doc key ──
                    # Priority: row.number → extract from filename → SRC:{id}
                    kb_number = (
                        normalize_kb_number(row.number)
                        or extract_kb_number_from_name(raw_title)
                    )
                    doc_key = kb_number or f"SRC:{row.id}"

                    if not doc_key:
                        logger.warning("Skipping row with missing (number,id). name=%r", raw_title)
                        continue

                    updated_at = row.updated_at
                    if updated_at is None:
                        updated_at = datetime.fromtimestamp(0, tz=timezone.utc)
                    else:
                        updated_at = updated_at.replace(tzinfo=timezone.utc) if updated_at.tzinfo is None else updated_at

                    doc = DocumentInfo(
                        number=doc_key,
                        name=norm_title,
                        updated_at=updated_at,
                        content=row.content,
                        is_deleted=bool(row.is_deleted),
                    )

                    existing = documents.get(doc_key)
                    if existing is None or doc.updated_at >= existing.updated_at:
                        if existing is not None:
                            logger.warning(
                                "Duplicate source doc key %s detected; keeping newer row (old=%s new=%s)",
                                doc_key,
                                existing.updated_at,
                                doc.updated_at,
                            )
                        documents[doc_key] = doc

            logger.info(
                "Found %d non-deleted documents in knowledgebase_document_assist "
                "AFTER scenario-skip (skipped %d scenarios)",
                len(documents),
                skipped_scenarios,
            )
            return documents

        except Exception as e:
            logger.error(f"Error fetching knowledgebase documents: {e}")
            raise

    async def fetch_vector_summaries(self) -> Dict[str, DocumentInfo]:
        """
        Fetch documents from vva_kb_summaries table.
        Keys by doc_id (string).
        """
        try:
            table_name = self.pg_client.get_table_name("kb_summaries")
            query = text(
                f"""
                SELECT doc_id, updated_at
                FROM {table_name}
                WHERE kb_collection_id = :collection_id
                """
            )

            summaries: Dict[str, DocumentInfo] = {}

            with self.pg_client.get_session() as session:
                result = session.execute(query, {"collection_id": self.kb_collection_id})
                for row in result:
                    doc_key = str(row.doc_id).strip()
                    if not doc_key:
                        continue

                    updated_at = row.updated_at
                    if updated_at is None:
                        updated_at = datetime.fromtimestamp(0, tz=timezone.utc)
                    else:
                        updated_at = updated_at.replace(tzinfo=timezone.utc) if updated_at.tzinfo is None else updated_at

                    summaries[doc_key] = DocumentInfo(number=doc_key, updated_at=updated_at)

            logger.info(
                "Found %d summaries in kb_summaries for collection=%s",
                len(summaries),
                self.kb_collection_id,
            )
            return summaries

        except Exception as e:
            logger.error(f"Error fetching vector summaries: {e}")
            raise

    def compare_documents(self, source_docs: Dict[str, DocumentInfo], vector_docs: Dict[str, DocumentInfo]) -> ComparisonResult:
        logger.info("Comparing documents...")

        source_ids: Set[str] = set(source_docs.keys())
        vector_ids: Set[str] = set(vector_docs.keys())

        new_doc_ids = source_ids - vector_ids
        new_documents = [source_docs[doc_id] for doc_id in new_doc_ids]

        deleted_doc_ids = vector_ids - source_ids
        deleted_documents = list(deleted_doc_ids)

        common_ids = source_ids & vector_ids
        updated_documents: List[Tuple[DocumentInfo, DocumentInfo]] = []

        for doc_id in common_ids:
            source_doc = source_docs[doc_id]
            vector_doc = vector_docs[doc_id]
            if source_doc.updated_at > vector_doc.updated_at:
                updated_documents.append((source_doc, vector_doc))

        result = ComparisonResult(
            new_documents=new_documents,
            updated_documents=updated_documents,
            deleted_documents=deleted_documents,
        )

        logger.info(result.summary())
        return result

    def _extract_counts_from_batch_stats(self, stats: Dict[str, int]) -> Tuple[int, int, int, int]:
        """
        Normalizes inserter stats across old/new formats.
        Returns: (inserted, updated, skipped, failed)
        """
        inserted = int(stats.get("inserted", 0))
        updated = int(stats.get("updated", 0))
        skipped = int(stats.get("skipped", 0))
        failed = int(stats.get("failed", 0))

        if "successful" in stats and (inserted + updated + skipped) == 0:
            inserted = int(stats.get("successful", 0))

        return inserted, updated, skipped, failed

    def _build_article_payloads(self, docs: List[DocumentInfo]) -> Tuple[List[Dict[str, str]], int]:
        articles: List[Dict[str, str]] = []
        skipped = 0
        for doc in tqdm(
            docs,
            desc="Preparing articles",
            unit="article",
            leave=False,
            dynamic_ncols=True,
            disable=not docs,
        ):
            if not (doc.number and str(doc.number).strip()):
                logger.warning("Skipping doc with missing kbid/number. name=%r", doc.name)
                skipped += 1
                continue

            content = clean_kb_content(doc.content or "", max_chars=100000)
            if not content.strip():
                logger.warning("Skipping doc %s because content is empty. name=%r", doc.number, doc.name)
                skipped += 1
                continue

            articles.append(
                {
                    "kbid": str(doc.number).strip(),
                    "title": doc.name or "",
                    "content": content,
                }
            )
        articles.sort(key=lambda item: item["kbid"])
        return articles, skipped

    def _log_section(self, title: str, count: Optional[int] = None) -> None:
        suffix = f" ({count})" if count is not None else ""
        logger.info("")
        logger.info("=" * 72)
        logger.info("%s%s", title, suffix)
        logger.info("=" * 72)

    def _log_doc_preview(self, label: str, lines: List[str]) -> None:
        if not lines:
            return
        logger.info("%s:", label)
        for line in lines:
            logger.info("  %s", line)

    def sync_new_articles(self, new_docs: List[DocumentInfo]) -> Tuple[int, int, int]:
        if not new_docs:
            logger.info("No new articles to sync")
            return 0, 0, 0

        self._log_section("SYNCING NEW ARTICLES", len(new_docs))

        articles_to_insert, skipped_from_prep = self._build_article_payloads(new_docs)

        stats = self.inserter.insert_batch_articles(
            articles=articles_to_insert,
            update_if_exists=False,
            max_workers=6,
            progress_desc="Indexing new articles",
        )

        inserted, updated, skipped, failed = self._extract_counts_from_batch_stats(stats)
        inserted_total = inserted + updated
        logger.info(
            "New article sync finished | inserted=%d skipped=%d failed=%d duration=%.3fs",
            inserted_total,
            skipped + skipped_from_prep,
            failed,
            stats.get("duration_s") or 0.0,
        )
        return inserted_total, skipped + skipped_from_prep, failed

    def sync_updated_articles(self, updated_docs: List[Tuple[DocumentInfo, DocumentInfo]]) -> Tuple[int, int, int]:
        if not updated_docs:
            logger.info("No updated articles to sync")
            return 0, 0, 0

        self._log_section("SYNCING UPDATED ARTICLES", len(updated_docs))

        articles_to_update, skipped_from_prep = self._build_article_payloads(
            [source_doc for source_doc, _ in updated_docs]
        )

        stats = self.inserter.insert_batch_articles(
            articles=articles_to_update,
            update_if_exists=True,
            max_workers=6,
            progress_desc="Reindexing updated articles",
        )

        inserted, updated, skipped, failed = self._extract_counts_from_batch_stats(stats)
        updated_total = updated + inserted
        logger.info(
            "Updated article sync finished | updated=%d skipped=%d failed=%d duration=%.3fs",
            updated_total,
            skipped + skipped_from_prep,
            failed,
            stats.get("duration_s") or 0.0,
        )
        return updated_total, skipped + skipped_from_prep, failed

    def sync_deleted_articles(self, deleted_doc_ids: List[str]) -> Tuple[int, int]:
        if not deleted_doc_ids:
            logger.info("No deleted articles to sync")
            return 0, 0

        self._log_section("SYNCING DELETED ARTICLES", len(deleted_doc_ids))

        stats = self.inserter.delete_batch_articles(
            deleted_doc_ids,
            progress_desc="Deleting removed articles",
        )

        success = int(stats.get("successful", 0))
        failed = int(stats.get("failed", 0))
        logger.info("Deleted article sync finished | deleted=%d failed=%d", success, failed)
        return success, failed

    async def run_comparison_and_sync(self, collection_id: str, org_id: str) -> SyncResult:
        self._log_section("STARTING KB COMPARISON AND SYNC")
        logger.info("Collection: %s", collection_id)
        logger.info("Organization: %s", org_id)
        if MAX_KB_DOCS > 0:
            logger.info("MAX_KB_DOCS: %d (random sampling enabled)", MAX_KB_DOCS)

        sync_result = SyncResult()

        try:
            logger.info("Step 1: Fetching documents from source (is_deleted=False only)...")
            source_docs = await self.fetch_knowledgebase_documents(collection_id, org_id)

            if MAX_KB_DOCS > 0 and len(source_docs) > MAX_KB_DOCS:
                original_count = len(source_docs)
                sampled_keys = random.sample(list(source_docs.keys()), MAX_KB_DOCS)
                source_docs = {k: source_docs[k] for k in sampled_keys}
                logger.info(
                    "MAX_KB_DOCS=%d applied: randomly sampled %d docs from %d total (dropped %d)",
                    MAX_KB_DOCS, len(source_docs), original_count, original_count - MAX_KB_DOCS,
                )

            logger.info("Step 2: Fetching documents from vector database...")
            vector_docs = await self.fetch_vector_summaries()

            logger.info("Step 3: Comparing documents...")
            comparison = self.compare_documents(source_docs, vector_docs)

            self.print_detailed_results(comparison)

            total_vector = len(vector_docs)
            total_deleted = len(comparison.deleted_documents)
            total_new = len(comparison.new_documents)
            total_updated = len(comparison.updated_documents)

            allow_full_purge = os.getenv("ALLOW_FULL_KB_PURGE", "false").lower() in ("1", "true", "yes")

            if (
                total_vector > 0
                and total_deleted == total_vector
                and total_new == 0
                and total_updated == 0
                and not allow_full_purge
            ):
                logger.warning("⚠️ Safety check triggered: this sync would delete ALL documents in the vector DB and write nothing.")
                logger.warning("Skipping deletion step. Set ALLOW_FULL_KB_PURGE=true to allow full purge intentionally.")
                comparison.deleted_documents = []

            if MAX_KB_DOCS > 0 and comparison.deleted_documents:
                logger.info(
                    "MAX_KB_DOCS is set — skipping deletion of %d doc(s) to avoid "
                    "removing documents outside the current sample.",
                    len(comparison.deleted_documents),
                )
                comparison.deleted_documents = []

            logger.info("\nStep 4: Syncing new articles...")
            new_inserted, new_skipped, new_failed = self.sync_new_articles(comparison.new_documents)
            sync_result.new_inserted = new_inserted
            sync_result.new_failed = new_failed
            sync_result.skipped += new_skipped

            logger.info("\nStep 5: Syncing updated articles...")
            updated_success, updated_skipped, updated_failed = self.sync_updated_articles(comparison.updated_documents)
            sync_result.updated_success = updated_success
            sync_result.updated_failed = updated_failed
            sync_result.skipped += updated_skipped

            logger.info("\nStep 6: Syncing deleted articles...")
            deleted_success, deleted_failed = self.sync_deleted_articles(comparison.deleted_documents)
            sync_result.deleted_success = deleted_success
            sync_result.deleted_failed = deleted_failed

            self._log_section("SYNC COMPLETE")
            logger.info(sync_result.summary())

            return sync_result

        except Exception as e:
            logger.error(f"Error during comparison and sync: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def print_detailed_results(self, result: ComparisonResult):
        self._log_section("DETAILED COMPARISON RESULTS")
        logger.info(result.summary())

        if result.new_documents:
            preview = [
                f"{doc.number} | {doc.name} | updated={doc.updated_at.isoformat()}"
                for doc in result.new_documents[:5]
            ]
            if len(result.new_documents) > 5:
                preview.append(f"... and {len(result.new_documents) - 5} more")
            self._log_doc_preview("New documents", preview)

        if result.updated_documents:
            preview = [
                (
                    f"{source_doc.number} | src={source_doc.updated_at.isoformat()} | "
                    f"vec={vector_doc.updated_at.isoformat()} | "
                    f"delta={source_doc.updated_at - vector_doc.updated_at}"
                )
                for source_doc, vector_doc in result.updated_documents[:5]
            ]
            if len(result.updated_documents) > 5:
                preview.append(f"... and {len(result.updated_documents) - 5} more")
            self._log_doc_preview("Updated documents", preview)

        if result.deleted_documents:
            preview = [f"{doc_number}" for doc_number in result.deleted_documents[:10]]
            if len(result.deleted_documents) > 10:
                preview.append(f"... and {len(result.deleted_documents) - 10} more")
            self._log_doc_preview("Deleted documents", preview)

    def cleanup(self):
        logger.info("Cleaning up database connections...")
        self.pg_client.close()
        self.inserter.close()
        logger.info("✓ Cleanup complete")


async def main():
    COLLECTION_ID = indexing_config.collection_id
    ORG_ID = indexing_config.org_id

    service = None
    try:
        service = KBComparisonService(kb_collection_id=COLLECTION_ID)

        result = await service.run_comparison_and_sync(COLLECTION_ID, ORG_ID)

        try:
            logger.info("Rebuilding per-shard SW/HW knowledge base tables...")
            shard_results = extract_sw_hw_all_shards_parallel(
                collection_id=COLLECTION_ID,
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
            total_items = sum(len(r.get("knowledge_base", [])) for r in shard_results.values())
            logger.info(
                "SW/HW KB refresh complete: %d shard(s), %d total unique items",
                len(shard_results), total_items,
            )
            for sid, r in sorted(shard_results.items()):
                logger.info("  shard=%d -> %d items", sid, len(r.get("knowledge_base", [])))

            if _reload_enrichment_kb_cache is not None:
                _reload_enrichment_kb_cache()
                logger.info("Query-enrichment SW/HW KB cache invalidated.")

        except Exception as e:
            logger.error("Failed to rebuild software/hardware configuration KB: %s", e)

        try:
            logger.info("Generating test queries...")
            if generate_test_queries_pgvec is None or run_test_queries_pgvec is None:
                logger.info("Synthetic query test modules are not installed; skipping post-sync tests.")
            else:
                await generate_test_queries_pgvec()
                logger.info("Running test on synthetic queries...")
                await run_test_queries_pgvec()
                logger.info("Test complete.")
        except Exception as e:
            logger.error("Failed to run test on synthetic queries: %s", e)

        return result

    finally:
        if service:
            service.cleanup()


async def main_with_idle():
    try:
        await main()
    except Exception:
        logger.exception("KB sync failed; keeping container alive for debugging.")

    logger.info("KB sync finished; going into idle loop to satisfy k8s health checks.")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    start_healthcheck_server()
    asyncio.run(main_with_idle())
