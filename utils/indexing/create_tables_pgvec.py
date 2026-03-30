"""
Insert New Knowledge Base Articles into Supabase PgVector
Schema-aligned with LanceDB tables:
- {prefix}kb_keywords (keyword-per-row, multiple embedding columns)
- {prefix}kb_summaries
- {prefix}kb_shard_registry  ← NEW: tracks shard capacity per collection

Sharding strategy:
- Each collection is split into shards of at most KB_SHARD_MAX_SIZE documents (default 200).
- A new shard is automatically created once the current one fills up.
- Shards are tracked in kb_shard_registry with doc_count / max_shard_size.
- All inserts/deletes keep shard counts accurate via SELECT FOR UPDATE transactions.
- Retrieval callers can use get_all_shard_ids() to fan-out queries across shards.

Other features:
- Create schema if not exist
- Create tables if not exist (schema-aligned)
- Connection pooling (shared)
- Fast exists checks
- Traceability includes failure/skip reason
"""

import os
import sys
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Set, Tuple

from dotenv import load_dotenv

from indexing_config import indexing_config
from kb_utils.kb_article_processor import KnowledgeBaseArticleProcessor
from kb_utils.kb_db import EMBEDDING_DIM, PgVectorPool, TraceabilityTracker, quote_ident
from kb_utils.kb_storage import KnowledgeBaseStorage
from kb_utils.shard_repository import ShardRepository

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("kb_pgvector")

# Maximum documents per shard — env-driven, defaults to 200.
KB_SHARD_MAX_SIZE = indexing_config.kb_shard_max_size


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeBaseInserter  (shard-aware)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBaseInserter:
    """
    Inserts / updates / deletes KB articles in a sharded PgVector store.

    Sharding model
    ──────────────
    Each (collection_id, shard_id) pair is an independent logical "vector store".
    shard_id is a zero-based integer that increments automatically when a shard
    reaches max_shard_size documents.  The kb_shard_registry table is the single
    source of truth for shard membership; its doc_count is kept in sync with
    actual rows via SELECT FOR UPDATE transactions.

    Retrieval side
    ──────────────
    Call get_all_shard_ids() to obtain the list of active shard IDs, then issue
    one ANN query per shard and merge / re-rank the results in your retrieval
    layer.  This keeps each shard dense at ≤ max_shard_size docs, so IVFFlat
    probes stay efficient as the collection grows.
    """

    def __init__(
        self,
        database_url: str,
        kb_collection_id: str,
        schema: str = "public",
        table_prefix: str = "vva_",
        create_tables_if_not_exist: bool = True,
        pool_minconn: int = 1,
        pool_maxconn: int = 10,
        create_vector_indexes: bool = True,
        max_shard_size: Optional[int] = None,
    ):
        self.database_url = database_url
        self.kb_collection_id = kb_collection_id
        self.schema = schema or "public"
        self.table_prefix = table_prefix or ""
        self.create_vector_indexes = bool(create_vector_indexes)

        # Allow per-instance override; fall back to env/module constant.
        self.max_shard_size = int(max_shard_size or KB_SHARD_MAX_SIZE)

        # Table names
        self.keywords_table_name      = f"{self.table_prefix}kb_keywords"
        self.summaries_table_name     = f"{self.table_prefix}kb_summaries"
        self.shard_registry_table_name = f"{self.table_prefix}kb_shard_registry"

        # Quoted, schema-qualified identifiers
        self.schema_q                  = quote_ident(self.schema)
        self.keywords_table_q          = f"{self.schema_q}.{quote_ident(self.keywords_table_name)}"
        self.summaries_table_q         = f"{self.schema_q}.{quote_ident(self.summaries_table_name)}"
        self.shard_registry_table_q    = f"{self.schema_q}.{quote_ident(self.shard_registry_table_name)}"

        self.traceability = TraceabilityTracker()

        logger.info("Creating PostgreSQL pool...")
        self.pool = PgVectorPool(database_url, minconn=pool_minconn, maxconn=pool_maxconn)
        logger.info("✓ PostgreSQL pool ready")

        logger.info(
            f"Shard config: max_shard_size={self.max_shard_size} docs per shard "
            f"(KB_SHARD_MAX_SIZE env={KB_SHARD_MAX_SIZE})"
        )

        self.shards = ShardRepository(
            pool=self.pool,
            kb_collection_id=self.kb_collection_id,
            shard_registry_table_q=self.shard_registry_table_q,
            summaries_table_q=self.summaries_table_q,
            max_shard_size=self.max_shard_size,
        )
        self.storage = KnowledgeBaseStorage(
            pool=self.pool,
            kb_collection_id=self.kb_collection_id,
            keywords_table_q=self.keywords_table_q,
            summaries_table_q=self.summaries_table_q,
        )
        self.article_processor = KnowledgeBaseArticleProcessor(
            kb_collection_id=self.kb_collection_id,
        )

        if create_tables_if_not_exist:
            self.create_tables_if_not_exist()

    def _with_conn(self):
        return self.pool.getconn()

    # ─────────────────────────────────────────────────────────────────────────
    # Schema / table bootstrap
    # ─────────────────────────────────────────────────────────────────────────

    def create_tables_if_not_exist(self):
        conn = self._with_conn()
        cur = conn.cursor()
        try:
            logger.info(f"Ensuring schema exists: {self.schema!r} ...")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_q};")

            logger.info("Ensuring pgvector extension is enabled...")
            ext_schema = None
            try:
                cur.execute('CREATE SCHEMA IF NOT EXISTS "extensions";')
            except Exception:
                pass
            try:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA "extensions";')
            except Exception:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                """
                SELECT n.nspname
                FROM pg_extension e
                JOIN pg_namespace n ON n.oid = e.extnamespace
                WHERE e.extname = 'vector'
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if row and row[0]:
                ext_schema = str(row[0])

            if ext_schema:
                ext_schema_q = '"' + ext_schema.replace('"', '""') + '"'
                cur.execute(f"SET search_path TO {self.schema_q}, public, {ext_schema_q};")
            else:
                cur.execute(f"SET search_path TO {self.schema_q}, public;")

            # ── Shard registry ────────────────────────────────────────────────
            logger.info(f"Creating {self.shard_registry_table_q} if not exists...")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.shard_registry_table_q} (
                    id             BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    kb_collection_id TEXT NOT NULL,
                    shard_id       INT  NOT NULL,
                    doc_count      INT  NOT NULL DEFAULT 0,
                    max_shard_size INT  NOT NULL DEFAULT {self.max_shard_size},
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (kb_collection_id, shard_id)
                );
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {quote_ident(f"idx_{self.shard_registry_table_name}_collection")}
                ON {self.shard_registry_table_q} (kb_collection_id);
                """
            )

            # ── kb_keywords ───────────────────────────────────────────────────
            logger.info(f"Creating {self.keywords_table_q} if not exists...")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.keywords_table_q} (
                    id                          BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    kbid                        TEXT NOT NULL,
                    kb_collection_id            TEXT NOT NULL,
                    shard_id                    INT  NOT NULL DEFAULT 0,
                    category                    TEXT,
                    article_type                TEXT,
                    primary_topic               TEXT,
                    specific_keyword            TEXT,
                    generic_keyword             TEXT,
                    usecase                     TEXT,
                    secondary_entity            TEXT,
                    acronym                     TEXT,
                    specific_keyword_embedding  vector({EMBEDDING_DIM}),
                    generic_keyword_embedding   vector({EMBEDDING_DIM}),
                    usecase_embedding           vector({EMBEDDING_DIM}),
                    secondary_entity_embedding  vector({EMBEDDING_DIM}),
                    acronym_embedding           vector({EMBEDDING_DIM}),
                    created_at                  TIMESTAMPTZ,
                    updated_at                  TIMESTAMPTZ
                );
                """
            )

            logger.info("Ensuring kb_keywords columns exist (migrations)...")
            for ddl in [
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS shard_id INT NOT NULL DEFAULT 0;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS category TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS article_type TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS primary_topic TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS specific_keyword TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS generic_keyword TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS usecase TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS secondary_entity TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS acronym TEXT;",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS specific_keyword_embedding vector({EMBEDDING_DIM});",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS generic_keyword_embedding vector({EMBEDDING_DIM});",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS usecase_embedding vector({EMBEDDING_DIM});",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS secondary_entity_embedding vector({EMBEDDING_DIM});",
                f"ALTER TABLE {self.keywords_table_q} ADD COLUMN IF NOT EXISTS acronym_embedding vector({EMBEDDING_DIM});",
            ]:
                cur.execute(ddl)

            # Composite index: collection + shard for fast per-shard ANN queries
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {quote_ident(f"idx_{self.keywords_table_name}_kbid_collection")}
                ON {self.keywords_table_q} (kbid, kb_collection_id);
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {quote_ident(f"idx_{self.keywords_table_name}_collection_shard")}
                ON {self.keywords_table_q} (kb_collection_id, shard_id);
                """
            )

            if self.create_vector_indexes:
                logger.info("Creating per-shard-friendly ivfflat vector indexes...")

                def create_vec_index(index_name: str, col: str):
                    # NOTE: IVFFlat indexes cover all shards in the table.
                    # For true per-shard isolation consider partitioning by shard_id,
                    # or use HNSW (no lists= parameter, works well at any size).
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {quote_ident(index_name)}
                        ON {self.keywords_table_q}
                        USING ivfflat ({col} vector_cosine_ops)
                        WITH (lists = 100)
                        WHERE {col} IS NOT NULL;
                        """
                    )

                create_vec_index(f"idx_{self.keywords_table_name}_spec_vec",      "specific_keyword_embedding")
                create_vec_index(f"idx_{self.keywords_table_name}_gen_vec",       "generic_keyword_embedding")
                create_vec_index(f"idx_{self.keywords_table_name}_usecase_vec",   "usecase_embedding")
                create_vec_index(f"idx_{self.keywords_table_name}_secondary_vec", "secondary_entity_embedding")
                create_vec_index(f"idx_{self.keywords_table_name}_acronym_vec",   "acronym_embedding")

            # ── kb_summaries ──────────────────────────────────────────────────
            logger.info(f"Creating {self.summaries_table_q} if not exists...")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.summaries_table_q} (
                    id                  BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    doc_id              TEXT NOT NULL,
                    kb_collection_id    TEXT NOT NULL,
                    shard_id            INT  NOT NULL DEFAULT 0,
                    article_title       TEXT,
                    primary_intent      TEXT,
                    query_triggers      TEXT,
                    secondary_mentions  TEXT,
                    summary             TEXT,
                    content             TEXT,
                    created_at          TIMESTAMPTZ,
                    updated_at          TIMESTAMPTZ,
                    UNIQUE (doc_id, kb_collection_id)
                );
                """
            )

            logger.info("Ensuring kb_summaries columns exist (migrations)...")
            for ddl in [
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS shard_id INT NOT NULL DEFAULT 0;",
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS article_title TEXT;",
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS primary_intent TEXT;",
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS query_triggers TEXT;",
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS secondary_mentions TEXT;",
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS summary TEXT;",
                f"ALTER TABLE {self.summaries_table_q} ADD COLUMN IF NOT EXISTS content TEXT;",
            ]:
                cur.execute(ddl)

            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {quote_ident(f"idx_{self.summaries_table_name}_doc_collection")}
                ON {self.summaries_table_q} (doc_id, kb_collection_id);
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {quote_ident(f"idx_{self.summaries_table_name}_collection_shard")}
                ON {self.summaries_table_q} (kb_collection_id, shard_id);
                """
            )

            conn.commit()
            logger.info("✓ Schema + tables + shard registry created/verified successfully")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tables: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            cur.close()
            self.pool.putconn(conn)

    # ─────────────────────────────────────────────────────────────────────────
    # Shard management
    # ─────────────────────────────────────────────────────────────────────────

    def _assign_shard_for_new_doc(self) -> int:
        return self.shards.assign_shard_for_new_doc()

    def _release_shard_slot(self, shard_id: int) -> None:
        self.shards.release_shard_slot(shard_id)

    def _get_shard_for_kbid(self, kbid: str) -> Optional[int]:
        return self.shards.get_shard_for_kbid(kbid)

    def get_all_shard_ids(self) -> List[int]:
        return self.shards.get_all_shard_ids()

    def get_shard_info(self) -> List[Dict[str, Any]]:
        return self.shards.get_shard_info()

    # ─────────────────────────────────────────────────────────────────────────
    # Existence checks
    # ─────────────────────────────────────────────────────────────────────────

    def check_article_exists(self, kbid: str) -> bool:
        return self.storage.check_article_exists(kbid)

    def _fetch_existing_kbids(self, kbids: List[str]) -> Dict[str, int]:
        return self.storage.fetch_existing_kbids(kbids)

    # ─────────────────────────────────────────────────────────────────────────
    # Delete helpers
    # ─────────────────────────────────────────────────────────────────────────

    def delete_existing_article_keywords(self, kbid: str) -> int:
        return self.storage.delete_existing_article_keywords(kbid)

    def delete_existing_article_summary(self, kbid: str) -> int:
        return self.storage.delete_existing_article_summary(kbid)

    # ─────────────────────────────────────────────────────────────────────────
    # LLM processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_single_article(
        self, kbid: str, kb_title: str, kb_content: str
    ) -> Dict[str, Any]:
        return self.article_processor.process_single_article(kbid, kb_title, kb_content)

    # ─────────────────────────────────────────────────────────────────────────
    # Row-level inserts (now shard_id-aware)
    # ─────────────────────────────────────────────────────────────────────────

    def insert_keyword_rows(self, rows: List[Dict[str, Any]], shard_id: int) -> bool:
        return self.storage.insert_keyword_rows(rows, shard_id)

    def insert_summary_record(self, data: Dict[str, Any], shard_id: int) -> bool:
        return self.storage.insert_summary_record(data, shard_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Core insert logic
    # ─────────────────────────────────────────────────────────────────────────

    def _insert_single_article_core(
        self,
        *,
        kbid: str,
        title: str,
        content: str,
        exists: bool,
        update_if_exists: bool,
        existing_shard_id: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Full insert / update pipeline for a single article.

        Parameters
        ----------
        kbid              : Unique article identifier.
        title             : Article title.
        content           : Full article text.
        exists            : Whether the article is already in the vector store.
        update_if_exists  : Re-process and re-index if the article already exists.
        existing_shard_id : The shard the article currently lives in (if exists=True).
                            Pass None to let the method look it up (one extra DB query).
        """
        kbid    = str(kbid).strip()
        title   = title or ""
        content = content or ""

        if not kbid:
            self.traceability.add_record(kbid="(missing)", status="failed", reason="Missing kbid")
            return False, "failed"

        if not content.strip():
            self.traceability.add_record(kbid=kbid, status="failed", reason="Missing content")
            return False, "failed"

        if exists and not update_if_exists:
            self.traceability.add_record(
                kbid=kbid, status="skipped",
                reason="Exists and update_if_exists=False (batch)",
            )
            return True, "skipped"

        # ── Delete existing rows (update path) ───────────────────────────────
        if exists and update_if_exists:
            # Resolve shard before deleting so we can release the slot.
            if existing_shard_id is None:
                existing_shard_id = self._get_shard_for_kbid(kbid)

            self.delete_existing_article_keywords(kbid)
            self.delete_existing_article_summary(kbid)

            # Release the old slot so the shard can accept another document.
            if existing_shard_id is not None:
                self._release_shard_slot(existing_shard_id)

        # ── Assign a shard for this (re-)insert ──────────────────────────────
        try:
            shard_id = self._assign_shard_for_new_doc()
        except Exception as e:
            self.traceability.add_record(
                kbid=kbid, status="failed",
                reason=f"Shard assignment failed: {e}",
            )
            return False, "failed"

        # ── LLM processing (outside any DB transaction) ───────────────────────
        result = self.process_single_article(kbid, title, content)
        if not result.get("success"):
            # Roll back the shard slot we just reserved.
            self._release_shard_slot(shard_id)
            self.traceability.add_record(
                kbid=kbid,
                status="failed",
                reason=f"Processing failed at stage '{result.get('stage')}': {result.get('error')}",
                details={"stage": result.get("stage"), "error": result.get("error")},
            )
            return False, "failed"

        # ── DB inserts ────────────────────────────────────────────────────────
        if not self.insert_keyword_rows(result["keyword_rows"], shard_id):
            self._release_shard_slot(shard_id)
            self.traceability.add_record(
                kbid=kbid, status="failed",
                reason="Failed inserting kb_keywords",
                details=result.get("metadata"),
            )
            return False, "failed"

        if not self.insert_summary_record(result["summary_data"], shard_id):
            # Keywords were inserted — best-effort cleanup.
            self.delete_existing_article_keywords(kbid)
            self._release_shard_slot(shard_id)
            self.traceability.add_record(
                kbid=kbid, status="failed",
                reason="Failed inserting kb_summaries",
                details=result.get("metadata"),
            )
            return False, "failed"

        details = result.get("metadata", {}) or {}
        op = "updated" if exists else "inserted"
        details["operation"] = op
        details["shard_id"] = shard_id
        self.traceability.add_record(kbid=kbid, status="success", details=details)
        return True, op

    # ─────────────────────────────────────────────────────────────────────────
    # Batch insert
    # ─────────────────────────────────────────────────────────────────────────

    def insert_batch_articles(
        self,
        *,
        articles: List[Dict[str, Any]],
        update_if_exists: bool = True,
        max_workers: Optional[int] = None,
        stop_on_error: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()
        articles = articles or []

        normalized = []
        kbids: List[str] = []
        for a in articles:
            if not a:
                continue
            kbid    = str(a.get("kbid") or "").strip()
            title   = a.get("title")   or ""
            content = a.get("content") or ""
            if not kbid:
                continue
            normalized.append({"kbid": kbid, "title": title, "content": content})
            kbids.append(kbid)

        # {kbid: shard_id} for every already-indexed doc in this batch.
        existing: Dict[str, int] = self._fetch_existing_kbids(kbids)

        stats: Dict[str, Any] = {
            "total":    len(normalized),
            "inserted": 0,
            "updated":  0,
            "skipped":  0,
            "failed":   0,
            "duration_s": None,
            "errors":   [],
        }

        def _run_one(a: Dict[str, Any]) -> Tuple[str, bool, str, Optional[str]]:
            kbid = a["kbid"]
            try:
                ok, op = self._insert_single_article_core(
                    kbid=kbid,
                    title=a.get("title", ""),
                    content=a.get("content", ""),
                    exists=(kbid in existing),
                    update_if_exists=update_if_exists,
                    existing_shard_id=existing.get(kbid),
                )
                return kbid, ok, op, None
            except Exception as e:
                self.traceability.add_record(
                    kbid=kbid, status="failed",
                    reason=f"Unexpected error (batch): {e}",
                )
                return kbid, False, "failed", str(e)

        if not max_workers or max_workers <= 1:
            for a in normalized:
                kbid, ok, op, err = _run_one(a)
                if op in ("inserted", "updated", "skipped", "failed"):
                    stats[op] += 1
                else:
                    stats["failed"] += 1
                if not ok and err:
                    stats["errors"].append({"kbid": kbid, "error": err})
                    if stop_on_error:
                        stats["duration_s"] = round(time.time() - t0, 3)
                        raise RuntimeError(f"Batch insert failed for kbid={kbid}: {err}")
        else:
            max_workers = max(1, min(int(max_workers), 64))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_run_one, a) for a in normalized]
                for fut in as_completed(futs):
                    kbid, ok, op, err = fut.result()
                    if op in ("inserted", "updated", "skipped", "failed"):
                        stats[op] += 1
                    else:
                        stats["failed"] += 1
                    if not ok and err:
                        stats["errors"].append({"kbid": kbid, "error": err})
                        if stop_on_error:
                            stats["duration_s"] = round(time.time() - t0, 3)
                            raise RuntimeError(f"Batch insert failed for kbid={kbid}: {err}")

        stats["duration_s"] = round(time.time() - t0, 3)

        # Backward-compatible aliases
        stats["successful"] = (
            int(stats.get("inserted", 0))
            + int(stats.get("updated", 0))
            + int(stats.get("skipped", 0))
        )
        stats["success"]  = stats["successful"]
        stats["failures"] = int(stats.get("failed", 0))

        # Log shard utilisation after every batch so operators can monitor growth.
        try:
            for sinfo in self.get_shard_info():
                logger.info(
                    f"Shard {sinfo['shard_id']}: "
                    f"{sinfo['doc_count']}/{sinfo['max_shard_size']} docs "
                    f"({sinfo['utilization_pct']}% full)"
                )
        except Exception:
            pass

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Batch delete
    # ─────────────────────────────────────────────────────────────────────────

    def delete_batch_articles(self, kbids: List[str]) -> Dict[str, Any]:
        """
        Delete articles from the vector store and release their shard slots.
        Returns {"successful": int, "failed": int}.
        """
        successful = 0
        failed = 0
        kbids = [str(x).strip() for x in (kbids or []) if str(x).strip()]
        if not kbids:
            return {"successful": 0, "failed": 0}

        for kbid in kbids:
            try:
                # Find shard before deleting — we need it to decrement the counter.
                shard_id = self._get_shard_for_kbid(kbid)

                self.delete_existing_article_keywords(kbid)
                self.delete_existing_article_summary(kbid)

                if shard_id is not None:
                    self._release_shard_slot(shard_id)

                self.traceability.add_record(
                    kbid=kbid, status="deleted",
                    details={"shard_id": shard_id},
                )
                successful += 1
            except Exception as e:
                self.traceability.add_record(
                    kbid=kbid, status="failed",
                    reason=f"Delete failed: {e}",
                )
                failed += 1

        return {"successful": successful, "failed": failed}

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def close(self):
        try:
            self.pool.closeall()
        except Exception:
            pass
        logger.info("PostgreSQL pool closed")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point (unchanged behaviour)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    database_url = indexing_config.pgvector_database_url
    schema = indexing_config.pgvector_schema
    table_prefix = indexing_config.pgvector_table_prefix or "vva_"

    kb_collection_id = indexing_config.collection_id or os.getenv("KB_COLLECTION_ID")
    if not kb_collection_id:
        kb_collection_id = str(__import__("uuid").uuid4())
        logger.info(f"Generated new collection ID: {kb_collection_id}")
    else:
        logger.info(f"Using collection ID: {kb_collection_id}")

    if not database_url:
        logger.error("PGVECTOR_DATABASE_URL not set in environment")
        sys.exit(1)

    INGEST_WORKERS = indexing_config.ingest_workers
    INGEST_WORKERS = max(1, min(INGEST_WORKERS, 64))

    inserter = KnowledgeBaseInserter(
        database_url=database_url,
        kb_collection_id=kb_collection_id,
        schema=schema,
        table_prefix=table_prefix,
        create_tables_if_not_exist=True,
        pool_minconn=1,
        pool_maxconn=max(10, INGEST_WORKERS),
        create_vector_indexes=True,
        # max_shard_size is read from KB_SHARD_MAX_SIZE env var automatically.
    )

    try:
        kb_file_path = "./kb_out/kb_docs.json"
        if not os.path.exists(kb_file_path):
            logger.error(f"KB file not found at {kb_file_path}")
            sys.exit(1)

        with open(kb_file_path, "r", encoding="utf-8") as f:
            kb_articles = json.load(f)

        logger.info(f"Loaded {len(kb_articles)} KB articles from {kb_file_path}")
        logger.info(f"Processing with {INGEST_WORKERS} workers...")

        tasks = []
        with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as ex:
            for article in kb_articles:
                kbid    = str(article.get("KB_number") or "").strip()
                content = article.get("processed_content", "") or ""
                title   = article.get("name", "") or ""

                if not kbid:
                    logger.warning(f"Skipping article with missing KB_number: {title or 'Unknown'}")
                    continue
                if not content.strip():
                    logger.warning(f"Skipping article {kbid} with missing processed_content")
                    continue

                tasks.append(
                    ex.submit(
                        inserter._insert_single_article_core,
                        kbid=kbid,
                        title=title,
                        content=content,
                        exists=False,
                        update_if_exists=True,
                    )
                )

            for fut in as_completed(tasks):
                try:
                    _ = fut.result()
                except Exception as e:
                    logger.error(f"Worker task failed: {e}")

        # Print final shard layout.
        logger.info("\nFinal shard utilisation:")
        for sinfo in inserter.get_shard_info():
            logger.info(
                f"  Shard {sinfo['shard_id']}: "
                f"{sinfo['doc_count']}/{sinfo['max_shard_size']} docs "
                f"({sinfo['utilization_pct']}% full)"
            )

    finally:
        inserter.traceability.print_summary()
        inserter.close()


if __name__ == "__main__":
    main()
