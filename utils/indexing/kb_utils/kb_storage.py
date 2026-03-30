import logging
from typing import Any, Dict, List

from psycopg2.extras import execute_values

from kb_utils.kb_db import EMBEDDING_DIM, as_f32_list


logger = logging.getLogger(__name__)


class KnowledgeBaseStorage:
    def __init__(
        self,
        *,
        pool,
        kb_collection_id: str,
        keywords_table_q: str,
        summaries_table_q: str,
    ):
        self.pool = pool
        self.kb_collection_id = kb_collection_id
        self.keywords_table_q = keywords_table_q
        self.summaries_table_q = summaries_table_q

    def check_article_exists(self, kbid: str) -> bool:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM {self.summaries_table_q}
                    WHERE doc_id = %s AND kb_collection_id = %s
                    LIMIT 1
                );
                """,
                (kbid, self.kb_collection_id),
            )
            return bool(cur.fetchone()[0])
        finally:
            cur.close()
            self.pool.putconn(conn)

    def fetch_existing_kbids(self, kbids: List[str]) -> Dict[str, int]:
        kbids = [str(x).strip() for x in (kbids or []) if str(x).strip()]
        if not kbids:
            return {}

        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT doc_id, shard_id
                FROM {self.summaries_table_q}
                WHERE kb_collection_id = %s
                  AND doc_id = ANY(%s)
                """,
                (self.kb_collection_id, kbids),
            )
            rows = cur.fetchall() or []
            return {
                str(row[0]): int(row[1])
                for row in rows
                if row and row[0] is not None and row[1] is not None
            }
        finally:
            cur.close()
            self.pool.putconn(conn)

    def delete_existing_article_keywords(self, kbid: str) -> int:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"DELETE FROM {self.keywords_table_q} WHERE kbid = %s AND kb_collection_id = %s",
                (kbid, self.kb_collection_id),
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            return 0
        finally:
            cur.close()
            self.pool.putconn(conn)

    def delete_existing_article_summary(self, kbid: str) -> int:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"DELETE FROM {self.summaries_table_q} WHERE doc_id = %s AND kb_collection_id = %s",
                (kbid, self.kb_collection_id),
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            return 0
        finally:
            cur.close()
            self.pool.putconn(conn)

    def insert_keyword_rows(self, rows: List[Dict[str, Any]], shard_id: int) -> bool:
        if not rows:
            logger.warning("No keyword rows to insert")
            return True

        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            records = []
            for row in rows:
                records.append(
                    (
                        str(row["kbid"]),
                        str(row["kb_collection_id"]),
                        shard_id,
                        row.get("category"),
                        row.get("article_type"),
                        row.get("primary_topic"),
                        row.get("specific_keyword"),
                        row.get("generic_keyword"),
                        row.get("usecase"),
                        row.get("secondary_entity"),
                        row.get("acronym"),
                        as_f32_list(row.get("specific_keyword_embedding"), dim=EMBEDDING_DIM),
                        as_f32_list(row.get("generic_keyword_embedding"), dim=EMBEDDING_DIM),
                        as_f32_list(row.get("usecase_embedding"), dim=EMBEDDING_DIM),
                        as_f32_list(row.get("secondary_entity_embedding"), dim=EMBEDDING_DIM),
                        as_f32_list(row.get("acronym_embedding"), dim=EMBEDDING_DIM),
                        row.get("created_at"),
                        row.get("updated_at"),
                    )
                )

            execute_values(
                cur,
                f"""
                INSERT INTO {self.keywords_table_q} (
                    kbid, kb_collection_id, shard_id,
                    category, article_type, primary_topic,
                    specific_keyword, generic_keyword, usecase,
                    secondary_entity, acronym,
                    specific_keyword_embedding, generic_keyword_embedding,
                    usecase_embedding, secondary_entity_embedding, acronym_embedding,
                    created_at, updated_at
                ) VALUES %s
                """,
                records,
                page_size=200,
            )
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.error("Error inserting keyword rows (shard_id=%s): %s", shard_id, exc)
            return False
        finally:
            cur.close()
            self.pool.putconn(conn)

    def insert_summary_record(self, data: Dict[str, Any], shard_id: int) -> bool:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO {self.summaries_table_q} (
                    doc_id, kb_collection_id, shard_id,
                    article_title, primary_intent, query_triggers,
                    secondary_mentions, summary, content,
                    created_at, updated_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (doc_id, kb_collection_id) DO UPDATE SET
                    shard_id           = EXCLUDED.shard_id,
                    article_title      = EXCLUDED.article_title,
                    primary_intent     = EXCLUDED.primary_intent,
                    query_triggers     = EXCLUDED.query_triggers,
                    secondary_mentions = EXCLUDED.secondary_mentions,
                    summary            = EXCLUDED.summary,
                    content            = EXCLUDED.content,
                    updated_at         = EXCLUDED.updated_at
                """,
                (
                    str(data["doc_id"]),
                    str(data["kb_collection_id"]),
                    shard_id,
                    data.get("article_title"),
                    data.get("primary_intent"),
                    data.get("query_triggers"),
                    data.get("secondary_mentions"),
                    data.get("summary"),
                    data.get("content"),
                    data.get("created_at"),
                    data.get("updated_at"),
                ),
            )
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.error(
                "Error inserting summary for %s (shard_id=%s): %s",
                data.get("doc_id"),
                shard_id,
                exc,
            )
            return False
        finally:
            cur.close()
            self.pool.putconn(conn)
