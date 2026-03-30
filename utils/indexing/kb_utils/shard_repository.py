import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ShardRepository:
    def __init__(
        self,
        *,
        pool,
        kb_collection_id: str,
        shard_registry_table_q: str,
        summaries_table_q: str,
        max_shard_size: int,
    ):
        self.pool = pool
        self.kb_collection_id = kb_collection_id
        self.shard_registry_table_q = shard_registry_table_q
        self.summaries_table_q = summaries_table_q
        self.max_shard_size = max_shard_size

    def assign_shard_for_new_doc(self) -> int:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT shard_id, doc_count, max_shard_size
                FROM {self.shard_registry_table_q}
                WHERE kb_collection_id = %s
                ORDER BY shard_id ASC
                FOR UPDATE
                """,
                (self.kb_collection_id,),
            )
            rows = cur.fetchall()

            for row in rows:
                shard_id, doc_count, max_sz = int(row[0]), int(row[1]), int(row[2])
                if doc_count < max_sz:
                    cur.execute(
                        f"""
                        UPDATE {self.shard_registry_table_q}
                        SET doc_count = doc_count + 1
                        WHERE kb_collection_id = %s AND shard_id = %s
                        """,
                        (self.kb_collection_id, shard_id),
                    )
                    conn.commit()
                    logger.debug(
                        "Assigned shard %s (doc_count %s/%s) for collection %s",
                        shard_id,
                        doc_count + 1,
                        max_sz,
                        self.kb_collection_id,
                    )
                    return shard_id

            next_shard_id = len(rows)
            cur.execute(
                f"""
                INSERT INTO {self.shard_registry_table_q}
                    (kb_collection_id, shard_id, doc_count, max_shard_size, created_at)
                VALUES (%s, %s, 1, %s, %s)
                ON CONFLICT (kb_collection_id, shard_id) DO UPDATE
                    SET doc_count = {self.shard_registry_table_q}.doc_count + 1
                RETURNING shard_id
                """,
                (self.kb_collection_id, next_shard_id, self.max_shard_size, datetime.now()),
            )
            result_row = cur.fetchone()
            conn.commit()
            assigned_id = int(result_row[0]) if result_row else next_shard_id
            logger.info(
                "Created new shard %s (max_shard_size=%s) for collection %s",
                assigned_id,
                self.max_shard_size,
                self.kb_collection_id,
            )
            return assigned_id
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            self.pool.putconn(conn)

    def release_shard_slot(self, shard_id: int) -> None:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                UPDATE {self.shard_registry_table_q}
                SET doc_count = GREATEST(0, doc_count - 1)
                WHERE kb_collection_id = %s AND shard_id = %s
                """,
                (self.kb_collection_id, shard_id),
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("Could not release shard slot (shard_id=%s): %s", shard_id, exc)
        finally:
            cur.close()
            self.pool.putconn(conn)

    def get_shard_for_kbid(self, kbid: str) -> Optional[int]:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT shard_id
                FROM {self.summaries_table_q}
                WHERE doc_id = %s AND kb_collection_id = %s
                LIMIT 1
                """,
                (kbid, self.kb_collection_id),
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            cur.close()
            self.pool.putconn(conn)

    def get_all_shard_ids(self) -> List[int]:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT shard_id
                FROM {self.shard_registry_table_q}
                WHERE kb_collection_id = %s
                ORDER BY shard_id ASC
                """,
                (self.kb_collection_id,),
            )
            return [int(r[0]) for r in (cur.fetchall() or [])]
        finally:
            cur.close()
            self.pool.putconn(conn)

    def get_shard_info(self) -> List[Dict[str, Any]]:
        conn = self.pool.getconn()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT shard_id, doc_count, max_shard_size, created_at
                FROM {self.shard_registry_table_q}
                WHERE kb_collection_id = %s
                ORDER BY shard_id ASC
                """,
                (self.kb_collection_id,),
            )
            rows = cur.fetchall() or []
            info = []
            for row in rows:
                shard_id, doc_count, max_sz, created_at = row
                doc_count = int(doc_count or 0)
                max_sz = int(max_sz or self.max_shard_size)
                info.append(
                    {
                        "shard_id": int(shard_id),
                        "doc_count": doc_count,
                        "max_shard_size": max_sz,
                        "utilization_pct": round(doc_count / max_sz * 100, 1) if max_sz else 0.0,
                        "created_at": created_at.isoformat() if created_at else None,
                    }
                )
            return info
        finally:
            cur.close()
            self.pool.putconn(conn)
