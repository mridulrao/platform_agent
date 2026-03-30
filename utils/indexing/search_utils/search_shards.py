import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text


logger = logging.getLogger(__name__)


class ShardRegistry:
    def __init__(self, pg_client, shard_registry_table: str, kb_collection_id: str):
        self._pg_client = pg_client
        self._table = shard_registry_table
        self._collection_id = kb_collection_id
        self._shard_ids: Optional[List[int]] = None
        self._lock = threading.Lock()

    def refresh(self) -> List[int]:
        with self._lock:
            self._shard_ids = self._load()
            return list(self._shard_ids)

    def get(self) -> List[int]:
        if self._shard_ids is None:
            with self._lock:
                if self._shard_ids is None:
                    self._shard_ids = self._load()
        return list(self._shard_ids)

    def _load(self) -> List[int]:
        try:
            q = text(
                f"""
                SELECT shard_id
                FROM {self._table}
                WHERE kb_collection_id = :cid
                ORDER BY shard_id ASC
                """
            )
            with self._pg_client.get_engine().connect() as conn:
                rows = conn.execute(q, {"cid": self._collection_id}).fetchall()
            ids = [int(r[0]) for r in rows if r[0] is not None]
            if not ids:
                logger.info(
                    "ShardRegistry: no shards found for collection %r; defaulting to [0].",
                    self._collection_id,
                )
                return [0]
            logger.info(
                "ShardRegistry: loaded %d shard(s) for collection %r: %s",
                len(ids),
                self._collection_id,
                ids,
            )
            return ids
        except Exception as exc:
            logger.warning(
                "ShardRegistry: could not load shards (%s); defaulting to [0].",
                exc,
            )
            return [0]


def collection_shard_clause(
    kb_collection_id: str,
    shard_id: int,
    table_alias: str = "",
) -> Tuple[str, Dict[str, Any]]:
    prefix = f"{table_alias}." if table_alias else ""
    params: Dict[str, Any] = {"sid": shard_id}
    if kb_collection_id:
        params["cid"] = kb_collection_id
        clause = f"{prefix}kb_collection_id = :cid AND {prefix}shard_id = :sid"
    else:
        clause = f"{prefix}shard_id = :sid"
    return clause, params
