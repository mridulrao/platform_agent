import logging
import threading
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from search_utils.search_fusion import merge_vector_results
from search_utils.search_shards import collection_shard_clause


logger = logging.getLogger(__name__)

VECTOR_EXACT_MATCH_MULT = 0.65
VECTOR_PHRASE_MATCH_MULT = 0.82


class VectorSearcher:
    def __init__(
        self,
        *,
        pg_client,
        keywords_table: str,
        kb_collection_id: str,
        oversample_mult: int,
        category_boost_oversample_mult: int,
        vector_cat_match_mult: float,
        vector_cat_both_mult: float,
        vector_cat_other_mult: float,
    ):
        self.pg_client = pg_client
        self.keywords_table = keywords_table
        self.kb_collection_id = kb_collection_id
        self.oversample_mult = oversample_mult
        self.category_boost_oversample_mult = category_boost_oversample_mult
        self.vector_cat_match_mult = vector_cat_match_mult
        self.vector_cat_both_mult = vector_cat_both_mult
        self.vector_cat_other_mult = vector_cat_other_mult
        self._vector_type_sql: Optional[str] = None
        self._vector_schema_q: Optional[str] = None
        self._vector_type_lock = threading.Lock()

    def resolve_vector_type_sql(self) -> str:
        if self._vector_type_sql is not None:
            return self._vector_type_sql
        with self._vector_type_lock:
            if self._vector_type_sql is not None:
                return self._vector_type_sql
            q = text(
                """
                SELECT n.nspname AS schema_name
                FROM pg_type t
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE t.typname = 'vector'
                ORDER BY
                  CASE
                    WHEN n.nspname = 'extensions' THEN 0
                    WHEN n.nspname = 'public' THEN 1
                    ELSE 2
                  END,
                  n.nspname
                LIMIT 1;
                """
            )
            with self.pg_client.get_engine().connect() as conn:
                schema_name = conn.execute(q).scalar()
            if not schema_name:
                raise RuntimeError(
                    "pgvector type `vector` was not found in this database.\n"
                    "Fix:\n"
                    "- Enable the 'vector' extension (Supabase: Database -> Extensions)\n"
                    "- Or ensure you're connecting to the correct database.\n"
                )
            schema_q = '"' + str(schema_name).replace('"', '""') + '"'
            self._vector_schema_q = schema_q
            self._vector_type_sql = f'{schema_q}."vector"'
            logger.info("Resolved pgvector type as: %s", self._vector_type_sql)
            return self._vector_type_sql

    @property
    def vector_schema_q(self) -> Optional[str]:
        return self._vector_schema_q

    @staticmethod
    def embedding_to_pgvector_literal(emb: List[float]) -> str:
        out = []
        for x in emb:
            try:
                fx = float(x)
            except Exception:
                fx = 0.0
            if fx != fx or fx == float("inf") or fx == float("-inf"):
                fx = 0.0
            out.append(repr(fx))
        return "[" + ",".join(out) + "]"

    def search_single_shard(
        self,
        *,
        query_embedding: List[float],
        query_text: str = "",
        column_name: str,
        shard_id: int,
        category_filter: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not query_embedding:
            return []
        try:
            vector_type_sql = self.resolve_vector_type_sql()
        except Exception as exc:
            logger.error("Vector search disabled (pgvector not available): %s", exc)
            return []

        vector_str = self.embedding_to_pgvector_literal(query_embedding)
        op_schema_q = self._vector_schema_q or '"public"'

        lim = top_k * self.oversample_mult
        if category_filter and str(category_filter).strip() and category_filter.lower() != "both":
            lim *= self.category_boost_oversample_mult

        where_clause, params = collection_shard_clause(self.kb_collection_id, shard_id)
        params.update({"qvec": vector_str, "lim": lim})

        q = text(
            f"""
            SELECT
                kbid,
                category,
                article_type,
                primary_topic,
                specific_keyword,
                generic_keyword,
                usecase,
                acronym,
                secondary_entity,
                {column_name} OPERATOR({op_schema_q}.<=>) CAST(:qvec AS {vector_type_sql}) AS distance
            FROM {self.keywords_table}
            WHERE {column_name} IS NOT NULL
              AND {where_clause}
            ORDER BY distance ASC
            LIMIT :lim
            """
        )

        try:
            with self.pg_client.get_engine().connect() as conn:
                rows = conn.execute(q, params).fetchall()

            best: Dict[str, Dict[str, Any]] = {}
            q_cat = str(category_filter).strip().lower() if category_filter else ""
            normalized_query = " ".join(str(query_text or "").strip().lower().split())
            for row in rows:
                kbid = str(row[0])
                dist = float(row[9]) if row[9] is not None else 1e9
                row_cat = str(row[1]).strip().lower() if row[1] is not None else ""
                mult = 1.0
                if q_cat and q_cat != "both":
                    if row_cat == q_cat:
                        mult = self.vector_cat_match_mult
                    elif row_cat == "both":
                        mult = self.vector_cat_both_mult
                    else:
                        mult = self.vector_cat_other_mult
                field_values = [
                    str(row[3] or "").strip().lower(),
                    str(row[4] or "").strip().lower(),
                    str(row[5] or "").strip().lower(),
                    str(row[6] or "").strip().lower(),
                    str(row[7] or "").strip().lower(),
                    str(row[8] or "").strip().lower(),
                ]
                exact_hit = bool(normalized_query) and any(val == normalized_query for val in field_values if val)
                phrase_hit = bool(normalized_query) and any(
                    normalized_query in val or val in normalized_query for val in field_values if val
                )

                boosted_dist = dist * mult
                if exact_hit:
                    boosted_dist *= VECTOR_EXACT_MATCH_MULT
                elif phrase_hit:
                    boosted_dist *= VECTOR_PHRASE_MATCH_MULT
                if kbid not in best or boosted_dist < best[kbid]["vector_distance"]:
                    best[kbid] = {
                        "KBID": kbid,
                        "Category": row[1],
                        "Article_Type": row[2],
                        "Primary_Topic": row[3],
                        "vector_distance": boosted_dist,
                        "vector_distance_raw": dist,
                        "vector_exact_match": exact_hit,
                        "vector_phrase_match": phrase_hit,
                        "source": f"vector:{column_name}",
                        "shard_id": shard_id,
                    }
            return sorted(best.values(), key=lambda x: x["vector_distance"])
        except Exception as exc:
            logger.exception("Vector search failed (shard=%s, col=%s): %s", shard_id, column_name, exc)
            return []

    def merge(self, per_shard: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return merge_vector_results(per_shard)
