import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from search_utils.search_fusion import merge_fts_keyword_results, merge_fts_summary_results
from search_utils.search_shards import collection_shard_clause


logger = logging.getLogger(__name__)

KEYWORD_EXACT_MATCH_BONUS = 1.25
KEYWORD_PHRASE_MATCH_BONUS = 0.35
SUMMARY_EXACT_MATCH_BONUS = 0.80
SUMMARY_PHRASE_MATCH_BONUS = 0.25


class FTSSearcher:
    def __init__(
        self,
        *,
        pg_client,
        keywords_table: str,
        summaries_table: str,
        kb_collection_id: str,
        oversample_mult: int,
        category_boost_oversample_mult: int,
        fts_cat_match_mult: float,
        fts_cat_both_mult: float,
        fts_cat_other_mult: float,
    ):
        self.pg_client = pg_client
        self.keywords_table = keywords_table
        self.summaries_table = summaries_table
        self.kb_collection_id = kb_collection_id
        self.oversample_mult = oversample_mult
        self.category_boost_oversample_mult = category_boost_oversample_mult
        self.fts_cat_match_mult = fts_cat_match_mult
        self.fts_cat_both_mult = fts_cat_both_mult
        self.fts_cat_other_mult = fts_cat_other_mult

    def fts_query_fn(self) -> str:
        return "websearch_to_tsquery"

    def search_keywords_single_shard(
        self,
        *,
        query_text: str,
        shard_id: int,
        category_filter: Optional[str],
        top_k: int,
        use_generated_column: bool = True,
    ) -> List[Dict[str, Any]]:
        if not query_text or not str(query_text).strip():
            return []
        qfn = self.fts_query_fn()
        doc_vec = (
            "fts_keywords"
            if use_generated_column
            else "(setweight(to_tsvector('simple', coalesce(specific_keyword,'')), 'A') ||"
            " setweight(to_tsvector('simple', coalesce(generic_keyword,'')), 'B') ||"
            " setweight(to_tsvector('simple', coalesce(usecase,'')), 'B') ||"
            " setweight(to_tsvector('simple', coalesce(primary_topic,'')), 'B') ||"
            " setweight(to_tsvector('simple', coalesce(acronym,'')), 'C') ||"
            " setweight(to_tsvector('simple', coalesce(secondary_entity,'')), 'D'))"
        )
        lim = top_k * self.oversample_mult
        if category_filter and str(category_filter).strip() and category_filter.lower() != "both":
            lim *= self.category_boost_oversample_mult
        where_clause, params = collection_shard_clause(self.kb_collection_id, shard_id)
        normalized_query = str(query_text).strip().lower()
        params.update(
            {
                "qtxt": str(query_text),
                "qtxt_norm": normalized_query,
                "qpattern": f"%{normalized_query}%",
                "lim": lim,
            }
        )
        q = text(
            f"""
            WITH q AS (
              SELECT {qfn}('simple', :qtxt) AS tsq
            )
            SELECT
                kbid,
                max(ts_rank_cd({doc_vec}, q.tsq)) AS fts_rank,
                max(category) AS category,
                max(article_type) AS article_type,
                max(primary_topic) AS primary_topic,
                max(
                    CASE WHEN lower(trim(coalesce(specific_keyword, ''))) = :qtxt_norm
                           OR lower(trim(coalesce(acronym, ''))) = :qtxt_norm
                           OR lower(trim(coalesce(primary_topic, ''))) = :qtxt_norm
                           OR lower(trim(coalesce(usecase, ''))) = :qtxt_norm
                      THEN 1 ELSE 0 END
                ) AS exact_hit,
                max(
                    CASE WHEN lower(coalesce(specific_keyword, '')) LIKE :qpattern
                           OR lower(coalesce(generic_keyword, '')) LIKE :qpattern
                           OR lower(coalesce(usecase, '')) LIKE :qpattern
                           OR lower(coalesce(primary_topic, '')) LIKE :qpattern
                           OR lower(coalesce(acronym, '')) LIKE :qpattern
                           OR lower(coalesce(secondary_entity, '')) LIKE :qpattern
                      THEN 1 ELSE 0 END
                ) AS phrase_hit
            FROM {self.keywords_table}, q
            WHERE {doc_vec} @@ q.tsq
              AND {where_clause}
            GROUP BY kbid
            ORDER BY fts_rank DESC
            LIMIT :lim
            """
        )
        try:
            with self.pg_client.get_engine().connect() as conn:
                rows = conn.execute(q, params).fetchall()
            q_cat = str(category_filter).strip().lower() if category_filter else ""
            out = []
            for row in rows:
                base_rank = float(row[1] or 0.0)
                row_cat = str(row[2]).strip().lower() if row[2] is not None else ""
                exact_hit = bool(row[5])
                phrase_hit = bool(row[6])
                mult = 1.0
                if q_cat and q_cat != "both":
                    if row_cat == q_cat:
                        mult = self.fts_cat_match_mult
                    elif row_cat == "both":
                        mult = self.fts_cat_both_mult
                    else:
                        mult = self.fts_cat_other_mult
                exact_bonus = KEYWORD_EXACT_MATCH_BONUS if exact_hit else 0.0
                phrase_bonus = KEYWORD_PHRASE_MATCH_BONUS if phrase_hit else 0.0
                out.append(
                    {
                        "KBID": str(row[0]),
                        "fts_keywords_rank": base_rank * mult + exact_bonus + phrase_bonus,
                        "fts_keywords_rank_raw": base_rank,
                        "fts_keywords_exact_match": exact_hit,
                        "fts_keywords_phrase_match": phrase_hit,
                        "fts_keywords_exact_bonus": exact_bonus,
                        "fts_keywords_phrase_bonus": phrase_bonus,
                        "Category": row[2],
                        "Article_Type": row[3],
                        "Primary_Topic": row[4],
                        "source": "fts:keywords",
                        "shard_id": shard_id,
                    }
                )
            out.sort(key=lambda x: x.get("fts_keywords_rank", 0.0), reverse=True)
            return out[:top_k]
        except Exception as exc:
            msg = str(exc).lower()
            if use_generated_column and ("fts_keywords" in msg or "column" in msg):
                return self.search_keywords_single_shard(
                    query_text=query_text,
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=top_k,
                    use_generated_column=False,
                )
            logger.exception("FTS keywords search failed (shard=%s): %s", shard_id, exc)
            return []

    def search_summaries_single_shard(
        self,
        *,
        query_text: str,
        shard_id: int,
        top_k: int,
        use_generated_column: bool = True,
    ) -> List[Dict[str, Any]]:
        if not query_text or not str(query_text).strip():
            return []
        qfn = self.fts_query_fn()
        doc_vec = (
            "fts_summaries"
            if use_generated_column
            else "(setweight(to_tsvector('simple', coalesce(primary_intent,'')), 'A') ||"
            " setweight(to_tsvector('simple', coalesce(query_triggers,'')), 'A') ||"
            " setweight(to_tsvector('simple', coalesce(summary,'')), 'A') ||"
            " setweight(to_tsvector('simple', coalesce(content,'')), 'B') ||"
            " setweight(to_tsvector('simple', coalesce(secondary_mentions,'')), 'D'))"
        )
        where_clause, params = collection_shard_clause(self.kb_collection_id, shard_id)
        normalized_query = str(query_text).strip().lower()
        params.update(
            {
                "qtxt": str(query_text),
                "qtxt_norm": normalized_query,
                "qpattern": f"%{normalized_query}%",
                "lim": top_k * self.oversample_mult,
            }
        )
        q = text(
            f"""
            WITH q AS (
              SELECT {qfn}('simple', :qtxt) AS tsq
            )
            SELECT
                doc_id,
                ts_rank_cd({doc_vec}, q.tsq) AS fts_rank,
                primary_intent,
                CASE WHEN lower(trim(coalesce(primary_intent, ''))) = :qtxt_norm
                          OR lower(trim(coalesce(query_triggers, ''))) = :qtxt_norm
                     THEN 1 ELSE 0 END AS exact_hit,
                CASE WHEN lower(coalesce(primary_intent, '')) LIKE :qpattern
                          OR lower(coalesce(query_triggers, '')) LIKE :qpattern
                          OR lower(coalesce(summary, '')) LIKE :qpattern
                          OR lower(coalesce(content, '')) LIKE :qpattern
                     THEN 1 ELSE 0 END AS phrase_hit
            FROM {self.summaries_table}, q
            WHERE {doc_vec} @@ q.tsq
              AND {where_clause}
            ORDER BY fts_rank DESC
            LIMIT :lim
            """
        )
        try:
            with self.pg_client.get_engine().connect() as conn:
                rows = conn.execute(q, params).fetchall()
            return [
                {
                    "KBID": str(row[0]),
                    "fts_summaries_rank": (
                        float(row[1] or 0.0)
                        + (SUMMARY_EXACT_MATCH_BONUS if bool(row[3]) else 0.0)
                        + (SUMMARY_PHRASE_MATCH_BONUS if bool(row[4]) else 0.0)
                    ),
                    "fts_summaries_rank_raw": float(row[1] or 0.0),
                    "fts_summaries_exact_match": bool(row[3]),
                    "fts_summaries_phrase_match": bool(row[4]),
                    "primary_intent": row[2],
                    "source": "fts:summaries",
                    "shard_id": shard_id,
                }
                for row in rows
            ]
        except Exception as exc:
            msg = str(exc).lower()
            if use_generated_column and ("fts_summaries" in msg or "column" in msg):
                return self.search_summaries_single_shard(
                    query_text=query_text,
                    shard_id=shard_id,
                    top_k=top_k,
                    use_generated_column=False,
                )
            logger.exception("FTS summaries search failed (shard=%s): %s", shard_id, exc)
            return []

    def merge_keywords(self, per_shard: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return merge_fts_keyword_results(per_shard)

    def merge_summaries(self, per_shard: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return merge_fts_summary_results(per_shard)
