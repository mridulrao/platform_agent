"""
Knowledge Base Searcher using PgVector + Postgres Full Text Search (FTS)

Hybrid:
- Vector similarity search via pgvector (<=>)
- Full Text Search (FTS) via ts_rank_cd / websearch_to_tsquery
- Combine via weighted RRF (rank fusion) to avoid score-scale headaches
Returns KBIDs WITH scores.

Shard-awareness (added):
- Every collection is split into N shards of ≤ KB_SHARD_MAX_SIZE docs each.
- All vector / FTS searches fan out across shards in parallel, then merge
  results (keeping best score per KBID) before the final RRF fusion step.
- Shard IDs are loaded once per searcher instance and cached; call
  refresh_shards() if you need to pick up newly created shards at runtime.
- Pass kb_collection_id to __init__ (or set PGVECTOR_KB_COLLECTION_ID in env)
  to scope all queries to a single collection.  Previously queries had no
  collection filter at all, which caused cross-collection result bleed.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Tuple

from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from indexing_config import indexing_config
from pgvec_client import get_shared_pgvector_client
from search_utils.search_fts import FTSSearcher
from search_utils.search_fusion import weighted_rrf
from search_utils.search_shards import ShardRegistry
from search_utils.search_vector import VectorSearcher

load_dotenv()
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config-driven knobs
# -----------------------------------------------------------------------------
DEFAULT_TOP_K = indexing_config.kb_top_k
OVERSAMPLE_MULT = indexing_config.kb_oversample_mult

CATEGORY_BOOST_OVERSAMPLE_MULT = indexing_config.category_boost_oversample_mult

VECTOR_CAT_MATCH_MULT = indexing_config.vector_cat_match_mult
VECTOR_CAT_BOTH_MULT = indexing_config.vector_cat_both_mult
VECTOR_CAT_OTHER_MULT = indexing_config.vector_cat_other_mult

FTS_CAT_MATCH_MULT = indexing_config.fts_cat_match_mult
FTS_CAT_BOTH_MULT = indexing_config.fts_cat_both_mult
FTS_CAT_OTHER_MULT = indexing_config.fts_cat_other_mult

DEFAULT_VECTOR_WEIGHT = indexing_config.vector_weight
DEFAULT_FTS_KEYWORDS_WEIGHT = indexing_config.fts_keywords_weight
DEFAULT_FTS_SUMMARIES_WEIGHT = indexing_config.fts_summaries_weight
DEFAULT_RRF_K = indexing_config.rrf_k
SEARCH_PHRASE_TOPK_MULT = indexing_config.search_phrase_topk_mult
ACRONYM_TOPK_MULT = indexing_config.acronym_topk_mult

def get_cached_pg_client():
    return get_shared_pgvector_client()


def _adaptive_top_k(top_k: int, signal_name: str) -> int:
    if signal_name == "search_phrases":
        return max(top_k, top_k * SEARCH_PHRASE_TOPK_MULT)
    if signal_name == "acronyms":
        return max(top_k, top_k * ACRONYM_TOPK_MULT)
    return top_k


# -----------------------------------------------------------------------------
# Shard registry reader
# -----------------------------------------------------------------------------

class PgVectorFTSHybridKnowledgeBaseSearcher:
    """
    Triple-hybrid KB searcher (vector + FTS-keywords + FTS-summaries) with
    shard fan-out and collection scoping.

    Parameters
    ----------
    kb_collection_id : str, optional
        Scope all queries to this collection. Reads from indexing config /
        PGVECTOR_KB_COLLECTION_ID if not supplied.
    max_workers : int
        Thread-pool size for blocking DB/embedding calls.
    """

    def __init__(
        self,
        kb_collection_id: Optional[str] = None,
        max_workers: int = 10,
    ):
        self.pg_client = get_cached_pg_client()

        # Collection scoping ───────────────────────────────────────────────────
        self.kb_collection_id: str = (
            kb_collection_id
            or indexing_config.collection_id
            or ""
        )
        if not self.kb_collection_id:
            logger.warning(
                "PgVectorFTSHybridKnowledgeBaseSearcher: kb_collection_id is not set. "
                "Queries will NOT be scoped to a collection and may return mixed results. "
                "Set COLLECTION_ID in indexing config or pass kb_collection_id= explicitly."
            )

        # Table names ──────────────────────────────────────────────────────────
        self.keywords_table       = self.pg_client.get_table_name("kb_keywords")
        self.summaries_table      = self.pg_client.get_table_name("kb_summaries")
        self.shard_registry_table = self.pg_client.get_table_name("kb_shard_registry")

        # Shard registry ───────────────────────────────────────────────────────
        self._shard_registry = ShardRegistry(
            pg_client=self.pg_client,
            shard_registry_table=self.shard_registry_table,
            kb_collection_id=self.kb_collection_id,
        )

        # Azure OpenAI clients ─────────────────────────────────────────────────
        self.client = indexing_config.get_llm_client()
        self.embedding_client = indexing_config.get_embedding_client()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.vector_searcher = VectorSearcher(
            pg_client=self.pg_client,
            keywords_table=self.keywords_table,
            kb_collection_id=self.kb_collection_id,
            oversample_mult=OVERSAMPLE_MULT,
            category_boost_oversample_mult=CATEGORY_BOOST_OVERSAMPLE_MULT,
            vector_cat_match_mult=VECTOR_CAT_MATCH_MULT,
            vector_cat_both_mult=VECTOR_CAT_BOTH_MULT,
            vector_cat_other_mult=VECTOR_CAT_OTHER_MULT,
        )
        self.fts_searcher = FTSSearcher(
            pg_client=self.pg_client,
            keywords_table=self.keywords_table,
            summaries_table=self.summaries_table,
            kb_collection_id=self.kb_collection_id,
            oversample_mult=OVERSAMPLE_MULT,
            category_boost_oversample_mult=CATEGORY_BOOST_OVERSAMPLE_MULT,
            fts_cat_match_mult=FTS_CAT_MATCH_MULT,
            fts_cat_both_mult=FTS_CAT_BOTH_MULT,
            fts_cat_other_mult=FTS_CAT_OTHER_MULT,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Shard management helpers
    # ─────────────────────────────────────────────────────────────────────────

    def refresh_shards(self) -> List[int]:
        """
        Reload shard IDs from the DB.  Call this if you know a new shard was
        recently created and you want the searcher to pick it up immediately
        without restarting the process.
        """
        ids = self._shard_registry.refresh()
        logger.info(f"Shards refreshed: {ids}")
        return ids

    def _get_shard_ids(self) -> List[int]:
        return self._shard_registry.get()

    def get_shard_ids(self) -> List[int]:
        """
        Public accessor for shard IDs.
        Used by the two-level rerank pipeline in filter_useful_kb to discover
        how many shards to fan out across before running per-shard retrieval.
        """
        return self._get_shard_ids()

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    def get_embedding(self, text_in: str) -> List[float]:
        try:
            if not text_in or not str(text_in).strip():
                return []
            resp = self.embedding_client.embeddings.create(
                model=indexing_config.embedding_model,
                input=str(text_in),
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.exception(f"Embedding failed: {e}")
            return []

    async def get_embedding_async(self, text_in: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.get_embedding, text_in)

    def vector_search(
        self,
        query_embedding: List[float],
        query_text: str,
        column_name: str,
        category_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Fan out _vector_search_single_shard across all shards, merge, and
        return the top_k results by boosted distance.
        """
        shard_ids = self._get_shard_ids()

        per_shard = [
            self.vector_searcher.search_single_shard(
                query_embedding=query_embedding,
                query_text=query_text,
                column_name=column_name,
                shard_id=sid,
                category_filter=category_filter,
                top_k=top_k,
            )
            for sid in shard_ids
        ]
        merged = self.vector_searcher.merge(per_shard)
        return merged[:top_k]

    async def vector_search_async(self, *args, **kwargs) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: self.vector_search(*args, **kwargs))

    def fts_search_keywords(
        self,
        query_text: str,
        category_filter: Optional[str] = None,
        top_k: int = 5,
        use_generated_column: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fan out FTS keyword search across all shards and merge."""
        shard_ids = self._get_shard_ids()

        per_shard = [
            self.fts_searcher.search_keywords_single_shard(
                query_text=query_text,
                shard_id=sid,
                category_filter=category_filter,
                top_k=top_k,
                use_generated_column=use_generated_column,
            )
            for sid in shard_ids
        ]
        merged = self.fts_searcher.merge_keywords(per_shard)
        return merged[:top_k]

    def fts_search_summaries(
        self,
        query_text: str,
        top_k: int = 5,
        use_generated_column: bool = True,
    ) -> List[Dict[str, Any]]:
        """Fan out FTS summary search across all shards and merge."""
        shard_ids = self._get_shard_ids()

        per_shard = [
            self.fts_searcher.search_summaries_single_shard(
                query_text=query_text,
                shard_id=sid,
                top_k=top_k,
                use_generated_column=use_generated_column,
            )
            for sid in shard_ids
        ]
        merged = self.fts_searcher.merge_summaries(per_shard)
        return merged[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Parallel dual-FTS (keywords + summaries) across all shards
    # ─────────────────────────────────────────────────────────────────────────

    async def fts_search_dual_async(
        self,
        query_text: str,
        category_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run FTS keyword search and FTS summary search in parallel.
        Each internally fans out across shards and merges before returning.
        """
        loop = asyncio.get_running_loop()

        kw_task = loop.run_in_executor(
            self.executor,
            self.fts_search_keywords,
            query_text, category_filter, top_k,
        )
        sm_task = loop.run_in_executor(
            self.executor,
            self.fts_search_summaries,
            query_text, top_k,
        )

        kw_res, sm_res = await asyncio.gather(kw_task, sm_task)
        return {"keywords": kw_res, "summaries": sm_res}

    def _rrf(
        self,
        vector_results: List[Dict[str, Any]],
        fts_kw_results: List[Dict[str, Any]],
        fts_sm_results: List[Dict[str, Any]],
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        fts_kw_weight: float = DEFAULT_FTS_KEYWORDS_WEIGHT,
        fts_sm_weight: float = DEFAULT_FTS_SUMMARIES_WEIGHT,
        # Legacy params kept so existing callers with ranked_lists= don't break
        ranked_lists: List[List[Dict[str, Any]]] = None,
        weights: List[float] = None,
        k: int = DEFAULT_RRF_K,
    ) -> List[Dict[str, Any]]:
        """
        Score-based fusion (replaces original rank-based RRF internally).

        WHY NOT RANK-BASED RRF:
        RRF computes w/(k+rank).  At k=60, rank-1 scores 0.0074 and rank-5
        scores 0.0069 — a 7% difference regardless of whether the actual
        vector distance was 0.05 (near-perfect) or 0.45 (weak).  This
        washes out the semantic signal and was responsible for ~43% of
        observed ranking failures where the correct doc was in the pool
        at positions 2-20 but lost to a weaker doc at rank 1.

        HOW THIS WORKS:
        Min-max normalize each signal to [0,1] within the current batch,
        then take a weighted sum.  Score magnitude is preserved:
          - vector_distance: inverted  (0.05 → ~1.0,  0.45 → ~0.0)
          - fts_keywords_rank: direct  (higher ts_rank → higher score)
          - fts_summaries_rank: direct
        A presence bonus rewards docs that multiple signals agree on,
        replacing the implicit multi-list bonus RRF had.

        Function signature accepts both the new explicit form and the old
        ranked_lists= form so no other files need to change.
        """
        return weighted_rrf(
            vector_results=vector_results,
            fts_kw_results=fts_kw_results,
            fts_sm_results=fts_sm_results,
            vector_weight=vector_weight,
            fts_kw_weight=fts_kw_weight,
            fts_sm_weight=fts_sm_weight,
            match_bonus=float(os.getenv("FUSION_MATCH_BONUS", "0.15")),
            ranked_lists=ranked_lists,
            weights=weights,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Triple hybrid search
    # ─────────────────────────────────────────────────────────────────────────

    async def triple_hybrid_search(
        self,
        keywords: List[str],
        embedding_column: str,
        category_filter: Optional[str] = None,
        top_k: int = 5,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        fts_keywords_weight: float = DEFAULT_FTS_KEYWORDS_WEIGHT,
        fts_summaries_weight: float = DEFAULT_FTS_SUMMARIES_WEIGHT,
    ) -> List[Dict[str, Any]]:
        if not keywords:
            return []

        query_text = " ".join(str(k) for k in keywords if str(k).strip()).strip()
        if not query_text:
            return []

        # Embed once, then fan-out vector + FTS in parallel.
        query_embedding = await self.get_embedding_async(query_text)

        vector_task = self.vector_search_async(
            query_embedding, query_text, embedding_column, category_filter, top_k=top_k
        )
        fts_task = self.fts_search_dual_async(query_text, category_filter, top_k=top_k)

        vector_results, fts = await asyncio.gather(vector_task, fts_task)
        fts_kw = fts.get("keywords",  [])
        fts_sm = fts.get("summaries", [])

        fused = self._rrf(
            vector_results=vector_results,
            fts_kw_results=fts_kw,
            fts_sm_results=fts_sm,
            vector_weight=vector_weight,
            fts_kw_weight=fts_keywords_weight,
            fts_sm_weight=fts_summaries_weight,
        )

        # Annotate which retrieval sources matched each KBID.
        vec_ids = {x.get("KBID") for x in vector_results}
        kw_ids  = {x.get("KBID") for x in fts_kw}
        sm_ids  = {x.get("KBID") for x in fts_sm}

        out = []
        for r in fused[:top_k]:
            kbid = r["KBID"]
            matched_in = []
            if kbid in vec_ids:
                matched_in.append("vector")
            if kbid in kw_ids:
                matched_in.append("fts_keywords")
            if kbid in sm_ids:
                matched_in.append("fts_summaries")
            r["matched_in"] = matched_in
            out.append(r)

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Triple hybrid search — SINGLE shard variant
    # ─────────────────────────────────────────────────────────────────────────

    async def _triple_hybrid_search_single_shard(
        self,
        keywords: List[str],
        embedding_column: str,
        shard_id: int,
        category_filter: Optional[str] = None,
        top_k: int = 5,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        fts_keywords_weight: float = DEFAULT_FTS_KEYWORDS_WEIGHT,
        fts_summaries_weight: float = DEFAULT_FTS_SUMMARIES_WEIGHT,
    ) -> List[Dict[str, Any]]:
        """
        Identical to triple_hybrid_search but scoped to ONE shard only.

        Bypasses the cross-shard fan-out in vector_search / fts_search_* and
        calls the _single_shard variants directly so the caller controls which
        shard is searched.  Called by process_response_for_shard, which is in
        turn called by _process_single_shard in the two-level rerank pipeline.
        """
        if not keywords:
            return []

        query_text = " ".join(str(k) for k in keywords if str(k).strip()).strip()
        if not query_text:
            return []

        # Embed once, then fire vector + both FTS searches in parallel.
        query_embedding = await self.get_embedding_async(query_text)

        loop = asyncio.get_running_loop()

        vector_task = loop.run_in_executor(
            self.executor,
                lambda: self.vector_searcher.search_single_shard(
                    query_embedding=query_embedding,
                    query_text=query_text,
                    column_name=embedding_column,
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=top_k,
                ),
            )
        kw_task = loop.run_in_executor(
            self.executor,
                lambda: self.fts_searcher.search_keywords_single_shard(
                    query_text, shard_id, category_filter, top_k
                ),
            )
        sm_task = loop.run_in_executor(
            self.executor,
                lambda: self.fts_searcher.search_summaries_single_shard(
                    query_text, shard_id, top_k
                ),
            )

        vector_results, fts_kw, fts_sm = await asyncio.gather(vector_task, kw_task, sm_task)

        fused = self._rrf(
            vector_results=vector_results,
            fts_kw_results=fts_kw,
            fts_sm_results=fts_sm,
            vector_weight=vector_weight,
            fts_kw_weight=fts_keywords_weight,
            fts_sm_weight=fts_summaries_weight,
        )

        vec_ids = {x.get("KBID") for x in vector_results}
        kw_ids  = {x.get("KBID") for x in fts_kw}
        sm_ids  = {x.get("KBID") for x in fts_sm}

        out = []
        for r in fused[:top_k]:
            kbid = r["KBID"]
            matched_in = []
            if kbid in vec_ids:
                matched_in.append("vector")
            if kbid in kw_ids:
                matched_in.append("fts_keywords")
            if kbid in sm_ids:
                matched_in.append("fts_summaries")
            r["matched_in"] = matched_in
            out.append(r)

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry: process_response_async
    # ─────────────────────────────────────────────────────────────────────────

    async def process_response_async(
        self,
        response: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = DEFAULT_TOP_K
        else:
            top_k = max(1, int(top_k))

        category = (response.get("category") or "").strip()
        keywords = response.get("keywords") or {}
        usecases = response.get("usecase") or []
        search_phrases = response.get("search_phrases") or []
        acronyms = response.get("acronyms") or []

        category_filter = None if category.lower() == "both" else category

        specific = keywords.get("specific", []) or keywords.get("primary_specific", []) or []
        generic  = keywords.get("generic",  []) or keywords.get("primary_generic",  []) or []

        tasks = []
        if specific:
            tasks.append(
                self.triple_hybrid_search(
                    keywords=specific,
                    embedding_column="specific_keyword_embedding",
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "specific"),
                )
            )
        if generic:
            tasks.append(
                self.triple_hybrid_search(
                    keywords=generic,
                    embedding_column="generic_keyword_embedding",
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "generic"),
                )
            )
        if usecases:
            tasks.append(
                self.triple_hybrid_search(
                    keywords=usecases,
                    embedding_column="usecase_embedding",
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "usecase"),
                )
            )
        if search_phrases:
            tasks.append(
                self.triple_hybrid_search(
                    keywords=search_phrases,
                    embedding_column="usecase_embedding",
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "search_phrases"),
                )
            )
        if acronyms:
            tasks.append(
                self.triple_hybrid_search(
                    keywords=acronyms,
                    embedding_column="specific_keyword_embedding",
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "acronyms"),
                )
            )

        if not tasks:
            return []

        lists = await asyncio.gather(*tasks)

        # Dedup across the three keyword types; keep best rrf_score per KBID.
        best: Dict[str, Dict[str, Any]] = {}
        for lst in lists:
            for r in lst:
                kbid = r.get("KBID")
                if not kbid:
                    continue
                if kbid not in best or float(r.get("rrf_score", 0.0)) > float(best[kbid].get("rrf_score", 0.0)):
                    best[kbid] = r

        out = sorted(best.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return out[:top_k]

    # Sync wrapper
    def process_response(
        self,
        response: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.process_response_async(response, top_k=top_k))

    # ─────────────────────────────────────────────────────────────────────────
    # Shard-scoped entry: process_response_for_shard
    # ─────────────────────────────────────────────────────────────────────────

    async def process_response_for_shard(
        self,
        response: Dict[str, Any],
        shard_id: int,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Same as process_response_async but scoped to a SINGLE shard.

        Called by _process_single_shard in the two-level rerank pipeline
        (filter_useful_kb_agent_pgvec_deterministic_v2.py).  By restricting
        all three hybrid arms to one shard we avoid cross-shard score bleed
        and keep each shard's result list independent before the final rerank.

        Parameters
        ----------
        response  : structured query JSON with keys category / keywords / usecase
        shard_id  : which shard to restrict all DB queries to
        top_k     : number of results to return (default: DEFAULT_TOP_K)

        Returns
        -------
        Merged, RRF-fused list of candidates from this shard only,
        deduped by KBID, sorted by rrf_score descending.
        """
        top_k = max(1, int(top_k or DEFAULT_TOP_K))

        category = (response.get("category") or "").strip()
        keywords = response.get("keywords") or {}
        usecases = response.get("usecase") or []
        search_phrases = response.get("search_phrases") or []
        acronyms = response.get("acronyms") or []

        category_filter = None if category.lower() == "both" else category

        specific = keywords.get("specific", []) or keywords.get("primary_specific", []) or []
        generic  = keywords.get("generic",  []) or keywords.get("primary_generic",  []) or []

        tasks = []
        if specific:
            tasks.append(
                self._triple_hybrid_search_single_shard(
                    keywords=specific,
                    embedding_column="specific_keyword_embedding",
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "specific"),
                )
            )
        if generic:
            tasks.append(
                self._triple_hybrid_search_single_shard(
                    keywords=generic,
                    embedding_column="generic_keyword_embedding",
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "generic"),
                )
            )
        if usecases:
            tasks.append(
                self._triple_hybrid_search_single_shard(
                    keywords=usecases,
                    embedding_column="usecase_embedding",
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "usecase"),
                )
            )
        if search_phrases:
            tasks.append(
                self._triple_hybrid_search_single_shard(
                    keywords=search_phrases,
                    embedding_column="usecase_embedding",
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "search_phrases"),
                )
            )
        if acronyms:
            tasks.append(
                self._triple_hybrid_search_single_shard(
                    keywords=acronyms,
                    embedding_column="specific_keyword_embedding",
                    shard_id=shard_id,
                    category_filter=category_filter,
                    top_k=_adaptive_top_k(top_k, "acronyms"),
                )
            )

        if not tasks:
            return []

        lists = await asyncio.gather(*tasks)

        # Dedup across the three keyword types; keep best rrf_score per KBID.
        best: Dict[str, Dict[str, Any]] = {}
        for lst in lists:
            for r in lst:
                kbid = r.get("KBID")
                if not kbid:
                    continue
                if kbid not in best or float(r.get("rrf_score", 0.0)) > float(best[kbid].get("rrf_score", 0.0)):
                    best[kbid] = r

        out = sorted(best.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return out[:top_k]

    def __del__(self):
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------

async def search_knowledge_base_pgvec_fts(
    response: Dict[str, Any],
    top_k: Optional[int] = None,
    kb_collection_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    s = PgVectorFTSHybridKnowledgeBaseSearcher(kb_collection_id=kb_collection_id)
    return await s.process_response_async(response, top_k=top_k)


def search_knowledge_base_pgvec_fts_sync(
    response: Dict[str, Any],
    top_k: Optional[int] = None,
    kb_collection_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    s = PgVectorFTSHybridKnowledgeBaseSearcher(kb_collection_id=kb_collection_id)
    return s.process_response(response, top_k=top_k)


# -----------------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------------

async def test_search():
    sample = {
        "category": "software",
        "keywords": {"specific": ["EndNote 21"], "generic": ["installation"]},
        "usecase": ["install EndNote 21"],
    }
    res = await search_knowledge_base_pgvec_fts(sample, top_k=5)
    for r in res:
        print(r)
    return res


if __name__ == "__main__":
    asyncio.run(test_search())
