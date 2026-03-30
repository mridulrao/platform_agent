from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from env_config import config as env_config


load_dotenv()


@dataclass(frozen=True)
class AzureClientSettings:
    api_version: str
    azure_endpoint: str
    api_key: str
    azure_deployment: Optional[str] = None


class IndexingConfig:
    def __init__(self) -> None:
        self._llm_client: Optional[AzureOpenAI] = None
        self._embedding_client: Optional[AzureOpenAI] = None
        self._llm_lock = threading.Lock()
        self._embedding_lock = threading.Lock()

        self.llm_settings = AzureClientSettings(
            api_version=env_config.LLM_AZURE_OPENAI_API_VERSION,
            azure_endpoint=env_config.LLM_AZURE_OPENAI_ENDPOINT,
            api_key=env_config.LLM_AZURE_OPENAI_API_KEY,
            azure_deployment=getattr(env_config, "LLM_AZURE_OPENAI_DEPLOYMENT_NAME", None),
        )
        self.embedding_settings = AzureClientSettings(
            api_version=env_config.AZURE_OPENAI_EMBEDDING_MODEL_VERSION,
            azure_endpoint=env_config.AZURE_OPENAI_EMBEDDING_MODEL_ENDPOINT,
            api_key=env_config.LLM_AZURE_OPENAI_API_KEY,
            azure_deployment=getattr(env_config, "AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT", None),
        )

        self.llm_model = str(getattr(env_config, "LLM_AZURE_OPENAI_MODEL", "gpt-4o"))
        self.llm_deployment_name = str(
            getattr(
                env_config,
                "LLM_AZURE_OPENAI_DEPLOYMENT_NAME",
                getattr(env_config, "LLM_AZURE_OPENAI_DEPLOYMENT", self.llm_model),
            )
        )
        self.embedding_model = str(
            getattr(env_config, "AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT", "text-embedding-3-large")
        )
        self.embedding_dim = int(getattr(env_config, "AZURE_OPENAI_EMBEDDING_DIM", 1536))

        self.collection_id = getattr(env_config, "COLLECTION_ID", "")
        self.org_id = getattr(env_config, "ORG_ID", "")

        self.pgvector_database_url = getattr(env_config, "PGVECTOR_DATABASE_URL", None) or getattr(
            env_config, "DATABASE_URL", None
        )
        self.pgvector_schema = getattr(env_config, "PGVECTOR_SCHEMA", "public")
        self.pgvector_table_prefix = getattr(env_config, "PGVECTOR_TABLE_PREFIX", "")
        self.pgvector_extensions_schema = getattr(env_config, "PGVECTOR_EXTENSIONS_SCHEMA", "extensions")
        self.pgvector_pool_size = int(getattr(env_config, "PGVECTOR_POOL_SIZE", 1))
        self.pgvector_max_overflow = int(getattr(env_config, "PGVECTOR_MAX_OVERFLOW", 0))
        self.pgvector_pool_recycle = int(getattr(env_config, "PGVECTOR_POOL_RECYCLE", 300))
        self.ivfprobes = int(getattr(env_config, "IVFPROBES", 10))

        self.kb_top_k = max(1, int(getattr(env_config, "PGVECTOR_KB_TOP_K", getattr(env_config, "KB_TOP_K", 8))))
        self.kb_oversample_mult = max(1, int(getattr(env_config, "PGVECTOR_KB_OVERSAMPLE_MULT", 6)))
        self.category_boost_oversample_mult = max(
            1, int(getattr(env_config, "PGVECTOR_CATEGORY_BOOST_OVERSAMPLE_MULT", 2))
        )
        self.vector_cat_match_mult = float(getattr(env_config, "PGVECTOR_VECTOR_CATEGORY_MATCH_MULT", 0.85))
        self.vector_cat_both_mult = float(getattr(env_config, "PGVECTOR_VECTOR_CATEGORY_BOTH_MULT", 0.90))
        self.vector_cat_other_mult = float(getattr(env_config, "PGVECTOR_VECTOR_CATEGORY_OTHER_MULT", 1.00))
        self.fts_cat_match_mult = float(getattr(env_config, "PGVECTOR_FTS_CATEGORY_MATCH_MULT", 1.15))
        self.fts_cat_both_mult = float(getattr(env_config, "PGVECTOR_FTS_CATEGORY_BOTH_MULT", 1.10))
        self.fts_cat_other_mult = float(getattr(env_config, "PGVECTOR_FTS_CATEGORY_OTHER_MULT", 1.00))
        self.vector_weight = float(getattr(env_config, "PGVECTOR_VECTOR_WEIGHT", 0.45))
        self.fts_keywords_weight = float(getattr(env_config, "PGVECTOR_FTS_KEYWORDS_WEIGHT", 0.25))
        self.fts_summaries_weight = float(getattr(env_config, "PGVECTOR_FTS_SUMMARIES_WEIGHT", 0.30))
        self.rrf_k = int(getattr(env_config, "PGVECTOR_RRF_K", 60))
        self.search_phrase_topk_mult = max(1, int(getattr(env_config, "PGVECTOR_SEARCH_PHRASE_TOPK_MULT", 2)))
        self.acronym_topk_mult = max(1, int(getattr(env_config, "PGVECTOR_ACRONYM_TOPK_MULT", 2)))
        self.per_shard_top_k = max(1, int(getattr(env_config, "PGVECTOR_PER_SHARD_TOP_K", 25)))
        self.retrieval_top_k = max(1, int(getattr(env_config, "PGVECTOR_RETRIEVAL_TOP_K", 50)))
        self.content_snippet_chars = int(getattr(env_config, "PGVECTOR_CONTENT_SNIPPET_CHARS", 1800))
        self.ingest_workers = int(getattr(env_config, "PGVECTOR_INGEST_WORKERS", 6))
        self.kb_shard_max_size = int(getattr(env_config, "KB_SHARD_MAX_SIZE", 200))

        self.indexing_max_retries = int(getattr(env_config, "INDEXING_MAX_RETRIES", 5))
        self.indexing_retry_delay_seconds = float(
            getattr(env_config, "INDEXING_RETRY_DELAY_SECONDS", 2)
        )
        self.kb_validation_throttle_seconds = float(getattr(env_config, "KB_VALIDATION_THROTTLE_SECONDS", 0.0))
        self.openai_max_retries = int(getattr(env_config, "OPENAI_MAX_RETRIES", 6))
        self.openai_backoff_initial = float(getattr(env_config, "OPENAI_BACKOFF_INITIAL", 1.0))
        self.openai_backoff_max = float(getattr(env_config, "OPENAI_BACKOFF_MAX", 60.0))
        self.openai_backoff_multiplier = float(getattr(env_config, "OPENAI_BACKOFF_MULTIPLIER", 2.0))
        self.openai_backoff_jitter = float(getattr(env_config, "OPENAI_BACKOFF_JITTER", 0.25))
        self.openai_retry_base_seconds = float(getattr(env_config, "OPENAI_RETRY_BASE_SECONDS", 1.0))
        self.openai_retry_max_seconds = float(getattr(env_config, "OPENAI_RETRY_MAX_SECONDS", 30.0))
        self.openai_retry_jitter = float(getattr(env_config, "OPENAI_RETRY_JITTER", 0.25))

        self.rerank_top_n = int(getattr(env_config, "PGVECTOR_RERANK_TOP_N", 5))
        self.use_llm_rerank = bool(int(getattr(env_config, "PGVECTOR_USE_LLM_RERANK", 0)))
        self.llm_group_size = int(getattr(env_config, "PGVECTOR_LLM_GROUP_SIZE", 5))
        self.llm_rerank_timeout = float(getattr(env_config, "PGVECTOR_LLM_RERANK_TIMEOUT", 15.0))
        self.llm_rerank_model = str(
            getattr(env_config, "PGVECTOR_LLM_RERANK_MODEL", getattr(env_config, "LLM_AZURE_OPENAI_DEPLOYMENT", self.llm_model))
        )
        self.llm_small_pool_threshold = int(getattr(env_config, "PGVECTOR_LLM_SMALL_POOL_THRESHOLD", 10))
        self.content_snippet_in_prompt = int(getattr(env_config, "PGVECTOR_CONTENT_SNIPPET_IN_PROMPT", 400))
        self.use_ce_rerank = bool(int(getattr(env_config, "PGVECTOR_USE_CE_RERANK", 1)))
        self.ce_model_name = str(getattr(env_config, "PGVECTOR_CE_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        self.ce_batch_size = int(getattr(env_config, "PGVECTOR_CE_BATCH_SIZE", 16))
        self.ce_tie_dedup_enabled = bool(int(getattr(env_config, "PGVECTOR_CE_TIE_DEDUP_ENABLED", 1)))
        self.ce_tie_margin = float(getattr(env_config, "PGVECTOR_CE_TIE_MARGIN", 0.08))
        self.ce_tie_topk = int(getattr(env_config, "PGVECTOR_CE_TIE_TOPK", 2))
        self.ce_intra_family_ce = bool(int(getattr(env_config, "PGVECTOR_CE_INTRA_FAMILY_CE", 1)))
        self.rerank_strategy = str(getattr(env_config, "PGVECTOR_RERANK_STRATEGY", "ce")).strip().lower()
        self.top2_vs_rest_ratio = float(getattr(env_config, "KB_TOP2_VS_REST_RATIO", 1.5))
        self.keep_score_ratio = float(getattr(env_config, "KB_KEEP_SCORE_RATIO", 0.8))
        self.min_candidates = int(getattr(env_config, "KB_MIN_CANDIDATES", 2))
        self.max_candidates = int(getattr(env_config, "KB_MAX_CANDIDATES", 10))
        self.debug_rerank = bool(int(getattr(env_config, "PGVECTOR_DEBUG_RERANK", 1)))
        self.debug_rerank_topm = int(getattr(env_config, "PGVECTOR_DEBUG_RERANK_TOPM", 10))

    def _build_client(self, settings: AzureClientSettings) -> AzureOpenAI:
        kwargs = {
            "api_version": settings.api_version,
            "azure_endpoint": settings.azure_endpoint,
            "api_key": settings.api_key,
        }
        if settings.azure_deployment:
            kwargs["azure_deployment"] = settings.azure_deployment
        return AzureOpenAI(**kwargs)

    def get_llm_client(self) -> AzureOpenAI:
        if self._llm_client is None:
            with self._llm_lock:
                if self._llm_client is None:
                    self._llm_client = self._build_client(self.llm_settings)
        return self._llm_client

    def get_embedding_client(self) -> AzureOpenAI:
        if self._embedding_client is None:
            with self._embedding_lock:
                if self._embedding_client is None:
                    self._embedding_client = self._build_client(self.embedding_settings)
        return self._embedding_client

    def refresh_from_env(self) -> None:
        self.__init__()


indexing_config = IndexingConfig()
