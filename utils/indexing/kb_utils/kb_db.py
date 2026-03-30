import json
import logging
import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg2 import pool
from pgvector.psycopg2 import register_vector

from indexing_config import indexing_config


logger = logging.getLogger(__name__)

EMBEDDING_DIM = indexing_config.embedding_dim

OPENAI_MAX_RETRIES = indexing_config.openai_max_retries
OPENAI_RETRY_BASE_SECONDS = indexing_config.openai_retry_base_seconds
OPENAI_RETRY_MAX_SECONDS = indexing_config.openai_retry_max_seconds
OPENAI_RETRY_JITTER = indexing_config.openai_retry_jitter


def is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    http_status = getattr(exc, "http_status", None)
    if http_status == 429:
        return True
    code = getattr(exc, "code", None)
    if code in (429, "429", "rate_limit_exceeded", "RateLimitError"):
        return True
    msg = str(exc).lower()
    return (
        "too many requests" in msg
        or "rate limit" in msg
        or "ratelimit" in msg
        or ("429" in msg and ("rate" in msg or "limit" in msg or "too many" in msg))
    )


def exp_backoff_sleep(attempt: int) -> float:
    base = OPENAI_RETRY_BASE_SECONDS * (2 ** attempt)
    delay = min(base, OPENAI_RETRY_MAX_SECONDS)
    if OPENAI_RETRY_JITTER and OPENAI_RETRY_JITTER > 0:
        jitter = 1.0 + random.uniform(-OPENAI_RETRY_JITTER, OPENAI_RETRY_JITTER)
        delay *= max(0.0, jitter)
    return delay


def call_with_429_retries(fn, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc):
                raise
            if attempt >= OPENAI_MAX_RETRIES:
                raise
            sleep_s = exp_backoff_sleep(attempt)
            logger.warning(
                "OpenAI 429/rate-limit. Retry %d/%d in %.2fs. Error=%s",
                attempt + 1,
                OPENAI_MAX_RETRIES,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
    raise last_exc if last_exc else RuntimeError("OpenAI call failed")


def as_f32_list(vec: Any, *, dim: int) -> Optional[List[float]]:
    if vec is None:
        return None
    if isinstance(vec, (list, tuple)) and len(vec) == 0:
        return None
    try:
        lst = list(vec)
    except Exception as exc:
        raise ValueError(f"Embedding is not list-convertible: type={type(vec)}") from exc
    if len(lst) == 0:
        return None
    if len(lst) != dim:
        raise ValueError(f"Embedding length mismatch: expected {dim}, got {len(lst)}")
    return [float(x) for x in lst]


def quote_ident(name: str) -> str:
    if name is None:
        raise ValueError("Identifier cannot be None")
    return '"' + str(name).replace('"', '""') + '"'


class TraceabilityTracker:
    def __init__(self, log_file: str = "vva_indexing_traceability.json"):
        self.log_file = log_file
        self.records: List[Dict[str, Any]] = []
        self.errors_by_kbid: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def add_record(
        self,
        kbid: str,
        status: str,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        rec = {
            "kbid": kbid,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "details": details or {},
        }
        with self._lock:
            self.records.append(rec)
            if status in ("failed", "skipped"):
                self.errors_by_kbid.setdefault(kbid, []).append(
                    {"status": status, "reason": reason, "details": details or {}}
                )

    def print_summary(self):
        with self._lock:
            failed = [r for r in self.records if r["status"] == "failed"]
            skipped = [r for r in self.records if r["status"] == "skipped"]
            deleted = [r for r in self.records if r["status"] == "deleted"]
            success = [r for r in self.records if r["status"] == "success"]

            logger.info("\n" + "=" * 70)
            logger.info("TRACEABILITY SUMMARY")
            logger.info("=" * 70)
            logger.info("Total records: %d", len(self.records))
            logger.info("Success: %d", len(success))
            logger.info("Failed: %d", len(failed))
            logger.info("Skipped: %d", len(skipped))
            logger.info("Deleted: %d", len(deleted))
            if self.errors_by_kbid:
                logger.info("\nError details by KB (failed / skipped):")
                try:
                    logger.info(json.dumps(self.errors_by_kbid, indent=2, default=str))
                except Exception as exc:
                    logger.info("(Could not JSON-serialize error dict: %s)", exc)
            logger.info("=" * 70 + "\n")


class PgVectorPool:
    def __init__(self, database_url: str, minconn: int = 1, maxconn: int = 10):
        self.database_url = database_url
        self._pool = pool.ThreadedConnectionPool(minconn, maxconn, dsn=database_url)

    def getconn(self):
        conn = self._pool.getconn()
        try:
            register_vector(conn)
        except Exception:
            pass
        return conn

    def putconn(self, conn):
        self._pool.putconn(conn)

    def closeall(self):
        self._pool.closeall()
