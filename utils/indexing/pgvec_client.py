"""
PgVector Client for Supabase - Multi-Instance Support with Connection Pool Caching
"""

import time
import threading
import logging
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Optional, Dict

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from indexing_config import indexing_config

logger = logging.getLogger(__name__)


def _sanitize_database_url(database_url: str) -> str:
    """Normalize database URLs for SQLAlchemy+psycopg2.

    - Convert postgres:// -> postgresql:// (fixes: Can't load plugin: sqlalchemy.dialects:postgres)
    - Strip query args that psycopg2 rejects (e.g. pgbouncer=true)
    """
    if not database_url:
        return database_url

    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]

    try:
        u = urlparse(database_url)
        q = [(k, v) for (k, v) in parse_qsl(u.query, keep_blank_values=True) if k.lower() != "pgbouncer"]
        return urlunparse(u._replace(query=urlencode(q)))
    except Exception:
        return database_url


def _quote_ident(name: str) -> str:
    if name is None:
        raise ValueError("Identifier cannot be None")
    return '"' + str(name).replace('"', '""') + '"'


class PgVectorClient:
    """
    PostgreSQL connection pool manager with pgvector extension.
    Supports multiple instances for different databases.
    """

    def __init__(
        self,
        database_url: str,
        # ── CHANGED: conservative defaults to avoid exhausting Supabase pooler
        # and macOS FD limits under concurrent workloads.
        # pool_size=5 + max_overflow=2 → max 7 real connections per process.
        # pool_recycle=300 → recycle every 5 min (Supabase closes idle ~10 min).
        pool_size: int = 5,         # was 25
        max_overflow: int = 2,      # was 10
        schema: str = "public",
        table_prefix: str = "",
        pool_pre_ping: bool = True,
        pool_recycle: int = 300,    # was 3600
        echo: bool = False,
        extensions_schema: str = "extensions",  # Supabase commonly uses this
        set_search_path: bool = True,
    ):
        self.database_url = _sanitize_database_url(database_url)
        self.schema = schema or "public"
        self.table_prefix = table_prefix or ""
        self.extensions_schema = extensions_schema or "extensions"
        self.set_search_path = bool(set_search_path)

        self._engine = None
        self._session_factory = None
        self._lock = threading.Lock()
        self._pgvector_initialized = False
        self._vector_type_sql: Optional[str] = None  # cached resolved type name for SQL

        logger.info(
            f"Initializing PgVector client for schema='{self.schema}', "
            f"extensions_schema='{self.extensions_schema}', prefix='{self.table_prefix}', "
            f"pool_size={pool_size}, max_overflow={max_overflow}, pool_recycle={pool_recycle}s..."
        )

        self._engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            echo=echo,
        )

        @event.listens_for(self._engine, "connect")
        def set_pgvector_params(dbapi_conn, conn_record):
            cur = dbapi_conn.cursor()
            cur.execute(f"SET ivfflat.probes = {indexing_config.ivfprobes};")
            cur.close()

        if self.set_search_path:
            self._attach_search_path_hooks()

        self._session_factory = sessionmaker(bind=self._engine)

        logger.info("✓ PgVector client initialized successfully")

    # ---------------------- connection hooks ----------------------

    def _attach_search_path_hooks(self):
        """
        Ensure search_path is correct even under PgBouncer transaction pooling.

        We set it on:
        - connect: when a DBAPI connection is created
        - checkout: every time a connection is checked out of the pool
          (important when the underlying server connection can vary/reset)

        CHANGED: _on_checkout now calls connection_record.invalidate() and
        re-raises on failure instead of silently swallowing the error.
        A silent swallow lets SQLAlchemy hand out a dead/stale connection
        which later causes DNS-like OperationalErrors mid-script.
        """
        target_schema = self.schema
        ext_schema = self.extensions_schema

        search_path_sql = f"SET search_path TO {_quote_ident(target_schema)}, public, {_quote_ident(ext_schema)};"

        def _set_search_path(dbapi_connection):
            cur = None
            try:
                cur = dbapi_connection.cursor()
                cur.execute(search_path_sql)
            finally:
                if cur is not None:
                    try:
                        cur.close()
                    except Exception:
                        pass

        @event.listens_for(self._engine, "connect")
        def _on_connect(dbapi_connection, connection_record):
            try:
                _set_search_path(dbapi_connection)
            except Exception as e:
                logger.warning(f"Failed to set search_path on connect: {e}")

        @event.listens_for(self._engine, "checkout")
        def _on_checkout(dbapi_connection, connection_record, connection_proxy):
            # ── CHANGED: invalidate stale connections instead of silently ignoring
            # the error.  SQLAlchemy will discard this connection and get a fresh
            # one from the pool (or create a new one) for the caller.
            try:
                _set_search_path(dbapi_connection)
            except Exception as e:
                logger.warning(
                    f"Failed to set search_path on checkout — invalidating stale connection: {e}"
                )
                connection_record.invalidate()
                raise  # re-raise so SQLAlchemy replaces this connection

        logger.info(
            f"✓ Attached hooks to set search_path: {target_schema}, public, {ext_schema}"
        )

    # ---------------------- pgvector init / discovery ----------------------

    def _ensure_schema_exists(self, schema_name: str) -> None:
        schema_q = _quote_ident(schema_name)
        with self._engine.connect() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_q};"))
            conn.commit()

    def _vector_type_exists_anywhere(self) -> bool:
        q = text(
            """
            SELECT EXISTS (
              SELECT 1
              FROM pg_type t
              JOIN pg_namespace n ON n.oid = t.typnamespace
              WHERE t.typname = 'vector'
            ) AS exists;
            """
        )
        with self._engine.connect() as conn:
            return bool(conn.execute(q).scalar())

    def _resolve_vector_type_sql(self) -> str:
        """
        Resolve a schema-qualified SQL type name for pgvector casts, e.g.:

          - '"extensions"."vector"'  (Supabase common)
          - '"public"."vector"'
          - '"some_schema"."vector"'

        We do NOT rely on search_path or to_regtype('vector') because with
        transaction poolers (PgBouncer), search_path can be reset per-transaction.
        """
        ext_schema = (self.extensions_schema or "extensions").replace("'", "''")
        q = text(
            f"""
            SELECT n.nspname AS schema_name
            FROM pg_type t
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE t.typname = 'vector'
            ORDER BY
              CASE
                WHEN n.nspname = '{ext_schema}' THEN 0
                WHEN n.nspname = 'public' THEN 1
                ELSE 2
              END,
              n.nspname
            LIMIT 1;
            """
        )
        with self._engine.connect() as conn:
            schema_name = conn.execute(q).scalar()

        if not schema_name:
            return "vector"

        return f"{_quote_ident(schema_name)}.{_quote_ident('vector')}"

    def _initialize_pgvector_extension(self):
        with self._lock:
            if self._pgvector_initialized:
                return

            try:
                self._ensure_schema_exists(self.extensions_schema)

                with self._engine.connect() as conn:
                    try:
                        conn.execute(
                            text(
                                f"CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA {_quote_ident(self.extensions_schema)};"
                            )
                        )
                        conn.commit()
                        logger.info(f"✓ pgvector extension ensured (schema={self.extensions_schema})")
                    except Exception as e_schema:
                        conn.rollback()
                        try:
                            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                            conn.commit()
                            logger.info("✓ pgvector extension ensured (default schema)")
                        except Exception as e_plain:
                            conn.rollback()
                            raise RuntimeError(
                                "Failed to enable pgvector extension. "
                                "On Supabase, enable it in Dashboard → Database → Extensions.\n"
                                f"Schema attempt error: {e_schema}\n"
                                f"Plain attempt error: {e_plain}"
                            )

                if not self._vector_type_exists_anywhere():
                    raise RuntimeError(
                        "pgvector appears not available: type 'vector' does not exist.\n"
                        "Fix:\n"
                        "- Supabase dashboard → Database → Extensions → enable 'vector'\n"
                        "- Ensure you are connecting to the correct database\n"
                    )

                self._vector_type_sql = self._resolve_vector_type_sql()
                logger.info(f"✓ Resolved vector type for SQL casts: {self._vector_type_sql}")

                self._pgvector_initialized = True

            except Exception as e:
                logger.error(f"pgvector initialization failed: {e}")
                raise

    # ---------------------- public API ----------------------

    def get_session(self) -> Session:
        if not self._pgvector_initialized:
            self._initialize_pgvector_extension()
        if self._session_factory is None:
            raise RuntimeError("PgVectorClient not initialized properly.")
        return self._session_factory()

    def get_engine(self):
        if not self._pgvector_initialized:
            self._initialize_pgvector_extension()
        if self._engine is None:
            raise RuntimeError("PgVectorClient engine not initialized.")
        return self._engine

    def get_vector_type_sql(self) -> str:
        """Return SQL type name to use in casts: 'vector' or '"extensions".vector'"""
        if not self._pgvector_initialized:
            self._initialize_pgvector_extension()
        return self._vector_type_sql or "vector"

    def get_table_name(self, base_name: str) -> str:
        table_name = f"{self.table_prefix}{base_name}"
        return f'{_quote_ident(self.schema)}.{_quote_ident(table_name)}'

    def execute_query(self, query: str, params: Optional[dict] = None, retries: int = 3):
        # ── CHANGED: added retry + engine.dispose() on transient failures.
        # Previously a single connection error killed the call permanently.
        last_err = None
        for attempt in range(retries):
            try:
                with self.get_engine().connect() as conn:
                    result = conn.execute(text(query), params or {})
                    conn.commit()
                    return result
            except Exception as e:
                last_err = e
                logger.warning(f"execute_query attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)   # 1s, 2s
                    self._engine.dispose()     # flush stale pool connections
        raise last_err

    def close(self):
        if self._engine:
            logger.info("Closing PgVector connection pool...")
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._pgvector_initialized = False
            self._vector_type_sql = None
            logger.info("✓ Connection pool closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PgVectorConnectionPoolManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._pools: Dict[str, PgVectorClient] = {}
        self._pool_lock = threading.Lock()
        logger.info("✓ PgVectorConnectionPoolManager initialized")

    def _generate_pool_key(self, database_url: str, schema: str, table_prefix: str, extensions_schema: str) -> str:
        key_string = f"{database_url}|{schema}|{table_prefix}|{extensions_schema}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get_client(
        self,
        database_url: str,
        schema: str = "public",
        table_prefix: str = "",
        pool_size: int = 5,         # was 25
        max_overflow: int = 2,      # was 10
        pool_pre_ping: bool = True,
        pool_recycle: int = 300,    # was 3600
        echo: bool = False,
        extensions_schema: str = "extensions",
        set_search_path: bool = True,
    ) -> PgVectorClient:
        pool_key = self._generate_pool_key(database_url, schema, table_prefix, extensions_schema)

        if pool_key in self._pools:
            return self._pools[pool_key]

        with self._pool_lock:
            if pool_key in self._pools:
                return self._pools[pool_key]

            logger.info(f"Creating new connection pool: {pool_key}")
            client = PgVectorClient(
                database_url=database_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                schema=schema,
                table_prefix=table_prefix,
                pool_pre_ping=pool_pre_ping,
                pool_recycle=pool_recycle,
                echo=echo,
                extensions_schema=extensions_schema,
                set_search_path=set_search_path,
            )
            self._pools[pool_key] = client
            return client

    def close_all(self):
        with self._pool_lock:
            logger.info(f"Closing {len(self._pools)} connection pools...")
            for pool_key, client in self._pools.items():
                try:
                    client.close()
                except Exception as e:
                    logger.error(f"Error closing pool {pool_key}: {e}")
            self._pools.clear()
            logger.info("✓ All connection pools closed")


class PgVectorClientSingleton:
    _instance = None
    _client = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._initialize_client()
        return cls._instance

    @classmethod
    def _initialize_client(cls):
        logger.info("Initializing default PgVector client...")

        database_url = indexing_config.pgvector_database_url
        if not database_url:
            raise ValueError("PGVECTOR_DATABASE_URL or DATABASE_URL is not configured.")

        pool_size = indexing_config.pgvector_pool_size
        max_overflow = indexing_config.pgvector_max_overflow
        pool_recycle = indexing_config.pgvector_pool_recycle
        schema = indexing_config.pgvector_schema
        table_prefix = indexing_config.pgvector_table_prefix
        extensions_schema = indexing_config.pgvector_extensions_schema

        pool_manager = PgVectorConnectionPoolManager()
        cls._client = pool_manager.get_client(
            database_url=database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            schema=schema,
            table_prefix=table_prefix,
            extensions_schema=extensions_schema,
            set_search_path=True,
        )

    def __getattr__(self, name):
        if self._client is None:
            raise RuntimeError("PgVectorClientSingleton not initialized.")
        return getattr(self._client, name)


def get_pgvector_client():
    return PgVectorClientSingleton()


def get_shared_pgvector_client():
    """Return the process-wide shared PgVector client."""
    return get_pgvector_client()


def create_pgvector_client(database_url: str, **kwargs) -> PgVectorClient:
    pool_manager = PgVectorConnectionPoolManager()
    return pool_manager.get_client(database_url=database_url, **kwargs)


def close_all_pools():
    PgVectorConnectionPoolManager().close_all()
