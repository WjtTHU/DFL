import atexit
from concurrent.futures import Future, ThreadPoolExecutor, wait
from threading import Lock
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

import polars as pl
import yaml
from loguru import logger
from psycopg_pool import ConnectionPool

from wip_analysis.utils.env_num import env_num
from wip_analysis.utils.logging import format_polars_df

_connection_pool: Optional[ConnectionPool] = None
_writer_executor: Optional[ThreadPoolExecutor] = None
_writer_futures: set[Future[None]] = set()
_writer_futures_lock = Lock()


_POOL_MIN_SIZE = env_num("WIP_ANALYSIS_DB_POOL_MIN_SIZE", 16)
_POOL_MAX_SIZE = max(_POOL_MIN_SIZE, env_num("WIP_ANALYSIS_DB_POOL_MAX_SIZE", 96))
_WRITER_MAX_WORKERS = env_num("WIP_ANALYSIS_DB_WRITER_WORKERS", 64)


def get_conn_pool(config_path: str = "config/postgres.yaml") -> ConnectionPool:
    """
    Get or create a connection pool instance.
    Thread-safe singleton pattern for connection pool.
    """
    global _connection_pool

    if _connection_pool is None:
        config = load_db_config(config_path)

        user = quote(config["user"])
        password = quote(config["password"])
        host, port = config["host"], config["port"]
        dbname = config["dbname"]

        connection_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        _connection_pool = ConnectionPool(
            conninfo=connection_uri,
            min_size=_POOL_MIN_SIZE,  # Minimum connections to maintain
            max_size=_POOL_MAX_SIZE,  # Maximum connections allowed
            timeout=30.0,  # Timeout for getting a connection from pool (seconds)
            max_idle=300.0,  # Max time a connection can be idle (5 minutes)
            max_lifetime=3600.0,  # Max lifetime of a connection (1 hour)
        )
        logger.info(
            f"Connection pool created: min_size={_POOL_MIN_SIZE}, max_size={_POOL_MAX_SIZE}, "
            "timeout=30s, max_idle=300s, max_lifetime=3600s",
        )

    return _connection_pool


def close_conn_pool() -> None:
    """
    Close the connection pool. Should be called during application shutdown.
    """
    global _connection_pool

    if _connection_pool is not None:
        logger.info("Closing database connection pool...")
        _connection_pool.close()
        _connection_pool = None
        logger.info("Database connection pool closed successfully")


def load_db_config(config_path="config/postgres.yaml"):
    """Loads database configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "database" not in config:
            raise ValueError("YAML config file must contain a 'database' section.")
        logger.opt(lazy=True).debug("Database configuration loaded from {}", lambda: config_path)
        return config["database"]
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred loading configuration: {e}")
        raise


def execute_query(
    query: Any,
    params: Optional[Sequence[Any]] = None,
    config_path: str = "config/postgres.yaml",
) -> None:
    """
    Execute INSERT/UPDATE/DELETE queries without returning data.
    """
    pool = get_conn_pool(config_path)

    with pool.connection() as conn:
        try:
            with conn.cursor() as cursor:
                logger.debug("Executing write query:\n{}\nWith Parameters:\n{}", query, params)
                if params is None:
                    cursor.execute(query)
                else:
                    cursor.execute(query, params)
                conn.commit()
                logger.debug("Query executed successfully: {} rows affected", cursor.rowcount)
        except Exception as e:
            conn.rollback()
            logger.error(f"Query execution failed: {e}")
            raise


def execute_batch(
    query: Any,
    params_list: Sequence[Sequence[Any]],
    config_path: str = "config/postgres.yaml",
) -> None:
    """
    Execute batch INSERT/UPDATE/DELETE queries for multiple rows.
    Uses executemany for efficient batch operations.

    Args:
        query: SQL query with placeholders (e.g., INSERT INTO table VALUES (%s, %s))
        params_list: List of parameter tuples, one per row
        config_path: Path to database configuration file
    """
    pool = get_conn_pool(config_path)

    with pool.connection() as conn:
        try:
            with conn.cursor() as cursor:
                logger.debug("Executing batch query:\n{}\nWith {} rows", query, len(params_list))
                cursor.executemany(query, params_list)
                conn.commit()
                logger.debug(
                    "Batch query executed successfully: {} rows affected",
                    cursor.rowcount,
                )
        except Exception as e:
            conn.rollback()
            logger.error(f"Batch query execution failed: {e}")
            raise


def _get_writer_executor() -> ThreadPoolExecutor:
    """Lazily build the singleton ThreadPoolExecutor used for fire-and-forget writes."""
    global _writer_executor
    if _writer_executor is None:
        _writer_executor = ThreadPoolExecutor(
            max_workers=_WRITER_MAX_WORKERS,
            thread_name_prefix="pg-writer",
        )
        atexit.register(shutdown_async_writes)
        logger.info("Async DB writer executor created (max_workers={})", _WRITER_MAX_WORKERS)
    return _writer_executor


def _run_async_write(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Async DB write failed: {e}")


def _submit_async_write(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[None]:
    future: Future[None] = _get_writer_executor().submit(_run_async_write, fn, *args, **kwargs)
    with _writer_futures_lock:
        _writer_futures.add(future)
    future.add_done_callback(_discard_writer_future)
    return future


def _discard_writer_future(future: Future[None]) -> None:
    with _writer_futures_lock:
        _writer_futures.discard(future)


def execute_query_async(
    query: Any,
    params: Optional[Sequence[Any]] = None,
    config_path: str = "config/postgres.yaml",
) -> Future[None]:
    """Fire-and-forget version of execute_query. Errors are logged in the worker."""
    return _submit_async_write(execute_query, query, params, config_path)


def execute_batch_async(
    query: Any,
    params_list: Sequence[Sequence[Any]],
    config_path: str = "config/postgres.yaml",
) -> Future[None]:
    """Fire-and-forget version of execute_batch. Errors are logged in the worker."""
    return _submit_async_write(execute_batch, query, params_list, config_path)


def wait_for_async_writes(timeout: float | None = None) -> bool:
    """Wait for async writes submitted before this call to finish.

    Returns True if all observed writes completed before the timeout.
    """
    with _writer_futures_lock:
        futures = set(_writer_futures)

    if not futures:
        return True

    done, not_done = wait(futures, timeout=timeout)
    if not_done:
        logger.warning(
            "Timed out waiting for async DB writes to drain (done={}, pending={})",
            len(done),
            len(not_done),
        )
        return False
    return True


def shutdown_async_writes(wait: bool = True) -> None:
    """Drain the async writer executor. Call before close_conn_pool() so pending writes
    can still acquire connections. Safe to call more than once."""
    global _writer_executor
    if _writer_executor is None:
        return
    logger.info("Shutting down async writer executor (wait={})", wait)
    _writer_executor.shutdown(wait=wait)
    _writer_executor = None


def fetch_data(
    query: str,
    config_path: str = "config/postgres.yaml",
    *,
    schema_overrides: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """
    Fetches data from the PostgreSQL database using a given SQL query.

    Args:
        query: The SQL query to execute. **Shouldn't contain blank line in the end or would cause error!!!**
        config_path: Path to the configuration YAML file.
        schema_overrides: Optional Polars schema overrides for columns with nullable leading rows.

    Returns:
        A polars DataFrame containing the query results. contains:
            - lot_id: str, unique identifier for each lot
            - created_at: datetime, timestamp of the record
            - product_id: str, identifier for the product
            - lot_status: str, status of the lot (e.g., "Run", "Hold", "Completed")
            - stage: str, current stage of the lot in the production process
            - current_wafer_qty: int, number of wafers currently in the lot
    """
    pool = get_conn_pool(config_path)
    try:
        logger.debug("Executing query via connection pool:\n{}", query)

        with pool.connection() as conn:
            df = pl.read_database(
                query=query,
                connection=conn,
                schema_overrides=schema_overrides,
            )

        # Convert timestamp of snapshot if necessary
        if "created_at" in df.columns and df["created_at"].dtype != pl.Datetime:
            df = df.with_columns(df["created_at"].cast(pl.Datetime).alias("created_at"))

        logger.debug("Successfully fetched {} rows via connection pool.", len(df))
        logger.opt(lazy=True).trace("Fetched DataFrame:\n{}", lambda: format_polars_df(df))
        return df

    except Exception as e:
        logger.error(f"An unexpected error occurred during data fetching via connection pool: {e}")
        raise


# Example usage
if __name__ == "__main__":
    try:
        # Make sure config/config.yaml exists and is correctly populated
        # Ensure your config.yaml has: user, password, host, port, dbname under 'database'
        from wip_analysis.data_loader.sql_template import render_sql

        test_query = render_sql("get_full_data.sql.j2")
        data = fetch_data(test_query)
        print("Sample data fetched:")
        print(data.head())
    except Exception as e:
        print(f"Failed to fetch data: {e}")
