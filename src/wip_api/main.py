import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from wip_analysis.data_loader.postgres_loader import (
    close_conn_pool,
    get_conn_pool,
    shutdown_async_writes,
)
from wip_analysis.data_loader.schema_validation import (
    validate_sql_schema_config,
    validate_sql_schema_config_against_database,
)
from wip_api.routes import (
    bottleneck,
    health,
    hwm_lot,
    hwm_prod,
    metrics,
    paths_stats,
    pid_date_out,
    stage_cap,
    stage_dur,
    stage_load,
    super_hot_lot,
)
from wip_api.utils.inflight import inflight_requests, request_route_key
from wip_api.utils.logging import setup_logging
from wip_api.utils.on_shut import cleanup

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_level = os.environ.get("WIP_ANALYSIS_LOG_LEVEL", "INFO")
    setup_logging(log_level)
    logger.info("WIP Analysis API starting up...")

    try:
        validate_sql_schema_config()
        logger.info("SQL schema config validated successfully")
    except Exception as e:
        logger.error("SQL schema config validation failed; aborting startup:\n{}", e)
        raise

    try:
        pool = get_conn_pool()
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database connection pool: {e}")
        raise

    try:
        validate_sql_schema_config_against_database(pool)
        logger.info("Database schema validated successfully")
    except Exception as e:
        logger.error(
            "Database schema validation failed; Should re-generate your table/views. "
            "Aborting startup:\n{}",
            e,
        )
        try:
            close_conn_pool()
        except Exception as close_error:
            logger.exception(f"Error closing database connection pool: {close_error}")
        raise

    yield

    logger.info("WIP Analysis API shutting down...")

    try:
        shutdown_async_writes()
    except Exception as e:
        logger.exception(f"Error draining async writer executor: {e}")

    try:
        close_conn_pool()
    except Exception as e:
        logger.exception(f"Error closing database connection pool: {e}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Error during CUDA cleanup: {e}")

    try:
        cleanup()

    except Exception as e:
        logger.warning(f"Error during garbage collection: {e}")

    logger.info("Cleaning up logging handlers...")
    logger.remove()


app = FastAPI(
    title="WIP Analysis API",
    description="API for WIP analysis and lot progress analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    start = time.perf_counter()
    response = None
    route_key = request_route_key(request)
    should_track = route_key not in {"GET /api/v1/metrics", "HEAD /api/v1/metrics"}

    if should_track:
        inflight_requests.increment(route_key)

    try:
        response = await call_next(request)
        return response
    finally:
        if should_track:
            inflight_requests.decrement(route_key)

        elapsed_ms = (time.perf_counter() - start) * 1000
        if response is not None:
            logger.info(
                "{} {} {} - took {:.1f} ms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )


# Routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

app.include_router(hwm_lot.router, prefix="/api/v1", tags=["hwm"])
app.include_router(pid_date_out.router, prefix="/api/v1", tags=["analysis"])
app.include_router(stage_cap.router, prefix="/api/v1", tags=["analysis"])
app.include_router(stage_dur.router, prefix="/api/v1", tags=["analysis"])
app.include_router(stage_load.router, prefix="/api/v1", tags=["analysis"])

app.include_router(super_hot_lot.router, prefix="/api/v1", tags=["hwm"])
app.include_router(paths_stats.router, prefix="/api/v1", tags=["analysis"])
app.include_router(hwm_prod.router, prefix="/api/v1", tags=["hwm"])
app.include_router(bottleneck.router, prefix="/api/v1", tags=["analysis"])
