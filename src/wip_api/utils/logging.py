import logging
import sys
import time
from pathlib import Path

from loguru import logger

from wip_analysis.utils.logging import InterceptHandler


def setup_logging(log_level: str = "INFO"):
    log_dir = Path("logs/backend_api")
    log_file = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.log"

    log_level = log_level.upper()

    logger.remove()
    logger.add(
        log_file, level=log_level, backtrace=True, diagnose=True, enqueue=False, rotation="5GB"
    )
    logger.add(sys.stdout, level=log_level)

    # Setup InterceptHandler for standard logging
    intercept_handler = InterceptHandler()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [intercept_handler]
    root_logger.setLevel(getattr(logging, log_level, logging.DEBUG))

    # Configure specific loggers
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "sqlalchemy"]:
        specific_logger = logging.getLogger(logger_name)
        specific_logger.handlers = [intercept_handler]
        specific_logger.setLevel(getattr(logging, log_level, logging.DEBUG))
        specific_logger.propagate = False

    logging.captureWarnings(True)
