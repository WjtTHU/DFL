import inspect
import logging
from typing import Any

import polars as pl
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def format_polars_df(
    df: pl.DataFrame,
    *,
    tbl_rows: int = -1,
    tbl_cols: int = -1,
    tbl_width_chars: int = -1,
    fmt_str_lengths: int = 100000,
) -> str:
    """Format a Polars DataFrame for a lazy log argument."""
    with pl.Config(
        tbl_rows=tbl_rows,
        tbl_cols=tbl_cols,
        tbl_width_chars=tbl_width_chars,
        fmt_str_lengths=fmt_str_lengths,
    ):
        return str(df)


def format_sequence(values: Any) -> str:
    """Format large sequences only when the enclosing log record is emitted."""
    return str(values)
