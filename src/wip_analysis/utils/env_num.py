import json
import os
from typing import TypeVar, cast

from loguru import logger

T = TypeVar("T")


def env_json(name: str, default: T) -> T:
    """Read and JSON-decode an environment variable.

    Invalid JSON falls back to ``default`` with a warning, matching ``env_num``.
    Callers remain responsible for validating the decoded value's schema.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        return cast(T, json.loads(raw))
    except (TypeError, ValueError):
        logger.warning("Invalid JSON for {}={!r}; using default {!r}", name, raw, default)
        return default


def env_num(name: str, default, min_value=None):
    """Read a number from the environment, matching the type of `default`.

    The return type follows `default`: an `int` default parses/returns `int`,
    a `float` default parses/returns `float`. `min_value` clamps the result;
    when left as `None` it defaults to `1` for int settings (e.g. worker counts,
    which must be >= 1) and `0.0` for float settings (e.g. a tolerance that can be
    disabled with 0). Invalid values fall back to `default` with a warning.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default

    caster = type(default)
    try:
        value = caster(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid value for {}={!r}; using default {}", name, raw, default)
        return default

    if min_value is None:
        min_value = 1 if isinstance(default, int) else 0.0
    return max(value, caster(min_value))
