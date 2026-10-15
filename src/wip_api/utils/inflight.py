from __future__ import annotations

from threading import Lock
from typing import Any

from starlette.requests import Request
from starlette.routing import Match

UNMATCHED_ROUTE = "<unmatched>"
UNKNOWN_ROUTE = "<unknown>"


class InflightRequestTracker:
    """Track in-flight request counts by route key within the current process."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = Lock()

    def increment(self, route_key: str) -> None:
        with self._lock:
            self._counts[route_key] = self._counts.get(route_key, 0) + 1

    def decrement(self, route_key: str) -> None:
        with self._lock:
            count = self._counts.get(route_key, 0)
            if count <= 1:
                self._counts.pop(route_key, None)
                return
            self._counts[route_key] = count - 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            by_route = dict(sorted(self._counts.items()))

        return {
            "total": sum(by_route.values()),
            "by_route": by_route,
        }

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()


inflight_requests = InflightRequestTracker()


def request_route_key(request: Request) -> str:
    """Return a stable METHOD + route-template key without using the raw path."""
    return f"{request.method.upper()} {_route_template(request)}"


def _route_template(request: Request) -> str:
    app = request.app
    scope = request.scope

    for route in app.routes:
        matches = getattr(route, "matches", None)
        if matches is None:
            continue

        match, _ = matches(scope)
        if match == Match.FULL:
            return _route_path(route)

    return UNMATCHED_ROUTE


def _route_path(route: Any) -> str:
    path_format = getattr(route, "path_format", None)
    if path_format:
        return str(path_format)

    path = getattr(route, "path", None)
    if path:
        return str(path)

    return UNKNOWN_ROUTE
