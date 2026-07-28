"""Prometheus instrumentation for the FastAPI gateway."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response as StarletteResponse

REQUEST_COUNT = Counter(
    "call_center_api_requests_total",
    "Total HTTP requests received by the serving gateway.",
    ["method", "path", "status_code"],
)

REQUEST_ERRORS = Counter(
    "call_center_api_errors_total",
    "Total HTTP requests that completed with 5xx responses.",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "call_center_api_request_latency_seconds",
    "HTTP request latency for the serving gateway.",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)


def normalized_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or request.url.path


def instrument_app(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = normalized_path(request)
        method = request.method
        start = time.perf_counter()
        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
            REQUEST_COUNT.labels(method=method, path=path, status_code=status_code).inc()
            if status_code.startswith("5"):
                REQUEST_ERRORS.labels(method=method, path=path, status_code=status_code).inc()

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> StarletteResponse:
        return StarletteResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
