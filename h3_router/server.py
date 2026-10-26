from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}
MAX_REQUEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RouterSettings:
    fl2va_urls: tuple[str, ...]
    ref2va_urls: tuple[str, ...]
    host: str = "0.0.0.0"
    port: int = 8080
    connect_timeout: float = 10

    @classmethod
    def from_env(cls) -> "RouterSettings":
        def urls(name: str) -> tuple[str, ...]:
            return tuple(item.strip().rstrip("/") for item in os.getenv(name, "").split(",") if item.strip())

        fl2va = urls("H3_FL2VA_SERVICE_URLS")
        return cls(
            fl2va_urls=fl2va,
            ref2va_urls=urls("H3_REF2VA_SERVICE_URLS"),
            host=os.getenv("H3_ROUTER_HOST", "0.0.0.0"),
            port=int(os.getenv("H3_ROUTER_PORT", "8080")),
            connect_timeout=float(os.getenv("H3_ROUTER_CONNECT_TIMEOUT_SECONDS", "10")),
        )


class BackendPool:
    def __init__(self, urls: tuple[str, ...]):
        self.urls = urls
        self._counter = itertools.count()

    def candidates(self) -> list[str]:
        if not self.urls:
            return []
        start = next(self._counter) % len(self.urls)
        return list(self.urls[start:] + self.urls[:start])


def _error(status: int, kind: str, message: str):
    return JSONResponse(
        {"type": "error", "error": {"type": kind, "message": message, "http_code": str(status)}, "request_id": uuid4().hex},
        status_code=status,
    )


def _is_ref_request(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("role") in {"reference_image", "reference_video", "reference_audio"}
        for item in content
    )


def create_app(settings: RouterSettings | None = None) -> FastAPI:
    settings = settings or RouterSettings.from_env()
    fl_pool = BackendPool(settings.fl2va_urls)
    ref_pool = BackendPool(settings.ref2va_urls)
    shared_pool = BackendPool(tuple(dict.fromkeys(settings.fl2va_urls + settings.ref2va_urls)))
    app = FastAPI(title="MiniMax H3 Stateless Router", version="0.1.0")

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "fl2va_targets": len(settings.fl2va_urls), "ref2va_targets": len(settings.ref2va_urls)}

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS"])
    async def proxy(path: str, request: Request):
        try:
            declared_length = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            return _error(400, "bad_request_error", "invalid Content-Length")
        if declared_length < 0 or declared_length > MAX_REQUEST_BYTES:
            return _error(400, "bad_request_error", "request body exceeds 64 MB")
        body_buffer = bytearray()
        async for chunk in request.stream():
            body_buffer.extend(chunk)
            if len(body_buffer) > MAX_REQUEST_BYTES:
                return _error(400, "bad_request_error", "request body exceeds 64 MB")
        body = bytes(body_buffer)
        is_create = request.method == "POST" and path == "v2/video_generation"
        if is_create:
            pool = ref_pool if _is_ref_request(body) else fl_pool
            if pool is ref_pool and not settings.ref2va_urls:
                return _error(501, "not_implemented_error", "ref2va is not enabled")
        else:
            pool = shared_pool
        candidates = pool.candidates()
        if not candidates:
            return _error(503, "server_error", "no backend service is configured")

        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_BY_HOP}
        headers["x-forwarded-proto"] = request.headers.get(
            "x-forwarded-proto", request.url.scheme
        ).split(",", 1)[0].strip()
        headers["x-forwarded-host"] = request.headers.get(
            "x-forwarded-host", request.headers.get("host", "")
        ).split(",", 1)[0].strip()
        client_host = request.client.host if request.client else ""
        forwarded_for = request.headers.get("x-forwarded-for", "").strip()
        headers["x-forwarded-for"] = (
            f"{forwarded_for}, {client_host}"
            if forwarded_for and client_host
            else forwarded_for or client_host
        )
        query = request.url.query
        last_error = "backend unavailable"
        client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=settings.connect_timeout), follow_redirects=False)
        response = None
        for backend in candidates:
            url = f"{backend}/{path}" + (f"?{query}" if query else "")
            try:
                upstream_request = client.build_request(request.method, url, headers=headers, content=body)
                response = await client.send(upstream_request, stream=True)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = str(exc)
                continue
        if response is None:
            await client.aclose()
            return _error(503, "server_error", last_error)

        response_headers = {
            key: value for key, value in response.headers.items()
            if key.lower() not in HOP_BY_HOP
        }

        async def stream():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(stream(), status_code=response.status_code, headers=response_headers)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = RouterSettings.from_env()
    uvicorn.run(create_app(settings), host=args.host or settings.host, port=args.port or settings.port)


if __name__ == "__main__":
    main()
