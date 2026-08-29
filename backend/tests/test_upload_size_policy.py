"""Multipart admission limits that run before form parsing and authentication."""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from backend.config import MAX_UPLOAD_SIZE
from backend.middleware.upload_size import (
    MULTIPART_ENVELOPE_ALLOWANCE,
    UploadSizeLimitMiddleware,
)


def _sink_app(limit: int, *, with_base_middleware: bool = False):
    reached: list[bytes] = []
    app = FastAPI()

    @app.api_route("/sink", methods=["POST", "PUT", "PATCH"])
    async def sink(request: Request):
        reached.append(await request.body())
        return PlainTextResponse("ok")

    if with_base_middleware:
        async def passthrough(request, call_next):
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=passthrough)
    app.add_middleware(UploadSizeLimitMiddleware, max_body_bytes=limit)
    return TestClient(app), reached


def _multipart_headers(**extra: str) -> dict[str, str]:
    return {"Content-Type": "multipart/form-data; boundary=test", **extra}


def test_declared_oversized_multipart_is_rejected_before_inner_app_runs():
    client, reached = _sink_app(32)

    response = client.post(
        "/sink",
        headers=_multipart_headers(**{"Content-Length": "64"}),
        content=b"small",
    )

    assert response.status_code == 413
    assert reached == []


def test_chunked_multipart_overflow_across_base_http_middleware_is_413():
    client, reached = _sink_app(32, with_base_middleware=True)

    def chunks():
        for _ in range(4):
            yield b"x" * 16

    response = client.post(
        "/sink",
        headers=_multipart_headers(),
        content=chunks(),
    )

    assert response.status_code == 413
    assert reached == []


def test_multipart_body_at_limit_passes_unchanged():
    client, reached = _sink_app(32)
    body = b"x" * 32

    response = client.post("/sink", headers=_multipart_headers(), content=body)

    assert response.status_code == 200
    assert reached == [body]


def test_non_multipart_body_above_limit_is_not_rejected():
    client, reached = _sink_app(32)
    body = b"x" * 64

    response = client.post(
        "/sink",
        headers={"Content-Type": "application/octet-stream"},
        content=body,
    )

    assert response.status_code == 200
    assert reached == [body]


def test_full_app_multipart_413_retains_cors_headers():
    from backend.main import app

    response = TestClient(app).post(
        "/api/session/missing/upload/data",
        headers={
            **_multipart_headers(
                **{
                    "Content-Length": str(
                        MAX_UPLOAD_SIZE + MULTIPART_ENVELOPE_ALLOWANCE + 1
                    )
                }
            ),
            "Origin": "https://example.test",
        },
        content=b"small",
    )

    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "https://example.test"
