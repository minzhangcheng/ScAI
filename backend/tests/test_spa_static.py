from pathlib import Path
import os
import sys
from importlib.metadata import version as package_version

import httpx
import pytest
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient

from backend.main import app, frontend_spa_fallback
from backend.routes.root import api_root, frontend_file_response, root, router as root_router


def _version_tuple(distribution: str) -> tuple[int, ...]:
    return tuple(
        int(part) for part in package_version(distribution).split(".")
        if part.isdigit()
    )


_OLD_ANYIO_PY313_FILE_IO = (
    sys.version_info >= (3, 13) and _version_tuple("anyio") < (4, 11)
)


def _contract_app() -> FastAPI:
    contract_app = FastAPI()
    contract_app.include_router(root_router)

    async def websocket_probe(websocket: WebSocket, session_id: str):
        await websocket.accept()
        await websocket.send_text(session_id)
        await websocket.close()

    contract_app.add_api_websocket_route("/ws/{session_id}", websocket_probe)
    contract_app.add_api_route(
        "/{spa_path:path}",
        frontend_spa_fallback,
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )
    return contract_app


def _route_paths_in_order(routes):
    for route in routes:
        candidates = getattr(route, "effective_candidates", None)
        if callable(candidates):
            yield from _route_paths_in_order(candidates())
            low_priority = getattr(route, "effective_low_priority_routes", None)
            if callable(low_priority):
                yield from _route_paths_in_order(low_priority())
            continue
        source = getattr(route, "original_route", route)
        path = (
            getattr(route, "path", None)
            or getattr(getattr(route, "starlette_route", None), "path", None)
            or getattr(source, "path", None)
        )
        if path:
            yield path


@pytest.mark.asyncio
async def test_production_spa_and_backend_route_priority(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<html>ScAI SPA</html>", encoding="utf-8")
    (assets / "app-ABCDEFGH.js").write_text("console.log('scai')", encoding="utf-8")
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    root_response = await root()
    assert isinstance(root_response, FileResponse)
    assert Path(root_response.path) == dist / "index.html"
    assert root_response.headers["cache-control"] == "no-cache"

    deep_link = await frontend_spa_fallback("login")
    assert isinstance(deep_link, FileResponse)
    assert Path(deep_link.path) == dist / "index.html"

    asset = await frontend_spa_fallback("assets/app-ABCDEFGH.js")
    assert isinstance(asset, FileResponse)
    assert Path(asset.path) == assets / "app-ABCDEFGH.js"
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"

    # Concrete backend routes are registered before the SPA catch-all and the
    # metadata handler remains JSON at /api.
    paths = list(_route_paths_in_order(app.routes))
    assert paths.index("/api") < paths.index("/{spa_path:path}")
    assert paths.index("/ws/{session_id}") < paths.index("/{spa_path:path}")
    assert (await api_root())["message"] == "ScAI API"

    # Backend namespaces and missing filename-like assets remain real 404s.
    for missing in ("api/not-a-route", "ws/not-an-http-route", "assets/missing.js"):
        with pytest.raises(HTTPException) as exc_info:
            await frontend_spa_fallback(missing)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.skipif(
    _OLD_ANYIO_PY313_FILE_IO,
    reason=(
        "this host combines Python 3.13 with AnyIO <4.11, whose worker-thread "
        "file I/O hangs; the supported project venv carries a newer AnyIO"
    ),
)
async def test_spa_real_asgi_http_contract(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<html>ScAI SPA</html>", encoding="utf-8")
    (assets / "app-ABCDEFGH.js").write_text("console.log('scai')", encoding="utf-8")
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    transport = httpx.ASGITransport(app=_contract_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        root_response = await client.get("/")
        assert root_response.status_code == 200
        assert "ScAI SPA" in root_response.text
        assert root_response.headers["cache-control"] == "no-cache"

        deep_link = await client.get("/results/session-1")
        assert deep_link.status_code == 200
        assert "ScAI SPA" in deep_link.text
        assert deep_link.headers["cache-control"] == "no-cache"

        head = await client.head("/results/session-1")
        assert head.status_code == 200
        assert head.content == b""

        asset = await client.get("/assets/app-ABCDEFGH.js")
        assert asset.status_code == 200
        assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"

        api_health = await client.get("/api/health")
        assert api_health.status_code == 200
        assert api_health.json()["status"] == "ok"

        unknown_api = await client.get("/api/not-a-route")
        assert unknown_api.status_code == 404
        assert not unknown_api.headers["content-type"].startswith("text/html")
        assert (await client.get("/ws/not-an-http-route")).status_code == 404
        assert (await client.get("/assets/missing.js")).status_code == 404
        assert (await client.post("/results/session-1")).status_code == 405


@pytest.mark.skipif(
    _OLD_ANYIO_PY313_FILE_IO,
    reason=(
        "this host combines Python 3.13 with AnyIO <4.11, whose worker-thread "
        "file I/O hangs; the supported project venv carries a newer AnyIO"
    ),
)
def test_spa_testclient_and_websocket_route_contract(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>ScAI SPA</html>", encoding="utf-8")
    (dist / "assets" / "app-ABCDEFGH.js").write_text("asset", encoding="utf-8")
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    with TestClient(_contract_app()) as client:
        assert client.get("/").status_code == 200
        assert client.head("/login").status_code == 200
        assert client.get("/assets/app-ABCDEFGH.js").status_code == 200
        unknown_api = client.get("/api/not-a-route")
        assert unknown_api.status_code == 404
        assert not unknown_api.headers["content-type"].startswith("text/html")
        assert client.post("/login").status_code == 405
        assert client.get("/api/health").json()["status"] == "ok"
        with client.websocket_connect("/ws/session") as websocket:
            assert websocket.receive_text() == "session"


def test_spa_file_resolution_rejects_escape(tmp_path, monkeypatch):
    from backend.routes.root import (
        _open_regular_beneath,
        _safe_relative_parts,
        frontend_file_response,
    )

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("index", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (dist / "escape.txt").symlink_to(outside)
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    assert frontend_file_response("../outside.txt") is None
    assert frontend_file_response("escape.txt") is None
    assert _safe_relative_parts("assets/bad\x00.js") is None
    assert _open_regular_beneath(dist, ("bad\x00.js",)) is None


@pytest.mark.asyncio
async def test_public_spa_route_returns_404_for_nul_path(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("index", encoding="utf-8")
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    transport = httpx.ASGITransport(app=_contract_app())
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test",
    ) as client:
        response = await client.get("/%00")

    assert response.status_code == 404


async def _consume_asgi_response(response, method: str = "GET"):
    messages = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await response(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": "/asset",
            "raw_path": b"/asset",
            "query_string": b"",
            "headers": [],
            # A pathsend-capable server must still receive bytes from the
            # descriptor, never an instruction to reopen the validated path.
            "extensions": {"http.response.pathsend": {}},
        },
        receive,
        send,
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return messages, body


def _run_descriptor_reads_inline(monkeypatch):
    """Avoid this host's Python 3.13/old-AnyIO worker-thread file-I/O hang."""
    import importlib

    root_routes = importlib.import_module("backend.routes.root")

    async def inline_run_sync(function, *args, **_kwargs):
        return function(*args)

    monkeypatch.setattr(root_routes.anyio.to_thread, "run_sync", inline_run_sync)


@pytest.mark.asyncio
async def test_spa_response_consumes_the_opened_inode_after_path_rebinding(
    tmp_path, monkeypatch
):
    _run_descriptor_reads_inline(monkeypatch)
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    asset = assets / "app-ABCDEFGH.js"
    trusted = b"console.log('trusted build')"
    attacker = b"console.log('rebound path')"
    asset.write_bytes(trusted)
    replacement = tmp_path / "replacement.js"
    replacement.write_bytes(attacker)
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    response = frontend_file_response("assets/app-ABCDEFGH.js")
    assert isinstance(response, FileResponse)
    os.replace(replacement, asset)

    messages, body = await _consume_asgi_response(response)

    assert body == trusted
    assert asset.read_bytes() == attacker
    assert not any(message["type"] == "http.response.pathsend" for message in messages)
    # Both spellings are valid JavaScript media types and browsers accept
    # either; which one appears depends on the mimetypes database the
    # deployment happens to carry, and this test is about the response
    # body and ASGI shape, not about that database.
    assert response.headers["content-type"].split(";")[0].strip() in {
        "text/javascript", "application/javascript",
    }
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


@pytest.mark.asyncio
async def test_spa_opened_response_preserves_head_and_rejects_internal_symlinks(
    tmp_path, monkeypatch
):
    _run_descriptor_reads_inline(monkeypatch)
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    asset = assets / "head-ABCDEFGH.js"
    asset.write_bytes(b"trusted head body")
    (assets / "linked-ABCDEFGH.js").symlink_to(asset)
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    response = frontend_file_response("assets/head-ABCDEFGH.js")
    assert response is not None
    messages, body = await _consume_asgi_response(response, method="HEAD")

    assert body == b""
    assert response.headers["content-length"] == str(len(b"trusted head body"))
    assert messages[-1] == {
        "type": "http.response.body", "body": b"", "more_body": False
    }
    assert "accept-ranges" not in response.headers
    assert frontend_file_response("assets/linked-ABCDEFGH.js") is None


@pytest.mark.asyncio
async def test_spa_response_does_not_depend_on_file_response_asgi_internals(
    tmp_path, monkeypatch
):
    _run_descriptor_reads_inline(monkeypatch)
    dist = tmp_path / "dist"
    dist.mkdir()
    asset = dist / "compat-ABCDEFGH.js"
    asset.write_bytes(b"minimum-starlette-compatible")
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(dist))

    response = frontend_file_response(asset.name)
    assert response is not None

    async def incompatible_file_response(*_args, **_kwargs):
        raise AssertionError("base FileResponse ASGI internals must not be called")

    monkeypatch.setattr(FileResponse, "__call__", incompatible_file_response)
    _messages, body = await _consume_asgi_response(response)

    assert body == b"minimum-starlette-compatible"
    assert response.headers["content-length"] == str(len(body))
    # Both spellings are valid JavaScript media types and browsers accept
    # either; which one appears depends on the mimetypes database the
    # deployment happens to carry, and this test is about the response
    # body and ASGI shape, not about that database.
    assert response.headers["content-type"].split(";")[0].strip() in {
        "text/javascript", "application/javascript",
    }


@pytest.mark.asyncio
async def test_spa_dist_ancestor_rebinding_keeps_the_opened_directory_chain(
    tmp_path, monkeypatch
):
    _run_descriptor_reads_inline(monkeypatch)
    trusted_parent = tmp_path / "trusted-tree"
    attacker_parent = tmp_path / "attacker-tree"
    trusted_dist = trusted_parent / "dist"
    attacker_dist = attacker_parent / "dist"
    trusted_dist.mkdir(parents=True)
    attacker_dist.mkdir(parents=True)
    asset_name = "ancestor-ABCDEFGH.js"
    (trusted_dist / asset_name).write_bytes(b"trusted ancestor")
    (attacker_dist / asset_name).write_bytes(b"rebound ancestor")
    parked_parent = tmp_path / "parked-trusted-tree"
    monkeypatch.setenv("SCAI_FRONTEND_DIST", str(trusted_dist))

    import importlib

    root_routes = importlib.import_module("backend.routes.root")
    original_open = root_routes.os.open
    rebound = False

    def open_and_rebind(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal rebound
        if dir_fd is None:
            fd = original_open(path, flags, mode)
        else:
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if str(path) == trusted_parent.name and not rebound:
            rebound = True
            trusted_parent.rename(parked_parent)
            attacker_parent.rename(trusted_parent)
        return fd

    monkeypatch.setattr(root_routes.os, "open", open_and_rebind)
    response = frontend_file_response(asset_name)
    assert response is not None
    _messages, body = await _consume_asgi_response(response)

    assert rebound is True
    assert body == b"trusted ancestor"
    assert (trusted_dist / asset_name).read_bytes() == b"rebound ancestor"
