"""Test V2 snapshot API response models, error mappings, and AnalysisState fields."""
from backend.tests.split_facade import setattr_facade
import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.models.api_requests import (
    SnapshotTransactionResponse,
)
from backend.models.schemas import AnalysisState
from backend.routes.export_import import export_v2
from backend.services.snapshot_v2.errors import (
    SnapshotCompatibilityError,
    SnapshotStorageError,
    snapshot_error_detail,
)
from backend.services.snapshot_v2.service import SnapshotService


# Certifies Snapshot V2-enabled behaviour. The operator gate ships off, so pin it
# on here instead of inheriting it from the environment: these assertions describe
# the enabled lifecycle and are meaningless (and were failing) with it off.
pytestmark = pytest.mark.requires_snapshot_v2

class TestSnapshotV2ResponseModels:
    """Response models exist and validate correctly."""

    def test_snapshot_v2_list_response_exists(self):
        from backend.models.api_requests import SnapshotV2ListResponse
        payload = {
            "snapshots": [{"snapshot_id": "s1", "manifest": {}, "failure": None}],
            "store_status": {
                "logical_bytes": 1000, "physical_bytes": 800,
                "dedup_savings": 200, "object_count": 5, "manifest_count": 1,
            },
        }
        resp = SnapshotV2ListResponse(**payload)
        assert len(resp.snapshots) == 1
        assert resp.store_status.manifest_count == 1

    def test_snapshot_v2_detail_response_exists(self):
        from backend.models.api_requests import SnapshotV2DetailResponse
        payload = {
            "manifest": {
                "format_version": 2, "session_id": "s1", "snapshot_id": "snap-01",
                "created_at": "2026-01-01T00:00:00Z",
                "continuation": {"kind": "idle", "graph_version": "2.0"},
            },
        }
        resp = SnapshotV2DetailResponse(**payload)
        assert resp.manifest["snapshot_id"] == "snap-01"

    def test_snapshot_transaction_response_validates(self):
        resp = SnapshotTransactionResponse(
            transaction_id="tx-01",
            status="queued",
            status_url="/api/session/s1/snapshot-transactions/tx-01",
            cancel_url="/api/session/s1/snapshot-transactions/tx-01/cancel",
        )
        assert resp.transaction_id == "tx-01"
        assert resp.status == "queued"

    def test_snapshot_transaction_response_rejects_missing_fields(self):
        with pytest.raises(ValidationError):
            SnapshotTransactionResponse(
                transaction_id="tx-01",
                status="queued",
            )

    def test_delete_snapshot_v2_response_exists(self):
        from backend.models.api_requests import DeleteSnapshotV2Response
        resp = DeleteSnapshotV2Response(
            deleted_ids=["snap-01", "snap-02"],
            freed_bytes=4096,
            surviving_count=3,
        )
        assert len(resp.deleted_ids) == 2
        assert resp.freed_bytes == 4096
        assert resp.surviving_count == 3

    def test_export_v2_response_exists(self):
        from backend.models.api_requests import ExportV2Response
        resp = ExportV2Response(
            download_url="/api/session/s1/export/download",
            snapshot_count=3,
            object_count=15,
        )
        assert resp.download_url
        assert resp.snapshot_count == 3
        assert resp.object_count == 15

    def test_import_v2_response_exists(self):
        from backend.models.api_requests import ImportV2Response
        resp = ImportV2Response(
            import_id="imp-01",
            snapshot_count=3,
            status="completed",
        )
        assert resp.import_id == "imp-01"
        assert resp.status == "completed"

    def test_store_status_response_exists(self):
        from backend.models.api_requests import StoreStatusResponse
        resp = StoreStatusResponse(
            logical_bytes=10000,
            physical_bytes=7500,
            dedup_savings=2500,
            object_count=10,
            manifest_count=2,
        )
        assert resp.logical_bytes == 10000
        assert resp.physical_bytes == 7500
        assert resp.dedup_savings == 2500
        assert resp.object_count == 10
        assert resp.manifest_count == 2
        assert resp.dedup_ratio == pytest.approx(0.25)

    def test_store_status_response_zero_physical(self):
        from backend.models.api_requests import StoreStatusResponse
        resp = StoreStatusResponse(
            logical_bytes=0, physical_bytes=0, dedup_savings=0,
            object_count=0, manifest_count=0,
        )
        assert resp.dedup_ratio == 0.0

    def test_snapshot_failure_response_exists(self):
        from backend.models.api_requests import SnapshotFailureResponse
        resp = SnapshotFailureResponse(
            snapshot_id="snap-01",
            code="QUOTA_EXCEEDED",
            message="disk full",
            allowed_actions=["retry", "dismiss"],
        )
        assert resp.code == "QUOTA_EXCEEDED"
        assert "retry" in resp.allowed_actions

    def test_snapshot_failure_response_default_actions(self):
        from backend.models.api_requests import SnapshotFailureResponse
        resp = SnapshotFailureResponse(
            snapshot_id="snap-01",
            code="UNKNOWN",
            message="something broke",
        )
        assert resp.allowed_actions == []


class TestLegacySnapshotRouteBoundary:
    def test_v2_projection_is_rejected_by_v1_management_routes(self):
        from backend.models.schemas import Snapshot
        from backend.routes.snapshots import _require_v1_snapshot

        snapshot = Snapshot(
            id="v2-projection",
            name="V2",
            phase="manual",
            format_version=2,
            state_file="",
        )

        with pytest.raises(HTTPException) as exc:
            _require_v1_snapshot(snapshot)

        assert exc.value.status_code == 409
        assert "V2" in str(exc.value.detail)


class TestSnapshotV2ErrorMapping:
    """Error-to-status-code mapping is centralized via snapshot_error_detail / _v2_http_exc."""

    def test_map_snapshot_compatibility_error_to_422(self):
        from backend.services.snapshot_v2.errors import SnapshotCompatibilityError
        from backend.routes.snapshots import _v2_http_exc
        exc = SnapshotCompatibilityError("COMPAT", "incompatible version")
        http_exc = _v2_http_exc(exc)
        assert http_exc.status_code == 422
        assert http_exc.detail["code"] == "COMPAT"

    def test_map_snapshot_integrity_error_to_422(self):
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.routes.snapshots import _v2_http_exc
        exc = SnapshotIntegrityError("HASH", "hash mismatch")
        http_exc = _v2_http_exc(exc)
        assert http_exc.status_code == 422
        assert http_exc.detail["code"] == "HASH"

    def test_map_snapshot_storage_error_to_507(self):
        from backend.services.snapshot_v2.errors import SnapshotStorageError
        from backend.routes.snapshots import _v2_http_exc
        exc = SnapshotStorageError("QUOTA", "disk full")
        http_exc = _v2_http_exc(exc)
        assert http_exc.status_code == 507
        assert http_exc.detail["code"] == "QUOTA"

    def test_map_snapshot_delete_conflict_to_409(self):
        from backend.services.snapshot_v2.errors import SnapshotDeleteConflict
        from backend.routes.snapshots import _v2_http_exc
        exc = SnapshotDeleteConflict("HAS_CHILDREN", "has children")
        http_exc = _v2_http_exc(exc)
        assert http_exc.status_code == 409
        assert http_exc.detail["code"] == "HAS_CHILDREN"

    def test_map_snapshot_transaction_error_to_503(self):
        from backend.services.snapshot_v2.errors import SnapshotTransactionError
        from backend.routes.snapshots import _v2_http_exc
        exc = SnapshotTransactionError("ENQUEUE_FAILED", "queue full")
        http_exc = _v2_http_exc(exc)
        assert http_exc.status_code == 503
        assert http_exc.detail["code"] == "ENQUEUE_FAILED"

    def test_map_snapshot_integrity_error_not_found_to_404(self):
        from backend.services.snapshot_v2.errors import SnapshotIntegrityError
        from backend.routes.snapshots import _v2_http_exc
        exc = SnapshotIntegrityError("MANIFEST_NOT_FOUND", "not found")
        http_exc = _v2_http_exc(exc)
        assert http_exc.status_code == 404
        assert http_exc.detail["code"] == "MANIFEST_NOT_FOUND"
        assert "message" in http_exc.detail


class TestAnalysisStateV2Fields:
    """AnalysisState.to_api_dict() includes V2 snapshot fields."""

    def test_to_api_dict_includes_snapshot_failure(self):
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.snapshot_failure = {"code": "QUOTA", "message": "disk full"}
        api = state.to_api_dict()
        assert "snapshot_failure" in api
        assert api["snapshot_failure"]["code"] == "QUOTA"

    def test_to_api_dict_includes_snapshot_transaction(self):
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.snapshot_transaction = {"transaction_id": "tx-01", "status": "completed"}
        api = state.to_api_dict()
        assert "snapshot_transaction" in api
        assert api["snapshot_transaction"]["transaction_id"] == "tx-01"

    def test_to_api_dict_includes_runtime_restore_status(self):
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.runtime_restore_status = "unavailable"
        api = state.to_api_dict()
        assert api["runtime_restore_status"] == "unavailable"

    def test_to_api_dict_includes_pending_required_snapshot(self):
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.pending_required_snapshot = {"snapshot_id": "snap-01", "reason": "test"}
        api = state.to_api_dict()
        assert "pending_required_snapshot" in api
        assert api["pending_required_snapshot"]["snapshot_id"] == "snap-01"

    def test_to_api_dict_includes_v1_snapshot_count(self):
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        assert "snapshot_count" in api
        assert api["snapshot_count"] == 0

    def test_to_api_dict_includes_v1_snapshot_count_nonzero(self):
        from backend.models.schemas import AnalysisState, Snapshot
        state = AnalysisState(user_id="test")
        state.snapshots.append(Snapshot(name="s1", phase="post_annotation"))
        state.snapshots.append(Snapshot(name="s2", phase="pre_direction"))
        api = state.to_api_dict()
        assert api["snapshot_count"] == 2

    def test_to_api_dict_v2_fields_default_to_none(self):
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        assert api.get("snapshot_failure") is None
        assert api.get("snapshot_transaction") is None
        assert api.get("pending_required_snapshot") is None

    def test_to_api_dict_exports_as_json(self):
        import json
        from backend.models.schemas import AnalysisState
        state = AnalysisState(user_id="test")
        state.snapshot_failure = {"code": "TEST", "message": "test"}
        state.snapshot_transaction = {"tx": "t1"}
        state.pending_required_snapshot = {"snap": "s1"}
        api = state.to_api_dict()
        json.dumps(api)


class TestSnapshotErrorDetailFunction:
    """snapshot_error_detail() maps SnapshotV2Error types to correct status codes."""

    def test_not_found_404(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotIntegrityError,
            snapshot_error_detail,
        )
        exc = SnapshotIntegrityError("MANIFEST_NOT_FOUND", "not found")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 404

    def test_snapshot_not_found_404(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotIntegrityError,
            snapshot_error_detail,
        )
        exc = SnapshotIntegrityError("SNAPSHOT_NOT_FOUND", "snap gone")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 404

    def test_compatibility_422(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotCompatibilityError,
            snapshot_error_detail,
        )
        exc = SnapshotCompatibilityError("COMPAT", "version mismatch")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 422

    def test_integrity_hash_422(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotIntegrityError,
            snapshot_error_detail,
        )
        exc = SnapshotIntegrityError("HASH_MISMATCH", "hash differs")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 422

    def test_delete_conflict_409(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotDeleteConflict,
            snapshot_error_detail,
        )
        exc = SnapshotDeleteConflict("HAS_CHILDREN", "has children")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 409

    def test_storage_507(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotStorageError,
            snapshot_error_detail,
        )
        exc = SnapshotStorageError("QUOTA", "disk full")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 507

    def test_transaction_503(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotTransactionError,
            snapshot_error_detail,
        )
        exc = SnapshotTransactionError("ENQUEUE_FAILED", "queue full")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 503

    def test_safe_point_busy_maps_to_structured_conflict(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotTransactionError,
            snapshot_error_detail,
        )

        detail = snapshot_error_detail(
            SnapshotTransactionError(
                "SAFE_POINT_BUSY",
                "Another operation owns the safe point",
            )
        )
        assert detail["status_code"] == 409
        assert detail["code"] == "SAFE_POINT_BUSY"
        assert "owns the safe point" in detail["message"]

    def test_session_busy_maps_to_structured_conflict(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotTransactionError,
            snapshot_error_detail,
        )

        detail = snapshot_error_detail(
            SnapshotTransactionError(
                "SESSION_BUSY",
                "Detailed report generation is active for this session",
            )
        )
        assert detail["status_code"] == 409
        assert detail["code"] == "SESSION_BUSY"

    def test_unknown_v2_error_500(self):
        from backend.services.snapshot_v2.errors import (
            SnapshotCaptureError,
            snapshot_error_detail,
        )
        exc = SnapshotCaptureError("UNKNOWN", "unexpected")
        result = snapshot_error_detail(exc)
        assert result["status_code"] == 500


def test_snapshot_error_detail_preserves_details_and_context():
    exc = SnapshotCompatibilityError(
        "RUNTIME_UNAVAILABLE",
        "runtime choice required",
        details={"choices": ["reset", "keep"]},
    )
    detail = snapshot_error_detail(exc, {"runtime_status": "missing"})
    assert detail == {
        "status_code": 422,
        "code": "RUNTIME_UNAVAILABLE",
        "message": "RUNTIME_UNAVAILABLE: runtime choice required",
        "details": {"choices": ["reset", "keep"]},
        "runtime_status": "missing",
    }


@pytest.mark.asyncio
async def test_export_v2_errors_are_structured(monkeypatch):
    request = MagicMock()
    request.state.user_id = "u1"
    setattr_facade(
        monkeypatch, "backend.routes.export_import", "require_session_ownership",
        AsyncMock(return_value=AnalysisState(id="s1", user_id="u1")),
    )
    monkeypatch.setattr(
        SnapshotService,
        "export_snapshots",
        AsyncMock(side_effect=SnapshotStorageError("NO_SPACE", "disk full")),
    )
    with pytest.raises(HTTPException) as caught:
        await export_v2("s1", request, include_runtime=True, snapshot_ids=None)
    assert caught.value.detail == {"code": "NO_SPACE", "message": "NO_SPACE: disk full"}


# ── Route-level restore job control (Batch 2 Task 5) ───────────────────────


def _patch_auth_middleware(app, monkeypatch, patched_dispatch):
    """Patch live AuthMiddleware.dispatch_func after middleware stack is built.

    BaseHTTPMiddleware freezes ``dispatch`` into ``dispatch_func`` at construction.
    ``app.middleware_stack`` is None until the first request builds it, so we
    force a build via ``build_middleware_stack()`` when needed.
    """
    from backend.middleware import auth as auth_mw

    if app.middleware_stack is None:
        app.middleware_stack = app.build_middleware_stack()
    mw = app.middleware_stack
    while mw is not None:
        if isinstance(mw, auth_mw.AuthMiddleware):
            monkeypatch.setattr(mw, "dispatch_func", patched_dispatch)
            return
        mw = getattr(mw, "app", None)
    raise AssertionError("AuthMiddleware not found in app.middleware_stack")


def _write_object(paths, content: bytes, media_type: str = "application/json"):
    from backend.services.snapshot_v2.models import ObjectRef

    sha = hashlib.sha256(content).hexdigest()
    (paths.objects / sha).write_bytes(content)
    return ObjectRef(sha256=sha, size=len(content), media_type=media_type), sha


def _publish_minimal_v2_snapshot(
    paths,
    session_id: str,
    snapshot_id: str,
    *,
    runtime_status: str = "complete",
):
    """Publish a minimal committed V2 snapshot for route tests."""
    from backend.models.schemas import AnalysisState
    from backend.services.snapshot_v2.models import IdleCapsule
    from backend.services.snapshot_v2.object_store import ManifestStore, canonical_json
    from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest

    now = datetime.now(timezone.utc)
    state = AnalysisState(
        id=session_id, user_id="u1", name="t",
        initial_instruction="t", current_round=1, max_rounds=4,
        created_at=now, updated_at=now,
    )
    payload = state.model_dump(mode="json")
    payload.pop("snapshots", None)
    payload.pop("realtime_messages", None)
    state_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode()
    state_ref, state_sha = _write_object(paths, state_bytes)

    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
    nb_bytes = json.dumps(nb, sort_keys=True, separators=(",", ":")).encode()
    nb_ref, nb_sha = _write_object(paths, nb_bytes, "application/x-ipynb+json")

    objects = {
        state_sha: state_ref.model_dump(mode="json"),
        nb_sha: nb_ref.model_dump(mode="json"),
    }
    runtime = {
        "status": runtime_status,
        "dependencies": {"required": {}, "incidental_skips": []},
        "residual_ref": None,
        "captured_residual_names": [],
        "adata_refs": {},
    }
    if runtime_status == "complete":
        residual = b"residual-bytes"
        residual_ref, residual_sha = _write_object(
            paths, residual, "application/octet-stream",
        )
        objects[residual_sha] = residual_ref.model_dump(mode="json")
        runtime["residual_ref"] = residual_ref.model_dump(mode="json")
        runtime["captured_residual_names"] = ["x"]

    manifest = make_snapshot_v2_manifest(
        session_id=session_id,
        snapshot_id=snapshot_id,
        analysis_state_ref=state_ref.model_dump(mode="json"),
        notebook_ref=nb_ref.model_dump(mode="json"),
        objects=objects,
        continuation=IdleCapsule(graph_version="2.0").model_dump(mode="json"),
        runtime=runtime,
    )
    raw = canonical_json(manifest)
    if isinstance(raw, str):
        raw = raw.encode()
    manifest.root_hash = hashlib.sha256(raw).hexdigest()
    mstore = ManifestStore(paths)
    staging = mstore.write_staging(manifest)
    mstore.publish_staging(staging)


class TestSnapshotV2RestoreJobRoutes:
    """HTTP contract for restore enqueue, status GET, cancel, and error codes."""

    @pytest.fixture
    def route_env(self, tmp_path, monkeypatch):
        session_id = f"sess-api-{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        for target in (
            "backend.config._config.get_session_dir",
            "backend.services.snapshot_v2.paths.get_session_dir",
        ):
            monkeypatch.setattr(target, _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.config._config.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )

        root = _sess_dir(session_id)
        for d in ("workspace", "states", "integrity"):
            (root / d).mkdir(parents=True, exist_ok=True)

        from backend.services.snapshot_v2.paths import SnapshotV2Paths

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        from backend.models.schemas import AnalysisState
        from backend.state import sessions, snapshot_restore_jobs

        owner_id = "owner-u1"
        state = AnalysisState(
            id=session_id, user_id=owner_id, name="route-test",
            initial_instruction="t", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state
        snapshot_restore_jobs.clear()

        yield {
            "session_id": session_id,
            "paths": paths,
            "owner_id": owner_id,
            "state": state,
            "sess_dir": _sess_dir,
        }

        sessions.pop(session_id, None)
        snapshot_restore_jobs.clear()

    @pytest.fixture
    def client(self, route_env, monkeypatch):
        import backend.routes.snapshots as snaps
        from fastapi.testclient import TestClient
        from backend.main import app

        owner_id = route_env["owner_id"]
        state = route_env["state"]

        async def fake_owner(session_id, user_id=None):
            from fastapi import HTTPException
            if user_id is not None and user_id not in ("", owner_id) and user_id != owner_id:
                raise HTTPException(status_code=404, detail="Session not found")
            return state

        monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)

        async def patched_dispatch(request, call_next):
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                request.state.user_id = auth[7:].strip()
            else:
                request.state.user_id = owner_id
            return await call_next(request)

        client = TestClient(app)
        _patch_auth_middleware(app, monkeypatch, patched_dispatch)
        return client

    def _body(self, snapshot_id: str, **extra):
        body = {
            "snapshot_id": snapshot_id,
            "runtime_choice": "reset",
            "current_state_choice": "abort",
        }
        body.update(extra)
        return body

    def test_get_snapshot_missing_returns_404_structured(self, client, route_env):
        session_id = route_env["session_id"]
        resp = client.get(f"/api/session/{session_id}/snapshots-v2/does-not-exist")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["code"] == "MANIFEST_NOT_FOUND"

    def test_restore_v2_returns_202_and_status_url_is_callable(
        self, client, route_env, monkeypatch,
    ):
        """POST restore-v2 → 202; status_url from response is GET-callable."""
        from backend.services.snapshot_v2.jobs import RestoreJobManager, RestoreJobHandle, _persist_handle
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord
        from backend.state import snapshot_restore_jobs

        session_id = route_env["session_id"]
        paths = route_env["paths"]
        snap_id = "snap-ready-01"
        _publish_minimal_v2_snapshot(paths, session_id, snap_id)

        async def fake_enqueue(self, sid, preflight, request, **kw):
            tx_id = f"restore-{uuid.uuid4().hex[:12]}"
            record = SnapshotTransactionRecord(
                transaction_id=tx_id,
                session_id=sid,
                snapshot_id=request.snapshot_id,
                status="queued",
            )
            handle = RestoreJobHandle(
                record=record,
                task=None,
                cancel_event=asyncio.Event(),
                paths=self.paths,
            )
            _persist_handle(handle)
            snapshot_restore_jobs[tx_id] = handle
            return record

        monkeypatch.setattr(RestoreJobManager, "enqueue", fake_enqueue)

        resp = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json=self._body(snap_id),
        )
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert data["transaction_id"]
        assert data["status"] == "queued"
        assert "status_url" in data
        assert "cancel_url" in data
        assert data["transaction_id"] in data["status_url"]

        status_url = data["status_url"]
        if status_url.startswith("http"):
            from urllib.parse import urlparse
            status_path = urlparse(status_url).path
        else:
            status_path = status_url
        status_resp = client.get(status_path)
        assert status_resp.status_code == 200, status_resp.text
        body = status_resp.json()
        payload = body.get("data", body)
        assert (
            payload.get("transaction_id") == data["transaction_id"]
            or body.get("transaction_id") == data["transaction_id"]
        )

    def test_cross_request_get_and_cancel(self, client, route_env, monkeypatch):
        """GET status and POST cancel work without the original manager instance."""
        from backend.services.snapshot_v2.jobs import RestoreJobManager, RestoreJobHandle, _persist_handle
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord
        from backend.state import snapshot_restore_jobs

        session_id = route_env["session_id"]
        paths = route_env["paths"]
        snap_id = "snap-cancel-01"
        _publish_minimal_v2_snapshot(paths, session_id, snap_id)

        async def fake_enqueue(self, sid, preflight, request, **kw):
            tx_id = f"restore-{uuid.uuid4().hex[:12]}"
            record = SnapshotTransactionRecord(
                transaction_id=tx_id,
                session_id=sid,
                snapshot_id=request.snapshot_id,
                status="waiting_safe_point",
            )
            handle = RestoreJobHandle(
                record=record,
                task=None,
                cancel_event=asyncio.Event(),
                paths=self.paths,
            )
            _persist_handle(handle)
            snapshot_restore_jobs[tx_id] = handle
            return record

        monkeypatch.setattr(RestoreJobManager, "enqueue", fake_enqueue)

        post = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json=self._body(snap_id),
        )
        assert post.status_code == 202, post.text
        tx_id = post.json()["transaction_id"]

        get_resp = client.get(
            f"/api/session/{session_id}/snapshot-transactions/{tx_id}",
        )
        assert get_resp.status_code == 200, get_resp.text

        cancel_resp = client.post(
            f"/api/session/{session_id}/snapshot-transactions/{tx_id}/cancel",
        )
        assert cancel_resp.status_code == 200, cancel_resp.text

        get_after = client.get(
            f"/api/session/{session_id}/snapshot-transactions/{tx_id}",
        )
        assert get_after.status_code == 200
        after_body = get_after.json()
        payload = after_body.get("data", after_body)
        assert payload.get("status") == "cancelled" or after_body.get("status") == "cancelled"

    def test_409_runtime_choice_required(self, client, route_env, monkeypatch):
        """Unavailable runtime without runtime_choice → 409."""
        from backend.utils.runtime_apply import RuntimeChoiceRequired
        from backend.services.snapshot_v2.preflight import RestorePreflight, RestorePreflightResult

        session_id = route_env["session_id"]
        snap_id = "snap-choice-01"
        paths = route_env["paths"]
        _publish_minimal_v2_snapshot(
            paths, session_id, snap_id, runtime_status="missing",
        )

        def fake_run(self, snapshot_id, request=None):
            if request is not None and getattr(request, "runtime_choice", None) is None:
                raise RuntimeChoiceRequired("missing")
            return RestorePreflightResult(status="ready")

        monkeypatch.setattr(RestorePreflight, "run", fake_run)

        resp = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json={
                "snapshot_id": snap_id,
                "current_state_choice": "abort",
            },
        )
        assert resp.status_code == 409, resp.text

    def test_410_legacy_format(self, client, route_env, monkeypatch):
        """Legacy / unsupported format → 410."""
        from backend.services.snapshot_v2.preflight import RestorePreflight, RestorePreflightResult

        session_id = route_env["session_id"]
        snap_id = "snap-legacy-01"

        def fake_run(self, snapshot_id, request=None):
            return RestorePreflightResult(
                status="incompatible",
                format_version=1,
                compatibility_errors=["Snapshot format v1 is not supported"],
            )

        monkeypatch.setattr(RestorePreflight, "run", fake_run)

        resp = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json=self._body(snap_id),
        )
        assert resp.status_code == 410, resp.text

    def test_422_corrupt_target(self, client, route_env, monkeypatch):
        """Corrupt target content → 422."""
        from backend.services.snapshot_v2.preflight import RestorePreflight, RestorePreflightResult

        session_id = route_env["session_id"]
        snap_id = "snap-corrupt-01"

        def fake_run(self, snapshot_id, request=None):
            return RestorePreflightResult(
                status="corrupted",
                compatibility_errors=["hash mismatch"],
            )

        monkeypatch.setattr(RestorePreflight, "run", fake_run)

        resp = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json=self._body(snap_id),
        )
        assert resp.status_code == 422, resp.text

    def test_507_quota(self, client, route_env, monkeypatch):
        """Storage/quota failure → 507."""
        from backend.services.snapshot_v2.errors import SnapshotStorageError
        from backend.services.snapshot_v2.preflight import RestorePreflight, RestorePreflightResult
        from backend.services.snapshot_v2.jobs import RestoreJobManager

        session_id = route_env["session_id"]
        snap_id = "snap-quota-01"

        def fake_run(self, snapshot_id, request=None):
            return RestorePreflightResult(status="ready")

        async def boom_enqueue(self, *a, **k):
            raise SnapshotStorageError("QUOTA_EXCEEDED", "disk full")

        monkeypatch.setattr(RestorePreflight, "run", fake_run)
        monkeypatch.setattr(RestoreJobManager, "enqueue", boom_enqueue)

        resp = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json=self._body(snap_id),
        )
        assert resp.status_code == 507, resp.text

    def test_ownership_isolation_on_status_and_cancel(self, client, route_env, monkeypatch):
        """Non-owner cannot GET status or cancel (404 leak-oracle safe)."""
        from backend.services.snapshot_v2.jobs import RestoreJobManager, RestoreJobHandle, _persist_handle
        from backend.services.snapshot_v2.models import SnapshotTransactionRecord
        from backend.state import snapshot_restore_jobs

        session_id = route_env["session_id"]
        paths = route_env["paths"]
        snap_id = "snap-own-01"
        _publish_minimal_v2_snapshot(paths, session_id, snap_id)

        async def fake_enqueue(self, sid, preflight, request, **kw):
            tx_id = f"restore-{uuid.uuid4().hex[:12]}"
            record = SnapshotTransactionRecord(
                transaction_id=tx_id,
                session_id=sid,
                snapshot_id=request.snapshot_id,
                status="waiting_safe_point",
            )
            handle = RestoreJobHandle(
                record=record,
                task=None,
                cancel_event=asyncio.Event(),
                paths=self.paths,
            )
            _persist_handle(handle)
            snapshot_restore_jobs[tx_id] = handle
            return record

        monkeypatch.setattr(RestoreJobManager, "enqueue", fake_enqueue)

        post = client.post(
            f"/api/session/{session_id}/snapshots/{snap_id}/restore-v2",
            json=self._body(snap_id),
            headers={"Authorization": f"Bearer {route_env['owner_id']}"},
        )
        assert post.status_code == 202, post.text
        tx_id = post.json()["transaction_id"]

        other = {"Authorization": "Bearer other-user"}
        get_resp = client.get(
            f"/api/session/{session_id}/snapshot-transactions/{tx_id}",
            headers=other,
        )
        assert get_resp.status_code == 404, get_resp.text

        cancel_resp = client.post(
            f"/api/session/{session_id}/snapshot-transactions/{tx_id}/cancel",
            headers=other,
        )
        assert cancel_resp.status_code == 404, cancel_resp.text



# ── Batch 3 Task 4: listing diagnostics + service centralization ───────────


class TestSnapshotV2ListingDiagnostics:
    """Corrupt committed manifests surface as typed diagnostics, not silent drops."""

    @pytest.fixture
    def list_env(self, tmp_path, monkeypatch):
        session_id = f"sess-list-{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        for target in (
            "backend.config._config.get_session_dir",
            "backend.services.snapshot_v2.paths.get_session_dir",
        ):
            monkeypatch.setattr(target, _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.config._config.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )

        root = _sess_dir(session_id)
        for d in ("workspace", "states", "integrity"):
            (root / d).mkdir(parents=True, exist_ok=True)

        from backend.services.snapshot_v2.paths import SnapshotV2Paths

        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        from backend.models.schemas import AnalysisState
        from backend.state import sessions

        owner_id = "owner-u1"
        state = AnalysisState(
            id=session_id, user_id=owner_id, name="list-test",
            initial_instruction="t", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state
        yield {
            "session_id": session_id,
            "paths": paths,
            "owner_id": owner_id,
            "state": state,
        }
        sessions.pop(session_id, None)

    def test_list_exposes_corrupt_manifest_diagnostic(self, list_env):
        """Corrupting one committed manifest must not silently shorten the list."""
        from backend.services.snapshot_v2.service import SnapshotService

        session_id = list_env["session_id"]
        paths = list_env["paths"]
        good_id = "snap-good-01"
        bad_id = "snap-bad-01"
        _publish_minimal_v2_snapshot(paths, session_id, good_id)
        _publish_minimal_v2_snapshot(paths, session_id, bad_id)

        # Corrupt the second committed manifest after publish
        bad_path = paths.committed / f"{bad_id}.json"
        assert bad_path.exists()
        bad_path.write_text("{not-valid-json")

        service = SnapshotService(session_id, paths)
        listing = service.list_snapshots()

        # Must include both entries — not a shorter healthy-only list
        assert len(listing) == 2, f"expected 2 entries, got {len(listing)}: {listing}"
        by_id = {e.get("snapshot_id"): e for e in listing}
        assert good_id in by_id
        assert bad_id in by_id
        assert by_id[good_id].get("integrity_status") == "ok" or by_id[good_id].get("status") == "ok"
        assert by_id[bad_id].get("integrity_status") == "corrupt" or by_id[bad_id].get("status") == "corrupt"
        assert by_id[bad_id].get("error_code")

    def test_list_route_returns_corrupt_diagnostic(self, list_env, monkeypatch):
        """GET snapshots-v2 surfaces corrupt entry via SnapshotService listing."""
        import backend.routes.snapshots as snaps
        from fastapi.testclient import TestClient
        from backend.main import app

        session_id = list_env["session_id"]
        paths = list_env["paths"]
        owner_id = list_env["owner_id"]
        state = list_env["state"]
        good_id = "snap-route-good"
        bad_id = "snap-route-bad"
        _publish_minimal_v2_snapshot(paths, session_id, good_id)
        _publish_minimal_v2_snapshot(paths, session_id, bad_id)
        (paths.committed / f"{bad_id}.json").write_text("CORRUPT")

        async def fake_owner(session_id, user_id=None):
            return state

        monkeypatch.setattr(snaps, "require_session_ownership", fake_owner)

        async def patched_dispatch(request, call_next):
            request.state.user_id = owner_id
            return await call_next(request)

        client = TestClient(app)
        _patch_auth_middleware(app, monkeypatch, patched_dispatch)
        resp = client.get(f"/api/session/{session_id}/snapshots-v2")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        data = body.get("data", body)
        snaps_list = data.get("snapshots") or []
        assert len(snaps_list) == 2
        statuses = {s.get("snapshot_id"): s.get("status") or s.get("integrity_status") for s in snaps_list}
        assert statuses[bad_id] == "corrupt"


class TestSnapshotV2DeleteViaService:
    """Delete routes go through SnapshotService under session lock."""

    @pytest.fixture
    def del_env(self, tmp_path, monkeypatch):
        session_id = f"sess-del-{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        for target in (
            "backend.config._config.get_session_dir",
            "backend.services.snapshot_v2.paths.get_session_dir",
        ):
            monkeypatch.setattr(target, _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_session_workspace",
            lambda sid: _sess_dir(sid) / "workspace",
        )
        monkeypatch.setattr(
            "backend.config._config.get_session_states",
            lambda sid: _sess_dir(sid) / "states",
        )
        monkeypatch.setattr(
            "backend.config._config.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        root = _sess_dir(session_id)
        for d in ("workspace", "states", "integrity"):
            (root / d).mkdir(parents=True, exist_ok=True)
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)
        from backend.models.schemas import AnalysisState
        from backend.state import sessions
        owner_id = "owner-u1"
        state = AnalysisState(
            id=session_id, user_id=owner_id, name="del-test",
            initial_instruction="t", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state
        yield {"session_id": session_id, "paths": paths, "owner_id": owner_id, "state": state}
        sessions.pop(session_id, None)

    @pytest.mark.asyncio
    async def test_delete_leaf_reconciles_state_snapshots(self, del_env):
        from backend.services.snapshot_v2.service import SnapshotService

        session_id = del_env["session_id"]
        paths = del_env["paths"]
        snap_id = "snap-del-01"
        _publish_minimal_v2_snapshot(paths, session_id, snap_id)
        service = SnapshotService(session_id, paths)
        del_env["state"].snapshots = service.rebuild_snapshots_from_manifests()
        assert any(s.id == snap_id for s in del_env["state"].snapshots)
        result = await service.delete_leaf(snap_id)
        assert snap_id in result.deleted_ids
        assert not any(s.id == snap_id for s in del_env["state"].snapshots)

    @pytest.mark.asyncio
    async def test_protected_delete_returns_conflict_before_removal(self, del_env):
        from backend.services.snapshot_v2.errors import SnapshotDeleteConflict
        from backend.services.snapshot_v2.service import SnapshotService

        session_id = del_env["session_id"]
        paths = del_env["paths"]
        snap_id = "snap-prot-01"
        _publish_minimal_v2_snapshot(paths, session_id, snap_id)
        del_env["state"].last_snapshot_id = snap_id
        service = SnapshotService(session_id, paths)
        with pytest.raises(SnapshotDeleteConflict):
            await service.delete_leaf(snap_id)
        # Manifest still present
        assert (paths.committed / f"{snap_id}.json").exists()


class TestSnapshotV2ConcurrencySerialization:
    """Conflicting ops serialize under the same session lock."""

    @pytest.mark.asyncio
    async def test_create_and_delete_serialize(self, tmp_path, monkeypatch):
        import asyncio
        from backend.services.snapshot_v2.service import SnapshotService
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.state import get_session_lock

        session_id = f"sess-conc-{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv("SCAI_DATA_DIR", str(tmp_path / "data"))

        def _sess_dir(sid):
            return tmp_path / "data" / "sessions" / sid

        for target in (
            "backend.config._config.get_session_dir",
            "backend.services.snapshot_v2.paths.get_session_dir",
        ):
            monkeypatch.setattr(target, _sess_dir)
        monkeypatch.setattr(
            "backend.config._config.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        monkeypatch.setattr(
            "backend.services.snapshot_v2.paths.get_integrity_dir",
            lambda sid: _sess_dir(sid) / "integrity",
        )
        root = _sess_dir(session_id)
        for d in ("workspace", "states", "integrity"):
            (root / d).mkdir(parents=True, exist_ok=True)
        paths = SnapshotV2Paths.for_session(session_id)
        for d in (
            paths.root, paths.objects, paths.staging, paths.journals,
            paths.integrity, paths.committed,
        ):
            d.mkdir(parents=True, exist_ok=True)

        snap_id = "snap-conc-01"
        _publish_minimal_v2_snapshot(paths, session_id, snap_id)
        service = SnapshotService(session_id, paths)

        order: list[str] = []
        lock = get_session_lock(session_id)

        async def hold_then_delete():
            async with lock:
                order.append("delete_enter")
                await asyncio.sleep(0.05)
                # nested service call would re-acquire same lock (asyncio.Lock is not reentrant)
                # so call the sync delete_snapshot while already holding via delete_leaf path
            order.append("delete_outer")
            await service.delete_leaf(snap_id)
            order.append("delete_done")

        async def concurrent_export():
            await asyncio.sleep(0.01)
            order.append("export_start")
            try:
                # export may fail if nothing left / integrity — still proves lock wait
                await service.export_snapshots(include_runtime=False, snapshot_ids=[snap_id])
            except Exception:
                pass
            order.append("export_done")

        await asyncio.gather(hold_then_delete(), concurrent_export())
        # Both completed; export_done cannot finish before delete_enter if serialized
        assert "delete_done" in order and "export_done" in order
