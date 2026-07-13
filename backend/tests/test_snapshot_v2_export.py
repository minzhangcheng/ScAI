"""Tests: V2 snapshot export with explicit runtime policy."""
from backend.tests.snapshot_v2_fixtures import v2_manifest, write_manifest_to_store
import hashlib
import json
import tarfile
import io
from pathlib import Path

import pytest

from backend.services.snapshot_v2.models import (
    FORMAT_VERSION,
    IdleCapsule,
    ObjectRef,
    RuntimeGenerationManifest,
    SnapshotManifestV2,
    WorkspaceEntry,
)


# ── Fixtures (mirrors test_snapshot_v2_deletion.py) ─────────────────────


@pytest.fixture
def v2_paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging,
                      self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def build_manifest(
    session_id: str,
    snapshot_id: str,
    *,
    objects: dict[str, ObjectRef] | None = None,
    workspace: list[WorkspaceEntry] | None = None,
    parent_id: str | None = None,
    runtime_capture: RuntimeGenerationManifest | None = None,
) -> SnapshotManifestV2:
    obj = objects or {}
    ws = workspace or []
    return v2_manifest(
        format_version=2,
        session_id=session_id,
        snapshot_id=snapshot_id,
        captured_at="2026-07-11T12:00:00Z",
        branch="main",
        generation=1,
        parent_id=parent_id,
        root_hash="",
        capsule=IdleCapsule(graph_version="2.0"),
        workspace=ws,
        objects=obj,
        runtime_snapshot_capture=runtime_capture,
    )


def build_object_ref(
    sha256: str | None = None,
    size: int = 1024,
    media_type: str = "application/octet-stream",
) -> ObjectRef:
    if sha256 is None:
        sha256 = hashlib.sha256(f"obj-{size}".encode()).hexdigest()
    if not all(c in "0123456789abcdef" for c in sha256):
        sha256 = hashlib.sha256(sha256.encode()).hexdigest()
    return ObjectRef(sha256=sha256, size=size, media_type=media_type)


def write_object(paths, sha256: str, size: int = 1024) -> None:
    obj_path = paths.objects / sha256
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_bytes(b"x" * size)


# ── Tests ──────────────────────────────────────────────────────────────


class TestSnapshotV2Export:
    """V2 export: base/workspace always, runtime conditional, protection."""

    SESSION_ID = "test-session-export"

    def _setup_two_snapshots(
        self, v2_paths, session_id: str | None = None,
    ) -> tuple[SnapshotManifestV2, SnapshotManifestV2, str, str, str, str]:
        """Create two committed manifests with workspace + runtime objects.

        Returns (manifest_a, manifest_b, ws_hash, rt_hash, extra_hash)
        """
        sid = session_id or self.SESSION_ID

        # Object hashes must match the content write_object() writes: b"x" * size
        ws_hash = hashlib.sha256(b"x" * 256).hexdigest()
        write_object(v2_paths, ws_hash, size=256)

        rt_hash = hashlib.sha256(b"x" * 512).hexdigest()
        write_object(v2_paths, rt_hash, size=512)

        extra_hash = hashlib.sha256(b"x" * 128).hexdigest()
        write_object(v2_paths, extra_hash, size=128)

        rt_capture = RuntimeGenerationManifest(
            capture_command="capture",
            capture_timestamp="2026-07-11T12:00:00Z",
            kernel_id="kernel-1",
            captured_adata_refs={
                "adata": ObjectRef(sha256=rt_hash, size=512, media_type="application/x-hdf5"),
            },
        )

        manifest_a = build_manifest(
            sid, "snap-a",
            objects={
                ws_hash: ObjectRef(sha256=ws_hash, size=256, media_type="application/octet-stream"),
                rt_hash: ObjectRef(sha256=rt_hash, size=512, media_type="application/x-hdf5"),
            },
            workspace=[
                WorkspaceEntry(path="data/file.txt", object_hash=ws_hash, size=256, mode=0o644),
            ],
            runtime_capture=rt_capture,
        )
        write_manifest_to_store(manifest_a, v2_paths.committed, v2_paths.integrity)

        manifest_b = build_manifest(
            sid, "snap-b",
            objects={
                ws_hash: ObjectRef(sha256=ws_hash, size=256, media_type="application/octet-stream"),
                extra_hash: ObjectRef(sha256=extra_hash, size=128, media_type="text/plain"),
            },
            workspace=[
                WorkspaceEntry(path="data/other.txt", object_hash=extra_hash, size=128, mode=0o644),
            ],
        )
        write_manifest_to_store(manifest_b, v2_paths.committed, v2_paths.integrity)

        return manifest_a, manifest_b, ws_hash, rt_hash, extra_hash

    # ── RED tests (will fail before exporter is implemented) ─────────

    def test_export_includes_base_objects_without_runtime_flag(self, v2_paths):
        """Base/workspace objects always export even with include_runtime=False."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        self._setup_two_snapshots(v2_paths)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )
        exporter = SnapshotExporter(self.SESSION_ID, paths)
        export_path = exporter.export(
            self.SESSION_ID, include_runtime=False,
        )

        with tarfile.open(export_path, "r:gz") as tf:
            names = tf.getnames()

            # snap-v2-export.json metadata always present
            assert "snapshot-v2-export.json" in names

            # Read metadata
            meta = json.loads(tf.extractfile("snapshot-v2-export.json").read())

        # Workspace hashes from both manifests are included
        assert meta["runtime_included"] is False
        assert meta["omitted_runtime_hashes"]

        # Workspace object is in the archive
        ws_hash = meta["included_hashes"][0] if meta["included_hashes"] else None
        if ws_hash:
            assert f"objects/{ws_hash}" in names

        # Verify the metadata structure
        assert "included_snapshot_ids" in meta
        assert "included_hashes" in meta
        assert "omitted_runtime_hashes" in meta
        assert "object_count" in meta
        assert "total_size_bytes" in meta

    def test_export_runtime_only_with_flag(self, v2_paths):
        """Runtime objects included only when include_runtime=True."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        _, _, ws_hash, rt_hash, extra_hash = self._setup_two_snapshots(v2_paths)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )

        # Export without runtime
        exporter_a = SnapshotExporter(self.SESSION_ID, paths)
        path_a = exporter_a.export(self.SESSION_ID, include_runtime=False)

        with tarfile.open(path_a, "r:gz") as tf:
            meta_a = json.loads(tf.extractfile("snapshot-v2-export.json").read())
            names_a = tf.getnames()

        assert meta_a["runtime_included"] is False
        assert rt_hash in meta_a["omitted_runtime_hashes"]
        assert rt_hash not in meta_a["included_hashes"]
        assert ws_hash in meta_a["included_hashes"]

        # Export with runtime
        exporter_b = SnapshotExporter(self.SESSION_ID, paths)
        path_b = exporter_b.export(self.SESSION_ID, include_runtime=True)

        with tarfile.open(path_b, "r:gz") as tf:
            meta_b = json.loads(tf.extractfile("snapshot-v2-export.json").read())
            names_b = tf.getnames()

        assert meta_b["runtime_included"] is True
        assert rt_hash in meta_b["included_hashes"]
        assert rt_hash not in meta_b.get("omitted_runtime_hashes", [])

        # Runtime object file exists in archive
        assert f"objects/{rt_hash}" in names_b

    def test_archive_inventory_deterministic(self, v2_paths):
        """Archive contains deterministic inventory of objects."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        self._setup_two_snapshots(v2_paths)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )
        exporter = SnapshotExporter(self.SESSION_ID, paths)
        export_path = exporter.export(self.SESSION_ID, include_runtime=True)

        with tarfile.open(export_path, "r:gz") as tf:
            names = sorted(tf.getnames())
            meta = json.loads(tf.extractfile("snapshot-v2-export.json").read())

            # Metadata always present
            assert "snapshot-v2-export.json" in names

            # Every included hash has a corresponding object file
            for h in meta["included_hashes"]:
                assert f"objects/{h}" in names, f"Missing object file for hash {h}"

            # Object count matches
            object_count = sum(1 for n in names if n.startswith("objects/"))
            assert meta["object_count"] == object_count

            # Total size matches sum of object sizes
            total = 0
            for n in names:
                if n.startswith("objects/"):
                    total += tf.getmember(n).size
            assert meta["total_size_bytes"] == total

    def test_active_export_protects_from_deletion(self, v2_paths):
        """Snapshots under active export cannot be deleted."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter
        from backend.services.snapshot_v2.deletion import DeletionService
        from backend.services.snapshot_v2.errors import SnapshotDeleteConflict

        self._setup_two_snapshots(v2_paths)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )

        # Manually write a protection record (simulating active export)
        from backend.services.snapshot_v2.exporter import _write_export_protection
        export_id = "exp-test-protection"
        _write_export_protection(
            paths, export_id, self.SESSION_ID, ["snap-a", "snap-b"],
        )

        try:
            # Deletion should fail because snap-a is protected
            delete_service = DeletionService(self.SESSION_ID, paths)
            with pytest.raises(SnapshotDeleteConflict, match="protected"):
                delete_service.delete_leaf("snap-a")
        finally:
            # Clean up protection record
            from backend.services.snapshot_v2.exporter import _remove_export_protection
            _remove_export_protection(paths, export_id)

        # After protection is removed, deletion should succeed
        result = delete_service.delete_leaf("snap-a")
        assert "snap-a" in result.deleted_ids

    def test_export_with_specific_snapshot_ids(self, v2_paths):
        """Only specified snapshots are exported when snapshot_ids is provided."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        _, _, ws_hash, rt_hash, extra_hash = self._setup_two_snapshots(v2_paths)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )

        # Export only snap-a (which has ws_hash and rt_hash)
        exporter = SnapshotExporter(self.SESSION_ID, paths)
        export_path = exporter.export(
            self.SESSION_ID, snapshot_ids=["snap-a"], include_runtime=False,
        )

        with tarfile.open(export_path, "r:gz") as tf:
            meta = json.loads(tf.extractfile("snapshot-v2-export.json").read())

        assert meta["included_snapshot_ids"] == ["snap-a"]
        # extra_hash is only in snap-b, should not be included
        if extra_hash in meta["included_hashes"]:
            pytest.fail("extra_hash from snap-b should not be in included_hashes")

    def test_export_metadata_legacy_label(self, v2_paths):
        """V1 metadata is labelled legacy in the export format."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter

        self._setup_two_snapshots(v2_paths)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )
        exporter = SnapshotExporter(self.SESSION_ID, paths)
        export_path = exporter.export(self.SESSION_ID, include_runtime=False)

        with tarfile.open(export_path, "r:gz") as tf:
            meta = json.loads(tf.extractfile("snapshot-v2-export.json").read())

        assert meta.get("format_version") == 2
        # The format is explicitly V2, so existing V1 metadata should be
        # labelled as legacy
        assert "legacy" not in meta  # no legacy metadata here since this is pure V2

    def test_export_verifies_object_integrity(self, v2_paths):
        """Export verifies object hashes before including them."""
        from backend.services.snapshot_v2.paths import SnapshotV2Paths
        from backend.services.snapshot_v2.exporter import SnapshotExporter, ExportIntegrityError

        # Create manifest pointing to a non-existent object
        bad_hash = "a" * 64
        manifest = build_manifest(
            self.SESSION_ID, "snap-bad",
            objects={
                bad_hash: ObjectRef(sha256=bad_hash, size=64, media_type="text/plain"),
            },
            workspace=[
                WorkspaceEntry(path="missing.txt", object_hash=bad_hash, size=64, mode=0o644),
            ],
        )
        write_manifest_to_store(manifest, v2_paths.committed, v2_paths.integrity)

        paths = SnapshotV2Paths(
            root=v2_paths.root,
            objects=v2_paths.objects,
            staging=v2_paths.staging,
            journals=v2_paths.journals,
            integrity=v2_paths.integrity,
            committed=v2_paths.committed,
        )
        exporter = SnapshotExporter(self.SESSION_ID, paths)

        with pytest.raises(ExportIntegrityError, match="OBJECT_NOT_FOUND|not found"):
            exporter.export(self.SESSION_ID, include_runtime=False)


@pytest.fixture
def client(app_db):
    """TestClient with app_db initialized (auth middleware needs DB)."""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


class TestSnapshotV2ExportRoute:
    """Route-level tests for the V2 export endpoint (auth-enforcement only)."""

    def test_export_v2_route_requires_auth(self, client):
        """POST /{session_id}/export-v2 without auth returns 401."""
        response = client.post("/api/session/nonexistent/export-v2")
        assert response.status_code in (401, 403, 404)

    def test_export_v2_route_with_runtime_flag_requires_auth(self, client):
        """POST /{session_id}/export-v2?include_runtime=true without auth returns 401."""
        response = client.post(
            "/api/session/nonexistent/export-v2",
            params={"include_runtime": True},
        )
        assert response.status_code in (401, 403, 404)

    def test_export_v2_delegates_to_snapshot_service(self, monkeypatch, tmp_path, app_db):
        """Authenticated export path calls SnapshotService.export_snapshots."""
        import backend.routes.export_import as exp_mod
        from backend.middleware import auth as auth_mw
        from backend.models.schemas import AnalysisState
        from backend.state import sessions
        from fastapi.testclient import TestClient
        from backend.main import app

        session_id = "sess-export-route-01"
        state = AnalysisState(
            id=session_id, user_id="owner", name="e",
            initial_instruction="t", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state
        called = {}

        async def fake_owner(session_id, user_id=None):
            return state

        async def fake_export(self, *, include_runtime=True, snapshot_ids=None):
            called["include_runtime"] = include_runtime
            called["snapshot_ids"] = snapshot_ids
            out = tmp_path / "fake-export.tar.gz"
            import tarfile
            with tarfile.open(out, "w:gz") as tf:
                import io, json
                data = json.dumps({"format_version": 2}).encode()
                info = tarfile.TarInfo(name="snapshot-v2-export.json")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            return out

        monkeypatch.setattr(exp_mod, "require_session_ownership", fake_owner)
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.SnapshotService.export_snapshots",
            fake_export,
        )
        # Avoid moving into real UPLOAD_TEMP that may not exist
        monkeypatch.setattr(exp_mod, "UPLOAD_TEMP_DIR", tmp_path)

        # BaseHTTPMiddleware freezes dispatch into dispatch_func at construction.
        # Patch the live middleware instance so this works even if earlier tests
        # already built app.middleware_stack with the original method.
        async def patched_dispatch(request, call_next):
            request.state.user_id = "owner"
            return await call_next(request)

        client = TestClient(app)
        mw = app.middleware_stack
        while mw is not None:
            if isinstance(mw, auth_mw.AuthMiddleware):
                monkeypatch.setattr(mw, "dispatch_func", patched_dispatch)
                break
            mw = getattr(mw, "app", None)
        else:
            raise AssertionError("AuthMiddleware not found in app.middleware_stack")

        try:
            resp = client.post(f"/api/session/{session_id}/export-v2")
            assert resp.status_code == 200, resp.text
            assert "include_runtime" in called
        finally:
            sessions.pop(session_id, None)

    def test_export_v2_returns_export_token(self, monkeypatch, tmp_path, app_db):
        """V2 export response includes an export_token (not session_id key)."""
        import backend.routes.export_import as exp_mod
        from backend.middleware import auth as auth_mw
        from backend.models.schemas import AnalysisState
        from backend.state import sessions
        from fastapi.testclient import TestClient
        from backend.main import app

        session_id = "sess-export-token"
        state = AnalysisState(
            id=session_id, user_id="owner", name="t",
            initial_instruction="t", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state

        async def fake_owner(session_id, user_id=None):
            return state

        async def fake_export(self, *, include_runtime=True, snapshot_ids=None):
            out = tmp_path / "fake-export-token.tar.gz"
            import tarfile, io, json
            with tarfile.open(out, "w:gz") as tf:
                data = json.dumps({"format_version": 2}).encode()
                info = tarfile.TarInfo(name="snapshot-v2-export.json")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            return out

        monkeypatch.setattr(exp_mod, "require_session_ownership", fake_owner)
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.SnapshotService.export_snapshots",
            fake_export,
        )
        monkeypatch.setattr(exp_mod, "UPLOAD_TEMP_DIR", tmp_path)

        async def patched_dispatch(request, call_next):
            request.state.user_id = "owner"
            return await call_next(request)

        client = TestClient(app)
        mw = app.middleware_stack
        while mw is not None:
            if isinstance(mw, auth_mw.AuthMiddleware):
                monkeypatch.setattr(mw, "dispatch_func", patched_dispatch)
                break
            mw = getattr(mw, "app", None)

        try:
            resp = client.post(f"/api/session/{session_id}/export-v2")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "export_token" in body, "Missing export_token in response"
            assert isinstance(body["export_token"], str)
            assert len(body["export_token"]) > 0
            assert "download_url" in body
            assert body["export_token"] in body["download_url"], "token should appear in download URL"
        finally:
            sessions.pop(session_id, None)

    def test_export_v2_unknown_snapshot_ids_rejected(self, monkeypatch, tmp_path, app_db):
        """Export with unknown snapshot_ids must fail with 422."""
        import backend.routes.export_import as exp_mod
        from backend.middleware import auth as auth_mw
        from backend.models.schemas import AnalysisState
        from backend.state import sessions
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.services.snapshot_v2.exporter import ExportIntegrityError

        session_id = "sess-export-unknown"
        state = AnalysisState(
            id=session_id, user_id="owner", name="u",
            initial_instruction="t", current_round=1, max_rounds=4,
        )
        sessions[session_id] = state

        async def fake_owner(session_id, user_id=None):
            return state

        async def fake_export(self, *, include_runtime=True, snapshot_ids=None):
            raise ExportIntegrityError(
                "UNKNOWN_SNAPSHOT_IDS",
                "Snapshot IDs not found: snap-nonexistent",
            )

        monkeypatch.setattr(exp_mod, "require_session_ownership", fake_owner)
        monkeypatch.setattr(
            "backend.services.snapshot_v2.service.SnapshotService.export_snapshots",
            fake_export,
        )
        monkeypatch.setattr(exp_mod, "UPLOAD_TEMP_DIR", tmp_path)

        async def patched_dispatch(request, call_next):
            request.state.user_id = "owner"
            return await call_next(request)

        client = TestClient(app)
        mw = app.middleware_stack
        while mw is not None:
            if isinstance(mw, auth_mw.AuthMiddleware):
                monkeypatch.setattr(mw, "dispatch_func", patched_dispatch)
                break
            mw = getattr(mw, "app", None)

        try:
            resp = client.post(
                f"/api/session/{session_id}/export-v2",
                params={"snapshot_ids": "snap-nonexistent"},
            )
            assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
        finally:
            sessions.pop(session_id, None)
