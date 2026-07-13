"""Workspace capture scope tests for Snapshot V2.

Capture scope is workspace root files plus recursive figures/, tables/,
and reproducibility/ only. Unsafe symlinks fail rather than being skipped.
"""
import ast
import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def workspace_dir(tmp_path):
    """Set up a workspace directory with known structure."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True)

    (ws / "figures").mkdir()
    (ws / "tables").mkdir()
    (ws / "reproducibility").mkdir()
    (ws / "scratch").mkdir()

    (ws / "figures" / "plot.png").write_bytes(b"PNG content")
    (ws / "figures" / "heatmap.pdf").write_bytes(b"PDF content")
    (ws / "tables" / "cluster.csv").write_bytes(b"gene,cluster\nX,1\n")
    (ws / "tables" / "markers.csv").write_bytes(b"gene,pval\nY,0.01\n")
    (ws / "reproducibility" / "session.tar.gz").write_bytes(b"tarball data")
    (ws / "analysis.ipynb").write_bytes(b"{}")
    (ws / "summary.txt").write_bytes(b"summary content")
    (ws / "scratch" / "private.txt").write_bytes(b"should not capture")

    return ws


@pytest.fixture
def v2_paths(tmp_path, workspace_dir):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [self.root, self.objects, self.staging, self.journals, self.integrity, self.committed]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


class TestWorkspaceCaptureScope:
    def test_capture_scope_includes_root_and_declared_dirs_only(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()
        entries, warnings = capture.capture_scope(store, str(workspace_dir))

        paths = {e.path for e in entries}
        assert "figures/plot.png" in paths
        assert "figures/heatmap.pdf" in paths
        assert "tables/cluster.csv" in paths
        assert "tables/markers.csv" in paths
        assert "reproducibility/session.tar.gz" in paths
        assert "analysis.ipynb" in paths
        assert "summary.txt" in paths
        assert "scratch/private.txt" not in paths
        assert not any(p.startswith("scratch/") for p in paths)

    def test_capture_scope_excludes_nested_out_of_scope(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        (workspace_dir / "scratch" / "nested").mkdir()
        (workspace_dir / "scratch" / "nested" / "x.bin").write_bytes(b"x")
        (workspace_dir / "logs").mkdir()
        (workspace_dir / "logs" / "run.log").write_bytes(b"log")

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()
        entries, _ = capture.capture_scope(store, str(workspace_dir))
        paths = {e.path for e in entries}
        assert "logs/run.log" not in paths
        assert "scratch/nested/x.bin" not in paths

    def test_entries_have_correct_metadata(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()
        entries, _ = capture.capture_scope(store, str(workspace_dir))

        for entry in entries:
            assert len(entry.object_hash) == 64
            assert entry.size > 0
            assert entry.mode > 0
            assert entry.kind == "file"

        plot = next(e for e in entries if e.path == "figures/plot.png")
        assert plot.size == len(b"PNG content")

    def test_excludes_paths_outside_workspace(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()

        sibling = workspace_dir.parent / "uploads"
        sibling.mkdir()
        (sibling / "data.h5ad").write_bytes(b"secret data")

        entries, warnings = capture.capture_scope(store, str(workspace_dir))
        paths = {e.path for e in entries}
        assert "uploads/data.h5ad" not in paths

    def test_unsafe_symlink_fails_capture(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()

        outside = workspace_dir.parent / "outside_secret.txt"
        outside.write_bytes(b"secret")
        bad_link = workspace_dir / "leak.txt"
        bad_link.symlink_to(os.path.relpath(outside, workspace_dir))

        with pytest.raises(SnapshotCaptureError) as exc:
            capture.capture_scope(store, str(workspace_dir))
        assert "leak.txt" in str(exc.value) or "SYMLINK" in str(exc.value.code)

    def test_unsafe_absolute_symlink_fails_capture(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()

        outside = workspace_dir.parent / "secret2.txt"
        outside.write_bytes(b"secret")
        bad_link = workspace_dir / "leak2.txt"
        bad_link.symlink_to(str(outside))

        with pytest.raises(SnapshotCaptureError):
            capture.capture_scope(store, str(workspace_dir))

    def test_safe_internal_symlink_within_scope_is_rejected_or_resolved(
        self, workspace_dir, v2_paths,
    ):
        """Any symlink in scope fails closed (no following)."""
        from backend.services.snapshot_v2.errors import SnapshotCaptureError
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()
        target = workspace_dir / "summary.txt"
        link = workspace_dir / "figures" / "alias.png"
        link.symlink_to(os.path.relpath(target, link.parent))

        with pytest.raises(SnapshotCaptureError):
            capture.capture_scope(store, str(workspace_dir))

    def test_duplicate_paths_single_entry(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()

        (workspace_dir / "dup.txt").write_bytes(b"first")
        (workspace_dir / "dup.txt").write_bytes(b"second")

        entries, _ = capture.capture_scope(store, str(workspace_dir))
        dup = [e for e in entries if e.path == "dup.txt"]
        assert len(dup) == 1

    def test_empty_dirs_produce_no_file_entries(self, workspace_dir, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()

        (workspace_dir / "figures" / "empty_nested").mkdir()
        entries, _ = capture.capture_scope(store, str(workspace_dir))
        paths = {e.path for e in entries}
        assert "figures/empty_nested" not in paths

    def test_works_with_empty_workspace(self, tmp_path, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()

        empty = tmp_path / "empty_ws"
        empty.mkdir()
        entries, warnings = capture.capture_scope(store, str(empty))
        assert entries == []

    def test_legacy_capture_delegates_to_scope(self, workspace_dir, v2_paths):
        """capture() remains for callers; must use scoped semantics."""
        from backend.services.snapshot_v2.object_store import ObjectStore
        from backend.services.snapshot_v2.workspace import WorkspaceCapture

        store = ObjectStore(v2_paths)
        capture = WorkspaceCapture()
        entries, _ = capture.capture(store, str(workspace_dir))
        paths = {e.path for e in entries}
        assert "scratch/private.txt" not in paths
        assert "summary.txt" in paths


class TestObjectStoreContainsVerified:
    def test_contains_verified_true_for_existing_object(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        ref = store.put_bytes(b"hello-verified", "text/plain").ref
        assert store.contains_verified(ref.sha256) is True

    def test_contains_verified_false_for_missing(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        assert store.contains_verified("a" * 64) is False

    def test_contains_verified_false_for_corrupt(self, v2_paths):
        from backend.services.snapshot_v2.object_store import ObjectStore

        store = ObjectStore(v2_paths)
        ref = store.put_bytes(b"good-content", "text/plain").ref
        obj = v2_paths.objects / ref.sha256
        obj.write_bytes(b"tampered")
        assert store.contains_verified(ref.sha256) is False


class TestManifestStoreHostSecret:
    def test_no_session_key_method(self):
        from backend.services.snapshot_v2.object_store import ManifestStore

        assert not hasattr(ManifestStore, "_session_key")

    def test_sign_uses_integrity_secret(self, v2_paths):
        import hashlib
        import hmac
        from backend.services.snapshot_v2.object_store import ManifestStore
        from backend.utils.crypto import get_integrity_secret

        store = ManifestStore(v2_paths)
        data = b"payload"
        sig = store._sign(data)
        secret = get_integrity_secret()
        key = secret if isinstance(secret, bytes) else secret.encode()
        expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        assert sig == expected
        assert store._verify_signature(data, sig) is True
