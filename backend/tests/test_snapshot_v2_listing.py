"""Tests: UI listing uses diagnostics; safety paths use verify_all_committed."""
from unittest.mock import MagicMock

import pytest

from backend.tests.snapshot_v2_fixtures import make_snapshot_v2_manifest, write_manifest_to_store


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
            for d in [
                self.root, self.objects, self.staging,
                self.journals, self.integrity, self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


class TestListingDiagnostics:
    def test_list_snapshots_surfaces_corrupt_entries(self, v2_paths):
        from backend.services.snapshot_v2.service import SnapshotService

        good = make_snapshot_v2_manifest(snapshot_id="snap-good", session_id="s1")
        write_manifest_to_store(good, v2_paths.committed, v2_paths.integrity)
        (v2_paths.committed / "snap-bad.json").write_text("not-json")

        service = SnapshotService("s1", v2_paths)
        listed = service.list_snapshots()
        by_id = {e["snapshot_id"]: e for e in listed}
        assert by_id["snap-good"]["status"] == "ok"
        assert by_id["snap-bad"]["status"] == "corrupt"
        assert by_id["snap-bad"]["integrity_status"] == "corrupt"
        assert by_id["snap-bad"]["manifest"] is None

    def test_ui_list_uses_diagnostics_not_verify_all(self, v2_paths, monkeypatch):
        from backend.services.snapshot_v2.object_store import ManifestStore
        from backend.services.snapshot_v2.service import SnapshotService

        calls: list[str] = []
        orig_diag = ManifestStore.list_committed_with_diagnostics
        orig_verify = ManifestStore.verify_all_committed

        def tracking_diag(self):
            calls.append("list_committed_with_diagnostics")
            return orig_diag(self)

        def tracking_verify(self):
            calls.append("verify_all_committed")
            return orig_verify(self)

        monkeypatch.setattr(
            ManifestStore, "list_committed_with_diagnostics", tracking_diag,
        )
        monkeypatch.setattr(ManifestStore, "verify_all_committed", tracking_verify)

        SnapshotService("s1", v2_paths).list_snapshots()
        assert "list_committed_with_diagnostics" in calls
        assert "verify_all_committed" not in calls


class TestSafetyPathsUseVerifyAll:
    def test_exporter_uses_verify_all_committed(self, v2_paths, monkeypatch):
        from backend.services.snapshot_v2.exporter import SnapshotExporter
        from backend.services.snapshot_v2.object_store import ManifestStore

        calls: list[str] = []
        orig_verify = ManifestStore.verify_all_committed
        orig_diag = ManifestStore.list_committed_with_diagnostics

        def tracking_verify(self):
            calls.append("verify_all_committed")
            return orig_verify(self)

        def tracking_diag(self):
            calls.append("list_committed_with_diagnostics")
            return orig_diag(self)

        monkeypatch.setattr(ManifestStore, "verify_all_committed", tracking_verify)
        monkeypatch.setattr(
            ManifestStore, "list_committed_with_diagnostics", tracking_diag,
        )

        good = make_snapshot_v2_manifest(snapshot_id="snap-export", session_id="s1")
        write_manifest_to_store(good, v2_paths.committed, v2_paths.integrity)
        # Need objects for export; expect failure after enumeration if objects missing
        exporter = SnapshotExporter("s1", v2_paths)
        try:
            exporter.export("s1")
        except Exception:
            pass
        assert "verify_all_committed" in calls
        assert "list_committed_with_diagnostics" not in calls

    def test_deletion_load_all_uses_verify_all(self, v2_paths, monkeypatch):
        from backend.services.snapshot_v2.deletion import DeletionService
        from backend.services.snapshot_v2.object_store import ManifestStore

        calls: list[str] = []
        orig_verify = ManifestStore.verify_all_committed
        orig_diag = ManifestStore.list_committed_with_diagnostics

        def tracking_verify(self):
            calls.append("verify_all_committed")
            return orig_verify(self)

        def tracking_diag(self):
            calls.append("list_committed_with_diagnostics")
            return orig_diag(self)

        monkeypatch.setattr(ManifestStore, "verify_all_committed", tracking_verify)
        monkeypatch.setattr(
            ManifestStore, "list_committed_with_diagnostics", tracking_diag,
        )

        deletion = DeletionService("s1", v2_paths)
        deletion._load_all_committed()
        assert "verify_all_committed" in calls
        assert "list_committed_with_diagnostics" not in calls
