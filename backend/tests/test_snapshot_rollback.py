"""Tests for snapshot rollback enhancement: ipynb backup, workspace manifest, branch labels."""

import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import dill
import pytest
import zstandard as zstd

from backend.models.schemas import Snapshot, AnalysisState
from backend.utils.snapshot import SnapshotManager
from backend.utils.sandbox import ZSTD_MAGIC, COMPRESSION_LEVEL


@pytest.fixture
def temp_session(tmp_path):
    """Create a temporary session directory structure."""
    session_id = "test-session-001"
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "snapshots").mkdir()
    (session_dir / "states").mkdir()
    (session_dir / "workspace" / "figures").mkdir(parents=True)
    (session_dir / "workspace" / "tables").mkdir(parents=True)
    (session_dir / "workspace" / "reproducibility").mkdir(parents=True)
    (session_dir / "reports" / "figures").mkdir(parents=True)
    (session_dir / "reports" / "tables").mkdir(parents=True)

    # Create a minimal ipynb file
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('round 1')"))
    nb_path = session_dir / "states" / f"{session_id}.ipynb"
    nbformat.write(nb, str(nb_path))

    # Create a minimal pkl file
    pkl_path = session_dir / "states" / f"{session_id}_session.pkl"
    pkl_path.write_text("fake pkl content")

    return session_id, session_dir, tmp_path


@pytest.fixture
def mock_config(monkeypatch, temp_session):
    """Patch config functions to use temp directories."""
    session_id, session_dir, tmp_path = temp_session

    def mock_get_session_dir(sid):
        return tmp_path / "sessions" / sid

    def mock_get_session_states(sid):
        return tmp_path / "sessions" / sid / "states"

    monkeypatch.setattr("backend.utils.snapshot.get_session_dir", mock_get_session_dir)
    monkeypatch.setattr("backend.utils.snapshot.get_session_states", mock_get_session_states)

    return session_id, session_dir, tmp_path


class TestSnapshotCreateWithManifest:
    """Tests for create_snapshot with ipynb backup and workspace manifest."""

    @pytest.mark.asyncio
    async def test_create_snapshot_includes_notebook_backup(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        # Create workspace files
        (session_dir / "workspace" / "figures" / "fig_001.jpg").write_text("fake fig")
        (session_dir / "workspace" / "tables" / "markers.csv").write_text("a,b,c")

        # Create a minimal state
        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )

        snapshot = await mgr.create_snapshot(state, "Test", "pre_annotation")

        # Verify notebook backup exists
        nb_backup = mgr.snapshots_dir / f"{snapshot.id}_notebook.ipynb"
        assert nb_backup.exists(), "Notebook backup should be created"
        assert snapshot.notebook_file == f"{snapshot.id}_notebook.ipynb"

        # Verify manifest includes workspace files
        assert "figures" in snapshot.workspace_manifest
        assert "fig_001.jpg" in snapshot.workspace_manifest["figures"]
        assert "markers.csv" in snapshot.workspace_manifest["tables"]

    @pytest.mark.asyncio
    async def test_create_snapshot_manifest_empty_dirs(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )

        snapshot = await mgr.create_snapshot(state, "Test", "pre_annotation")

        # Empty dirs should produce empty lists
        assert snapshot.workspace_manifest == {
            "figures": [],
            "tables": [],
            "reproducibility": [],
            "root": [],
        }

    @pytest.mark.asyncio
    async def test_create_snapshot_with_branch_label(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )

        snapshot = await mgr.create_snapshot(
            state, "Branch Test", "pre_annotation", branch_label="branch-A"
        )
        assert snapshot.branch_label == "branch-A"


class TestSnapshotRestoreWithCleanup:
    """Tests for rollback workspace/reports cleanup."""

    def test_cleanup_workspace_files_deletes_new_files(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        # Create files that exist in the manifest
        (session_dir / "workspace" / "figures" / "fig_001.jpg").write_text("keep me")
        # Create files NOT in the manifest (should be deleted)
        (session_dir / "workspace" / "figures" / "fig_002.jpg").write_text("delete me")
        (session_dir / "workspace" / "tables" / "new_data.csv").write_text("delete me too")

        manifest = {
            "figures": ["fig_001.jpg"],
            "tables": [],
            "reproducibility": [],
            "root": [],
        }

        result = mgr.cleanup_workspace_files(manifest)

        # fig_001.jpg should survive
        assert (session_dir / "workspace" / "figures" / "fig_001.jpg").exists()
        # fig_002.jpg should be deleted
        assert not (session_dir / "workspace" / "figures" / "fig_002.jpg").exists()
        # new_data.csv should be deleted
        assert not (session_dir / "workspace" / "tables" / "new_data.csv").exists()

        assert result["figures"] == 1
        assert result["tables"] == 1

    def test_cleanup_workspace_root_files(self, mock_config):
        """Files in workspace root (e.g. .h5ad) should also be tracked and cleaned."""
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        # Create a file in workspace root
        (session_dir / "workspace" / "output.h5ad").write_text("h5ad data")
        (session_dir / "workspace" / "markers.csv").write_text("a,b,c")

        # Manifest tracks root files that should be kept
        manifest = {
            "figures": [],
            "tables": [],
            "reproducibility": [],
            "root": ["output.h5ad"],
        }

        result = mgr.cleanup_workspace_files(manifest)

        # output.h5ad is in manifest, should survive
        assert (session_dir / "workspace" / "output.h5ad").exists()
        # markers.csv not in manifest, should be deleted
        assert not (session_dir / "workspace" / "markers.csv").exists()

        assert result["root"] == 1

    def test_cleanup_reports_deletes_report_files(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        # Create report files (including report_embedded.html which was missing from old hardcoded list)
        (session_dir / "reports" / "detailed_report.html").write_text("<html>report</html>")
        (session_dir / "reports" / "report.html").write_text("<html>main</html>")
        (session_dir / "reports" / "report_embedded.html").write_text("<html>embedded</html>")
        (session_dir / "reports" / "report.json").write_text("{}")
        (session_dir / "reports" / "score_definitions.csv").write_text("csv")
        (session_dir / "reports" / "figures" / "fig.jpg").write_text("fig")
        (session_dir / "reports" / "tables" / "data.csv").write_text("csv")
        (session_dir / "reports" / f"{session_id}.ipynb").write_text("{}")

        mgr.cleanup_reports()

        # All report files should be deleted (html, json, csv, ipynb)
        assert not (session_dir / "reports" / "detailed_report.html").exists()
        assert not (session_dir / "reports" / "report.html").exists()
        assert not (session_dir / "reports" / "report_embedded.html").exists()
        assert not (session_dir / "reports" / "report.json").exists()
        assert not (session_dir / "reports" / "score_definitions.csv").exists()
        assert not (session_dir / "reports" / "figures" / "fig.jpg").exists()
        assert not (session_dir / "reports" / "tables" / "data.csv").exists()
        assert not (session_dir / "reports" / f"{session_id}.ipynb").exists()

        # Directory structure should be preserved
        assert (session_dir / "reports" / "figures").is_dir()
        assert (session_dir / "reports" / "tables").is_dir()


class TestBranchLabels:
    """Tests for branch label computation."""

    def test_main_branch_label(self):
        """Snapshot without a restored parent should get 'main' branch label."""
        parent = Snapshot(name="Round 1", phase="pre_annotation", round_number=1, parent_id=None)
        # No last_restored_snapshot_id → branch_label = "main"
        last_restored_snapshot_id = None
        snapshots = [parent]
        if last_restored_snapshot_id:
            existing_branches_from_parent = sum(
                1 for s in snapshots
                if s.parent_id == last_restored_snapshot_id
                and s.branch_label and s.branch_label != "main"
            )
            branch_letter = chr(ord('A') + existing_branches_from_parent)
            branch_label = f"branch-{branch_letter}"
        else:
            branch_label = "main"
        assert branch_label == "main"

    def test_branch_from_restore(self):
        """Snapshot after a restore should get branch-A.

        The main-line continuation (branch_label='main') should NOT be counted
        when computing the branch letter, so the first new branch is always 'A'.
        """
        parent = Snapshot(name="Round 1", phase="pre_annotation", round_number=1, parent_id=None)
        # Main-line continuation — should NOT be counted as a branch
        main_line = Snapshot(
            name="Round 2", phase="pre_direction", round_number=2,
            parent_id=parent.id, branch_label="main"
        )
        snapshots = [parent, main_line]

        # Only count non-main branches
        existing = sum(
            1 for s in snapshots
            if s.parent_id == parent.id and s.branch_label and s.branch_label != "main"
        )
        branch_letter = chr(ord('A') + existing)
        branch_label = f"branch-{branch_letter}"

        assert branch_label == "branch-A"

    def test_second_branch_from_same_parent(self):
        """Second branch from the same parent should get 'branch-B'."""
        parent = Snapshot(name="Round 1", phase="pre_annotation", round_number=1, parent_id=None)
        main_line = Snapshot(
            name="Round 2", phase="pre_direction", round_number=2,
            parent_id=parent.id, branch_label="main"
        )
        first_branch = Snapshot(
            name="Branch A", phase="pre_annotation", round_number=1,
            parent_id=parent.id, branch_label="branch-A"
        )
        snapshots = [parent, main_line, first_branch]

        # Only count non-main branches (first_branch counts, main_line doesn't)
        existing = sum(
            1 for s in snapshots
            if s.parent_id == parent.id and s.branch_label and s.branch_label != "main"
        )
        branch_letter = chr(ord('A') + existing)
        branch_label = f"branch-{branch_letter}"

        assert branch_label == "branch-B"


class TestDeleteBranch:
    """Tests for branch deletion."""

    @pytest.mark.asyncio
    async def test_delete_branch_removes_descendants(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )

        # Create a chain of snapshots: A -> B -> C
        snap_a = await mgr.create_snapshot(state, "A", "pre_annotation")
        snap_b = await mgr.create_snapshot(state, "B", "pre_direction", parent_id=snap_a.id)
        snap_c = await mgr.create_snapshot(state, "C", "pre_round_wait", parent_id=snap_b.id)

        all_snapshots = [snap_a, snap_b, snap_c]

        # Delete branch starting from B (should delete B and C)
        deleted_ids = await mgr.delete_branch(snap_b, all_snapshots)

        assert snap_b.id in deleted_ids
        assert snap_c.id in deleted_ids
        assert snap_a.id not in deleted_ids  # Root of a different branch

        # Verify files are removed
        assert not (mgr.snapshots_dir / snap_b.state_file).exists()
        assert not (mgr.snapshots_dir / snap_c.state_file).exists()


class TestPklCompressionSnapshotCompat:
    """Tests that snapshot pkl files (opaque byte blobs) survive compression
    format changes, and that the detection logic works in snapshot file paths."""

    @pytest.mark.asyncio
    async def test_snapshot_copies_compressed_pkl_as_is(self, mock_config):
        """Snapshot should copy a zstd-compressed pkl verbatim (byte-identical)."""
        session_id, session_dir, tmp_path = mock_config
        pkl_path = session_dir / "states" / f"{session_id}_session.pkl"

        # Write a zstd-compressed dill blob (mimics what save_session now produces)
        original_data = {'adata': 'mock', 'results': [1, 2, 3]}
        _cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL, threads=-1)
        with open(pkl_path, 'wb') as f:
            with _cctx.stream_writer(f) as zf:
                dill.dump(original_data, zf)

        mgr = SnapshotManager(session_id)
        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )

        snapshot = await mgr.create_snapshot(state, "Compressed Pkl", "pre_annotation")

        # Verify the snapshot pkl is byte-identical to the source
        snapshot_pkl = mgr.snapshots_dir / f"{snapshot.id}_session.pkl"
        assert snapshot_pkl.exists(), "Snapshot pkl should exist"
        assert snapshot_pkl.read_bytes() == pkl_path.read_bytes(), \
            "Snapshot pkl should be a byte-identical copy"

    @pytest.mark.asyncio
    async def test_snapshot_copies_plain_pkl_as_is(self, mock_config):
        """Snapshot should copy a plain (uncompressed) dill pkl verbatim."""
        session_id, session_dir, tmp_path = mock_config
        pkl_path = session_dir / "states" / f"{session_id}_session.pkl"

        # Write a plain dill blob (old format)
        original_data = {'adata': 'mock', 'results': [4, 5, 6]}
        with open(pkl_path, 'wb') as f:
            dill.dump(original_data, f)

        mgr = SnapshotManager(session_id)
        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )

        snapshot = await mgr.create_snapshot(state, "Plain Pkl", "pre_annotation")

        snapshot_pkl = mgr.snapshots_dir / f"{snapshot.id}_session.pkl"
        assert snapshot_pkl.exists()
        assert snapshot_pkl.read_bytes() == pkl_path.read_bytes(), \
            "Snapshot pkl should be a byte-identical copy of plain dill"

    def test_detection_on_snapshot_pkl_round_trip(self, tmp_path):
        """End-to-end detection: write compressed -> copy to snapshot path -> detect & load."""
        from backend.utils.sandbox import _is_zstd

        original_data = {'sparse': 'data', 'count': 42}

        # Write compressed to "source" path
        src = tmp_path / "source_session.pkl"
        _cctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL, threads=-1)
        with open(src, 'wb') as f:
            with _cctx.stream_writer(f) as zf:
                dill.dump(original_data, zf)

        # "Snapshot copy" -- just byte copy
        dst = tmp_path / "snapshot_session.pkl"
        import shutil
        shutil.copy2(src, dst)

        # Detect and load from snapshot copy
        with open(dst, 'rb') as f:
            header = f.read(4)
            f.seek(0)
            assert _is_zstd(header) is True
            _dctx = zstd.ZstdDecompressor()
            with _dctx.stream_reader(f) as zf:
                loaded = dill.load(zf)

        assert loaded == original_data
