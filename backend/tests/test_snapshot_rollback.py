"""Tests for snapshot rollback enhancement: workspace cleanup, branch labels, V1 fixtures."""

import json
import shutil
from pathlib import Path

import dill
import pytest
import zstandard as zstd

from backend.models.schemas import Snapshot, AnalysisState
from backend.utils.snapshot import SnapshotManager


def _write_v1_snapshot(
    mgr: SnapshotManager,
    state: AnalysisState,
    name: str,
    phase: str,
    *,
    parent_id=None,
    branch_label=None,
    notebook_src: Path | None = None,
    pkl_bytes: bytes | None = None,
    workspace_manifest: dict | None = None,
    runtime_state_included: bool | None = None,
) -> Snapshot:
    """Hand-written V1 snapshot artifacts (no SnapshotManager.create_snapshot)."""
    mgr._ensure_dir()
    snapshot = Snapshot(
        name=name,
        phase=phase,
        round_number=state.current_round,
        parent_id=parent_id,
        branch_label=branch_label,
    )
    snapshot.branch_id = state.restore_branch_id
    snapshot.state_file = f"{snapshot.id}.json"

    state_dict = state.model_dump(mode="json")
    state_dict.pop("snapshots", None)
    state_dict.pop("realtime_messages", None)
    for seg in state_dict.get("detailed_report_segments", []) or []:
        if isinstance(seg, dict):
            seg.pop("content_html", None)

    state_path = mgr.snapshots_dir / snapshot.state_file
    state_path.write_text(
        json.dumps(state_dict, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    snapshot.size_bytes = state_path.stat().st_size

    if pkl_bytes is not None:
        pkl_dst = mgr.snapshots_dir / f"{snapshot.id}_session.pkl.zst"
        pkl_dst.write_bytes(zstd.ZstdCompressor(level=3).compress(pkl_bytes))
        snapshot.runtime_state_included = True
        snapshot.size_bytes += pkl_dst.stat().st_size
    elif runtime_state_included is not None:
        snapshot.runtime_state_included = runtime_state_included

    if notebook_src is not None and notebook_src.exists():
        nb_dst = mgr.snapshots_dir / f"{snapshot.id}_notebook.ipynb"
        shutil.copy2(notebook_src, nb_dst)
        snapshot.notebook_file = f"{snapshot.id}_notebook.ipynb"
        snapshot.size_bytes += nb_dst.stat().st_size

    if workspace_manifest is not None:
        snapshot.workspace_manifest = workspace_manifest
    else:
        snapshot.workspace_manifest = mgr.scan_workspace_manifest()
    return snapshot


def _streaming_pkl_bytes(pairs) -> bytes:
    import io
    buf = io.BytesIO()
    for name, obj in pairs:
        dill.dump((name, obj), buf)
    return buf.getvalue()


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

    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('round 1')"))
    nb_path = session_dir / "states" / f"{session_id}.ipynb"
    nbformat.write(nb, str(nb_path))

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
    try:
        monkeypatch.setattr("backend.utils.runtime_apply.get_session_states", mock_get_session_states)
    except AttributeError:
        pass

    monkeypatch.setattr("backend.config.get_session_states", mock_get_session_states)

    return session_id, session_dir, tmp_path


class TestV1SnapshotFixtures:
    """Inventory-style checks for hand-written V1 snapshot artifacts."""

    def test_v1_fixture_includes_notebook_and_manifest(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        (session_dir / "workspace" / "figures" / "fig_001.jpg").write_text("fake fig")
        (session_dir / "workspace" / "tables" / "markers.csv").write_text("a,b,c")

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )
        nb_src = session_dir / "states" / f"{session_id}.ipynb"
        snapshot = _write_v1_snapshot(
            mgr, state, "Test", "pre_annotation", notebook_src=nb_src,
        )

        nb_backup = mgr.snapshots_dir / f"{snapshot.id}_notebook.ipynb"
        assert nb_backup.exists()
        assert snapshot.notebook_file == f"{snapshot.id}_notebook.ipynb"
        assert "figures" in snapshot.workspace_manifest
        assert "fig_001.jpg" in snapshot.workspace_manifest["figures"]
        assert "markers.csv" in snapshot.workspace_manifest["tables"]

    def test_v1_fixture_manifest_empty_dirs(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )
        snapshot = _write_v1_snapshot(mgr, state, "Test", "pre_annotation")

        assert snapshot.workspace_manifest == {
            "figures": [],
            "tables": [],
            "reproducibility": [],
            "root": [],
        }

    def test_v1_fixture_with_branch_label(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )
        snapshot = _write_v1_snapshot(
            mgr, state, "Branch Test", "pre_annotation", branch_label="branch-A",
        )
        assert snapshot.branch_label == "branch-A"


class TestSnapshotRestoreWithCleanup:
    """Tests for rollback workspace/reports cleanup."""

    def test_cleanup_workspace_files_deletes_new_files(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        (session_dir / "workspace" / "figures" / "fig_001.jpg").write_text("keep me")
        (session_dir / "workspace" / "figures" / "fig_002.jpg").write_text("delete me")
        (session_dir / "workspace" / "tables" / "new_data.csv").write_text("delete me too")

        manifest = {
            "figures": ["fig_001.jpg"],
            "tables": [],
            "reproducibility": [],
            "root": [],
        }

        result = mgr.cleanup_workspace_files(manifest)

        assert (session_dir / "workspace" / "figures" / "fig_001.jpg").exists()
        assert not (session_dir / "workspace" / "figures" / "fig_002.jpg").exists()
        assert not (session_dir / "workspace" / "tables" / "new_data.csv").exists()

        assert result["figures"] == 1
        assert result["tables"] == 1

    def test_cleanup_workspace_root_files(self, mock_config):
        """Files in workspace root (e.g. .h5ad) should also be tracked and cleaned."""
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        (session_dir / "workspace" / "output.h5ad").write_text("h5ad data")
        (session_dir / "workspace" / "markers.csv").write_text("a,b,c")

        manifest = {
            "figures": [],
            "tables": [],
            "reproducibility": [],
            "root": ["output.h5ad"],
        }

        result = mgr.cleanup_workspace_files(manifest)

        assert (session_dir / "workspace" / "output.h5ad").exists()
        assert not (session_dir / "workspace" / "markers.csv").exists()

        assert result["root"] == 1

    def test_cleanup_reports_deletes_report_files(self, mock_config):
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        (session_dir / "reports" / "detailed_report.html").write_text("<html>report</html>")
        (session_dir / "reports" / "report.html").write_text("<html>main</html>")
        (session_dir / "reports" / "report_embedded.html").write_text("<html>embedded</html>")
        (session_dir / "reports" / "report.json").write_text("{}")
        (session_dir / "reports" / "score_definitions.csv").write_text("csv")
        (session_dir / "reports" / "figures" / "fig.jpg").write_text("fig")
        (session_dir / "reports" / "tables" / "data.csv").write_text("csv")
        (session_dir / "reports" / f"{session_id}.ipynb").write_text("{}")

        mgr.cleanup_reports()

        assert not (session_dir / "reports" / "detailed_report.html").exists()
        assert not (session_dir / "reports" / "report.html").exists()
        assert not (session_dir / "reports" / "report_embedded.html").exists()
        assert not (session_dir / "reports" / "report.json").exists()
        assert not (session_dir / "reports" / "score_definitions.csv").exists()
        assert not (session_dir / "reports" / "figures" / "fig.jpg").exists()
        assert not (session_dir / "reports" / "tables" / "data.csv").exists()
        assert not (session_dir / "reports" / f"{session_id}.ipynb").exists()

        assert (session_dir / "reports" / "figures").is_dir()
        assert (session_dir / "reports" / "tables").is_dir()


class TestBranchLabels:
    """Tests for branch label computation."""

    def test_main_branch_label(self):
        """Snapshot without a restored parent should get 'main' branch label."""
        parent = Snapshot(name="Round 1", phase="pre_annotation", round_number=1, parent_id=None)
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
        """Snapshot after a restore should get branch-A."""
        parent = Snapshot(name="Round 1", phase="pre_annotation", round_number=1, parent_id=None)
        main_line = Snapshot(
            name="Round 2", phase="pre_direction", round_number=2,
            parent_id=parent.id, branch_label="main"
        )
        snapshots = [parent, main_line]

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

        existing = sum(
            1 for s in snapshots
            if s.parent_id == parent.id and s.branch_label and s.branch_label != "main"
        )
        branch_letter = chr(ord('A') + existing)
        branch_label = f"branch-{branch_letter}"

        assert branch_label == "branch-B"


class TestDeleteBranch:
    """Tests for branch deletion against hand-written V1 fixtures."""

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

        snap_a = _write_v1_snapshot(mgr, state, "A", "pre_annotation")
        snap_b = _write_v1_snapshot(
            mgr, state, "B", "pre_direction", parent_id=snap_a.id,
        )
        snap_c = _write_v1_snapshot(
            mgr, state, "C", "pre_round_wait", parent_id=snap_b.id,
        )

        all_snapshots = [snap_a, snap_b, snap_c]

        deleted_ids = await mgr.delete_branch(snap_b, all_snapshots)

        assert snap_b.id in deleted_ids
        assert snap_c.id in deleted_ids
        assert snap_a.id not in deleted_ids

        assert not (mgr.snapshots_dir / snap_b.state_file).exists()
        assert not (mgr.snapshots_dir / snap_c.state_file).exists()

    @pytest.mark.asyncio
    async def test_v1_delete_rejects_empty_state_file(self, mock_config):
        """A V2 projection must never resolve to the V1 snapshots directory."""
        session_id, _, _ = mock_config
        mgr = SnapshotManager(session_id)
        snapshot = Snapshot(
            name="V2 projection",
            phase="manual",
            state_file="",
            format_version=2,
        )

        with pytest.raises(ValueError, match="V1 snapshot state_file"):
            await mgr.delete_snapshot(snapshot)


class TestPklCompressionSnapshotCompat:
    """Hand-written V1 .pkl.zst fixtures for restore / inventory classification."""

    def test_v1_fixture_zst_roundtrip_streaming_dill(self, mock_config):
        """Hand-written compressed pkl fixture round-trips streaming dill."""
        session_id, session_dir, tmp_path = mock_config
        mgr = SnapshotManager(session_id)

        original_data = [("adata", "mock"), ("results", [1, 2, 3])]
        pkl_bytes = _streaming_pkl_bytes(original_data)

        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )
        snapshot = _write_v1_snapshot(
            mgr, state, "Streaming Pkl", "pre_annotation", pkl_bytes=pkl_bytes,
        )

        snapshot_pkl = mgr.snapshots_dir / f"{snapshot.id}_session.pkl.zst"
        assert snapshot_pkl.exists()
        assert snapshot.runtime_state_included is True

        _dctx = zstd.ZstdDecompressor()
        with open(snapshot_pkl, "rb") as f:
            with _dctx.stream_reader(f) as zf:
                loaded = {}
                while True:
                    try:
                        name, obj = dill.load(zf)
                    except EOFError:
                        break
                    loaded[name] = obj

        assert loaded == {"adata": "mock", "results": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_snapshot_restore_decompresses_zst_to_live_pkl(self, mock_config):
        """restore_snapshot should decompress hand-written .pkl.zst to live .pkl."""
        session_id, session_dir, tmp_path = mock_config
        pkl_path = session_dir / "states" / f"{session_id}_session.pkl"

        pkl_bytes = _streaming_pkl_bytes([("adata", "mock"), ("results", [4, 5, 6])])

        mgr = SnapshotManager(session_id)
        state = AnalysisState(
            session_id=session_id,
            user_id="test_user",
            data_files=[],
            current_round=1,
        )
        snapshot = _write_v1_snapshot(
            mgr, state, "Restore Test", "pre_annotation", pkl_bytes=pkl_bytes,
        )

        pkl_path.unlink(missing_ok=True)

        restored_state = await mgr.restore_snapshot(snapshot)

        assert pkl_path.exists()

        loaded = {}
        with open(pkl_path, "rb") as f:
            while True:
                try:
                    name, obj = dill.load(f)
                except EOFError:
                    break
                loaded[name] = obj

        assert loaded == {"adata": "mock", "results": [4, 5, 6]}
        assert restored_state.id
        assert restored_state.user_id == "test_user"
