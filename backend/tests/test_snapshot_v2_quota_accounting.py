"""Quota accounting for Snapshot V2 capture.

Locks real capture-path behaviour: first capture of new content must count
workspace (and other newly written) object sizes in ``physical_added_bytes``;
a second identical capture must not re-count those same physical objects.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.services.snapshot_v2.models import ObjectRef
from backend.services.snapshot_v2.object_store import ObjectStore


def test_physical_added_bytes_counts_created_runtime_refs():
    created_refs = []
    physical_added = 0

    def track(ref: ObjectRef, created: bool) -> None:
        nonlocal physical_added
        if created:
            physical_added += ref.size
        created_refs.append(ref.sha256)

    runtime_ref = ObjectRef(
        sha256="1" * 64,
        size=1024,
        media_type="application/octet-stream",
    )

    track(runtime_ref, created=True)

    assert created_refs == ["1" * 64]
    assert physical_added == 1024


def test_register_existing_ref_is_not_creation_accounting(tmp_path):
    objects = tmp_path / "objects"
    staging = tmp_path / "staging"
    objects.mkdir()
    staging.mkdir()
    store = ObjectStore(
        SimpleNamespace(
            objects=objects,
            staging=staging,
            root=tmp_path / "v2",
            journals=tmp_path / "journals",
            integrity=tmp_path / "integrity",
            committed=tmp_path / "committed",
        )
    )
    source = tmp_path / "runtime.pkl"
    source.write_bytes(b"x" * 1024)

    put_result = store.put_file(source, "application/octet-stream")

    assert put_result.created is True
    # Object now exists on disk; register_existing_ref succeeds — but that
    # must NOT be treated as "already existed before this capture".
    assert store.register_existing_ref(put_result.ref) is True
    assert put_result.ref.size == 1024


# ---- Real capture-path accounting -------------------------------------------


def _make_v2_paths(tmp_path: Path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [
                self.root,
                self.objects,
                self.staging,
                self.journals,
                self.integrity,
                self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def _make_analysis_state(session_id: str = "sess-quota"):
    from backend.models.schemas import AnalysisState

    return AnalysisState(
        id=session_id,
        name="Quota",
        user_id="u1",
        initial_instruction="analyze",
    )


def _make_capture_request(session_id: str = "sess-quota"):
    from backend.services.snapshot_v2.models import (
        IdleCapsule,
        SnapshotCaptureRequest,
        empty_complete_runtime,
    )

    return SnapshotCaptureRequest(
        session_id=session_id,
        reason="manual",
        capsule=IdleCapsule(graph_version="2.0"),
        runtime=empty_complete_runtime(),
    )


@pytest.mark.asyncio
async def test_prewrite_estimate_counts_known_workspace_and_state_artifacts(
    tmp_path, monkeypatch,
):
    from backend.services.snapshot_v2.service import SnapshotService

    paths = _make_v2_paths(tmp_path)
    service = SnapshotService("sess-quota", paths)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "table.csv").write_bytes(b"x" * 321)
    states = tmp_path / "states"
    states.mkdir()
    (states / "sess-quota.ipynb").write_bytes(b"n" * 123)
    monkeypatch.setattr(
        "backend.services.snapshot_v2.service.get_session_states",
        lambda _sid: states,
    )

    estimate = await service._estimate_capture_addition(
        _make_capture_request(),
        str(workspace),
        _make_analysis_state(),
    )

    assert estimate >= 321 + 123


def _workspace_object_sizes(workspace: Path) -> dict[str, int]:
    """sha256 -> size for scoped workspace files (root + figures/tables/repro)."""
    sizes: dict[str, int] = {}
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        # Match workspace capture scope used by production capture.
        if "/" in rel:
            top = rel.split("/", 1)[0]
            if top not in ("figures", "tables", "reproducibility"):
                continue
        data = path.read_bytes()
        sizes[hashlib.sha256(data).hexdigest()] = len(data)
    return sizes


@pytest.mark.asyncio
async def test_capture_path_physical_added_includes_workspace_and_is_not_recounted(
    tmp_path,
):
    """Real capture: first pass counts new workspace bytes; second pass does not."""
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions

    session_id = "sess-quota"
    v2_paths = _make_v2_paths(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Distinct payload so hashes are unique to this test run.
    payload = b"quota-workspace-bytes-" + (b"W" * 512)
    (workspace / "root.txt").write_bytes(payload)
    figures = workspace / "figures"
    figures.mkdir()
    (figures / "plot.png").write_bytes(b"png-" + payload)

    states = tmp_path / "states"
    states.mkdir()
    (states / f"{session_id}.ipynb").write_text(
        json.dumps(
            {
                "cells": [],
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
            }
        )
    )

    analysis_state = _make_analysis_state(session_id)
    expected_workspace = _workspace_object_sizes(workspace)
    assert expected_workspace, "fixture must produce workspace object hashes"
    expected_workspace_bytes = sum(expected_workspace.values())

    sessions[session_id] = analysis_state
    try:
        service = SnapshotService(session_id, v2_paths)
        with patch(
            "backend.services.snapshot_v2.service.get_session_states",
            return_value=states,
        ):
            # Cold store: no objects before first capture.
            assert not any(v2_paths.objects.iterdir())

            first = await service._capture_locked(
                _make_capture_request(session_id),
                workspace_root=str(workspace),
                workflow_state={"kernel_present": False},
                analysis_state=analysis_state,
                persist_session=False,
            )
            assert first.failure is None, f"first capture failed: {first.failure}"
            first_physical = first.manifest.physical_added_bytes
            # Logical size of every declared object — on a cold store every declared
            # object is newly written, so physical must equal this sum (not merely
            # analysis/notebook leftovers that make physical > 0).
            first_logical = sum(o.size for o in first.manifest.objects.values())
            assert first.manifest.logical_size_bytes == first_logical
            assert first_physical == first_logical

            # Newly written workspace objects must contribute to physical quota.
            assert first_physical >= expected_workspace_bytes
            for sha, size in expected_workspace.items():
                assert sha in first.manifest.objects
                assert first.manifest.objects[sha].size == size

            # All first-capture object hashes now exist on disk.
            first_hashes = set(first.manifest.objects)
            on_disk_after_first = {
                p.name for p in v2_paths.objects.iterdir() if p.is_file()
            }
            assert first_hashes.issubset(on_disk_after_first)

            second = await service._capture_locked(
                _make_capture_request(session_id),
                workspace_root=str(workspace),
                workflow_state={"kernel_present": False},
                analysis_state=analysis_state,
                persist_session=False,
            )
            assert second.failure is None, f"second capture failed: {second.failure}"
            second_physical = second.manifest.physical_added_bytes

            # Same workspace content: those hashes must not re-count as physical adds.
            shared = first_hashes & set(second.manifest.objects)
            assert expected_workspace.keys() <= shared

            # Non-vacuous second-capture check: physical equals only hashes that
            # were not already on disk after the first capture (not "h not in
            # on_disk_after_first" over workspace hashes that are already present).
            expected_second_physical = sum(
                o.size
                for h, o in second.manifest.objects.items()
                if h not in on_disk_after_first
            )
            assert second_physical == expected_second_physical
            # Workspace payload was counted in first physical and is fully reused.
            assert first_physical - second_physical >= expected_workspace_bytes
            assert second_physical < first_physical
            assert second_physical < second.manifest.logical_size_bytes
            assert second.manifest.logical_size_bytes > 0
    finally:
        sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_quota_failure_cannot_be_bypassed_by_identical_retry(tmp_path):
    from backend.services.snapshot_v2.service import SnapshotService
    from backend.state import sessions

    session_id = "sess-quota-retry"
    paths = _make_v2_paths(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_bytes(b"quota-retry-payload")
    states = tmp_path / "states"
    states.mkdir()
    (states / f"{session_id}.ipynb").write_text(
        json.dumps({"cells": [], "nbformat": 4, "nbformat_minor": 5})
    )
    state = _make_analysis_state(session_id)
    sessions[session_id] = state

    try:
        service = SnapshotService(session_id, paths, session_quota_bytes=1)
        failures = []
        with patch(
            "backend.services.snapshot_v2.service.get_session_states",
            return_value=states,
        ):
            for _ in range(2):
                record = await service._capture_locked(
                    _make_capture_request(session_id),
                    workspace_root=str(workspace),
                    workflow_state={"kernel_present": False},
                    analysis_state=state,
                    persist_session=False,
                )
                failures.append(record.failure)

        assert all(failure is not None for failure in failures)
        assert [failure.code for failure in failures] == [
            "snapshot_quota_exceeded",
            "snapshot_quota_exceeded",
        ]
        assert list(paths.committed.glob("*.json")) == []
        assert list(paths.objects.iterdir()) == []
    finally:
        sessions.pop(session_id, None)
