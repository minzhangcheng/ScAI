"""What ``include_runtime=true`` has to mean. Spec 5.4.

An archive that imports cleanly and cannot continue is the failure this batch
exists to remove, and it is the one a passing import does not reveal:
``import_state`` compares the archived pairing against what the archive carries
and retires both the runtime and the checkpoint when they disagree, reporting a
successful import either way.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"


class _Sandbox:
    """A kernel-bearing sandbox whose save can be told to fail."""

    def __init__(self, *, committed=True, raises=None):
        self.km = object()
        self._exec_lock = asyncio.Lock()
        self._initialized = True
        self.notebook_state = SimpleNamespace(save=lambda: None)
        self.sync_runtime_persistence_state = lambda _state: None
        self.committed = committed
        self.raises = raises
        self.saves = 0

    async def _save_session_locked(self):
        self.saves += 1
        if self.raises is not None:
            raise self.raises
        return self.committed


def _state(session_id="resumable", version=1):
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id=session_id, name="r", user_id="u")
    state.state_version = version
    return state


# Task 2: freeze, flush, verify


@pytest.mark.asyncio
async def test_a_resumable_export_refuses_when_the_flush_fails(monkeypatch):
    """No half-resumable archive. Spec 5.4.

    Falling back to the previous committed generation would ship a *new* state
    beside an *old* runtime -- exactly the incoherence 5.1 names -- and this
    design keeps no matching older state and checkpoint to pair it with.
    """
    from backend.services import resumable_export
    from backend.services.resumable_export import ResumableExportUnavailable

    for sandbox in (_Sandbox(committed=False), _Sandbox(raises=OSError("no space"))):
        monkeypatch.setattr(resumable_export, "live_sandbox", lambda _sid, s=sandbox: s)
        with pytest.raises(ResumableExportUnavailable) as refusal:
            await resumable_export.flush_for_resumable_export("resumable", _state())
        assert refusal.value.reason == "flush_failed"
        assert "without runtime" in refusal.value.detail


def test_the_execution_boundary_is_checked_across_the_whole_capture(monkeypatch):
    """State, runtime and checkpoint must describe one moment.

    A graph step between the flush and the checkpoint copy makes the archived
    pairing disagree with the archive's own files. The destination notices and
    retires both -- after reporting a successful import. Refusing here delivers
    the same verdict where the user can act on it.
    """
    from backend.services import resumable_export
    from backend.services.resumable_export import (
        ResumableExportUnavailable,
        assert_boundary_held,
    )

    state = _state(version=1)
    monkeypatch.setattr(
        resumable_export, "_pairing",
        lambda _sid, _state: {"state_version": 2, "checkpoint": "b"},
    )

    with pytest.raises(ResumableExportUnavailable) as refusal:
        assert_boundary_held(
            "resumable", state, {"state_version": 1, "checkpoint": "a"},
        )
    assert refusal.value.reason == "boundary_moved"


# Task 3: the checkpoint copy (green at the base ref, kept as regression guards)


# Task 4: the archive records the facts


def test_the_manifest_names_the_generation_its_coherence_and_its_coverage():
    """An archive is read long after its HTTP response is gone. Spec 5.4.

    The coverage half of this used to be handed in through
    `runtime_persistence["saved_names"]` -- a key nothing in production writes,
    so the assertion proved the plumbing would work if the key existed while
    the shipped field was always null. Coverage lives in the generation
    manifest beside the dump, so the fixture writes one.
    """
    from backend.config import get_session_states
    from backend.routes.export_import_parts.export_routes import _generation_facts
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )

    sid = "resumable"
    get_session_states(sid).mkdir(parents=True, exist_ok=True)
    write_runtime_manifest(
        runtime_manifest_path(sid),
        build_runtime_manifest(
            revision=4,
            state_revision=4,
            pkl_digest="d" * 64,
            saved_names=["adata", "markers"],
            skipped_names=[{
                "name": "model", "type": "Booster",
                "reason": "serialization_failed",
            }],
            serializer_version="dill-zstd-1",
        ),
    )

    facts = _generation_facts(sid, {
        "runtime_persistence": {
            "coherence": "current",
            "committed_revision": 4,
            "committed_digest": "d" * 64,
        },
    }, True)

    assert facts["runtime_coherence"] == "current"
    assert facts["runtime_committed_revision"] == 4
    assert facts["runtime_committed_digest"] == "d" * 64
    assert facts["runtime_saved_names"] == ["adata", "markers"]
    assert [s["name"] for s in facts["runtime_skipped_names"]] == ["model"]
    assert "runtime_omitted_reason" not in facts


# Task 4: a resumable archive's session-id collision


class _Context:
    def __init__(self, archive_session_id, session_id):
        self.archive_session_id = archive_session_id
        self.session_id = session_id


def test_a_resumable_archive_refuses_a_reallocated_session_id():
    """Spec 3.6. The checkpoint is keyed by session id and cannot be renamed.

    A reallocated id drops it: the runtime pkl still lands and the import still
    reports success, so the user learns about it only when Resume restarts the
    round. That is why this is refused rather than warned about.
    """
    from fastapi import HTTPException

    from backend.routes.export_import_parts.import_tar import (
        _refuse_resumable_collision,
    )

    with pytest.raises(HTTPException) as refusal:
        _refuse_resumable_collision(
            {"resumable": True}, _Context("original", "reallocated"),
        )
    assert refusal.value.status_code == 409
    assert "already exists" in str(refusal.value.detail)
    assert "without runtime" in str(refusal.value.detail)
