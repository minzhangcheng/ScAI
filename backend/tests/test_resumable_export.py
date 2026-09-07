"""What ``include_runtime=true`` has to mean. Spec 5.4.

An archive that imports cleanly and cannot continue is the failure this batch
exists to remove, and it is the one a passing import does not reveal:
``import_state`` compares the archived pairing against what the archive carries
and retires both the runtime and the checkpoint when they disagree, reporting a
successful import either way.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import pathlib
import tarfile

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


# ---------------------------------------------------------------------------
# Task 2: freeze, flush, verify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resumable_export_flushes_the_kernel_before_archiving(monkeypatch):
    """The archive must carry the generation the flush produced, not an older one."""
    from backend.services import resumable_export

    sandbox = _Sandbox()
    monkeypatch.setattr(resumable_export, "live_sandbox", lambda _sid: sandbox)
    monkeypatch.setattr(
        resumable_export, "_capture_descriptor", lambda _sid, _state: {"pairing": {"state_version": 1, "checkpoint": None}},
    )

    await resumable_export.flush_for_resumable_export("resumable", _state())

    assert sandbox.saves == 1


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


@pytest.mark.asyncio
async def test_a_cold_session_without_committed_runtime_is_refused(monkeypatch):
    from backend.services import resumable_export

    monkeypatch.setattr(resumable_export, "live_sandbox", lambda _sid: None)
    with pytest.raises(resumable_export.ResumableExportUnavailable) as refusal:
        await resumable_export.flush_for_resumable_export("resumable", _state())
    assert refusal.value.reason == "generation_unverified"


@pytest.mark.asyncio
async def test_a_results_only_export_does_not_flush():
    """`include_runtime=false` must not pause an analysis to build an archive."""
    source = (
        BACKEND / "routes" / "export_import_parts" / "export_routes.py"
    ).read_text()
    packer = source[source.index("async def _pack_session_archive("):]
    packer = packer[:packer.index("async def _retire_failed_export")]

    assert "if include_runtime:\n        async with frozen_resumable_export(" in packer
    freeze_at = packer.index("frozen_resumable_export(")
    guard_at = packer.index("if include_runtime:")
    assert guard_at < freeze_at


def test_an_unverifiable_generation_is_refused(tmp_path, monkeypatch):
    """Spec 4.4. A dump that fails its own HMAC is not a generation."""
    from backend.services.resumable_export import (
        ResumableExportUnavailable,
        verify_committed_generation,
    )
    from backend.utils import runtime_apply

    def _fail(_session_id):
        raise runtime_apply.RuntimeApplyError("bad signature")

    monkeypatch.setattr(runtime_apply, "verify_live_pkl_or_raise", _fail)

    with pytest.raises(ResumableExportUnavailable) as refusal:
        verify_committed_generation("resumable")
    assert refusal.value.reason == "generation_unverified"


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


def test_a_held_boundary_is_not_refused(monkeypatch):
    from backend.services import resumable_export
    from backend.services.resumable_export import assert_boundary_held

    pairing = {"state_version": 1, "checkpoint": "a"}
    monkeypatch.setattr(resumable_export, "_pairing", lambda _sid, _state: dict(pairing))

    assert_boundary_held("resumable", _state(), dict(pairing))


def test_the_boundary_check_runs_after_the_tar_is_built():
    """The checkpoint copy happens inside `_build_session_tar`.

    Checking before it would leave exactly the window this exists to close.
    """
    source = (
        BACKEND / "routes" / "export_import_parts" / "export_routes.py"
    ).read_text()
    packer = source[source.index("async def _pack_session_archive("):]
    packer = packer[:packer.index("async def _retire_failed_export")]

    capture_at = packer.index("result = await _pack_frozen_session_archive(")
    check_at = packer.index("assert_boundary_held(")
    assert capture_at < check_at
    assert packer.index("_build_session_tar,") < packer.index("verify_captured_archive,")


def test_the_refusal_is_a_409_the_caller_can_act_on():
    """Not a 500. Nothing is broken; a results-only export still works."""
    source = (
        BACKEND / "routes" / "export_import_parts" / "export_routes.py"
    ).read_text()

    assert source.count("except ResumableExportUnavailable as unavailable:") == 2
    assert "status_code=409" in source
    assert '"results_only_export_available": True' in source


# ---------------------------------------------------------------------------
# Task 3: the checkpoint copy (green at the base ref, kept as regression guards)
# ---------------------------------------------------------------------------


def test_the_archive_carries_a_backup_api_copy_of_the_checkpoint():
    """Green before this batch: `create_checkpoint_export_snapshot` already did
    this. Pinned because a raw copy of a live WAL database is not a snapshot and
    the failure would be silent."""
    source = (
        BACKEND / "routes" / "export_import_parts" / "tar_builders.py"
    ).read_text()

    assert "create_checkpoint_export_snapshot(" in source
    assert 'if state_file.name.startswith("checkpoints.sqlite"):' in source


def test_the_archive_carries_no_wal_or_shm_companion():
    """Shipping the companions of a database copied mid-transaction is how a
    successful import yields an unusable graph."""
    source = (
        BACKEND / "routes" / "export_import_parts" / "tar_builders.py"
    ).read_text()
    packer = source[source.index("def _add_states_to_tar("):]
    packer = packer[:packer.index("\ndef ", 10)]

    assert '-wal' not in packer
    assert '-shm' not in packer


# ---------------------------------------------------------------------------
# Task 4: the archive records the facts
# ---------------------------------------------------------------------------


def _manifest_of(tar_path, session_id):
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.extractfile(f"{session_id}/manifest.json")
        assert member is not None
        return json.loads(member.read().decode("utf-8"))


def test_the_manifest_names_the_generation_its_coherence_and_its_coverage():
    """An archive is read long after its HTTP response is gone. Spec 5.4."""
    from backend.routes.export_import_parts.export_routes import _generation_facts

    facts = _generation_facts("resumable", {
        "runtime_persistence": {
            "coherence": "current",
            "committed_revision": 4,
            "committed_digest": "d" * 64,
            "saved_names": ["adata", "markers"],
            "failed_variables": ["model"],
        },
    }, True)

    assert facts["runtime_coherence"] == "current"
    assert facts["runtime_committed_revision"] == 4
    assert facts["runtime_committed_digest"] == "d" * 64
    assert facts["runtime_saved_names"] == ["adata", "markers"]
    assert facts["runtime_skipped_names"] == ["model"]
    assert "runtime_omitted_reason" not in facts


def test_a_results_only_manifest_says_why_it_carries_no_runtime():
    from backend.routes.export_import_parts.export_routes import _generation_facts

    facts = _generation_facts("resumable", {}, False)
    assert facts["runtime_omitted_reason"] == "include_runtime=false"


def test_the_manifest_declares_whether_the_archive_can_continue_the_analysis():
    source = (
        BACKEND / "routes" / "export_import_parts" / "export_routes.py"
    ).read_text()
    assert '"resumable": bool(include_runtime),' in source
    assert "**_generation_facts(session_id, state_payload, include_runtime)," in source


# ---------------------------------------------------------------------------
# Task 4: a resumable archive's session-id collision
# ---------------------------------------------------------------------------


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


def test_a_resumable_archive_landing_on_its_own_id_is_allowed():
    from backend.routes.export_import_parts.import_tar import (
        _refuse_resumable_collision,
    )

    _refuse_resumable_collision({"resumable": True}, _Context("same", "same"))


def test_a_results_only_archive_still_reallocates():
    """It carries no checkpoint, so it loses nothing by landing under a new id.

    Refusing here would break re-importing a backup onto the installation that
    produced it, which is the common case and which the reallocation exists to
    make non-destructive.
    """
    from backend.routes.export_import_parts.import_tar import (
        _refuse_resumable_collision,
    )

    _refuse_resumable_collision(
        {"resumable": False}, _Context("original", "reallocated"),
    )


def test_an_archive_from_before_this_field_keeps_the_old_behaviour():
    """It makes no claim, and it has no checkpoint to lose."""
    from backend.routes.export_import_parts.import_tar import (
        _refuse_resumable_collision,
    )

    _refuse_resumable_collision({}, _Context("original", "reallocated"))


def test_the_reallocation_itself_is_unchanged_for_everything_else():
    """The refusal is scoped; the destructive-overwrite protection is not."""
    source = (
        BACKEND / "routes" / "export_import_parts" / "import_identity.py"
    ).read_text()
    allocate = source[source.index("def _allocate_import_session_id("):]
    allocate = allocate[:allocate.index("\ndef ", 10)] if "\ndef " in allocate[10:] else allocate

    assert "new_id = str(uuid.uuid4())" in allocate
    assert "already exists on this installation" in allocate
