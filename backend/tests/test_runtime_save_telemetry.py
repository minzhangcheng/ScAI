"""Bounded runtime-save telemetry: fixed keys, finite values, no paths."""
import asyncio
import math
import time

import pytest

from backend.utils.sandbox_parts.runtime_save import (
    SAVE_PHASE_NAMES,
    SavePhaseRecorder,
    _parse_save_phase_markers,
)


def test_phase_recorder_closes_the_interval_with_unattributed_overhead():
    recorder = SavePhaseRecorder()
    recorder.add("namespace_snapshot", 1.0)
    recorder.add("dill_serialization", 2.0)
    recorder.add("staged_write", 0.5)

    telemetry = recorder.snapshot(
        end_to_end_seconds=4.0,
        staged_bytes=1234,
        committed_bytes=1000,
    )

    phases = telemetry["last_save_phases_seconds"]
    assert set(phases) == set(SAVE_PHASE_NAMES) | {"unattributed_overhead"}
    assert phases["unattributed_overhead"] == 0.5
    assert math.isclose(
        sum(phases.values()),
        telemetry["last_save_end_to_end_seconds"],
        abs_tol=1e-6,
    )
    assert telemetry["last_save_staged_bytes"] == 1234
    assert telemetry["last_save_committed_bytes"] == 1000


def test_phase_recorder_omits_unknown_and_non_finite_values():
    recorder = SavePhaseRecorder()
    recorder.add("not_a_phase", 1000.0)
    recorder.add("cleanup", float("inf"))
    recorder.add("dill_serialization", -4.0)
    recorder.add("target_verification", 1.0)

    phases = recorder.snapshot(end_to_end_seconds=3.0)[
        "last_save_phases_seconds"
    ]
    assert "not_a_phase" not in phases
    assert phases["cleanup"] == 0.0
    assert phases["dill_serialization"] == 0.0
    assert phases["target_verification"] == 1.0
    assert phases["unattributed_overhead"] == 2.0
    for value in phases.values():
        assert math.isfinite(value) and value >= 0


def test_phase_recorder_rejects_material_over_count_instead_of_scaling():
    recorder = SavePhaseRecorder()
    recorder.add("namespace_snapshot", 2.0)
    recorder.add("dill_serialization", 2.0)

    with pytest.raises(ValueError, match="exceed end-to-end"):
        recorder.snapshot(end_to_end_seconds=1.0)


def test_phase_recorder_rejects_unbounded_aggregate_without_outer_interval():
    recorder = SavePhaseRecorder()
    for _ in range(8):
        recorder.add("namespace_snapshot", 86_400.0)

    with pytest.raises(ValueError, match="telemetry bound"):
        recorder.snapshot(end_to_end_seconds=None)


def test_kernel_marker_parser_accepts_one_complete_kernel_group():
    output = "\n".join([
        "noise",
        '__SCAI_SAVE_PHASE__:{"name":"namespace_snapshot","seconds":1.25}',
        '__SCAI_SAVE_PHASE__:{"name":"dill_serialization","seconds":0.75}',
        '__SCAI_SAVE_PHASE__:{"name":"staged_write","seconds":2.5,"bytes":42}',
    ])

    phases = _parse_save_phase_markers(output, outer_elapsed_seconds=5.0)

    assert [phase["name"] for phase in phases] == [
        "namespace_snapshot", "dill_serialization", "staged_write",
    ]
    assert phases[0]["seconds"] == 1.25
    assert phases[2]["bytes"] == 42
    recorder = SavePhaseRecorder()
    recorder.add_kernel_phases(phases)
    assert set(recorder.snapshot(end_to_end_seconds=5)[
        "last_save_phases_seconds"
    ]) == set(SAVE_PHASE_NAMES) | {"unattributed_overhead"}


def _valid_kernel_markers() -> list[str]:
    return [
        '__SCAI_SAVE_PHASE__:{"name":"namespace_snapshot","seconds":0.25}',
        '__SCAI_SAVE_PHASE__:{"name":"dill_serialization","seconds":0.5}',
        '__SCAI_SAVE_PHASE__:{"name":"staged_write","seconds":0.75}',
    ]


@pytest.mark.parametrize("mutate", [
    lambda lines: lines + [
        '__SCAI_SAVE_PHASE__:{"name":"namespace_snapshot","seconds":0.1}',
    ],
    lambda lines: lines + [
        '__SCAI_SAVE_PHASE__:{"name":"spoofed","seconds":0.1}',
    ],
    lambda lines: lines + ['__SCAI_SAVE_PHASE__:not-json'],
    lambda lines: lines[:2] + [
        '__SCAI_SAVE_PHASE__:{"name":"staged_write","seconds":-1}',
    ],
    lambda lines: lines[:2] + [
        '__SCAI_SAVE_PHASE__:{"name":"staged_write","seconds":NaN}',
    ],
    lambda lines: lines[:2] + [
        '__SCAI_SAVE_PHASE__:{"name":"staged_write","seconds":0.75,"bytes":"bad"}',
    ],
    lambda lines: lines[:2],
])
def test_kernel_marker_parser_rejects_untrusted_marker_groups(mutate):
    assert _parse_save_phase_markers(
        "\n".join(mutate(_valid_kernel_markers())),
        outer_elapsed_seconds=2.0,
    ) == []


def test_kernel_marker_parser_rejects_phase_total_over_outer_duration():
    assert _parse_save_phase_markers(
        "\n".join(_valid_kernel_markers()), outer_elapsed_seconds=1.0,
    ) == []


def test_kernel_marker_parser_rejects_invalid_outer_duration():
    assert _parse_save_phase_markers(
        "\n".join(_valid_kernel_markers()), outer_elapsed_seconds="bad",
    ) == []


def test_phase_vocabulary_is_fixed_and_complete():
    assert len(SAVE_PHASE_NAMES) == len(set(SAVE_PHASE_NAMES))
    for required in (
        "namespace_snapshot",
        "dill_serialization",
        "staged_write",
        "staged_digest_hmac",
        "previous_generation_backup",
        "metadata_publication",
        "graph_workspace_capture",
        "target_verification",
        "cleanup",
    ):
        assert required in SAVE_PHASE_NAMES


def test_notebook_state_projection_changes_only_telemetry_fields(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state = NotebookState(
        session_id="telemetry-session",
        state_dir=str(tmp_path / "states"),
    )
    state.runtime_persistence.update({
        "status": "current",
        "current_revision": 7,
        "saved_revision": 7,
        "committed_revision": 7,
        "runtime_digest": "a" * 64,
    })
    before = {
        name: state.runtime_persistence.get(name)
        for name in (
            "status", "current_revision", "saved_revision",
            "committed_revision", "runtime_digest",
        )
    }

    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={
            "namespace_snapshot": 1.0,
            "attacker_phase": 9.0,
            "helper_path": "/etc/passwd",
        },
        last_save_end_to_end_seconds=2.0,
        last_save_staged_bytes=123,
        last_save_committed_bytes=-5,
        persist=False,
    )

    assert returned["last_save_phases_seconds"] == {
        "namespace_snapshot": 1.0,
    }
    assert returned["last_save_end_to_end_seconds"] == 2.0
    assert returned["last_save_staged_bytes"] == 123
    assert returned["last_save_committed_bytes"] is None
    assert {
        name: returned.get(name) for name in before
    } == before
    serialized = repr(returned)
    assert "/etc/passwd" not in serialized


def test_notebook_state_rejects_material_over_count_without_mutating_state(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state = NotebookState(
        session_id="telemetry-over-count",
        state_dir=str(tmp_path / "states"),
    )
    state.update_runtime_persistence(
        "current", current_revision=2, saved_revision=2,
        runtime_digest="2" * 64, persist=False,
    )
    before = dict(state.runtime_persistence)

    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={
            "namespace_snapshot": 2.0,
            "dill_serialization": 2.0,
        },
        last_save_end_to_end_seconds=1.0,
        expected_revision=2,
        persist=False,
    )

    assert returned == before
    assert state.runtime_persistence == before


def test_notebook_state_rejects_unbounded_aggregate_without_outer_duration(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state = NotebookState(
        session_id="telemetry-unbounded",
        state_dir=str(tmp_path / "states"),
    )
    state.update_runtime_persistence(
        "current", current_revision=2, saved_revision=2,
        runtime_digest="3" * 64, persist=False,
    )
    before = dict(state.runtime_persistence)
    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={
            "namespace_snapshot": 86_400.0,
            "dill_serialization": 86_400.0,
            "staged_write": 86_400.0,
            "staged_digest_hmac": 86_400.0,
            "previous_generation_backup": 86_400.0,
            "metadata_publication": 86_400.0,
            "graph_workspace_capture": 86_400.0,
            "target_verification": 86_400.0,
        },
        expected_revision=2,
        persist=False,
    )

    assert returned == before
    assert state.runtime_persistence == before


def test_host_phase_recorder_closes_commit_phases(tmp_path):
    from backend.utils.sandbox_parts.regular_save import _commit_pkl_transaction, _verify_pkl_integrity

    live = tmp_path / "s_session.pkl"
    staged = tmp_path / "s_session.pkl.tmp"
    integrity = tmp_path / "integrity"
    staged.write_bytes(b"new-generation")

    recorder = SavePhaseRecorder()
    _commit_pkl_transaction(
        staged,
        live,
        integrity,
        lambda: None,
        phase_recorder=recorder,
    )
    assert _verify_pkl_integrity(live, integrity)

    telemetry = recorder.snapshot(end_to_end_seconds=1.0)
    phases = telemetry["last_save_phases_seconds"]
    assert "staged_digest_hmac" in phases
    assert "cleanup" in phases
    assert telemetry["last_save_end_to_end_seconds"] == 1.0
    assert math.isclose(
        sum(phases.values()), telemetry["last_save_end_to_end_seconds"],
        abs_tol=1e-6,
    )


def test_telemetry_publication_survives_reload(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    session_id = "telemetry-reload"
    state_dir = tmp_path / "states"
    state = NotebookState(session_id=session_id, state_dir=str(state_dir))
    state.record_runtime_save_telemetry(
        last_save_phases_seconds={"namespace_snapshot": 1.5},
        last_save_end_to_end_seconds=2.5,
        last_save_staged_bytes=100,
        last_save_committed_bytes=80,
    )

    reloaded = NotebookState(session_id=session_id, state_dir=str(state_dir))
    persistence = reloaded.runtime_persistence
    assert persistence["last_save_phases_seconds"] == {
        "namespace_snapshot": 1.5,
    }
    assert persistence["last_save_end_to_end_seconds"] == 2.5
    assert persistence["last_save_staged_bytes"] == 100
    assert persistence["last_save_committed_bytes"] == 80


def test_new_generation_clears_previous_profile_before_its_profile_is_recorded(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state = NotebookState(session_id="telemetry-generations", state_dir=str(tmp_path / "states"))
    state.update_runtime_persistence(
        "current", current_revision=1, saved_revision=1,
        runtime_digest="1" * 64, persist=False,
    )
    state.record_runtime_save_telemetry(
        last_save_phases_seconds={"cleanup": 1.0},
        last_save_end_to_end_seconds=1.0,
        expected_revision=1,
        persist=False,
    )
    assert state.runtime_persistence["last_save_phases_seconds"] == {"cleanup": 1.0}

    state.update_runtime_persistence(
        "current", current_revision=2, saved_revision=2,
        runtime_digest="2" * 64, persist=False,
    )
    assert all(key not in state.runtime_persistence for key in (
        "last_save_phases_seconds", "last_save_end_to_end_seconds",
        "last_save_staged_bytes", "last_save_committed_bytes",
    ))


def test_failed_profile_publication_keeps_committed_generation_without_profile(tmp_path, monkeypatch):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state_dir = tmp_path / "states"
    state = NotebookState(session_id="telemetry-failed-publication", state_dir=str(state_dir))
    state.update_runtime_persistence(
        "current", current_revision=3, saved_revision=3,
        runtime_digest="3" * 64,
    )
    monkeypatch.setattr(state, "save_metadata", lambda **_kwargs: False)
    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={"cleanup": 1.0},
        last_save_end_to_end_seconds=1.0,
        expected_revision=3,
    )
    assert returned["committed_revision"] == 3
    assert "last_save_phases_seconds" not in returned
    reloaded = NotebookState(
        session_id="telemetry-failed-publication", state_dir=str(state_dir),
    )
    assert reloaded.runtime_persistence["committed_revision"] == 3
    assert "last_save_phases_seconds" not in reloaded.runtime_persistence


def test_late_profile_writer_cannot_attach_old_generation_to_new_generation(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state_dir = tmp_path / "states"
    state = NotebookState(session_id="telemetry-late-writer", state_dir=str(state_dir))
    state.update_runtime_persistence(
        "current", current_revision=4, saved_revision=4,
        runtime_digest="4" * 64,
    )
    state.update_runtime_persistence(
        "current", current_revision=5, saved_revision=5,
        runtime_digest="5" * 64,
    )
    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={"cleanup": 4.0},
        last_save_end_to_end_seconds=4.0,
        expected_revision=4,
    )
    assert returned["committed_revision"] == 5
    assert "last_save_phases_seconds" not in returned


def test_same_revision_late_writer_cannot_replace_new_commit_identity(tmp_path):
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state = NotebookState(
        session_id="telemetry-same-revision", state_dir=str(tmp_path / "states"),
    )
    first = state.update_runtime_persistence(
        "current", current_revision=4, saved_revision=4,
        runtime_digest="a" * 64,
        committed_pairing={"state_version": 1},
    )
    first_identity = state._persistence_generation_identity(first)
    second = state.update_runtime_persistence(
        "current", current_revision=4, saved_revision=4,
        runtime_digest="a" * 64,
        committed_pairing={"state_version": 2},
    )

    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={"cleanup": 1.0},
        last_save_end_to_end_seconds=1.0,
        expected_revision=4,
        expected_generation_identity=first_identity,
    )

    assert returned == second
    assert returned["committed_digest"] == "a" * 64
    assert returned["committed_pairing"] == {"state_version": 2}
    assert "last_save_phases_seconds" not in returned


def test_profile_publication_binds_expected_lifecycle_generation(
    tmp_path, monkeypatch,
):
    from backend.services import session_lifecycle
    from backend.utils.sandbox_parts.notebook_state import NotebookState

    state = NotebookState(
        session_id="telemetry-lifecycle", state_dir=str(tmp_path / "states"),
    )
    committed = state.update_runtime_persistence(
        "current", current_revision=2, saved_revision=2,
        runtime_digest="c" * 64,
    )
    before = dict(state.runtime_persistence)
    seen = []

    def reject_replaced_generation(_session_id, *, expected_generation=None):
        seen.append(expected_generation)
        raise session_lifecycle.SessionGenerationError("generation replaced")

    monkeypatch.setattr(
        session_lifecycle,
        "assert_session_generation_active",
        reject_replaced_generation,
    )
    returned = state.record_runtime_save_telemetry(
        last_save_phases_seconds={"cleanup": 1.0},
        last_save_end_to_end_seconds=1.0,
        expected_revision=2,
        expected_generation_identity=(
            state._persistence_generation_identity(committed)
        ),
        expected_lifecycle_generation=7,
    )

    assert seen == [7]
    assert returned == before
    assert state.runtime_persistence == before


@pytest.mark.asyncio
async def test_no_staged_execute_save_attributes_compatibility_hmac(
    tmp_path, monkeypatch,
):
    import backend.utils.sandbox as sandbox_module
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    sandbox = LocalJupyterSandbox(
        session_id="telemetry-compat-hmac",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sandbox.runtime_revision = 1
    sandbox.saved_runtime_revision = 0
    sandbox._exec_lock = asyncio.Lock()

    async def execute_legacy_save(_code, **_kwargs):
        sandbox.notebook_state.session_path.write_bytes(b"legacy-runtime")
        return CodeExecutionResult(success=True, output="Session saved: 1 variables")

    sandbox._execute_unlocked = execute_legacy_save
    integrity_dir = tmp_path / "integrity"
    monkeypatch.setattr(
        sandbox_module, "_integrity_dir", lambda _session_id: integrity_dir,
    )
    write_signature = sandbox_module._write_pkl_signature

    def slow_signature(pkl_path, target_integrity_dir):
        time.sleep(0.01)
        return write_signature(pkl_path, target_integrity_dir)

    monkeypatch.setattr(
        sandbox_module, "_write_pkl_signature", slow_signature,
    )

    assert await sandbox.save_session() is True

    phases = sandbox.notebook_state.runtime_persistence[
        "last_save_phases_seconds"
    ]
    assert phases["staged_digest_hmac"] >= 0.01
