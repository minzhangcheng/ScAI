"""Which decision, how valuable it is, and what the listing says. Part B 3.3, 4.1, 4.7, 4.8.

Three things that have to agree and used to have no single owner: the gate the
workflow node is about to present, the gate the capture records, and the order
three separate consumers give up snapshots in when the disk fills.
"""
import json
import types

import pytest

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
)
from backend.models.schemas import AnalysisPhase

try:
    from backend.config import _config
    from backend.services.round_snapshots.gates import (
        deletion_sort_key,
        in_deletion_order,
        mark_superseded,
        retention_rank,
        round_boundary_gate,
        with_deletion_order,
    )
    from backend.services.round_snapshots.layout import (
        MANIFEST_NAME,
        SnapshotsDisabled,
        snapshots_root,
    )
    from backend.services.round_snapshots.listing import read_session_listing
    from backend.services.round_snapshots.manifest import write_snapshot_manifest
    from backend.services.round_snapshots.manifest import build_snapshot_manifest
    from backend.services.round_snapshots.layout import (
        CONTINUATION_NAME,
        PAIR_CHECKPOINT_RELPATH,
        PAIR_STATE_RELPATH,
        RUNTIME_MANIFEST_RELPATH,
        RUNTIME_PKL_RELPATH,
        SESSION_STATE_NAME,
    )
except ImportError as exc:  # the modules under test do not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_modules_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


SESSION = "source-session"


def _state(*, current_round=2, phase=AnalysisPhase.EXPLORATION, candidates=None):
    return types.SimpleNamespace(
        current_round=current_round,
        analysis_phase=phase,
        integration_gate_candidates=candidates or [],
    )


# ------------------------------------------------------------------ the gate


@pytest.mark.parametrize(
    "state,workflow_state,expected",
    [
        # The workflow state declares the integration gate outright.
        (_state(), {"semantic_gate": GATE_INTEGRATION_SELECTION}, GATE_INTEGRATION_SELECTION),
        # Or the phase plus the candidates the integration node published imply it.
        (
            _state(current_round=4, phase=AnalysisPhase.INTEGRATION, candidates=[{"direction": "d"}]),
            {},
            GATE_INTEGRATION_SELECTION,
        ),
        # Integration phase with no candidates is not a gate: nothing to choose.
        (_state(current_round=4, phase=AnalysisPhase.INTEGRATION), {}, GATE_ROUND_WAIT),
        # Round 1 in innovation is the post-Stage-2 direction gate.
        (_state(current_round=1, phase=AnalysisPhase.INNOVATION), {}, GATE_DIRECTION_SELECTION),
        # Round 1 still exploring is not.
        (_state(current_round=1, phase=AnalysisPhase.EXPLORATION), {}, GATE_ROUND_WAIT),
        # Innovation after round 1 is an ordinary round-acceptance wait.
        (_state(current_round=2, phase=AnalysisPhase.INNOVATION), {}, GATE_ROUND_WAIT),
        # No workflow state at all still answers.
        (_state(current_round=3), None, GATE_ROUND_WAIT),
        # The declaration wins over the round-1 direction test.
        (
            _state(current_round=1, phase=AnalysisPhase.INNOVATION),
            {"semantic_gate": GATE_INTEGRATION_SELECTION},
            GATE_INTEGRATION_SELECTION,
        ),
    ],
)
def test_the_helper_classifies_the_boundary_the_way_the_node_does(
    state, workflow_state, expected,
):
    assert round_boundary_gate(state, workflow_state) == expected


def test_the_node_asks_the_helper_instead_of_re_deriving_the_gate():
    """One owner for the predicates, or the capture and the node drift apart.

    The node used to spell them itself, forty lines apart: an integration test
    at the top and a round-1 innovation test further down. A capture that
    classified the same boundary differently would arm a restored session onto
    a dialog the source run never showed. The source-level assertion is the
    cheap half of that guarantee; the table above is the other half.
    """
    from backend.tests.split_facade import split_source

    source = split_source("backend.workflow.synthesis")
    assert "round_boundary_gate(" in source
    assert 'state.get("semantic_gate") == GATE_INTEGRATION_SELECTION' not in source
    assert "current_round == 1" not in source


# ------------------------------------------------------------------ retention


@pytest.mark.parametrize(
    "gate,rank",
    [
        (GATE_ANNOTATION_CONFIRMATION, 1),
        (GATE_DIRECTION_SELECTION, 2),
        (GATE_INTEGRATION_SELECTION, 3),
        (GATE_ROUND_WAIT, 4),
    ],
)
def test_each_decision_carries_the_priority_the_owner_set(gate, rank):
    """Decided 2026-09-08: annotation first, then direction, integration, round.

    The rank follows the decision, not the round. A re-run that reaches the
    annotation gate again in round 3 is still the decision every later label
    depends on, so it keeps rank 1.
    """
    assert retention_rank(gate) == rank


def test_an_incident_gate_has_no_retention_rank():
    with pytest.raises(ValueError):
        retention_rank("llm_failure")


def _record(snapshot_id, *, gate, round_number, revision, created_at):
    return {
        "snapshot_id": snapshot_id,
        "gate": gate,
        "round": round_number,
        "runtime_revision": revision,
        "created_at": created_at,
        "retention_rank": retention_rank(gate),
    }


def test_the_least_valuable_decision_is_given_up_first():
    records = [
        _record("r001-annotation-g57", gate=GATE_ANNOTATION_CONFIRMATION,
                round_number=1, revision=57, created_at="2026-09-08T10:00:00+00:00"),
        _record("r001-direction-g212", gate=GATE_DIRECTION_SELECTION,
                round_number=1, revision=212, created_at="2026-09-08T11:00:00+00:00"),
        _record("r004-integration-g512", gate=GATE_INTEGRATION_SELECTION,
                round_number=4, revision=512, created_at="2026-09-08T13:00:00+00:00"),
        _record("r003-round-g400", gate=GATE_ROUND_WAIT,
                round_number=3, revision=400, created_at="2026-09-08T12:00:00+00:00"),
    ]
    assert [row["snapshot_id"] for row in in_deletion_order(records)] == [
        "r003-round-g400",
        "r004-integration-g512",
        "r001-direction-g212",
        "r001-annotation-g57",
    ]


def test_within_one_rank_the_later_snapshot_goes_first():
    records = [
        _record("r002-round-g100", gate=GATE_ROUND_WAIT, round_number=2,
                revision=100, created_at="2026-09-08T10:00:00+00:00"),
        _record("r005-round-g900", gate=GATE_ROUND_WAIT, round_number=5,
                revision=900, created_at="2026-09-08T15:00:00+00:00"),
        _record("r003-round-g400", gate=GATE_ROUND_WAIT, round_number=3,
                revision=400, created_at="2026-09-08T12:00:00+00:00"),
    ]
    assert [row["snapshot_id"] for row in in_deletion_order(records)] == [
        "r005-round-g900", "r003-round-g400", "r002-round-g100",
    ]


def test_an_unreadable_timestamp_is_deleted_last_within_its_rank():
    """Erring the other way would give up a snapshot because of a bad clock."""
    records = [
        _record("r002-round-g100", gate=GATE_ROUND_WAIT, round_number=2,
                revision=100, created_at="not a timestamp"),
        _record("r003-round-g400", gate=GATE_ROUND_WAIT, round_number=3,
                revision=400, created_at="2026-09-08T12:00:00+00:00"),
    ]
    assert in_deletion_order(records)[-1]["snapshot_id"] == "r002-round-g100"


def test_the_order_is_total_so_three_consumers_cannot_disagree():
    """The browser, the CLI and the prune read this order, not their own."""
    records = [
        _record("r002-round-g100", gate=GATE_ROUND_WAIT, round_number=2,
                revision=100, created_at="2026-09-08T10:00:00+00:00"),
        _record("r002-round-g101", gate=GATE_ROUND_WAIT, round_number=2,
                revision=101, created_at="2026-09-08T10:00:00+00:00"),
    ]
    keys = [deletion_sort_key(row) for row in records]
    assert len(set(keys)) == len(keys)
    annotated = {row["snapshot_id"]: row["deletion_order"] for row in with_deletion_order(records)}
    assert annotated == {"r002-round-g101": 1, "r002-round-g100": 2}


def test_a_replaced_capture_is_marked_and_not_hidden():
    """A supplementary analysis re-presents the annotation gate.

    Both bundles stay restorable: the earlier one holds the namespace before
    the supplementary work, which is often the point of going back.
    """
    records = [
        _record("r001-annotation-g57", gate=GATE_ANNOTATION_CONFIRMATION,
                round_number=1, revision=57, created_at="2026-09-08T10:00:00+00:00"),
        _record("r001-annotation-g61", gate=GATE_ANNOTATION_CONFIRMATION,
                round_number=1, revision=61, created_at="2026-09-08T10:40:00+00:00"),
        _record("r001-direction-g212", gate=GATE_DIRECTION_SELECTION,
                round_number=1, revision=212, created_at="2026-09-08T11:00:00+00:00"),
    ]
    flagged = {row["snapshot_id"]: row["superseded"] for row in mark_superseded(records)}
    assert flagged == {
        "r001-annotation-g57": True,
        "r001-annotation-g61": False,
        "r001-direction-g212": False,
    }


# ------------------------------------------------------------------ config


def test_no_configured_directory_means_snapshots_are_off(monkeypatch):
    monkeypatch.delenv("SCAI_SNAPSHOT_DIR", raising=False)
    assert _config._resolve_snapshot_dir() is None


def test_a_relative_directory_is_refused_rather_than_resolved(monkeypatch):
    """It would resolve against whatever directory the process started in."""
    monkeypatch.setenv("SCAI_SNAPSHOT_DIR", "./snapshots")
    assert _config._resolve_snapshot_dir() is None


def test_an_absolute_directory_turns_snapshots_on(monkeypatch, tmp_path):
    monkeypatch.setenv("SCAI_SNAPSHOT_DIR", str(tmp_path / "snap"))
    resolved = _config._resolve_snapshot_dir()
    assert resolved == (tmp_path / "snap").resolve()


def test_a_directory_inside_the_data_root_is_allowed_and_warned(monkeypatch, caplog):
    """Kernel code can write below a session tree; the manifest is signed anyway.

    Refusing would break a small single-volume deployment for a hazard the
    signature already detects, so this warns and stays on.
    """
    inside = _config.DATA_BASE_DIR / "snapshots"
    monkeypatch.setenv("SCAI_SNAPSHOT_DIR", str(inside))
    with caplog.at_level("WARNING"):
        assert _config._resolve_snapshot_dir() == inside.resolve()
    assert any("inside the data directory" in record.message for record in caplog.records)


@pytest.mark.parametrize("raw", ["", "nonsense", "0", "-5", "1.5"])
def test_an_unusable_bound_falls_back_rather_than_refusing_to_start(monkeypatch, raw):
    """Import time. A typo must not be the difference between running and not."""
    if raw:
        monkeypatch.setenv("SCAI_SNAPSHOT_STALL_SECONDS", raw)
    else:
        monkeypatch.delenv("SCAI_SNAPSHOT_STALL_SECONDS", raising=False)
    assert _config._resolve_snapshot_seconds("SCAI_SNAPSHOT_STALL_SECONDS", 120) == 120


def test_a_usable_bound_is_honoured(monkeypatch):
    monkeypatch.setenv("SCAI_SNAPSHOT_MAX_SECONDS", "900")
    assert _config._resolve_snapshot_seconds("SCAI_SNAPSHOT_MAX_SECONDS", 1800) == 900


@pytest.mark.parametrize("raw", ["", "nonsense", "0", "-1"])
def test_without_a_usable_reserve_nothing_is_deleted_automatically(monkeypatch, raw):
    if raw:
        monkeypatch.setenv("SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB", raw)
    else:
        monkeypatch.delenv("SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB", raising=False)
    assert _config._resolve_snapshot_reserve_bytes() is None


def test_a_reserve_is_read_in_gibibytes(monkeypatch):
    monkeypatch.setenv("SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB", "50")
    assert _config._resolve_snapshot_reserve_bytes() == 50 * 1024 ** 3


def test_the_enabled_question_reads_the_live_setting(monkeypatch, tmp_path):
    from backend.config import snapshots_enabled

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    assert snapshots_enabled() is False
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path)
    assert snapshots_enabled() is True


def test_no_path_helper_answers_while_snapshots_are_off(monkeypatch):
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    with pytest.raises(SnapshotsDisabled):
        snapshots_root(SESSION)


# ------------------------------------------------------------------ listing


def _write_bundle(root, session, snapshot, *, gate, round_number, revision, created_at):
    directory = root / session / "snapshots" / snapshot
    directory.mkdir(parents=True)
    manifest = build_snapshot_manifest(
        snapshot_id=snapshot,
        source_session_id=session,
        source_session_name="Liver atlas",
        source_lifecycle_generation=1,
        scai_version="0.55.7",
        created_at=created_at,
        round_number=round_number,
        gate=gate,
        retention_rank=retention_rank(gate),
        runtime_revision=revision,
        pkl_digest="b" * 64,
        files={
            CONTINUATION_NAME: {"sha256": "0" * 64, "size": 220},
            SESSION_STATE_NAME: {"sha256": "1" * 64, "size": 4096},
            RUNTIME_PKL_RELPATH: {"sha256": "b" * 64, "size": 199775703},
            RUNTIME_MANIFEST_RELPATH: {"sha256": "2" * 64, "size": 2490},
            PAIR_STATE_RELPATH: {"sha256": "3" * 64, "size": 1519},
            PAIR_CHECKPOINT_RELPATH: {"sha256": "4" * 64, "size": 10256384},
        },
        uploads={},
        coverage={"saved_names": 271, "skipped_names": [], "carried_definitions": []},
        executor_checkpoint_present=False,
        bytes_total=210_000_000,
        capture_seconds=41.2,
    )
    write_snapshot_manifest(directory / MANIFEST_NAME, manifest)
    return directory


def test_an_installation_with_snapshots_off_lists_nothing(monkeypatch):
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", None)
    assert read_session_listing(SESSION) == {"snapshots": [], "failed": []}


def test_an_enabled_installation_with_no_captures_lists_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path)
    assert read_session_listing(SESSION) == {"snapshots": [], "failed": []}


def test_a_bundle_on_disk_is_listed_with_its_rank_and_deletion_order(monkeypatch, tmp_path):
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path)
    _write_bundle(tmp_path, SESSION, "r001-annotation-g57",
                  gate=GATE_ANNOTATION_CONFIRMATION, round_number=1, revision=57,
                  created_at="2026-09-08T10:00:00+00:00")
    _write_bundle(tmp_path, SESSION, "r003-round-g400",
                  gate=GATE_ROUND_WAIT, round_number=3, revision=400,
                  created_at="2026-09-08T12:00:00+00:00")

    listing = read_session_listing(SESSION)
    assert listing["failed"] == []
    rows = {row["snapshot_id"]: row for row in listing["snapshots"]}
    assert set(rows) == {"r001-annotation-g57", "r003-round-g400"}
    assert rows["r001-annotation-g57"]["retention_rank"] == 1
    assert rows["r001-annotation-g57"]["deletion_order"] == 2
    assert rows["r003-round-g400"]["retention_rank"] == 4
    assert rows["r003-round-g400"]["deletion_order"] == 1
    assert rows["r003-round-g400"]["superseded"] is False
    assert rows["r003-round-g400"]["gate"] == GATE_ROUND_WAIT
    assert rows["r003-round-g400"]["saved_names"] == 271


def test_a_failed_capture_is_listed_as_a_decision_with_no_restore_point(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path)
    root = tmp_path / SESSION / "snapshots"
    root.mkdir(parents=True)
    (root / "r002-round-g300.failed.json").write_text(
        json.dumps({"reason": "stall", "at": "2026-09-08T11:00:00+00:00",
                    "bytes_written": 4096}),
        encoding="utf-8",
    )
    listing = read_session_listing(SESSION)
    assert listing["snapshots"] == []
    assert listing["failed"] == [{
        "snapshot_id": "r002-round-g300",
        "round": 2,
        "gate": GATE_ROUND_WAIT,
        "runtime_revision": 300,
        "reason": "stall",
        "at": "2026-09-08T11:00:00+00:00",
        "bytes_written": 4096,
    }]


def test_a_bundle_whose_manifest_does_not_verify_is_reported_not_omitted(
    monkeypatch, tmp_path,
):
    """It occupies space and cannot be restored; silence would explain neither."""
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path)
    directory = _write_bundle(tmp_path, SESSION, "r001-direction-g212",
                              gate=GATE_DIRECTION_SELECTION, round_number=1,
                              revision=212, created_at="2026-09-08T11:00:00+00:00")
    (directory / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    listing = read_session_listing(SESSION)
    assert listing["snapshots"] == []
    assert [row["snapshot_id"] for row in listing["failed"]] == ["r001-direction-g212"]
    assert listing["failed"][0]["reason"] == "unverified"


def test_a_directory_that_is_not_a_snapshot_is_ignored(monkeypatch, tmp_path):
    """Staging and half-removed bundles live beside the real ones."""
    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path)
    root = tmp_path / SESSION / "snapshots"
    (root / ".deleting-r001-round-g4").mkdir(parents=True)
    assert read_session_listing(SESSION) == {"snapshots": [], "failed": []}


# ------------------------------------------------------------------ the route


def test_the_listing_route_is_registered():
    from backend.tests.test_route_registry import _current_routes

    assert ("/api/session/{session_id}/snapshots", ("GET",)) in set(_current_routes())


def test_no_route_creates_a_snapshot():
    """Capture is automatic at a decision point, spec 1.

    A create endpoint would be a second producer with different semantics: it
    could run mid-cell, against a generation nothing had committed, and produce
    a bundle whose pair describes a position the dump never held.
    """
    from backend.tests.test_route_registry import _current_routes

    creators = [
        (path, methods) for path, methods in _current_routes()
        if "snapshot" in path.lower() and "POST" in methods
    ]
    assert creators == []
