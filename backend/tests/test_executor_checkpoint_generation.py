"""The executor checkpoint's binding to a runtime generation. Spec §3.11.

The checkpoint carries the executor's position and its account of what it has
already done -- `step_number`, the message history, `code_history`. Nothing
bound it to a runtime generation, and `validate_checkpoint` compared exactly
two fields, `round_number` and `planning_stage`, neither of which can see
which namespace is in the kernel.
"""
from backend.agents.bioinformatics.executor_checkpoint import (
    build_checkpoint_payload,
    validate_checkpoint,
)


class _State:
    id = "s"
    current_round = 2


def _payload(**overrides):
    payload = build_checkpoint_payload(
        step_name="step_10",
        state=_State(),
        planning_stage="stage1",
        step_number=10,
        total_steps=40,
        attempt_start_step=0,
        active_plan_id="plan-1",
        messages=[{"role": "system", "content": "a"}, {"role": "user", "content": "b"}],
        findings=[],
        figures=[],
        code_history=[],
        analysis_log=[],
        analysis_context={},
        budget_exhausted=False,
        runtime_revision=overrides.pop("runtime_revision", 44),
    )
    payload.update(overrides)
    return payload


def test_a_checkpoint_records_the_generation_it_was_written_against():
    assert _payload()["runtime_revision"] == 44


def test_a_checkpoint_older_than_the_committed_generation_is_refused():
    """The namespace has already moved past this position.

    A checkpoint behind the committed revision describes a kernel that no
    longer exists: something restored or replaced the runtime after it was
    written. Resuming from it would replay against variables that have since
    been rebuilt.
    """
    errors = validate_checkpoint(
        _payload(runtime_revision=30),
        current_round=2,
        planning_stage="stage1",
        committed_revision=44,
    )
    assert any("runtime_revision" in error for error in errors), errors


def test_a_checkpoint_ahead_of_the_committed_generation_is_ordinary():
    """This is what a live kernel looks like between two saves.

    The executor checkpoints every five steps and the dump waits out the save
    floor, so the checkpoint is normally ahead. Refusing that would break every
    ordinary resume.
    """
    assert validate_checkpoint(
        _payload(runtime_revision=61),
        current_round=2,
        planning_stage="stage1",
        committed_revision=44,
    ) == []


def test_a_checkpoint_from_before_this_contract_still_validates():
    """No revision recorded is not evidence of a mismatch.

    Every checkpoint on disk today predates the field. Refusing them would
    strand the sessions this is meant to protect.
    """
    payload = _payload()
    payload.pop("runtime_revision")
    assert validate_checkpoint(
        payload,
        current_round=2,
        planning_stage="stage1",
        committed_revision=44,
    ) == []


def test_the_existing_two_fields_still_decide():
    assert validate_checkpoint(
        _payload(), current_round=3, planning_stage="stage1",
    ) != []
    assert validate_checkpoint(
        _payload(), current_round=2, planning_stage="stage2",
    ) != []
    assert validate_checkpoint(
        _payload(), current_round=2, planning_stage="stage1",
    ) == []


def test_the_executor_passes_its_live_revision_when_it_checkpoints():
    """A field nothing populates is documentation, not a binding."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    source = (
        repo / "backend" / "agents" / "bioinformatics" / "executor_loop.py"
    ).read_text()
    assert 'runtime_revision=getattr(self.sandbox, "runtime_revision", None)' in source


def test_the_reader_supplies_the_committed_revision_it_has():
    """The check is only worth having where the answer is known."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    source = (
        repo / "backend" / "agents" / "bioinformatics" / "executor_checkpoint.py"
    ).read_text()
    assert "committed_revision=" in source
    assert '.get("committed_revision")' in source


def test_the_exact_rule_is_driven_by_the_persisted_record_not_a_state_field():
    """The exact-generation marker survives through the notebook projection."""
    import pathlib as _pathlib

    repo = _pathlib.Path(__file__).resolve().parents[2]
    source = (
        repo / "backend" / "agents" / "bioinformatics" / "executor_checkpoint.py"
    ).read_text()
    assert '"pair_restore"' in source, (
        "the exact-revision rule must key off the persisted record"
    )
    assert "runtime_pair_restored_revision" not in source, (
        "a state field cannot carry this signal across a checkpoint resume"
    )


def test_a_pair_restore_makes_an_ahead_checkpoint_unacceptable():
    """Ahead is ordinary for a live kernel and wrong right after a restore."""
    record = {"pair_restore": {"restored_revision": 44}, "committed_revision": 44}
    errors = validate_checkpoint(
        _payload(runtime_revision=61),
        current_round=2,
        planning_stage="stage1",
        committed_revision=44,
        exact_revision=bool(record.get("pair_restore")),
    )
    assert any("runtime_revision" in error for error in errors), errors

    # ... and ordinary again once the record no longer says a restore happened.
    assert validate_checkpoint(
        _payload(runtime_revision=61),
        current_round=2,
        planning_stage="stage1",
        committed_revision=44,
        exact_revision=False,
    ) == []
