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
