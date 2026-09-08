"""§5.1's intra-round floor must actually govern a save.

Measured before the fix: across two runs and five completed rounds the
step-boundary branch fired zero times, so the exposure window inside a round was
the whole round -- 95 to 100 minutes -- while `SCAI_RUNTIME_SAVE_FLOOR_MINUTES`
advertised 30.
"""
from __future__ import annotations




def test_the_loop_asks_the_runtime_cadence_every_turn():
    """Not once every five steps of a counter that keeps restarting.

    `should_checkpoint` is `step % 5 == 0` and `step_count` starts at 0 on every
    `run_autonomous_analysis` invocation, which happens per analysis step. The
    cadence has to be asked independently or the floor governs nothing.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/agents/bioinformatics/executor_loop.py")
    turn = source.split("context.step_count += 1", 1)[1][:2400]

    assert "runtime_save_due" in turn, (
        "the per-turn path never asks the runtime cadence, so §5.1's floor "
        "still governs a branch that does not run"
    )
    assert "should_checkpoint(context.step_count) or runtime_due" in turn

    # And only when there is something to save. `should_save` treats "nothing
    # has ever committed" as due, which is right for the round boundary and
    # would otherwise dump an empty kernel on turn one of every run.
    assert 'record.get("status") == "live_dirty"' in turn, (
        "the per-turn cadence is not gated on the runtime being dirty, so a "
        "fresh session saves before it has executed anything"
    )


def test_the_floor_still_throttles_it():
    """Asking every turn must not mean saving every turn.

    `should_save` is the throttle: inside the floor a step boundary answers
    False, and only `round_boundary` and `closing` ignore it.
    """
    from backend.services.save_cadence import should_save

    now = 1_000_000.0
    floor = 30

    # Just after a commit: not due.
    assert not should_save(
        reason="step_boundary",
        last_committed_at=now - 60,
        now=now,
        floor_minutes=floor,
    )
    # Past the floor: due.
    assert should_save(
        reason="step_boundary",
        last_committed_at=now - (floor * 60) - 1,
        now=now,
        floor_minutes=floor,
    )
    # A round boundary never waits.
    assert should_save(
        reason="round_boundary",
        last_committed_at=now - 1,
        now=now,
        floor_minutes=floor,
    )


def test_a_cadence_failure_does_not_stop_the_analysis():
    """The cadence is an optimisation over the round boundary.

    A sandbox that cannot answer -- no kernel yet, a probe that raises -- must
    leave the turn running rather than take the analysis down with it.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/agents/bioinformatics/executor_loop.py")
    turn = source.split("context.step_count += 1", 1)[1][:2400]

    assert "except Exception:" in turn
    assert "runtime_due = False" in turn
