"""Spec 5.1's cadence, and the exposure bound that depends on it."""
from __future__ import annotations

import pathlib

import pytest

from backend.services.save_cadence import should_save

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"

HOUR = 3600.0


# The three boundaries, as wired


class _Notebook:
    def __init__(self, last_saved_at=None):
        self.runtime_persistence = {"last_saved_at": last_saved_at}


class _Sandbox:
    def __init__(self, last_saved_at=None, km=object()):
        self.notebook_state = _Notebook(last_saved_at)
        self.km = km
        self.saves = 0

    async def save_session(self):
        self.saves += 1
        return True

    def sync_runtime_persistence_state(self, _state):
        return {"status": "current"}


def test_two_step_boundaries_inside_the_floor_commit_once():
    """The floor is what makes a step boundary cheap enough to have at all."""
    import time

    from backend.services.save_cadence import runtime_save_due

    just_saved = _Sandbox(
        last_saved_at=__import__("datetime").datetime.fromtimestamp(
            time.time(),
        ).isoformat(),
    )
    assert runtime_save_due(just_saved, "step_boundary") is False
    assert runtime_save_due(just_saved, "step_boundary") is False

    long_ago = _Sandbox(
        last_saved_at=__import__("datetime").datetime.fromtimestamp(
            time.time() - 24 * 3600,
        ).isoformat(),
    )
    assert runtime_save_due(long_ago, "step_boundary") is True


@pytest.mark.asyncio
async def test_the_round_boundary_saves_and_publishes_the_record():
    from backend.workflow.synthesis_parts.phases import (
        _save_round_boundary_runtime,
    )

    class _Workflow:
        session_id = "cadence-round"

        def __init__(self, sandbox):
            self.sandbox = sandbox

    sandbox = _Sandbox()
    published = []
    sandbox.sync_runtime_persistence_state = lambda state: published.append(state)

    await _save_round_boundary_runtime(_Workflow(sandbox), object())

    assert sandbox.saves == 1
    assert len(published) == 1


@pytest.mark.asyncio
async def test_a_failed_round_boundary_save_does_not_hold_the_gate():
    """The user is about to be asked a question. A dump must not block it."""
    from backend.workflow.synthesis_parts.phases import (
        _save_round_boundary_runtime,
    )

    class _Failing(_Sandbox):
        async def save_session(self):
            raise RuntimeError("disk full")

    class _Workflow:
        session_id = "cadence-round-fail"

        def __init__(self, sandbox):
            self.sandbox = sandbox

    sandbox = _Failing()
    published = []
    sandbox.sync_runtime_persistence_state = lambda state: published.append(state)

    await _save_round_boundary_runtime(_Workflow(sandbox), object())

    # No exception escaped, and the record was published anyway.
    assert published == [published[0]]


# Spec 5.1's disclosure: what the projection tells the client


@pytest.mark.parametrize("reason, committed, elapsed, floor, expected", [
    ("round_boundary", 0, 1, 30, True),
    ("closing", 0, 1, 30, True),
    ("decision_boundary", 0, 1, 30, True),
    ("step_boundary", None, 1, 30, True),
    ("step_boundary", 0, 1799, 30, False),
    ("step_boundary", 0, 1800, 30, True),
    ("step_boundary", 0, 1, 0, True),
    ("future_boundary", 0, 1, 30, True),
])
def test_save_boundary_policy(reason, committed, elapsed, floor, expected):
    assert should_save(
        reason=reason, last_committed_at=committed,
        now=elapsed, floor_minutes=floor,
    ) is expected
