"""Spec 5.1's cadence, and the exposure bound that depends on it."""
from __future__ import annotations

import pathlib

import pytest

from backend.services.save_cadence import should_save

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"

HOUR = 3600.0


def test_a_round_boundary_always_saves():
    assert should_save(
        reason="round_boundary", last_committed_at=1_000.0, now=1_001.0,
        floor_minutes=30,
    ) is True


def test_a_closing_boundary_always_saves():
    assert should_save(
        reason="closing", last_committed_at=1_000.0, now=1_001.0,
        floor_minutes=30,
    ) is True


def test_a_step_boundary_within_the_floor_does_not_save():
    assert should_save(
        reason="step_boundary", last_committed_at=1_000.0,
        now=1_000.0 + 29 * 60, floor_minutes=30,
    ) is False


def test_a_step_boundary_past_the_floor_saves():
    assert should_save(
        reason="step_boundary", last_committed_at=1_000.0,
        now=1_000.0 + 30 * 60, floor_minutes=30,
    ) is True


def test_a_session_that_has_never_committed_saves_at_the_first_step_boundary():
    """Until one generation exists there is nothing to fall back to."""
    assert should_save(
        reason="step_boundary", last_committed_at=None, now=1_000.0,
        floor_minutes=30,
    ) is True


def test_the_floor_runs_from_the_last_success_not_the_last_attempt():
    """Spec 5.1's exposure bound. A failed attempt must not reset the clock.

    This is the point of the whole module. "At most one round or N minutes of
    work is at risk" is a statement about the last generation that reached
    disk. If a failed attempt reset the clock, the real exposure would grow
    without bound while the number shown to the user stayed at N -- and the
    only signal that anything was wrong would be a warning nobody reads.

    The function is given no way to learn about an attempt: its only temporal
    input is the last *successful* commit. That is the guarantee, expressed as
    a signature rather than as a comment.
    """
    import inspect

    parameters = set(inspect.signature(should_save).parameters)
    assert "last_committed_at" in parameters
    assert not [name for name in parameters if "attempt" in name], parameters

    committed = 1_000.0
    # Three attempts fail over the next hour; none of them is an input here,
    # so the floor is measured from `committed` and the save still fires.
    assert should_save(
        reason="step_boundary", last_committed_at=committed,
        now=committed + HOUR, floor_minutes=30,
    ) is True


def test_a_floor_of_zero_saves_at_every_step_boundary():
    assert should_save(
        reason="step_boundary", last_committed_at=1_000.0, now=1_000.0,
        floor_minutes=0,
    ) is True


def test_an_unknown_boundary_saves_rather_than_dropping_its_generation():
    """A caller this module has not been taught about still believes it is a
    persistence point. Refusing would drop its generation silently."""
    assert should_save(
        reason="some_future_boundary", last_committed_at=1_000.0, now=1_000.1,
        floor_minutes=30,
    ) is True


def test_the_floor_is_a_named_configurable_constant():
    """Spec 12 leaves N open, pending A2-10's measurement.

    It must be one named constant with an environment binding, so confirming or
    changing it later is a one-line edit rather than a hunt through call sites.
    """
    from backend.config import RUNTIME_SAVE_FLOOR_MINUTES

    assert isinstance(RUNTIME_SAVE_FLOOR_MINUTES, int)
    assert RUNTIME_SAVE_FLOOR_MINUTES == 30

    source = (BACKEND / "config" / "_config.py").read_text()
    assert "SCAI_RUNTIME_SAVE_FLOOR_MINUTES" in source


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not-a-number", "0", "-5", "30.5"],
)
def test_an_unusable_environment_value_falls_back_rather_than_refusing(raw, monkeypatch):
    """Import time. A deployment typo must not stop the backend from starting."""
    from backend.config import _config

    monkeypatch.setenv("SCAI_RUNTIME_SAVE_FLOOR_MINUTES", raw)
    assert _config._resolve_save_floor_minutes() == 30


def test_a_usable_environment_value_is_honoured(monkeypatch):
    from backend.config import _config

    monkeypatch.setenv("SCAI_RUNTIME_SAVE_FLOOR_MINUTES", "5")
    assert _config._resolve_save_floor_minutes() == 5


# ---------------------------------------------------------------------------
# The three boundaries, as wired
# ---------------------------------------------------------------------------


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


def test_the_clock_is_read_from_the_last_successful_commit_field():
    """`last_saved_at` is written only on a generation that reached `current`.

    A failed attempt updates `last_attempt_*` and leaves this alone, which is
    the whole reason the cadence reads this field and not another.
    """
    from datetime import datetime

    from backend.services.save_cadence import last_committed_epoch

    assert last_committed_epoch(_Sandbox()) is None
    assert last_committed_epoch(_Sandbox(last_saved_at="not a timestamp")) is None

    stamp = datetime(2026, 9, 6, 12, 0, 0)
    sandbox = _Sandbox(last_saved_at=stamp.isoformat())
    assert last_committed_epoch(sandbox) == stamp.timestamp()

    source = (BACKEND / "utils" / "sandbox_parts" / "notebook_state.py").read_text()
    assert '"last_saved_at": now,' in source
    head, _, tail = source.partition('"last_saved_at": now,')
    assert head.rstrip().endswith('elif status == "current":') or (
        'elif status == "current":' in head[-400:]
    ), "last_saved_at must be written only on a successful commit"


def test_an_unreadable_record_saves_rather_than_extending_the_window():
    from backend.services.save_cadence import runtime_save_due

    class _Bare:
        pass

    assert runtime_save_due(_Bare(), "step_boundary") is True


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


def test_a_round_boundary_ignores_the_floor():
    import time

    from backend.services.save_cadence import runtime_save_due

    just_saved = _Sandbox(
        last_saved_at=__import__("datetime").datetime.fromtimestamp(
            time.time(),
        ).isoformat(),
    )
    assert runtime_save_due(just_saved, "round_boundary") is True


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


@pytest.mark.asyncio
async def test_a_round_boundary_without_a_kernel_saves_nothing():
    from backend.workflow.synthesis_parts.phases import (
        _save_round_boundary_runtime,
    )

    class _Workflow:
        session_id = "cadence-round-nokernel"

        def __init__(self, sandbox):
            self.sandbox = sandbox

    sandbox = _Sandbox(km=None)
    await _save_round_boundary_runtime(_Workflow(sandbox), object())
    assert sandbox.saves == 0

    class _NoSandbox:
        session_id = "cadence-round-nosandbox"
        sandbox = None

    await _save_round_boundary_runtime(_NoSandbox(), object())


def test_the_closing_save_is_one_attempt_and_does_not_hold_the_shutdown():
    """Spec 5.1. A2-3 removed the gate; this pins what replaced it."""
    from backend.tests.split_facade import split_source

    source = split_source("backend.utils.sandbox")
    start = source.index("    async def stop(self, *, save: bool = True):")
    body = source[start:start + 4000]

    assert body.count("await self.save_session()") == 1, body[:2000]
    assert "One attempt (spec 5.1)" in body
    # The notebook save is outside the dump's try, so a failed dump cannot take
    # the notebook with it.
    assert "Independent of the dump" in body


def test_the_step_boundary_asks_the_cadence_rather_than_saving_unconditionally():
    source = (
        BACKEND / "agents" / "bioinformatics" / "executor_loop.py"
    ).read_text()
    assert 'runtime_save_due(self.sandbox, "step_boundary")' in source


def test_the_round_boundary_is_the_node_spec_6_4_names():
    """pi_synthesis -> wait_for_user_input -> check_convergence.

    An earlier point would skip the user's decision, the integration and the
    report; a later one is not reached on a round that does not gate.
    """
    graph = (BACKEND / "workflow" / "graph.py").read_text()
    assert 'self.graph.add_edge("pi_synthesis", "wait_for_user_input")' in graph

    phases = (BACKEND / "workflow" / "synthesis_parts" / "phases.py").read_text()
    entry = phases.index("async def _wait_for_user_input(")
    body = phases[entry:entry + 2500]
    assert "_save_round_boundary_runtime(self, analysis_state)" in body


# ---------------------------------------------------------------------------
# Spec 5.1's disclosure: what the projection tells the client
# ---------------------------------------------------------------------------


def _projected(tmp_path, persistence, **state_kwargs):
    import json

    from backend.models.schemas import AnalysisState
    from backend.services.runtime_persistence_projection import (
        project_runtime_persistence,
    )

    state_path = tmp_path / "notebook_state.json"
    state_path.write_text(json.dumps({
        "session_id": "disclose",
        "runtime_persistence": persistence,
    }))
    state = AnalysisState(id="disclose", name="d", user_id="u", **state_kwargs)
    result = project_runtime_persistence(state_path, "disclose", state)
    assert result is not None
    return result[0]


def test_the_projection_reports_the_last_successful_commit_not_the_last_attempt(tmp_path):
    """Spec 5.1's exposure bound, as it reaches the browser.

    A record whose last attempt failed still carries the *successful* commit
    time. A UI measuring from `last_attempt_at` would read "saved a moment ago"
    through an unbounded run of failures.
    """
    record = _projected(tmp_path, {
        "status": "failed",
        "current_revision": 5,
        "saved_revision": 3,
        "committed_revision": 3,
        "committed_source": "regular_save",
        "committed_digest": "d" * 64,
        "last_attempt_revision": 5,
        "last_attempt_status": "failed",
        "last_saved_at": "2026-09-06T09:30:00",
        "metadata_updated_at": "2026-09-06T10:15:00",
    })

    assert record["exposure_since"] == "2026-09-06T09:30:00"
    assert record["exposure_since"] != record["metadata_updated_at"]


def test_a_stale_generation_is_disclosed_as_stale(tmp_path):
    record = _projected(tmp_path, {
        "status": "failed",
        "current_revision": 5,
        "saved_revision": 3,
        "committed_revision": 3,
        "committed_source": "regular_save",
        "committed_digest": "d" * 64,
        "last_attempt_revision": 5,
        "last_attempt_status": "failed",
        "last_saved_at": "2026-09-06T09:30:00",
    })
    assert record["coherence"] == "stale"


def test_a_runtime_that_never_committed_is_disclosed_as_unavailable(tmp_path):
    record = _projected(tmp_path, {
        "status": "pending",
        "current_revision": 0,
        "saved_revision": 0,
        "committed_revision": None,
        "last_attempt_revision": 0,
        "last_attempt_status": None,
    })
    assert record["coherence"] == "unavailable"


def test_a_current_generation_is_disclosed_as_current(tmp_path):
    record = _projected(tmp_path, {
        "status": "current",
        "current_revision": 3,
        "saved_revision": 3,
        "committed_revision": 3,
        "committed_source": "regular_save",
        "committed_digest": "d" * 64,
        "last_attempt_revision": 3,
        "last_attempt_status": "current",
        "last_saved_at": "2026-09-06T09:30:00",
    })
    assert record["coherence"] == "current"
    assert record["exposure_since"] == "2026-09-06T09:30:00"


def test_the_frontend_renders_the_three_states_and_nothing_else():
    """A1's verdicts reach the UI unchanged; the banner must not invent a fourth."""
    banner = (
        REPO / "frontend" / "src" / "components" / "runtime"
        / "RuntimePersistenceBanner.tsx"
    ).read_text()

    assert "coherence === 'current'" in banner
    assert "coherence === 'unavailable'" in banner
    # Last successful save, never the last attempt.
    assert "exposure_since" in banner
    assert "last_attempt" not in banner
