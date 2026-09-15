"""Crash recovery must survive graph progress that ran no kernel code.

Spec goals 2 and 3 promise a session resumes from its committed runtime. The
pairing recorded with a generation is `(state_version, checkpoint identity)`,
and both advance on pure graph movement: `_send_state_sync` increments
`state_version` for every change type but the persistence commit, and LangGraph
writes a checkpoint row whenever a node returns.

So the ordinary sequence -- round-boundary save, user clicks Continue,
`round_start`, backend dies during planning -- leaves a perfectly good dump that
`state_runtime_coherence` calls `stale`, which `is_forward_resume_blocked`
treats as a hard block. The only accepted action then quarantines the dump.
"""
from __future__ import annotations

import pytest

from backend.models.schemas import AnalysisState
from backend.utils.runtime_apply import (
    is_forward_resume_blocked,
    state_runtime_coherence,
)


def _committed_state(*, state_version: int, current_revision: int = 4) -> AnalysisState:
    """A session whose newest generation committed cleanly at revision 4."""
    state = AnalysisState(id="coh-" + str(state_version), user_id="u", name="s")
    state.state_version = state_version
    state.runtime_persistence = {
        "committed_source": "regular_save",
        "committed_digest": "d" * 64,
        "committed_revision": 4,
        "current_revision": current_revision,
        "saved_revision": 4,
        "last_attempt_revision": 4,
        "last_attempt_status": "committed",
        # Paired at the moment of the save: state_version 7, no checkpoint file.
        "committed_pairing": {"state_version": 7, "checkpoint": None},
    }
    return state


def test_a_dump_is_current_when_only_the_graph_moved():
    """The kernel executed nothing, so the dump still describes it.

    Two syncs after the save -- Continue, then round_start -- and no cell in
    between. `current_revision` still equals `committed_revision`, which is the
    kernel's own statement that nothing changed.
    """
    state = _committed_state(state_version=9)

    assert state_runtime_coherence(state) == "current"
    assert not is_forward_resume_blocked("complete", coherence="current")


def test_a_dump_is_stale_when_the_kernel_actually_ran():
    """The distinction the check exists for, and it must survive the fix.

    Here a cell executed after the commit, so the dump is genuinely behind the
    namespace and forward resume must be refused.
    """
    state = _committed_state(state_version=7, current_revision=6)

    assert state_runtime_coherence(state) == "stale"
    assert is_forward_resume_blocked("complete", coherence="stale")


def test_a_state_version_that_went_backwards_is_still_inconsistent():
    """Regression, not progress: the record cannot belong to this session."""
    state = _committed_state(state_version=3)

    assert state_runtime_coherence(state) == "inconsistent"


@pytest.mark.asyncio
async def test_graph_execution_waits_for_each_checkpoint_commit():
    """The next node must not race a still-pending SQLite checkpoint write."""
    from unittest.mock import AsyncMock

    from backend.workflow.core_parts.execution import _ainvoke_durably

    graph = AsyncMock()
    graph.ainvoke.return_value = {"analysis_state": object()}
    config = {"configurable": {"thread_id": "s"}}

    result = await _ainvoke_durably(graph, {"analysis_state": object()}, config)

    assert "analysis_state" in result
    graph.ainvoke.assert_awaited_once()
    assert graph.ainvoke.await_args.kwargs == {"durability": "sync"}


def test_export_still_requires_a_matched_dump_and_checkpoint():
    """Relaxing coherence must not relax what may be shipped.

    The matched-pair property is a shipping concern: an archive that carries a
    dump and a checkpoint which disagree resumes into the wrong graph position
    on the far side. `_capture_descriptor` compares the manifest's pairing with
    a freshly captured one *and* requires coherence "current", so removing the
    pairing test from coherence leaves that guarantee where it belongs.

    Asserted on the source because the check is one line inside an async
    capture that needs a live sandbox to reach.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/services/resumable_export.py")
    assert 'manifest.get("pairing") != pairing' in source, (
        "the exporter no longer compares the manifest pairing, so nothing "
        "enforces that a shipped dump and checkpoint match"
    )
    assert 'state_runtime_coherence(state) != "current"' in source


def test_coherence_no_longer_reads_the_checkpoint_to_decide_staleness():
    """The checkpoint hash advances whenever a node returns.

    Consulting it here is what made pure graph progress read stale, so its
    absence from this function is the fix rather than an accident of
    refactoring.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/utils/runtime_apply.py")
    body = source.split("def state_runtime_coherence(", 1)[1]
    body = body.split("\ndef ", 1)[0]

    assert "checkpoint_identity" not in body, (
        "state_runtime_coherence consults the checkpoint again; graph progress "
        "will read as stale and block crash recovery"
    )
    assert 'return "inconsistent"' in body, "the regression case must survive"


# --- a held resume must leave an actionable session -------------------------


@pytest.mark.asyncio
async def test_a_held_resume_does_not_leave_the_session_running():
    """Holding is not running, and RUNNING is what traps the user.

    A held WebSocket reconnect used to return the state untouched. With RUNNING
    persisted, POST /run answers 400 "Analysis already running" and /resume acts
    only on PAUSED -- so the session could be reached only through Pause, while
    the action that resolves it (re-run from the input data) is exactly POST
    /run.
    """
    import sys

    from backend.models.schemas import TaskStatus
    from backend.workflow.core_parts import execution as execution_part
    import backend.workflow.core  # noqa: F401

    # bind_facade runs inside the wrappers, not at import, so a plain import
    # leaves `logger` and friends unbound in the part module.
    execution_part.bind_facade(sys.modules["backend.workflow.core"])

    state = AnalysisState(id="held", user_id="u", name="s")
    state.status = TaskStatus.RUNNING
    state.runtime_restore_status = "unavailable"

    class _Workflow:
        session_id = "held"

        def __init__(self):
            self.set_calls = []

        def _set_status(self, target, status):
            self.set_calls.append(status)
            target.status = status

    workflow = _Workflow()

    result = await execution_part.resume(workflow, "thread-1", state)

    assert result is state
    assert state.status == TaskStatus.PAUSED, (
        "a held resume left the session RUNNING; /run will refuse it as "
        "'already running' and /resume only acts on PAUSED"
    )
    assert TaskStatus.PAUSED in workflow.set_calls


def test_the_run_and_resume_guards_both_set_a_status():
    """The asymmetry that caused it, pinned.

    The run path's twin of this guard has always set a status before returning.
    The resume path returned untouched, which is the whole defect.
    """
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/workflow/core_parts/execution.py")
    for marker, label in (
        ("refusing to start the graph", "run guard"),
        ("holding without running the graph", "resume guard"),
    ):
        assert marker in source, label
        tail = source.split(marker, 1)[1][:1200]
        assert "_set_status" in tail, (
            f"the {label} returns without setting a status, so the session "
            "keeps whatever it had -- RUNNING traps the user"
        )
