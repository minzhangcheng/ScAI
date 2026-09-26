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


# --- a held resume must leave an actionable session -------------------------
