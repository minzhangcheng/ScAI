"""Where a decision snapshot is taken from. Part B spec 3.3, 4.5.

Two hooks, and each one's correctness is about *when* it fires rather than what
it writes. Hook A has to run before the round-boundary gate acts -- before the
dialog opens, and before an unattended run answers it on the user's behalf.
Hook B has to run inside the executor's annotation tool call, after the pending
state is published and before the wait, and it has to leave the executor a
position it can actually resume from.
"""
import asyncio
import json
import types

import pytest

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
)
from backend.models.schemas import AnalysisPhase, TaskStatus

try:
    import backend.workflow.synthesis  # noqa: F401  binds the split facade
    from backend.agents.bioinformatics.executor_checkpoint import (
        PendingToolCheckpoint,
        build_checkpoint_payload,
        validate_checkpoint,
    )
    from backend.services.save_cadence import runtime_save_due, should_save
    from backend.tests.split_facade import setattr_facade, split_source
    from backend.workflow.annotation_parts import callback as annotation_callback
    from backend.workflow.synthesis_parts import phases
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_modules_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


@pytest.fixture(autouse=True)
def _bound_facade(_require_modules_under_test):
    """Bind the split module the way its facade does before delegating.

    ``synthesis.py`` calls ``bind_facade`` inside ``_wait_for_user_input``
    itself, so a test that calls the split function directly gets a module
    whose globals were never populated -- every facade name is a NameError and
    the failure looks like a missing import rather than an unbound facade.
    """
    import sys

    import backend.workflow.synthesis  # noqa: F401

    phases.bind_facade(sys.modules["backend.workflow.synthesis"])


# ------------------------------------------------------------------ cadence


def test_a_decision_boundary_always_saves():
    """It has to: the snapshot copies a *committed* generation.

    A boundary that waited out the step floor would capture a dump older than
    the workspace beside it -- half of one decision and half of the next. The
    caller only asks at all when a cell has run since the last commit, so this
    being unconditional does not mean a dump per decision.
    """
    import time

    assert should_save(
        reason="decision_boundary",
        last_committed_at=time.time(),
        now=time.time(),
        floor_minutes=30,
    ) is True


# ------------------------------------------------------------------ hook A


class _Node:
    """Enough of the workflow for the no-wait path through the round gate."""

    session_id = "hook-a"
    sandbox = None  # so the round-boundary save returns early

    def __init__(self):
        self.syncs: list[str] = []
        self.statuses: list = []
        self._supplementary_queue = []
        self._continue_event = asyncio.Event()

    def _set_status(self, state, status):
        state.status = status
        self.statuses.append(status)

    async def _send_state_sync(self, state, kind):
        self.syncs.append(kind)


def _analysis_state(**overrides):
    from backend.models.schemas import AnalysisState

    payload = dict(
        id="hook-a-session",
        name="Liver atlas",
        user_id="owner-1",
        current_round=3,
        round_wait_seconds=0,
        max_rounds=5,
    )
    payload.update(overrides)
    return AnalysisState(**payload)


@pytest.fixture
def recorded_capture(monkeypatch, _bound_facade):
    """Replace the writer with a recorder that notes when it was called."""
    calls: list[dict] = []

    async def _capture(workflow, analysis_state, *, gate, workflow_state=None, prepare=None):
        calls.append({
            "gate": gate,
            "round": analysis_state.current_round,
            "status": analysis_state.status,
            "prepare": prepare,
            "workflow_state": dict(workflow_state or {}),
        })
        if prepare is not None:
            await prepare()
        return None

    setattr_facade(
        monkeypatch, "backend.workflow.synthesis", "capture_decision_snapshot", _capture,
    )
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_kwargs,workflow_state,expected",
    [
        ({"current_round": 3}, {}, GATE_ROUND_WAIT),
        (
            {"current_round": 1, "analysis_phase": AnalysisPhase.INNOVATION},
            {},
            GATE_DIRECTION_SELECTION,
        ),
        (
            {"current_round": 4},
            {"semantic_gate": GATE_INTEGRATION_SELECTION},
            GATE_INTEGRATION_SELECTION,
        ),
    ],
    ids=["round", "direction", "integration"],
)
async def test_the_round_gate_captures_the_decision_it_is_about_to_present(
    recorded_capture, state_kwargs, workflow_state, expected,
):
    state = _analysis_state(**state_kwargs)
    await phases._wait_for_user_input(
        _Node(), {"analysis_state": state, **workflow_state},
    )
    assert [call["gate"] for call in recorded_capture] == [expected]


@pytest.mark.asyncio
async def test_the_capture_precedes_the_answer_even_when_nobody_is_watching(
    recorded_capture,
):
    """``round_wait_seconds = 0`` hands the decision to the PI.

    At this layer that is the same decision made by a different party, so the
    restore point has to exist before it is made -- a user who comes back to an
    unattended run is exactly who a rollback is for. The status recorded at
    capture time is the proof of ordering: RUNNING is what the no-wait branch
    sets *after* deciding.
    """
    state = _analysis_state(current_round=2, round_wait_seconds=0)
    node = _Node()
    await phases._wait_for_user_input(node, {"analysis_state": state})

    assert len(recorded_capture) == 1
    assert recorded_capture[0]["status"] != TaskStatus.RUNNING
    assert node.statuses[-1] == TaskStatus.RUNNING
    assert "auto_continue" in node.syncs


# ------------------------------------------------------------------ hook B


@pytest.mark.asyncio
async def test_the_executor_handle_pairs_the_call_it_interrupts():
    """A checkpoint written mid-tool-call has an unanswered request in it.

    ``tool_calls_are_paired`` rejects that, and ``restore_executor_state`` then
    throws the transcript away and starts fresh -- so a restored session would
    lose the executor's whole account of the round it is resuming into. The
    synthetic result is what keeps the transcript loadable.
    """
    from backend.utils.tool_messages import tool_calls_are_paired

    live = [
        {"role": "system", "content": "you are"},
        {"role": "user", "content": "Analysis instruction: go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "annotate_clusters", "arguments": "{}"},
            }],
        },
    ]
    assert tool_calls_are_paired(live) is False

    captured = {}

    async def _save(step_name, *, extra_messages=None, **kwargs):
        captured["step_name"] = step_name
        captured["messages"] = [*live, *(extra_messages or [])]

    handle = PendingToolCheckpoint(
        tool_call_id="call-1", tool_name="annotate_clusters", save=_save,
    )
    await handle.checkpoint_now(
        "pre_annotation_confirmation", result_text="awaiting review",
    )
    assert captured["step_name"] == "pre_annotation_confirmation"
    assert tool_calls_are_paired(captured["messages"]) is True
    assert captured["messages"][-1] == {
        "role": "tool", "tool_call_id": "call-1", "content": "awaiting review",
    }


@pytest.mark.asyncio
async def test_the_checkpoint_belongs_to_the_generation_the_capture_commits():
    """``validate_checkpoint`` refuses a position behind its generation.

    The executor's checkpoint is written before the save that commits the dump,
    so the revision it records is the one that save then commits. Writing it
    after would leave a position one revision behind and the restore would
    refuse the checkpoint it had just installed.
    """
    payload = build_checkpoint_payload(
        step_name="pre_annotation_confirmation",
        state=types.SimpleNamespace(id="s", current_round=1),
        planning_stage="stage1",
        step_number=10,
        total_steps=200,
        attempt_start_step=0,
        active_plan_id="plan-1",
        messages=[{"role": "system", "content": "a"}, {"role": "user", "content": "b"}],
        findings=[],
        figures=[],
        code_history=[],
        analysis_log=[],
        analysis_context={},
        budget_exhausted=False,
        runtime_revision=57,
    )
    assert validate_checkpoint(
        payload, current_round=1, planning_stage="stage1",
        committed_revision=57, exact_revision=True,
    ) == []
    assert validate_checkpoint(
        payload, current_round=1, planning_stage="stage1",
        committed_revision=58, exact_revision=True,
    ) != []


@pytest.mark.asyncio
async def test_the_annotation_hook_asks_for_the_annotation_decision(monkeypatch):
    calls: list[dict] = []

    async def _capture(workflow, state, *, gate, workflow_state=None, prepare=None):
        calls.append({"gate": gate, "prepare": prepare, "workflow_state": workflow_state})
        if prepare is not None:
            await prepare()
        return "r001-annotation-g57"

    monkeypatch.setattr(
        "backend.services.round_snapshots.writer.capture_decision_snapshot", _capture,
    )
    checkpointed: list[str] = []

    async def _save(step_name, *, extra_messages=None, **kwargs):
        checkpointed.append(step_name)

    agent = types.SimpleNamespace(
        pending_tool_checkpoint=PendingToolCheckpoint(
            tool_call_id="call-1", tool_name="annotate_clusters", save=_save,
        )
    )
    workflow = types.SimpleNamespace(
        bioinformatics_agent=agent, _current_graph_state={"planning_stage": "stage1"},
    )

    await annotation_callback._capture_annotation_snapshot(workflow, object())

    assert [call["gate"] for call in calls] == [GATE_ANNOTATION_CONFIRMATION]
    assert calls[0]["workflow_state"] == {"planning_stage": "stage1"}
    assert checkpointed == ["pre_annotation_confirmation"]


@pytest.mark.asyncio
async def test_the_annotation_hook_still_captures_without_an_executor_handle(monkeypatch):
    """A callback reached outside the executor loop still has a decision to keep.

    The handle is how the executor's position is reached, not a precondition
    for the snapshot. Without one the capture goes ahead and the manifest
    records that it carries no executor checkpoint.
    """
    calls: list[dict] = []

    async def _capture(workflow, state, *, gate, workflow_state=None, prepare=None):
        calls.append({"gate": gate, "prepare": prepare})
        return None

    monkeypatch.setattr(
        "backend.services.round_snapshots.writer.capture_decision_snapshot", _capture,
    )
    workflow = types.SimpleNamespace(bioinformatics_agent=None, _current_graph_state=None)
    await annotation_callback._capture_annotation_snapshot(workflow, object())
    assert calls == [{"gate": GATE_ANNOTATION_CONFIRMATION, "prepare": None}]


def test_the_annotation_capture_precedes_the_wait():
    """Ordering, asserted on the source because the flow around it is a loop.

    Taken any later the snapshot could not restore into the dialog, because the
    wait is where the answer arrives; taken earlier it would not carry the
    pending annotation state the dialog is rebuilt from.
    """
    source = split_source("backend.workflow.annotation")
    capture_at = source.index("await _capture_annotation_snapshot(self, state)")
    wait_at = source.index("signaled = await wait_with_skip_time(")
    published_at = source.index('await self._send_state_sync(state, "annotation_confirmation")')
    assert published_at < capture_at < wait_at


def test_the_tool_handler_publishes_and_retracts_the_executor_handle():
    """Outside the call there is no position to checkpoint, so no handle.

    Leaving it set would let a later decision checkpoint the executor against a
    tool call that had already returned, pairing a synthetic result with a real
    one.
    """
    source = split_source("backend.agents.bioinformatics_agent")
    index = source.index("PendingToolCheckpoint(")
    following = source[index:index + 900]
    assert "finally:" in following
    assert "self.pending_tool_checkpoint = None" in following
