"""A restored session can actually be answered. Part B spec 3.4, B2.

The semantic-gate continuation is the one part of this design that was not
written for it: Snapshot V2 produced these capsules, Part A deleted that
producer, and every route that consumes one has had no caller since. Reading
the code says the path is intact. This is the evidence.

For each of the four decisions: build a real session by restoring a real
bundle, register no workflow, drive the HTTP action a user would, and assert
the outcome is an exact graph-node continuation at the node that decision
declares -- not a 404 on a missing workflow, and not the same gate presented
again.
"""
import types

import pytest

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
    SEMANTIC_GATE_ACTION_NEXT_NODES,
)
from backend.models.schemas import AnalysisPhase

try:
    from backend.config import _config
    from backend.services.round_snapshots.restore import restore_snapshot_into_new_session
    from backend.services.round_snapshots.writer import capture_decision_snapshot
    from backend.workflow.restore_entry import build_restore_entry_plan
except ImportError as exc:  # the module under test does not exist yet
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from backend.tests.test_round_snapshot_capture import (  # noqa: E402
    _Sandbox,
    _Workflow,
    _seed_session,
    _state,
)


@pytest.fixture(autouse=True)
def _require_module_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


OWNER = "owner-1"


def _request():
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=OWNER))


@pytest.fixture
def restored(monkeypatch, tmp_path):
    """A session restored from a snapshot of the named decision, and nothing else.

    No workflow is registered on purpose: that is the state a restored session
    opens in, and the state every one of these routes has to handle without
    reaching for one.
    """
    import uuid

    monkeypatch.setattr(_config, "SNAPSHOT_DIR", tmp_path / "snapshots")

    async def _build(gate, **state_kwargs):
        session_id = "src-" + uuid.uuid4().hex[:12]
        round_number = state_kwargs.pop("round_number", 3)
        digest = _seed_session(session_id, revision=57, round_number=round_number)
        workflow = _Workflow(
            _Sandbox(session_id, 57, digest, round_number=round_number)
        )
        state = _state(session_id, round_number=round_number)
        for name, value in state_kwargs.items():
            setattr(state, name, value)
        snapshot = await capture_decision_snapshot(
            workflow, state, gate=gate, workflow_state={"current_round": state.current_round},
        )
        assert snapshot is not None
        new_id = await restore_snapshot_into_new_session(
            source_session_id=session_id, snapshot=snapshot, user_id=OWNER,
        )
        from backend.state import workflows

        assert new_id not in workflows, "the proof is about a session with no workflow"
        return new_id

    return _build


async def _current(session_id):
    from backend.services.session_store import ensure_session_loaded

    return await ensure_session_loaded(session_id)


def _assert_continues_at(state, gate):
    """The capsule became an exact continuation at the node the gate declares.

    Not a fresh ``initialize_round`` -- that was the mistake revision 4 of the
    Part A spec had to withdraw, because it restores past the user's decision,
    past convergence, and possibly past the report.
    """
    expected = SEMANTIC_GATE_ACTION_NEXT_NODES[gate]
    assert state.restore_resume_kind == "graph_node"
    assert state.restore_resume_node == expected
    assert state.restore_resume_gate is None

    plan = build_restore_entry_plan(state)
    assert plan.intent == "graph_node"
    assert plan.target_node == expected
    # The channels the capture recorded are still there for the node to enter with.
    assert plan.workflow_state["current_round"] == state.current_round


@pytest.mark.asyncio
async def test_the_annotation_decision_can_be_answered_after_a_restore(restored):
    """The dialog the user came back for, answered without a live workflow.

    ``/confirm-annotations`` writes the overrides, converts the capsule to the
    bioinformatics node, and restarts the runner. The executor then resumes
    from the position the bundle carried and re-issues ``annotate_clusters``,
    which returns the confirmed annotations straight from the state.
    """
    from backend.models.api_requests import AnnotationConfirmation
    from backend.routes.interaction import confirm_annotations

    session_id = await _build_annotation(restored)
    response = await confirm_annotations(
        session_id, _request(),
        AnnotationConfirmation(annotations=[
            {"cluster_id": "0", "user_override": "Hepatocyte", "user_confirmed": True},
        ]),
    )
    assert response.success

    state = await _current(session_id)
    assert state.annotation_state.confirmed is True
    assert state.annotation_state.cluster_annotations[0].user_override == "Hepatocyte"
    _assert_continues_at(state, GATE_ANNOTATION_CONFIRMATION)


async def _build_annotation(restored):
    """A restored annotation gate, with the pending state its dialog needs."""
    from backend.models.schema_parts.session import (
        AnnotationConfirmationState,
        ClusterAnnotation,
    )

    return await restored(
        GATE_ANNOTATION_CONFIRMATION,
        round_number=1,
        annotation_state=AnnotationConfirmationState(
            cluster_annotations=[ClusterAnnotation(cluster_id="0", cell_type="Unknown")],
            tissue="liver", organism="mouse", pending=True,
        ),
    )


@pytest.mark.asyncio
async def test_the_direction_decision_can_be_answered_after_a_restore(restored):
    from backend.models.api_requests import DirectionChoice
    from backend.routes.interaction import choose_direction

    session_id = await restored(
        GATE_DIRECTION_SELECTION, round_number=1, analysis_phase=AnalysisPhase.INNOVATION,
    )
    response = await choose_direction(
        session_id, _request(),
        DirectionChoice(directions=[{"direction": "Kupffer cell heterogeneity", "notes": ""}]),
    )
    assert response.success

    state = await _current(session_id)
    assert [d["direction"] for d in state.user_chosen_directions] == [
        "Kupffer cell heterogeneity",
    ]
    _assert_continues_at(state, GATE_DIRECTION_SELECTION)


@pytest.mark.asyncio
async def test_the_integration_decision_can_be_answered_after_a_restore(restored):
    from backend.models.api_requests import DirectionChoice
    from backend.routes.interaction import choose_direction

    session_id = await restored(
        GATE_INTEGRATION_SELECTION,
        round_number=4,
        analysis_phase=AnalysisPhase.INTEGRATION,
        integration_gate_candidates=[{"direction": "cross-direction correlation"}],
    )
    response = await choose_direction(
        session_id, _request(), DirectionChoice(let_pi_decide=True),
    )
    assert response.success
    _assert_continues_at(await _current(session_id), GATE_INTEGRATION_SELECTION)


@pytest.mark.asyncio
async def test_the_round_acceptance_decision_can_be_answered_after_a_restore(restored):
    """``/resume`` is the action behind Continue when no workflow owns the session.

    It is also the one that would have been most quietly broken: with no live
    workflow it neither signals nor starts anything, so without the capsule
    conversion it would hold, flip to paused, and execute nothing -- which
    reads as a session that simply does not respond.
    """
    from backend.routes.lifecycle import resume_analysis

    session_id = await restored(GATE_ROUND_WAIT, round_number=3)
    response = await resume_analysis(session_id, _request())
    assert response.success
    assert response.data["needs_reconnect"] is True
    assert response.data["next_node"] == SEMANTIC_GATE_ACTION_NEXT_NODES[GATE_ROUND_WAIT]
    _assert_continues_at(await _current(session_id), GATE_ROUND_WAIT)


@pytest.mark.asyncio
async def test_an_answered_decision_is_not_presented_again(restored):
    """Spec 6.4's acceptance, stated as the thing that would go wrong.

    A restored session that re-presented the gate the user had just answered
    would ask the same question forever, and the capsule is what stops it: once
    converted it names a node, not a gate, so nothing routes back to the wait.
    """
    from backend.models.api_requests import DirectionChoice
    from backend.routes.interaction import choose_direction

    session_id = await restored(
        GATE_DIRECTION_SELECTION, round_number=1, analysis_phase=AnalysisPhase.INNOVATION,
    )
    await choose_direction(
        session_id, _request(), DirectionChoice(let_pi_decide=True),
    )
    state = await _current(session_id)
    assert state.restore_resume_gate is None
    assert build_restore_entry_plan(state).intent != "semantic_gate"


@pytest.mark.asyncio
async def test_a_restored_session_never_answers_its_own_gate(restored):
    """Whatever the source's skip time was, the fork waits.

    The source may have been running unattended with the PI deciding every
    gate. The fork exists because somebody wants to decide differently, and
    there is no workflow here running a timeout in any case.
    """
    session_id = await restored(GATE_ROUND_WAIT, round_number=3, round_wait_seconds=0)
    state = await _current(session_id)
    assert state.status.value == "waiting_input"
    assert state.restore_resume_kind == "semantic_gate"
    assert not state.user_chosen_direction
