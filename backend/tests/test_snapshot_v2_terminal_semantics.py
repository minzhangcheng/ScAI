"""Regression coverage for exact Snapshot V2 terminal semantics."""

from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.services.snapshot_v2.continuation import ContinuationRegistry
from backend.services.snapshot_v2.dispatcher import ContinuationDispatcher
from backend.services.snapshot_v2.models import TerminalCapsule
from backend.services.snapshot_v2.object_store import ObjectStore
from backend.services.snapshot_v2.restore_entry import build_restore_entry_plan
from backend.workflow import SingleCellAnalysisWorkflow
from backend.workflow.routing import RoutingMixin


@pytest.fixture
def object_store(tmp_path):
    class Paths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for path in (
                self.root, self.objects, self.staging, self.journals,
                self.integrity, self.committed,
            ):
                path.mkdir(parents=True, exist_ok=True)

    paths = Paths()
    paths.ensure_dirs()
    return ObjectStore(paths)


def _state(status: TaskStatus, session_id: str = "terminal-semantics"):
    return AnalysisState(
        id=session_id,
        user_id="u",
        initial_instruction="analyze",
        current_round=2,
        status=status,
    )


@pytest.mark.parametrize(
    ("task_status", "capsule_status"),
    [
        (TaskStatus.COMPLETED, "completed"),
        (TaskStatus.STOPPED, "stopped"),
        (TaskStatus.FAILED, "failed"),
    ],
)
def test_terminal_capsule_freeze_dispatch_and_schema_preserve_three_states(
    object_store, tmp_path, task_status, capsule_status
):
    state = _state(task_status, f"terminal-{capsule_status}")
    capsule = ContinuationRegistry(object_store).freeze_dynamic(
        {
            "kernel_present": False,
            "inactive": True,
            "status": capsule_status,
            "continuation_kind": "terminal",
        },
        analysis_state=state,
        workflow_state={},
    )

    assert capsule.status == capsule_status
    assert (
        TerminalCapsule.model_validate_json(capsule.model_dump_json()).status
        == capsule_status
    )

    staged = ContinuationDispatcher(object_store).stage_checkpoint(
        capsule,
        {},
        state,
        snapshot_id=f"snap-{capsule_status}",
        transaction_id=f"tx-{capsule_status}",
    )
    ContinuationDispatcher(object_store).activate(
        state, staged, workspace_path=tmp_path
    )

    assert staged.terminal_status == capsule_status
    assert state.status is task_status
    assert state.restore_resume_kind == "terminal"
    assert state.restore_resume_workflow_state == {
        "terminal_status": capsule_status,
    }


def test_live_failed_status_wins_an_older_completed_descriptor(object_store):
    capsule = ContinuationRegistry(object_store).freeze_dynamic(
        {"continuation_kind": "terminal", "status": "completed"},
        analysis_state=_state(TaskStatus.FAILED),
        workflow_state={},
    )

    assert capsule.status == "failed"


@pytest.mark.parametrize(
    ("task_status", "capsule_status"),
    [
        (TaskStatus.COMPLETED, "completed"),
        (TaskStatus.STOPPED, "stopped"),
        (TaskStatus.FAILED, "failed"),
        (TaskStatus.GENERATING_REPORT, "completed"),
    ],
)
def test_static_post_report_freeze_uses_terminal_status(
    object_store, task_status, capsule_status
):
    capsule = ContinuationRegistry(object_store).freeze(
        "post_report", _state(task_status), {}
    )
    assert capsule.status == capsule_status


class _ExecutionHarness:
    session_id = "terminal-execution"

    @staticmethod
    def _set_status(state, status):
        state.status = status

    @staticmethod
    def _consume_restore_resume_metadata(state):
        state.restore_resume_kind = None
        state.restore_resume_workflow_state = None


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        ("completed", TaskStatus.COMPLETED),
        ("stopped", TaskStatus.STOPPED),
        ("failed", TaskStatus.FAILED),
    ],
)
@pytest.mark.asyncio
async def test_workflow_run_and_resume_preserve_restored_terminal(
    monkeypatch, terminal, expected
):
    monkeypatch.setattr("backend.config.is_snapshot_v2_enabled", lambda: True)
    harness = _ExecutionHarness()

    run_state = _state(TaskStatus.RUNNING, f"run-{terminal}")
    run_state.restore_resume_kind = "terminal"
    run_state.restore_resume_workflow_state = {"terminal_status": terminal}
    returned = await SingleCellAnalysisWorkflow.run(
        harness, run_state, run_state.id
    )
    assert returned.status is expected
    assert returned.restore_resume_kind is None

    resume_state = _state(TaskStatus.RUNNING, f"resume-{terminal}")
    resume_state.restore_resume_kind = "terminal"
    resume_state.restore_resume_workflow_state = {"terminal_status": terminal}
    returned = await SingleCellAnalysisWorkflow.resume(
        harness, resume_state.id, resume_state
    )
    assert returned.status is expected
    assert returned.restore_resume_kind is None


def test_terminal_restore_entry_holds_without_regenerating_report():
    state = _state(TaskStatus.FAILED)
    state.restore_resume_kind = "terminal"
    state.restore_resume_workflow_state = {"terminal_status": "failed"}

    plan = build_restore_entry_plan(state)
    assert plan.intent == "terminal"
    assert plan.terminal_status == "failed"
    assert plan.target_node is None

    router = SimpleNamespace(session_id=state.id)
    route = RoutingMixin._route_from_restore_entry(
        router, {"_restore_entry_plan": plan.model_dump(mode="json")}
    )
    assert route == "hold"


def test_legacy_terminal_metadata_without_subtype_defaults_to_completed():
    state = _state(TaskStatus.RUNNING)
    state.restore_resume_kind = "terminal"
    state.restore_resume_workflow_state = None

    plan = build_restore_entry_plan(state)

    assert plan.terminal_status == "completed"
