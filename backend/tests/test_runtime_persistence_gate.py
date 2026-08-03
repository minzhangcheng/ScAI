import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisState, TaskStatus
from backend.services.runtime_persistence_gate import (
    RuntimePersistenceActionOutcome,
    build_runtime_persistence_failure,
    enter_runtime_persistence_gate,
    is_runtime_persistence_blocked,
    runtime_persistence_gate_identity,
    validate_runtime_persistence_action,
)


def _failed_persistence(**updates):
    payload = {
        "status": "failed",
        "current_revision": 8,
        "saved_revision": 7,
        "committed_revision": 7,
        "last_attempt_revision": 8,
        "last_attempt_status": "failed",
        "requires_recompute": True,
        "recommended_action": "reduce_runtime_footprint",
        "budget_violations": [
            {"resource": "rss_bytes", "observed": 900, "limit": 800},
        ],
        "top_offenders": [
            {
                "name": "adata_copy",
                "type": "AnnData",
                "bytes": 600,
                "value": "must-not-leak",
            },
        ],
    }
    payload.update(updates)
    return payload


def test_runtime_gate_payload_is_safe_and_actionable():
    failure = build_runtime_persistence_failure(
        "runtime-gate",
        _failed_persistence(),
    )

    assert failure["allowed_actions"] == ["retry", "manage", "terminate"]
    assert failure["retry_context"]["producer"] == "runtime_persistence"
    assert failure["details"]["budget_violations"] == [
        {"resource": "rss_bytes", "observed": 900, "limit": 800}
    ]
    assert failure["details"]["top_offenders"] == [
        {"name": "adata_copy", "type": "AnnData", "bytes": 600}
    ]
    assert "must-not-leak" not in str(failure)


def test_runtime_gate_identity_survives_state_json_roundtrip():
    state = AnalysisState(id="runtime-roundtrip", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    token, generation = runtime_persistence_gate_identity(state)

    restored = AnalysisState.model_validate_json(state.model_dump_json())

    assert runtime_persistence_gate_identity(restored) == (token, generation)
    assert restored.snapshot_failure["details"]["gate_token"] == token
    assert restored.snapshot_failure["retry_context"]["gate_token"] == token
    result = validate_runtime_persistence_action(
        restored,
        "manage",
        gate_token=token,
        generation=generation,
        active=True,
    )
    assert result.outcome is RuntimePersistenceActionOutcome.ACCEPTED


def test_runtime_gate_redacts_raw_names_paths_and_errors_before_bounding():
    secret = "synthetic-runtime-secret-value"
    failure = build_runtime_persistence_failure(
        "runtime-redaction",
        _failed_persistence(
            error_summary=(
                f"Bearer {secret} failed at "
                "/home/private-user/project/analysis.py"
            ),
            failed_variables=[f"api_key={secret}"],
            recompute_variables=["/workspace/private-project/adata"],
            budget_violations=[{
                "resource": f"rss api_key={secret}",
                "observed": 900,
                "limit": 800,
            }],
            top_offenders=[{
                "name": "/data/sessions/private-session/work/adata",
                "type": f"AnnData Bearer {secret}",
                "bytes": 600,
            }],
        ),
    )

    rendered = str(failure)
    assert secret not in rendered
    assert "private-user" not in rendered
    assert "private-project" not in rendered
    assert "private-session" not in rendered
    assert "******" in rendered


def test_degraded_recovery_discloses_attempted_and_committed_revisions():
    failure = build_runtime_persistence_failure(
        "runtime-degraded",
        _failed_persistence(degraded_recovery=True),
    )

    assert failure["code"] == "runtime_persistence_degraded_recovery"
    assert failure["details"]["degraded_recovery"] is True
    assert failure["details"]["attempted_revision"] == 8
    assert failure["details"]["committed_revision"] == 7


def test_snapshot_feature_disable_does_not_clear_runtime_gate():
    from backend.services.snapshot_v2.feature_gate import (
        normalize_disabled_snapshot_state,
    )

    state = AnalysisState(id="runtime-feature-independent", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())

    assert normalize_disabled_snapshot_state(state) is False
    assert is_runtime_persistence_blocked(state) is True
    assert state.status == TaskStatus.WAITING_INPUT


@pytest.mark.asyncio
async def test_runtime_gate_survives_websocket_reconnect():
    from backend.routes.websockets import _resend_wait_input_state_sync

    state = AnalysisState(id="runtime-reconnect", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    sent = []

    class FakeWebSocket:
        async def send_json(self, message):
            sent.append(message)

    await _resend_wait_input_state_sync(
        FakeWebSocket(), state, "runtime-reconnect"
    )

    content = sent[0]["content"]
    assert content["change_type"] == "snapshot_failure"
    assert content["snapshot_failure"]["code"] == "runtime_persistence_failure"
    assert content["wait_input_reason"] == "snapshot_failure"


@pytest.mark.asyncio
async def test_live_retry_uses_same_kernel_and_clears_gate():
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = AnalysisState(id="runtime-live-retry", user_id="owner")

    class FakeSandbox:
        def __init__(self):
            self.persistence = _failed_persistence()
            self.save_calls = 0

        def sync_runtime_persistence_state(self, target):
            target.runtime_persistence = dict(self.persistence)
            return dict(self.persistence)

        async def save_session(self):
            self.save_calls += 1
            self.persistence.update({
                "status": "current",
                "current_revision": 8,
                "saved_revision": 8,
                "committed_revision": 8,
                "requires_recompute": False,
                "degraded_recovery": False,
            })
            return True

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow.sandbox = FakeSandbox()
    workflow._runtime_persistence_event = asyncio.Event()
    workflow._runtime_persistence_action = None
    workflow._runtime_persistence_gate_active = False
    workflow.state_callback = AsyncMock()
    workflow.message_callback = AsyncMock()
    workflow.await_restore_safe_point = AsyncMock()

    waiter = asyncio.create_task(
        workflow._wait_for_runtime_persistence_intervention(state)
    )
    for _ in range(20):
        if workflow._runtime_persistence_gate_active and is_runtime_persistence_blocked(state):
            break
        await asyncio.sleep(0)

    gate_token, generation = runtime_persistence_gate_identity(state)
    delivery = workflow.signal_runtime_persistence_action(
        "retry",
        gate_token=gate_token,
        generation=generation,
    )
    assert delivery.outcome is RuntimePersistenceActionOutcome.ACCEPTED
    assert await asyncio.wait_for(waiter, timeout=1) == "retry"
    assert workflow.sandbox.save_calls == 1
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    assert state.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_pause_persistence_recovery_requests_durable_transitions():
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = AnalysisState(
        id="runtime-pause-durable",
        user_id="owner",
        status=TaskStatus.WAITING_INPUT,
    )
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow._runtime_persistence_pause_task = object()
    workflow._wait_for_runtime_persistence_intervention = AsyncMock(
        return_value="retry"
    )

    await workflow._recover_failed_pause_persistence(state.id, state)

    workflow._wait_for_runtime_persistence_intervention.assert_awaited_once_with(
        state,
        successful_status=TaskStatus.PAUSED,
        persist_transitions=True,
    )
    assert state.status is TaskStatus.PAUSED
    assert workflow._runtime_persistence_pause_task is None


@pytest.mark.asyncio
async def test_recompute_action_without_live_workflow_redrives_from_gate():
    from backend.routes import lifecycle

    state = AnalysisState(id="runtime-redrive", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    gate_token, generation = runtime_persistence_gate_identity(state)

    with (
        patch.object(
            lifecycle,
            "require_session_ownership",
            new=AsyncMock(return_value=state),
        ),
        patch.object(lifecycle, "_update_session_cache", new=AsyncMock()),
        patch.object(lifecycle, "async_save_session", new=AsyncMock()),
        patch.object(
            lifecycle,
            "_close_hitl_socket_for_redrive",
            new=AsyncMock(),
        ),
    ):
        response = await lifecycle._runtime_persistence_action_locked(
            state.id,
            "owner",
            "manage",
            gate_token=gate_token,
            generation=generation,
        )

    assert response.success is True
    assert state.runtime_recompute_pending is True
    assert state.snapshot_failure is None
    assert state.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_fallback_terminate_reports_committed_stop_when_cleanup_incomplete(
    monkeypatch,
):
    from backend.routes import lifecycle

    state = AnalysisState(id="runtime-terminate-cleanup", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    gate_token, generation = runtime_persistence_gate_identity(state)
    update_cache = AsyncMock()
    save_session = AsyncMock()
    teardown = AsyncMock(return_value=False)
    monkeypatch.setattr(
        lifecycle,
        "require_session_ownership",
        AsyncMock(return_value=state),
    )
    monkeypatch.setattr(lifecycle, "_update_session_cache", update_cache)
    monkeypatch.setattr(lifecycle, "async_save_session", save_session)
    monkeypatch.setattr(lifecycle, "teardown_running_workflow", teardown)
    lifecycle.workflows.pop(state.id, None)

    response = await lifecycle._runtime_persistence_action_locked(
        state.id,
        "owner",
        "terminate",
        gate_token=gate_token,
        generation=generation,
    )

    assert response.success is False
    assert response.data == {
        "status": TaskStatus.STOPPED.value,
        "still_blocked": False,
        "action_result": RuntimePersistenceActionOutcome.ACCEPTED.value,
        "cleanup_incomplete": True,
        "termination_committed": True,
    }
    assert state.status is TaskStatus.STOPPED
    assert state.termination_reason == "runtime_persistence_failure"
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
    update_cache.assert_awaited_once_with(state.id, state)
    save_session.assert_awaited_once_with(state.id, state, force=True)
    teardown.assert_awaited_once_with(state.id, discard_runtime=True)


def _live_gate_workflow(state):
    from backend.workflow.core import SingleCellAnalysisWorkflow

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow._runtime_persistence_event = asyncio.Event()
    workflow._runtime_persistence_gate_active = True
    workflow._runtime_persistence_action = None
    workflow._runtime_persistence_action_pending = False
    workflow._current_analysis_state = state
    token, generation = runtime_persistence_gate_identity(state)
    workflow._runtime_persistence_gate_token = token
    workflow._runtime_persistence_gate_generation = generation
    return workflow


def _state_payload(state):
    return state.model_dump(mode="json")


def test_runtime_action_classifier_has_four_stable_read_write_outcomes():
    accepted = AnalysisState(id="runtime-classifier-accepted", user_id="owner")
    enter_runtime_persistence_gate(accepted, _failed_persistence())
    token, generation = runtime_persistence_gate_identity(accepted)
    result = validate_runtime_persistence_action(
        accepted, "retry", gate_token=token, generation=generation, active=True,
    )
    assert result.outcome is RuntimePersistenceActionOutcome.ACCEPTED
    assert accepted.runtime_persistence["gate_action_pending"] is True

    duplicate_before = _state_payload(accepted)
    result = validate_runtime_persistence_action(
        accepted, "manage", gate_token=token, generation=generation, active=True,
    )
    assert result.outcome is RuntimePersistenceActionOutcome.DUPLICATE_OR_CONFLICT
    assert _state_payload(accepted) == duplicate_before

    stale_before = _state_payload(accepted)
    result = validate_runtime_persistence_action(
        accepted,
        "terminate",
        gate_token=f"{token}-old",
        generation=generation - 1,
        active=True,
    )
    assert result.outcome is RuntimePersistenceActionOutcome.STALE
    assert _state_payload(accepted) == stale_before

    inactive = AnalysisState(id="runtime-classifier-inactive", user_id="owner")
    inactive_before = _state_payload(inactive)
    result = validate_runtime_persistence_action(
        inactive,
        "retry",
        gate_token="old-token",
        generation=1,
        active=False,
    )
    assert result.outcome is RuntimePersistenceActionOutcome.INACTIVE
    assert _state_payload(inactive) == inactive_before


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("stale", RuntimePersistenceActionOutcome.STALE),
        ("duplicate", RuntimePersistenceActionOutcome.DUPLICATE_OR_CONFLICT),
        ("inactive", RuntimePersistenceActionOutcome.INACTIVE),
    ],
)
def test_live_runtime_action_producer_rejections_are_read_only(case, expected):
    state = AnalysisState(id=f"runtime-producer-{case}", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    token, generation = runtime_persistence_gate_identity(state)
    workflow = _live_gate_workflow(state)
    request_token, request_generation = token, generation
    if case == "stale":
        request_token, request_generation = f"{token}-old", generation - 1
    elif case == "duplicate":
        workflow._runtime_persistence_action_pending = True
    else:
        workflow._runtime_persistence_gate_active = False
    before = _state_payload(state)

    result = workflow.signal_runtime_persistence_action(
        "manage",
        gate_token=request_token,
        generation=request_generation,
    )

    assert result.outcome is expected
    assert _state_payload(state) == before
    assert workflow._runtime_persistence_event.is_set() is False


@pytest.mark.asyncio
async def test_duplicate_live_runtime_action_returns_409(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(id="runtime-duplicate", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    token, generation = runtime_persistence_gate_identity(state)
    workflow = _live_gate_workflow(state)
    monkeypatch.setattr(
        lifecycle, "require_session_ownership", AsyncMock(return_value=state),
    )
    update_cache = AsyncMock()
    save_session = AsyncMock()
    monkeypatch.setattr(lifecycle, "_update_session_cache", update_cache)
    monkeypatch.setattr(lifecycle, "async_save_session", save_session)
    lifecycle.workflows[state.id] = workflow
    try:
        accepted = await lifecycle._runtime_persistence_action_locked(
            state.id,
            "owner",
            "retry",
            gate_token=token,
            generation=generation,
        )
        assert accepted.data["action_result"] == "accepted"

        with pytest.raises(HTTPException) as caught:
            await lifecycle._runtime_persistence_action_locked(
                state.id,
                "owner",
                "retry",
                gate_token=token,
                generation=generation,
            )
        assert caught.value.status_code == 409
        assert caught.value.detail["code"] == (
            "RUNTIME_PERSISTENCE_ACTION_DUPLICATE_OR_CONFLICT"
        )
        assert caught.value.detail["action_result"] == "duplicate_or_conflict"
        assert update_cache.await_count == 1
        assert save_session.await_count == 1
    finally:
        lifecycle.workflows.pop(state.id, None)


@pytest.mark.asyncio
async def test_stale_live_runtime_gate_token_returns_409(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(id="runtime-stale", user_id="owner")
    enter_runtime_persistence_gate(state, _failed_persistence())
    token, generation = runtime_persistence_gate_identity(state)
    workflow = _live_gate_workflow(state)
    monkeypatch.setattr(
        lifecycle, "require_session_ownership", AsyncMock(return_value=state),
    )
    lifecycle.workflows[state.id] = workflow
    try:
        with pytest.raises(HTTPException) as caught:
            await lifecycle._runtime_persistence_action_locked(
                state.id,
                "owner",
                "manage",
                gate_token=token + "-stale",
                generation=generation - 1,
            )
        assert caught.value.status_code == 409
        assert caught.value.detail["code"] == "RUNTIME_PERSISTENCE_ACTION_STALE"
        assert caught.value.detail["action_result"] == "stale"
        assert state.runtime_persistence.get("gate_action_pending") is not True
    finally:
        lifecycle.workflows.pop(state.id, None)


@pytest.mark.asyncio
async def test_inactive_runtime_action_route_returns_stable_409(monkeypatch):
    from backend.routes import lifecycle

    state = AnalysisState(id="runtime-inactive-route", user_id="owner")
    before = _state_payload(state)
    update_cache = AsyncMock()
    save_session = AsyncMock()
    monkeypatch.setattr(
        lifecycle, "require_session_ownership", AsyncMock(return_value=state),
    )
    monkeypatch.setattr(lifecycle, "_update_session_cache", update_cache)
    monkeypatch.setattr(lifecycle, "async_save_session", save_session)

    with pytest.raises(HTTPException) as caught:
        await lifecycle._runtime_persistence_action_locked(
            state.id,
            "owner",
            "retry",
            gate_token="inactive-token",
            generation=1,
        )

    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "RUNTIME_PERSISTENCE_ACTION_INACTIVE"
    assert caught.value.detail["action_result"] == "inactive"
    assert _state_payload(state) == before
    update_cache.assert_not_awaited()
    save_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_save_failure_reconnects_to_actionable_same_kernel_gate():
    """Pause failure keeps the live sandbox and accepts a reconnect action."""
    from backend.routes.websockets import _resend_wait_input_state_sync
    from backend.workflow.core import SingleCellAnalysisWorkflow

    state = AnalysisState(id="runtime-pause-live", user_id="owner")

    class _Sandbox:
        def __init__(self):
            self.save_calls = 0
            self.persistence = _failed_persistence()
            self.notebook_state = SimpleNamespace(save=MagicMock())

        async def save_session(self):
            self.save_calls += 1
            if self.save_calls == 1:
                raise RuntimeError("synthetic pause save failure")
            self.persistence.update({
                "status": "current",
                "current_revision": 8,
                "saved_revision": 8,
                "committed_revision": 8,
                "requires_recompute": False,
                "degraded_recovery": False,
            })
            return True

        def sync_runtime_persistence_state(self, target):
            target.runtime_persistence = dict(self.persistence)
            return dict(self.persistence)

    sandbox = _Sandbox()
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow.session_id = state.id
    workflow.sandbox = sandbox
    workflow.compiled_graph = SimpleNamespace(
        aget_state=AsyncMock(return_value=SimpleNamespace(
            values={"analysis_state": state},
        )),
        aupdate_state=AsyncMock(),
    )
    workflow._pause_event = asyncio.Event()
    workflow._pause_event.set()
    workflow._continue_event = asyncio.Event()
    workflow._annotation_event = asyncio.Event()
    workflow._plan_review_event = asyncio.Event()
    workflow._species_event = asyncio.Event()
    workflow._species_user_response = None
    workflow._runtime_persistence_event = asyncio.Event()
    workflow._runtime_persistence_action = None
    workflow._runtime_persistence_action_pending = False
    workflow._runtime_persistence_gate_active = False
    workflow._runtime_persistence_pause_task = None
    workflow._set_status = lambda target, status: setattr(target, "status", status)
    workflow._send_state_sync = AsyncMock()
    workflow.await_restore_safe_point = AsyncMock()
    workflow.state_callback = AsyncMock()
    workflow.message_callback = AsyncMock()

    pause_result = await workflow.pause(state.id, analysis_state=state)
    assert pause_result["success"] is False
    assert pause_result["still_blocked"] is True
    assert state.status == TaskStatus.WAITING_INPUT
    assert workflow.sandbox is sandbox

    sent = []

    class _WebSocket:
        async def send_json(self, message):
            sent.append(message)

    await _resend_wait_input_state_sync(_WebSocket(), state, state.id)
    assert sent[0]["content"]["snapshot_failure"]["code"] == (
        "runtime_persistence_failure"
    )

    for _ in range(50):
        if workflow._runtime_persistence_gate_active:
            break
        await asyncio.sleep(0)
    token, generation = runtime_persistence_gate_identity(state)
    recovery_task = workflow._runtime_persistence_pause_task
    delivery = workflow.signal_runtime_persistence_action(
        "retry",
        gate_token=token,
        generation=generation,
    )
    assert delivery.outcome is RuntimePersistenceActionOutcome.ACCEPTED
    assert recovery_task is not None
    await asyncio.wait_for(recovery_task, timeout=1)

    assert sandbox.save_calls == 2
    assert workflow.sandbox is sandbox
    assert state.status == TaskStatus.PAUSED
    assert state.snapshot_failure is None
    assert state.wait_input_reason is None
