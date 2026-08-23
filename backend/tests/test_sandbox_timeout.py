"""L1: wall-clock deadline helper for the sandbox iopub loop. The loop must time
out on elapsed wall-clock time, not reset on every incoming message."""
import pytest

from backend.utils.sandbox import compute_remaining_timeout
from backend.utils.sandbox_parts import execution as execution_module


def test_remaining_positive_before_deadline():
    # deadline = 100, now = 40 → 60s left
    assert compute_remaining_timeout(100.0, 40.0) == 60.0


def test_remaining_zero_at_deadline():
    assert compute_remaining_timeout(100.0, 100.0) == 0.0


def test_remaining_negative_after_deadline():
    assert compute_remaining_timeout(100.0, 130.0) == -30.0


def test_expired_is_non_positive():
    # the loop treats remaining <= 0 as timed out
    assert compute_remaining_timeout(100.0, 100.0) <= 0
    assert compute_remaining_timeout(100.0, 101.0) <= 0


@pytest.mark.asyncio
async def test_interrupt_drain_has_overall_deadline_under_continuous_output(
    monkeypatch,
):
    calls = 0

    class ContinuousClient:
        def get_iopub_msg(self, *, timeout):
            nonlocal calls
            calls += 1
            return {"msg_type": "stream", "content": {"text": "still busy"}}

    async def direct_call(func, *args, **kwargs):
        return func(*args, **kwargs)

    ticks = iter((0.0, 0.1, 0.4, 0.8, 1.1))
    monkeypatch.setattr(execution_module, "_to_thread_drained", direct_call)

    returned_idle = await execution_module._drain_iopub_until_idle(
        ContinuousClient(),
        timeout_seconds=1.0,
        monotonic=lambda: next(ticks),
    )

    assert returned_idle is False
    assert calls == 3


@pytest.mark.asyncio
async def test_execution_timeout_overrides_default_without_changing_default(
    monkeypatch,
):
    from queue import Empty as QueueEmpty
    from types import SimpleNamespace
    interrupts = []

    class _KernelClient:
        def execute(self, _code):
            return "message-id"

        def get_iopub_msg(self, *, timeout):
            raise QueueEmpty

    class _KernelManager:
        def is_alive(self):
            return True

        def interrupt_kernel(self):
            interrupts.append(True)

    async def no_sleep(_seconds):
        return None

    async def no_drain(*_args, **_kwargs):
        return False

    async def immediate_queue_empty(*_args, **_kwargs):
        raise QueueEmpty

    monkeypatch.setattr(
        execution_module,
        "asyncio",
        SimpleNamespace(sleep=no_sleep),
    )
    monkeypatch.setattr(execution_module, "_drain_iopub_until_idle", no_drain)
    monkeypatch.setattr(execution_module, "_to_thread_drained", immediate_queue_empty)

    class _Sandbox:
        _kernel_dead = False
        kc = _KernelClient()
        km = _KernelManager()
        timeout = 9.0
        session_id = "execution-timeout-override"

        async def _restart_kernel(self):
            return None

    monkeypatch.setattr(
        execution_module,
        "compute_remaining_timeout",
        lambda *_args: -1.0,
    )
    override = await execution_module.ExecutionMixin._execute_unlocked(
        _Sandbox(),
        "pass",
        save_to_history=False,
        skip_safety_check=True,
        execution_timeout=0.25,
    )
    assert override.success is False
    assert override.error == "Execution timed out after 0.25 seconds"
    assert override.timed_out is True

    default = await execution_module.ExecutionMixin._execute_unlocked(
        _Sandbox(),
        "pass",
        save_to_history=False,
        skip_safety_check=True,
    )
    assert default.success is False
    assert default.error == "Execution timed out after 9.0 seconds"
    assert default.timed_out is True
    assert interrupts == [True, True]


@pytest.mark.asyncio
async def test_timeout_text_without_canonical_identity_is_not_a_save_budget_timeout(
    tmp_path, monkeypatch,
):
    from backend.utils.runtime_budget import RuntimeBudgetExceeded
    from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox

    budgets = {"max_save_seconds": 0.25}
    sb = LocalJupyterSandbox(
        session_id="timeout-text-without-identity",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
        runtime_budgets=budgets,
    )
    sb.runtime_revision = 1
    sb.saved_runtime_revision = 0

    async def fake_metrics():
        return {
            "namespace_count": 0,
            "largest_object_bytes": 0,
            "anndata_total_bytes": 0,
            "rss_bytes": 1,
            "backed_anndata_total_bytes": 0,
            "projected_materialized_rss_bytes": 1,
            "top_offenders": [],
        }

    async def fake_execute(code, **_kwargs):
        if "__USER_GLOBALS__:" in code:
            return CodeExecutionResult(
                success=True,
                output="__USER_GLOBALS__:[]",
            )
        result = CodeExecutionResult(
            success=False,
            error="Execution timed out after 0.25 seconds",
        )
        result.timed_out = False
        return result

    monkeypatch.setattr(sb, "_runtime_resource_metrics", fake_metrics)
    monkeypatch.setattr(sb, "_execute_unlocked", fake_execute)

    with pytest.raises(RuntimeError, match="Sandbox runtime save failed") as failure:
        await sb.save_session()
    assert not isinstance(failure.value, RuntimeBudgetExceeded)
