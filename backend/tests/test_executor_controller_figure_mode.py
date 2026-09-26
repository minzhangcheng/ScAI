"""Controller mode normalization keeps figure follow-up in its disposable runtime."""
import asyncio
from types import SimpleNamespace

import pytest


def _controller_harness(monkeypatch):
    from backend.agents import bioinformatics_agent as facade
    from backend.agents.bioinformatics import executor_controller, executor_loop

    class Sandbox:
        def __init__(self):
            self.executed = []

        async def execute(self, code):
            self.executed.append(code)

    class DisposableRuntime:
        instances = []

        def __init__(self, main_sandbox, state):
            self.main_sandbox = main_sandbox
            self.state = state
            self.sandbox = Sandbox()
            self.exit_call = None
            self.__class__.instances.append(self)

        async def __aenter__(self):
            return self.sandbox

        async def __aexit__(self, exc_type, exc, traceback):
            self.exit_call = (exc_type, exc, traceback)

    main_sandbox = Sandbox()
    parent_history = ["parent receipt"]
    parent_context = {"parent": "namespace"}
    parent_result = {"parent": "result"}
    agent = SimpleNamespace(
        sandbox=main_sandbox,
        code_history=parent_history,
        analysis_context=parent_context,
        last_result=parent_result,
    )
    state = SimpleNamespace(
        id="controller-session",
        code_history=["source receipt"],
        sandbox_state={},
        current_plan=None,
        bioinformatics_settlement_attempt_ids={},
    )
    monkeypatch.setattr(
        "backend.services.figure_followup_runtime.FigureFollowupRuntime",
        DisposableRuntime,
    )
    return SimpleNamespace(
        facade=facade,
        executor_controller=executor_controller,
        executor_loop=executor_loop,
        DisposableRuntime=DisposableRuntime,
        main_sandbox=main_sandbox,
        parent_history=parent_history,
        parent_context=parent_context,
        parent_result=parent_result,
        agent=agent,
        state=state,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("figure_flag", ["omitted", False])
@pytest.mark.parametrize("loop_exit", ["success", "exception", "cancellation"])
async def test_run_kind_alone_uses_disposable_runtime_and_restores_parent(
    monkeypatch, figure_flag, loop_exit,
):
    harness = _controller_harness(monkeypatch)
    prepared = {}

    async def prepare(_agent, _state, _instruction, **kwargs):
        prepared.update(kwargs)
        runtime = harness.DisposableRuntime.instances[-1]
        assert harness.agent.sandbox is runtime.sandbox
        return SimpleNamespace(agent=harness.agent)

    async def run_loop(context):
        await context.agent.sandbox.execute("plot existing evidence")
        context.agent.code_history.append("render-only receipt")
        if loop_exit == "exception":
            raise RuntimeError("figure loop failed")
        if loop_exit == "cancellation":
            raise asyncio.CancelledError()
        return {"status": "completed"}

    monkeypatch.setattr(harness.executor_loop, "prepare_loop_context", prepare)
    monkeypatch.setattr(harness.executor_loop, "run_turn_loop", run_loop)
    request = {"run_kind": "figure_followup"}
    if figure_flag != "omitted":
        request["figure_followup"] = figure_flag

    from backend.agents.bioinformatics.executor_context import ExecutorBindings

    async def invoke():
        return await harness.executor_controller.run_autonomous_analysis(
            harness.agent,
            harness.state,
            "Render the existing result.",
            bindings=ExecutorBindings(facade=harness.facade),
            **request,
        )

    if loop_exit == "success":
        result = await invoke()
        assert result["status"] == "completed"
    elif loop_exit == "exception":
        with pytest.raises(RuntimeError, match="figure loop failed"):
            await invoke()
    else:
        with pytest.raises(asyncio.CancelledError):
            await invoke()

    runtime = harness.DisposableRuntime.instances[-1]
    assert prepared["figure_followup"] is True
    assert prepared["run_kind"] == "figure_followup"
    assert runtime.sandbox.executed == ["plot existing evidence"]
    assert harness.main_sandbox.executed == []
    assert runtime.exit_call == (None, None, None)
    assert harness.agent.sandbox is harness.main_sandbox
    assert harness.agent.code_history is harness.parent_history
    assert harness.agent.analysis_context is harness.parent_context
    assert harness.agent.last_result is harness.parent_result
    assert harness.parent_history == ["parent receipt"]
    assert harness.state.code_history == ["source receipt"]
