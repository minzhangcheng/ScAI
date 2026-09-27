"""Keep existing analysis results discoverable by the restricted renderer."""
import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from backend.agents.bioinformatics.figure_context import figure_input_context
from backend.models.schemas import AnalysisState, CodeExecution, AnalysisStep
from backend.services.figure_followup import (
    figure_followup_output_names, figure_followup_source_prompt,
    validate_plotting_only_code,
)
from backend.tests.test_bioinfo_executor_replay import replay_context


def _sources():
    state = AnalysisState(user_id="figure-inputs", code_history=[CodeExecution(
        id="de-execution", code="de_results = compute()", plan_step_ids=[4],
        declared_persistent_outputs=["de_results", "missing_result"],
        realized_persistent_outputs=["de_results"],
    )], analysis_log=[AnalysisStep(
        step_number=1, step_type="execute_code", execution_id="de-execution",
        code_description="Compare conditions within each cell type",
    )])
    scope = {"tasks": [{"task_id": "volcano", "logical_figure_id": "hep-volcano",
                         "plan_step_id": 4, "eligible_execution_ids": ["de-execution"]}]}
    return state, scope


@pytest.mark.asyncio
async def test_followup_discovers_dictionary_result_without_analysis_or_mutation():
    state, scope = _sources()
    frame = pd.DataFrame({"names": ["gene"], "logfoldchanges": [1.5], "pvals_adj": [.01]})
    # More unrelated objects than the inventory limit must not hide the source.
    namespace = {f"a{i:03}": frame for i in range(70)}
    namespace["de_results"] = {"Hepatocyte": frame, "nested": {"hidden": frame}}
    original_names = set(namespace)

    async def execute(code, **kwargs):
        assert kwargs == dict(records_history=False, skip_safety_check=True, mutates_namespace=False)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(code, namespace)
        return SimpleNamespace(success=True, output=stream.getvalue())

    sandbox = SimpleNamespace(kc=object(), execute=execute)
    result = json.loads(await figure_input_context(
        sandbox, "", preferred_names=figure_followup_output_names(state, scope),
    ))
    assert result["objects"][0] == {
        "name": "de_results['Hepatocyte']", "type": "DataFrame", "shape": [1, 3],
        "columns": ["names", "logfoldchanges", "pvals_adj"], "column_count": 3,
    }
    assert len(result["objects"]) == 60
    assert result["missing_referenced_names"] == ["missing_result"]
    assert set(namespace) - {"__builtins__"} == original_names
    pd.testing.assert_frame_equal(frame, namespace["de_results"]["Hepatocyte"])
    assert "hidden" not in json.dumps(result)
    prompt = figure_followup_source_prompt(state, scope)
    assert "Compare conditions within each cell type" in prompt
    assert '"realized_persistent_outputs": ["de_results"]' in prompt
    assert validate_plotting_only_code("ax.scatter(de_results['Hepatocyte']['logfoldchanges'], [1])") is None
    assert validate_plotting_only_code("globals()['de_results']")
    assert validate_plotting_only_code("sc.tl.rank_genes_groups(adata, 'condition')")


@pytest.mark.asyncio
async def test_followup_request_refreshes_source_inventory_without_saving_it_in_history(replay_context):
    from backend.agents.bioinformatics import executor_loop

    context, client, response = replay_context
    state, scope = _sources()
    context.agent._current_analysis_state = state
    context.executor_run_kind = "figure_followup"
    context.figure_followup_scope = scope
    context.agent.sandbox.kc = object()
    context.agent.sandbox.execute = AsyncMock(return_value=SimpleNamespace(
        success=True, output='__SCAI_FIGURE_INPUTS__:{"objects":[{"name":"de_results[\\"Hepatocyte\\"]"}]}',
    ))
    client.chat_with_tools.return_value = response
    for _ in range(2):
        await executor_loop.start_turn(context)
        request = client.chat_with_tools.await_args.kwargs["messages"]
        anchors = [m["content"] for m in request if "Figure follow-up source contract:" in str(m.get("content"))]
        assert len(anchors) == 1
        assert "Hepatocyte" in anchors[0]
        assert "de-execution" in anchors[0]
        assert "Compare conditions" in anchors[0]
        assert all("Measured live input inventory" not in str(m) for m in context.messages)
    assert context.agent.sandbox.execute.await_count == 2


@pytest.mark.asyncio
async def test_unavailable_inventory_is_disclosed():
    sandbox = SimpleNamespace(kc=object(), execute=AsyncMock(side_effect=RuntimeError("offline")))
    assert "unavailable" in await figure_input_context(sandbox, "", preferred_names=["de_results"])
