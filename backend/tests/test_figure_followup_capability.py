"""Plotting-only capability checks for figure-follow-up renders."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.schemas import AnalysisState
from backend.services.figure_followup import validate_plotting_only_code


@pytest.mark.parametrize(
    "code",
    [
        'raw = open("counts.csv").read()',
        'data = sc.read_h5ad("input.h5ad")',
        "sc.pp.scrublet(adata)",
        "sc.tl.rank_genes_groups(adata, groupby='condition')",
        "model.fit_transform(values)",
        "adata.obs['cluster'] = labels",
        "adata.X[:, 0] *= 2",
        "del adata.obs['cluster']",
        "for adata.X[0] in values:\n    pass",
        "async def update(values):\n    async for adata.obs['cluster'] in values:\n        pass",
        "[value for adata.obs['cluster'] in values]",
        "{value for adata.layers['counts'][0] in values}",
        "{adata.obs['cluster']: value for adata.obs['cluster'] in values}",
        "(value for adata.var['score'] in values)",
        "adata.write_h5ad('updated.h5ad')",
        "eval('fit_model(data)')",
        "import numpy as np\nnp.linalg.svd(adata.X)",
        "import numpy as np\nnp.corrcoef(adata.X.T)",
        "import numpy.linalg as linear\nlinear.eig(values)",
        "from numpy.linalg import svd as decompose\ndecompose(values)",
        "frame.corr()",
        "frame.cov()",
    ],
)
def test_plotting_validator_rejects_analysis_reads_mutation_and_persistence(code):
    error = validate_plotting_only_code(code)

    assert error


def test_plotting_validator_allows_local_dataframe_and_array_preparation():
    code = """
import numpy as np
import pandas as pd
values = np.array([1.0, 2.0])
frame = pd.DataFrame({"value": values})
frame["scaled"] = frame["value"] * 2
ax.scatter(frame["value"], frame["scaled"])
"""

    assert validate_plotting_only_code(code) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        'raw = open("counts.csv").read()',
        "import numpy as np\nnp.linalg.svd(adata.X)",
        "import numpy as np\nnp.corrcoef(adata.X.T)",
    ],
)
async def test_initial_followup_render_rejects_forbidden_code_before_agent_validation(code):
    from backend.agents.bioinformatics.figure_preflight import (
        FigureRequestContext,
        prepare_figure_request,
    )

    agent = SimpleNamespace(_validate_code=MagicMock(return_value=(True, "")))
    logger = MagicMock()
    request = FigureRequestContext(
        agent=agent,
        code=code,
        figure_name="blocked",
        description="A blocked render",
        figure_type="volcano",
        state=AnalysisState(user_id="figure-capability"),
        review_callback=None,
        diagnostic=False,
        plan_step_id=2,
        purpose="result",
        logical_figure_id="logical-figure",
        source_execution_id="source-receipt",
        submitted_plan_step_id=2,
        plan_requirement_key="plan.2.1",
        supervisor_binding_decision="accept",
        supervisor_binding_reason="Matches the requested output",
        dependencies=SimpleNamespace(get=lambda _name: logger),
        producer_run_kind="figure_followup",
        producer_run_scope_id="figure-scope",
        figure_followup_task_id="figure-task",
        figure_followup_scope={},
    )

    attempt, error = await prepare_figure_request(request)

    assert attempt is None
    assert "Figure follow-up request rejected" in error
    agent._validate_code.assert_not_called()


@pytest.mark.asyncio
async def test_every_followup_revision_and_retry_is_revalidated_before_render():
    from backend.agents.bioinformatics.figure_revision import perform_revision

    logger = MagicMock()
    deps = SimpleNamespace(
        get=lambda name: {
            "logger": logger,
            "compact_traceback": lambda error, **_kwargs: str(error),
        }[name]
    )
    agent = SimpleNamespace(
        _notify=AsyncMock(),
        _validate_code=MagicMock(return_value=(True, "")),
        _revise_figure_code=AsyncMock(side_effect=[
            "sc.pp.scrublet(adata)",
            "model.fit(values)",
        ]),
        _environment_api_hint=AsyncMock(return_value=""),
        technique_agent=None,
        sandbox=SimpleNamespace(execute=AsyncMock()),
    )
    attempt = SimpleNamespace(
        request=SimpleNamespace(
            producer_run_kind="figure_followup",
            figure_name="blocked_revision",
            figure_type="volcano",
        ),
        agent=agent,
        deps=deps,
        state=AnalysisState(user_id="figure-capability"),
        figure=SimpleNamespace(
            revision_count=0,
            png_base64="",
            layout_violations=[],
        ),
        current_code="ax.scatter(values)",
        revision_count=0,
        revision_history=[],
        best_snapshot=None,
        jpg_path=None,
        pdf_path=None,
        staleness_note="",
        terminal_revision_reason="attempts_exhausted",
    )

    outcome = await perform_revision(
        attempt,
        feedback="Revise the style",
        suggestions=[],
        must_fix=[],
        quality_score=6,
    )

    assert outcome == "break"
    assert attempt.terminal_revision_reason == "revision_failed"
    assert agent._revise_figure_code.await_count == 2
    agent.sandbox.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np\nnp.linalg.svd(adata.X)",
        "import numpy as np\nnp.corrcoef(adata.X.T)",
    ],
)
async def test_numerical_analysis_revision_is_rejected_before_kernel_execution(code):
    from backend.agents.bioinformatics.figure_revision import (
        execute_revision_candidate,
    )

    execute = AsyncMock()
    attempt = SimpleNamespace(
        request=SimpleNamespace(producer_run_kind="figure_followup"),
        agent=SimpleNamespace(sandbox=SimpleNamespace(execute=execute)),
        deps=SimpleNamespace(get=lambda _name: MagicMock()),
    )

    result, physical_error = await execute_revision_candidate(attempt, code)

    assert physical_error is None
    assert result.error
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_technical_retry_numerical_analysis_is_rejected_before_kernel_execution(
    monkeypatch,
):
    import backend.agents.bioinformatics.figure_revision as figure_revision

    logger = MagicMock()
    deps = SimpleNamespace(
        get=lambda name: {
            "logger": logger,
            "compact_traceback": lambda error, **_kwargs: str(error),
        }[name]
    )
    agent = SimpleNamespace(
        _notify=AsyncMock(),
        _validate_code=MagicMock(return_value=(True, "")),
        _revise_figure_code=AsyncMock(side_effect=[
            "ax.scatter(values)",
            "import numpy as np\nnp.corrcoef(adata.X.T)",
        ]),
        _environment_api_hint=AsyncMock(return_value=""),
        technique_agent=None,
        sandbox=SimpleNamespace(execute=AsyncMock()),
    )
    attempt = SimpleNamespace(
        request=SimpleNamespace(
            producer_run_kind="figure_followup",
            figure_name="blocked_retry",
            figure_type="scatter",
        ),
        agent=agent,
        deps=deps,
        state=AnalysisState(user_id="figure-capability"),
        figure=SimpleNamespace(
            revision_count=0,
            png_base64="",
            layout_violations=[],
        ),
        current_code="ax.scatter(values)",
        revision_count=0,
        revision_history=[],
        best_snapshot=None,
        jpg_path=None,
        pdf_path=None,
        staleness_note="",
        terminal_revision_reason="attempts_exhausted",
    )
    real_execute_candidate = figure_revision.execute_revision_candidate
    candidate_calls = 0

    async def fail_first_candidate(current_attempt, source):
        nonlocal candidate_calls
        candidate_calls += 1
        if candidate_calls == 1:
            return SimpleNamespace(
                output="", error="RuntimeError: initial revision render failed"
            ), None
        return await real_execute_candidate(current_attempt, source)

    monkeypatch.setattr(
        figure_revision, "execute_revision_candidate", fail_first_candidate
    )

    outcome = await figure_revision.perform_revision(
        attempt,
        feedback="Correct the rendering error",
        suggestions=[],
        must_fix=[],
        quality_score=6,
    )

    assert outcome == "break"
    assert candidate_calls == 2
    assert agent._revise_figure_code.await_count == 2
    agent.sandbox.execute.assert_not_awaited()
