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
    assert all(call.kwargs["figure_followup"] is True
               for call in agent._revise_figure_code.await_args_list)
    agent.sandbox.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_image", [False, True])
async def test_followup_revision_prompt_includes_runtime_boundary(monkeypatch, tmp_path, with_image):
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.agents.bioinformatics import figure_context
    from backend.tests.test_bioinfo_context_scopes import _bare_agent

    agent = _bare_agent(monkeypatch, tmp_path)
    client = MagicMock()
    client.chat_with_vision = AsyncMock(return_value=SimpleNamespace(content="ax.plot(values)"))
    monkeypatch.setattr(BioinformaticsAgent, "llm_client", property(lambda self: client))
    monkeypatch.setattr(BioinformaticsAgent, "has_vision", property(lambda self: True))
    monkeypatch.setattr(figure_context, "figure_input_context", AsyncMock(return_value="values exist"))
    agent._call_figure_revision_llm = lambda call: call()
    agent._budgeted_text_call = AsyncMock(return_value=SimpleNamespace(content="ax.plot(values)"))

    code = await agent._revise_figure_code(
        "ax.plot(values)", "Improve labels", [], [], "line",
        AnalysisState(user_id="figure-capability"), "image" if with_image else None,
        figure_followup=True,
    )

    call = client.chat_with_vision.await_args if with_image else agent._budgeted_text_call.await_args
    prompt = call.kwargs["text_prompt"] if with_image else call.kwargs["messages"][0]["content"]
    assert "Do not assign into, delete from, or mutate a pre-existing AnnData" in prompt
    assert "Do not call savefig, save, show, or close" in prompt
    assert "Do not load files" in prompt
    assert code == "ax.plot(values)"
