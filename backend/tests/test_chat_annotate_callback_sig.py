"""Exercise the Chat callback with the executor's annotation arguments."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.routes.chat_parts import delegation


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_markers", [False, True])
async def test_chat_annotation_callback_accepts_executor_context(
    monkeypatch, tmp_path, empty_markers,
):
    from backend.agents import bioinformatics_agent, literature_agent, pi_agent
    from backend.utils import sandbox as sandbox_module

    state = AnalysisState(id="chat-annotation-callback", user_id="owner")
    markers = {} if empty_markers else {"0": ["GeneA", "GeneB"]}
    source = {"adata_var": "atlas", "cluster_key": "clusters", "partition_digest": "p1"}
    annotation = {"status": "success", "annotations": {"0": {"cell_type": "TypeA"}}}
    literature = SimpleNamespace(annotate_clusters=AsyncMock(return_value=annotation))
    sandbox = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), km=None)
    observed = []

    def make_bio_agent(**kwargs):
        async def run(actual_state, instruction, max_steps, **execution_options):
            assert actual_state is state
            assert max_steps == 60
            observed.append(await kwargs["annotate_clusters_callback"](
                markers, "tissue-A", "species-B",
                analysis_log=[], database_priors={"0": ["TypeA"]},
                annotation_source=source,
            ))
            return {"status": "completed", "findings": [], "figures": []}

        assert callable(kwargs["supervisor_callback"])
        assert callable(kwargs["figure_review_callback"])
        return SimpleNamespace(run_autonomous_analysis=run, set_session=MagicMock())

    monkeypatch.setattr(delegation, "resolve_config_for_user", AsyncMock(return_value=SystemConfig()))
    monkeypatch.setattr(delegation, "get_session_states", lambda _: tmp_path / "states")
    monkeypatch.setattr(delegation, "get_session_workspace", lambda _: tmp_path / "workspace")
    monkeypatch.setattr(delegation, "_collect_session_figures", lambda *a, **kw: [])
    monkeypatch.setattr(delegation, "_persist_chat_terminal", AsyncMock(return_value=True))
    from backend.utils import agent_llm
    monkeypatch.setattr(agent_llm, "get_agent_llm_client", MagicMock(
        return_value=SimpleNamespace(close=AsyncMock()),
    ))
    monkeypatch.setattr(sandbox_module, "LocalJupyterSandbox", MagicMock(return_value=sandbox))
    monkeypatch.setattr(literature_agent, "LiteratureAgent", MagicMock(return_value=literature))
    pi = SimpleNamespace(set_session=MagicMock(), review_chat_execution=AsyncMock(
        return_value={"decision": "accept", "rationale": "checked"},
    ))
    monkeypatch.setattr(pi_agent, "PIAgent", MagicMock(return_value=pi))
    monkeypatch.setattr(bioinformatics_agent, "BioinformaticsAgent", make_bio_agent)

    result, _ = await delegation._execute_chat_agent_inner(
        state.id, state, "bioinformatics", "Annotate this atlas", MagicMock(), AsyncMock(), [],
    )

    assert result["status"] == "completed"
    if empty_markers:
        assert observed[0]["status"] == "error"
        literature.annotate_clusters.assert_not_awaited()
    else:
        assert observed[0] is annotation
        literature.annotate_clusters.assert_awaited_once_with(
            cluster_markers=markers, tissue="tissue-A", organism="species-B", state=state,
        )
    assert source["adata_var"] == "atlas"
    sandbox.stop.assert_awaited_once_with(save=True)
    assert state.id not in delegation.chat_sandboxes
