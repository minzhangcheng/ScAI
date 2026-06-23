"""Integration-style tests for snapshot rollback idempotency cache guards."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.models.schemas import (
    AgentRoundCache,
    AgentType,
    AnalysisState,
    MessageRole,
)
from backend.workflow.helpers import cache_hit, cache_store


@pytest.fixture
def analysis_state():
    s = AnalysisState(user_id="test_user", initial_instruction="run analysis")
    s.current_round = 1
    return s


@pytest.fixture
def mock_workflow(analysis_state):
    """Create a stand-in workflow object with the methods _biochemistry_analysis uses.

    We avoid importing SingleCellAnalysisWorkflow (heavy) and instead bind the
    real bound method onto a lightweight stand-in via __get__.
    """
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.biochemistry_agent = AsyncMock()
    wf.biochemistry_agent.process = AsyncMock(
        return_value={"findings": ["bio_finding_1"], "status": "completed"}
    )
    wf._send_message = AsyncMock()
    wf._build_previous_findings_context = MagicMock(return_value="")
    # bind the real method
    wf._biochemistry_analysis = AgentNodesMixin._biochemistry_analysis.__get__(wf)
    return wf


@pytest.mark.asyncio
async def test_biochem_cache_hit_short_circuits_agent_call(mock_workflow, analysis_state):
    # Seed cache for current round
    cache_store(analysis_state, AgentType.BIOCHEMISTRY, {"findings": ["cached"], "status": "completed"})

    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"biochemistry_tasks": ["task"]},
    }
    result = await mock_workflow._biochemistry_analysis(state)

    # Agent NOT called
    mock_workflow.biochemistry_agent.process.assert_not_called()
    # Returns cached value as biochemistry_result
    assert result["biochemistry_result"] == {"findings": ["cached"], "status": "completed"}
    # failed_node cleared
    assert result["failed_node"] is None


@pytest.mark.asyncio
async def test_biochem_cache_miss_runs_agent_and_stores(mock_workflow, analysis_state):
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"biochemistry_tasks": ["task"]},
    }
    result = await mock_workflow._biochemistry_analysis(state)

    mock_workflow.biochemistry_agent.process.assert_called_once()
    assert result["biochemistry_result"]["findings"] == ["bio_finding_1"]
    # Cache populated
    assert cache_hit(analysis_state, AgentType.BIOCHEMISTRY)["findings"] == ["bio_finding_1"]


@pytest.mark.asyncio
async def test_biochem_disabled_agent_also_caches(analysis_state):
    """When the agent is None (disabled), the skipped result should also be cached."""
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.biochemistry_agent = None
    wf._send_message = AsyncMock()
    wf.ensure_agent_ready = MagicMock(return_value=False)
    wf._biochemistry_analysis = AgentNodesMixin._biochemistry_analysis.__get__(wf)

    state = {"analysis_state": analysis_state, "pi_instructions": {}}
    result = await wf._biochemistry_analysis(state)

    assert result["biochemistry_result"]["status"] == "skipped"
    # Cache should contain the skipped result
    assert cache_hit(analysis_state, AgentType.BIOCHEMISTRY) == {"findings": [], "status": "skipped"}


@pytest.mark.asyncio
async def test_pathology_cache_hit_skips_agent(analysis_state):
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.pathology_agent = AsyncMock()
    wf.pathology_agent.process = AsyncMock(return_value={"findings": ["should_not_be_called"]})
    wf._send_message = AsyncMock()
    wf._build_previous_findings_context = MagicMock(return_value="")
    wf._collect_current_round_results = MagicMock(return_value={})
    wf.pi_agent = AsyncMock()
    wf._pathology_analysis = AgentNodesMixin._pathology_analysis.__get__(wf)

    cache_store(analysis_state, AgentType.PATHOLOGY, {"findings": ["cached_path"], "status": "completed"})

    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"pathology_tasks": ["task"], "overview": "x"},
        "biochemistry_result": None,
    }
    result = await wf._pathology_analysis(state)

    wf.pathology_agent.process.assert_not_called()
    assert result["pathology_result"]["findings"] == ["cached_path"]
    assert result["failed_node"] is None


@pytest.mark.asyncio
async def test_pathology_no_data_path_caches_skipped_result(analysis_state):
    """no_data early return should populate cache too."""
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.pathology_agent = AsyncMock()
    wf._send_message = AsyncMock()
    wf._pathology_analysis = AgentNodesMixin._pathology_analysis.__get__(wf)

    # analysis_state has no pathology files by default
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"pathology_tasks": ["task"]},
    }
    result = await wf._pathology_analysis(state)

    assert result["pathology_result"]["status"] == "no_data"
    cached = cache_hit(analysis_state, AgentType.PATHOLOGY)
    assert cached is not None
    assert cached["status"] == "no_data"


@pytest.mark.asyncio
async def test_pathology_disabled_path_caches_skipped(analysis_state):
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.pathology_agent = None
    wf._send_message = AsyncMock()
    wf.ensure_agent_ready = MagicMock(return_value=False)
    wf._pathology_analysis = AgentNodesMixin._pathology_analysis.__get__(wf)

    state = {"analysis_state": analysis_state, "pi_instructions": {}}
    result = await wf._pathology_analysis(state)

    assert result["pathology_result"]["status"] == "skipped"
    assert cache_hit(analysis_state, AgentType.PATHOLOGY) == {"findings": [], "status": "skipped"}


@pytest.mark.asyncio
async def test_literature_cache_hit_skips_agent(analysis_state):
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.literature_agent = MagicMock()
    wf.literature_agent.is_deep_research_available = False
    wf.literature_agent.process = AsyncMock(return_value={"findings": ["x"], "report": "r"})
    wf._send_message = AsyncMock()
    wf._collect_current_round_results = MagicMock(return_value={})
    wf._build_previous_findings_context = MagicMock(return_value="")
    wf.pi_agent = AsyncMock()
    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf)
    wf._literature_research = AgentNodesMixin._literature_research.__get__(wf)

    cache_store(analysis_state, AgentType.LITERATURE, {"topics": ["t"], "findings": ["cached_lit"], "reports": []})

    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {"literature_topics": ["topic1"], "overview": "x"},
        "biochemistry_result": None,
        "pathology_result": None,
        "bioinformatics_result": None,
    }
    result = await wf._literature_research(state)

    wf.literature_agent.process.assert_not_called()
    assert result["literature_result"]["findings"] == ["cached_lit"]
    assert result["failed_node"] is None


@pytest.mark.asyncio
async def test_bioinformatics_cache_hit_skips_agent(analysis_state):
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.bioinformatics_agent = AsyncMock()
    wf.bioinformatics_agent.run_autonomous_analysis = AsyncMock(
        return_value={"findings": ["should_not_run"]}
    )
    wf._send_message = AsyncMock()
    wf._current_graph_state = None
    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf)
    wf._bioinformatics_analysis = AgentNodesMixin._bioinformatics_analysis.__get__(wf)

    cache_store(analysis_state, AgentType.BIOINFORMATICS, {
        "findings": ["cached_bio"],
        "figures": [],
        "summary": "cached summary",
        "steps_executed": 5,
        "status": "completed",
    })

    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {},
        "planning_stage": "full",
        "human_guidance": None,
    }
    result = await wf._bioinformatics_analysis(state)

    wf.bioinformatics_agent.run_autonomous_analysis.assert_not_called()
    assert result["bioinformatics_result"]["findings"] == ["cached_bio"]
    assert result["bioinformatics_result"]["summary"] == "cached summary"
    assert result["failed_node"] is None


@pytest.mark.asyncio
async def test_bioinformatics_waiting_data_path_caches_skipped(analysis_state):
    """waiting_data early return should populate cache so replay also short-circuits."""
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.bioinformatics_agent = AsyncMock()
    wf._send_message = AsyncMock()
    wf._current_graph_state = None
    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf)
    wf._bioinformatics_analysis = AgentNodesMixin._bioinformatics_analysis.__get__(wf)

    # analysis_state has no sequencing data by default → triggers waiting_data branch
    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {},
        "planning_stage": "full",
        "human_guidance": None,
    }
    result = await wf._bioinformatics_analysis(state)

    assert result["bioinformatics_result"]["status"] == "waiting_data"
    cached = cache_hit(analysis_state, AgentType.BIOINFORMATICS)
    assert cached is not None
    assert cached["status"] == "waiting_data"


def test_error_result_is_cached_and_replayed(analysis_state):
    """Spec §6 item 7: a result containing an error key is cached and
    replayed on subsequent entries — no auto-retry or special handling."""
    error_result = {"findings": [], "status": "failed", "error": "UpstreamExhaustedError: timeout"}
    cache_store(analysis_state, AgentType.BIOCHEMISTRY, error_result)

    hit = cache_hit(analysis_state, AgentType.BIOCHEMISTRY)
    assert hit is not None
    assert hit["status"] == "failed"
    assert "error" in hit


@pytest.mark.asyncio
async def test_bioinformatics_stage2_cache_hit_transitions_planning_stage_to_full(analysis_state):
    """When bioinfo cache hits on a stage2 replay, planning_stage must
    transition to "full" — mirroring what the normal stage2 path does."""
    from backend.workflow.agents import AgentNodesMixin

    wf = MagicMock()
    wf.session_id = "test"
    wf.bioinformatics_agent = AsyncMock()
    wf._send_message = AsyncMock()
    wf._current_graph_state = None
    wf._require_core_agent = AgentNodesMixin._require_core_agent.__get__(wf)
    wf._bioinformatics_analysis = AgentNodesMixin._bioinformatics_analysis.__get__(wf)

    cache_store(analysis_state, AgentType.BIOINFORMATICS, {
        "findings": ["stage2_finding"],
        "figures": [],
        "summary": "stage2 summary",
        "steps_executed": 3,
        "status": "completed",
    })

    state = {
        "analysis_state": analysis_state,
        "pi_instructions": {},
        "planning_stage": "stage2",
        "human_guidance": None,
    }
    result = await wf._bioinformatics_analysis(state)

    assert result["bioinformatics_result"]["findings"] == ["stage2_finding"]
    assert result["planning_stage"] == "full"
    assert result["planning_context"] is None
    assert result["failed_node"] is None
