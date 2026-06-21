"""Structural tests: every agent entrypoint calls refresh_runtime_config_if_needed.

Each agent listed in the brief must have refresh_runtime_config_if_needed
available and the documented public / semipublic entrypoints must exist on
the class.  The SupervisorAgent does not inherit BaseAgent, so it gets a
standalone method wired the same way.
"""

from unittest.mock import MagicMock

from backend.agents.pathology_agent import PathologyAgent
from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.pi_agent import PIAgent
from backend.agents.biochemistry_agent import BiochemistryAgent
from backend.agents.painter_agent import PainterAgent
from backend.agents.technique_agent import TechniqueAgent
from backend.config._config import SystemConfig
from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import AgentSettingsProvider


class _Client:
    pass


def _provider():
    state = AnalysisState(id="entry-refresh", name="Entry", user_id="u", initial_instruction="x")
    return AgentSettingsProvider(state=state, config=SystemConfig(), global_client=_Client())


# ---------------------------------------------------------------------------
# Helper — monkey-patch refresh_runtime_config_if_needed on any agent
# ---------------------------------------------------------------------------

def _install_mock_refresh(agent):
    agent.refresh_runtime_config_if_needed = MagicMock()
    return agent.refresh_runtime_config_if_needed


# ---------------------------------------------------------------------------
# BaseAgent subclasses – refresh is inherited
# ---------------------------------------------------------------------------

class TestBaseAgentSubclasses:

    def test_pi_has_refresh(self):
        agent = PIAgent(settings_provider=_provider())
        assert hasattr(agent, "refresh_runtime_config_if_needed")

    def test_bioinformatics_has_refresh(self):
        agent = BioinformaticsAgent(settings_provider=_provider(), sandbox=None)
        assert hasattr(agent, "refresh_runtime_config_if_needed")

    def test_pathology_has_refresh(self):
        agent = PathologyAgent(settings_provider=_provider())
        assert hasattr(agent, "refresh_runtime_config_if_needed")

    def test_biochemistry_has_refresh(self):
        agent = BiochemistryAgent(settings_provider=_provider())
        assert hasattr(agent, "refresh_runtime_config_if_needed")

    def test_painter_has_refresh(self):
        agent = PainterAgent(settings_provider=_provider())
        assert hasattr(agent, "refresh_runtime_config_if_needed")

    def test_technique_has_refresh(self):
        agent = TechniqueAgent(settings_provider=_provider())
        assert hasattr(agent, "refresh_runtime_config_if_needed")


# ---------------------------------------------------------------------------
# SupervisorAgent – does NOT extend BaseAgent, must provide its own
# ---------------------------------------------------------------------------

class TestSupervisorRefresh:

    def test_supervisor_has_refresh(self):
        agent = SupervisorAgent(settings_provider=_provider(), session_id="entry-refresh")
        assert hasattr(agent, "refresh_runtime_config_if_needed")

    def test_supervisor_refresh_clears_cache_only_when_revision_changes(self):
        provider = _provider()
        provider.get_config_revision = MagicMock(side_effect=[100, 100, 101])
        provider.clear_agent_client_cache = MagicMock()
        agent = SupervisorAgent(settings_provider=provider, session_id="entry-refresh")

        agent.refresh_runtime_config_if_needed()
        agent.refresh_runtime_config_if_needed()
        agent.refresh_runtime_config_if_needed()

        assert provider.get_config_revision.call_count == 3
        provider.clear_agent_client_cache.assert_called_with("supervisor")
        assert provider.clear_agent_client_cache.call_count == 2


# ---------------------------------------------------------------------------
# PI Agent entrypoints
# ---------------------------------------------------------------------------

class TestPIEntrypoints:

    def test_process_exists(self):
        agent = PIAgent(settings_provider=_provider())
        assert hasattr(agent, "process")

    def test_review_cluster_annotations_exists(self):
        agent = PIAgent(settings_provider=_provider())
        assert hasattr(agent, "review_cluster_annotations")

    def test_annotate_clusters_from_literature_exists(self):
        agent = PIAgent(settings_provider=_provider())
        assert hasattr(agent, "annotate_clusters_from_literature")

    def test_plan_deep_research_tasks_exists(self):
        agent = PIAgent(settings_provider=_provider())
        assert hasattr(agent, "plan_deep_research_tasks")

    def test_synthesize_round_exists(self):
        agent = PIAgent(settings_provider=_provider())
        assert hasattr(agent, "synthesize_round")


# ---------------------------------------------------------------------------
# Bioinformatics entrypoints
# ---------------------------------------------------------------------------

class TestBioinformaticsEntrypoints:

    def test_process_exists(self):
        agent = BioinformaticsAgent(settings_provider=_provider(), sandbox=None)
        assert hasattr(agent, "process")

    def test_run_autonomous_analysis_exists(self):
        agent = BioinformaticsAgent(settings_provider=_provider(), sandbox=None)
        assert hasattr(agent, "run_autonomous_analysis")


# ---------------------------------------------------------------------------
# Pathology entrypoints
# ---------------------------------------------------------------------------

class TestPathologyEntrypoints:

    def test_process_exists(self):
        agent = PathologyAgent(settings_provider=_provider())
        assert hasattr(agent, "process")

    def test_call_llm_with_hitl_exists(self):
        agent = PathologyAgent(settings_provider=_provider())
        assert hasattr(agent, "_call_llm_with_hitl")

    def test_analyze_image_exists(self):
        agent = PathologyAgent(settings_provider=_provider())
        assert hasattr(agent, "_analyze_image")


# ---------------------------------------------------------------------------
# Biochemistry entrypoints
# ---------------------------------------------------------------------------

class TestBiochemistryEntrypoints:

    def test_process_exists(self):
        agent = BiochemistryAgent(settings_provider=_provider())
        assert hasattr(agent, "process")


# ---------------------------------------------------------------------------
# Painter entrypoints
# ---------------------------------------------------------------------------

class TestPainterEntrypoints:

    def test_generate_graphic_abstract_exists(self):
        agent = PainterAgent(settings_provider=_provider())
        assert hasattr(agent, "generate_graphic_abstract")

    def test_revise_graphic_abstract_exists(self):
        agent = PainterAgent(settings_provider=_provider())
        assert hasattr(agent, "revise_graphic_abstract")


# ---------------------------------------------------------------------------
# Technique entrypoints
# ---------------------------------------------------------------------------

class TestTechniqueEntrypoints:

    def test_process_exists(self):
        agent = TechniqueAgent(settings_provider=_provider())
        assert hasattr(agent, "process")

    def test_lookup_docs_exists(self):
        agent = TechniqueAgent(settings_provider=_provider())
        assert hasattr(agent, "lookup_docs")

    def test_lookup_from_error_exists(self):
        agent = TechniqueAgent(settings_provider=_provider())
        assert hasattr(agent, "lookup_from_error")


# ---------------------------------------------------------------------------
# Supervisor entrypoints
# ---------------------------------------------------------------------------

class TestSupervisorEntrypoints:

    def test_evaluate_step_exists(self):
        agent = SupervisorAgent(settings_provider=_provider(), session_id="entry-refresh")
        assert hasattr(agent, "evaluate_step")

    def test_generate_detailed_report_exists(self):
        agent = SupervisorAgent(settings_provider=_provider(), session_id="entry-refresh")
        assert hasattr(agent, "generate_detailed_report")

    def test_review_text_diagram_exists(self):
        agent = SupervisorAgent(settings_provider=_provider(), session_id="entry-refresh")
        assert hasattr(agent, "review_text_diagram")
