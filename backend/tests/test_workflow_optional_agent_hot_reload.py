"""Tests for workflow optional-agent materialization on config change."""
from unittest.mock import MagicMock, patch

from backend.workflow.core import SingleCellAnalysisWorkflow


def _workflow_with_provider(model_map):
    wf = SingleCellAnalysisWorkflow.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "wf-hot"
    wf.config = MagicMock()
    wf.config.image_tile_size = 1024
    wf.config.api.proxy_url = None
    wf.sandbox = None
    wf._pause_event = MagicMock()
    wf._settings_provider = MagicMock()
    wf._settings_provider.get_model = MagicMock(side_effect=lambda agent: model_map.get(agent, ""))
    wf._settings_provider.get_vision_setting = MagicMock(return_value=True)
    wf._settings_provider.register_client = MagicMock()
    wf._settings_provider.clear_agent_client_cache = MagicMock()
    wf.pathology_agent = None
    wf.biochemistry_agent = None
    wf.painter_agent = None
    wf.supervisor_agent = None
    wf.technique_agent = None
    wf.pi_agent = MagicMock()
    wf.bioinformatics_agent = MagicMock()
    return wf


def test_ensure_agent_ready_creates_agent_that_was_disabled_at_initialize():
    wf = _workflow_with_provider({"pathology": "vision-model"})
    assert wf.pathology_agent is None
    assert wf.ensure_agent_ready("pathology") is True
    assert wf.pathology_agent is not None


def test_ensure_agent_ready_disables_existing_optional_agent_when_model_empty():
    wf = _workflow_with_provider({"pathology": ""})
    wf.pathology_agent = MagicMock()
    assert wf.ensure_agent_ready("pathology") is False
    assert wf.pathology_agent is None


def test_ensure_agent_ready_refreshes_existing_agent_when_enabled():
    wf = _workflow_with_provider({"pathology": "vision-model"})
    wf.pathology_agent = MagicMock()
    wf.pathology_agent.refresh_runtime_config_if_needed = MagicMock()
    assert wf.ensure_agent_ready("pathology") is True
    wf.pathology_agent.refresh_runtime_config_if_needed.assert_called_once()


def test_ensure_agent_ready_propagates_technique_agent_to_existing_dependents():
    wf = _workflow_with_provider({"technique": "tech-model"})
    wf.bioinformatics_agent = MagicMock()
    wf.bioinformatics_agent.technique_agent = None
    wf.supervisor_agent = MagicMock()
    wf.supervisor_agent.technique_agent = None

    assert wf.ensure_agent_ready("technique") is True

    assert wf.technique_agent is not None
    assert wf.bioinformatics_agent.technique_agent is wf.technique_agent
    assert wf.supervisor_agent.technique_agent is wf.technique_agent


def test_ensure_agent_ready_clears_technique_agent_from_existing_dependents_when_disabled():
    wf = _workflow_with_provider({"technique": ""})
    old_technique = MagicMock()
    wf.technique_agent = old_technique
    wf.bioinformatics_agent = MagicMock()
    wf.bioinformatics_agent.technique_agent = old_technique
    wf.supervisor_agent = MagicMock()
    wf.supervisor_agent.technique_agent = old_technique

    assert wf.ensure_agent_ready("technique") is False

    assert wf.technique_agent is None
    assert wf.bioinformatics_agent.technique_agent is None
    assert wf.supervisor_agent.technique_agent is None
