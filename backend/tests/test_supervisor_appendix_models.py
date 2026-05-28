"""C2 — appendix records resolved (provider) model/temperature, not raw overrides."""
import pytest


def _build_state_without_overrides():
    from backend.models.schemas import AnalysisState
    state = AnalysisState(
        id="sess-c2",
        name="test",
        initial_instruction="x",
        user_id="user-c2",
    )
    # No custom_models / custom_temperatures set — must resolve to config defaults.
    return state


def test_appendix_records_resolved_models_via_provider():
    from backend.agents.supervisor_agent import SupervisorAgent
    from backend.config._config import SystemConfig
    from backend.utils.settings_provider import AgentSettingsProvider

    state = _build_state_without_overrides()
    config = SystemConfig()
    provider = AgentSettingsProvider(state=state, config=config, global_client=None)

    sup = SupervisorAgent.__new__(SupervisorAgent)
    sup._provider = provider

    appendix = sup._build_appendix(state, stats={})

    models = appendix["agent_models"]
    temps = appendix["agent_temperatures"]

    # PI default model from SystemConfig must appear, even without override.
    assert "pi" in models
    assert models["pi"] == config.models.pi_model
    assert models["pi"]
    # All known agents present
    for agent in ("pi", "bioinformatics", "pathology", "biochemistry",
                  "literature", "painter", "supervisor"):
        assert agent in models
        assert agent in temps


def test_appendix_respects_explicit_override():
    from backend.agents.supervisor_agent import SupervisorAgent
    from backend.config._config import SystemConfig
    from backend.utils.settings_provider import AgentSettingsProvider
    from backend.models.schemas import AnalysisState

    state = AnalysisState(id="sess", name="t", initial_instruction="x", user_id="u1")
    state.custom_models = {"pi": "custom-pi-model"}
    state.custom_temperatures = {"pi": 0.9}

    provider = AgentSettingsProvider(state=state, config=SystemConfig(), global_client=None)
    sup = SupervisorAgent.__new__(SupervisorAgent)
    sup._provider = provider

    appendix = sup._build_appendix(state, stats={})
    assert appendix["agent_models"]["pi"] == "custom-pi-model"
    assert appendix["agent_temperatures"]["pi"] == 0.9