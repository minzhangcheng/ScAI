"""PI delivery is durable session state and never an analysis gate."""

from unittest.mock import AsyncMock

import pytest

from backend.config import SystemConfig
from backend.models.schemas import AnalysisState, ContextBudgetObservation
from backend.utils.context_budget import (
    acknowledge_observations,
    append_observation,
    pending_observation_anchor,
)
from backend.utils.settings_provider import AgentSettingsProvider
from backend.state import sessions


def _observation(**updates):
    values = {
        "agent": "bioinformatics",
        "measured_tokens": 100,
        "final_tokens": 90,
        "ceiling": 80,
        "requested_completion_reserve": 32,
        "effective_completion_reserve": 32,
        "actions": ["protected context remained above the ceiling"],
        "fits": False,
    }
    values.update(updates)
    return ContextBudgetObservation(**values)


def test_pending_observations_deduplicate_and_delivered_records_are_evicted_first():
    state = AnalysisState(user_id="user-1")
    first = append_observation(state, _observation())
    duplicate = append_observation(state, _observation())
    assert duplicate.id == first.id
    assert duplicate.occurrence_count == 2

    acknowledge_observations(state, [first.id])
    for index in range(20):
        append_observation(state, _observation(agent=f"agent-{index}"))

    assert len(state.context_budget_observations) == 20
    assert all(item.id != first.id for item in state.context_budget_observations)
    assert all(item.pi_delivered_at is None for item in state.context_budget_observations)


@pytest.mark.asyncio
async def test_pi_receives_pending_anchor_and_acknowledges_only_after_success():
    state = AnalysisState(user_id="user-1")
    observation = append_observation(state, _observation())
    provider = AgentSettingsProvider(
        state=state,
        config=SystemConfig(),
        global_client=None,
    )

    dispatch = await provider.budget_text_dispatch(
        "pi",
        [{"role": "system", "content": "PI"}, {"role": "user", "content": "Review"}],
    )
    assert dispatch.messages[1]["content"].startswith("Technical context observations")
    assert observation.pi_delivered_at is None

    await provider.acknowledge_text_dispatch("pi", dispatch)
    assert observation.pi_delivered_at is not None
    anchor, ids = pending_observation_anchor(state)
    assert anchor is None
    assert ids == ()


@pytest.mark.asyncio
async def test_malformed_legacy_window_falls_back_and_creates_pi_visible_observation():
    state = AnalysisState(user_id="user-1")
    state.agent_server_configs["literature"] = {"context_window": "broken"}
    provider = AgentSettingsProvider(state, SystemConfig(), global_client=None)

    dispatch = await provider.budget_text_dispatch(
        "literature",
        [{"role": "user", "content": "Search"}],
    )

    assert dispatch.report.ceiling == 200_000 - 16_384 - 512
    pending = [
        item for item in state.context_budget_observations
        if item.pi_delivered_at is None
    ]
    assert len(pending) == 1
    assert pending[0].fits is True
    assert "configuration_fallback" in pending[0].actions[0]


@pytest.mark.asyncio
async def test_failed_pi_assembly_does_not_acknowledge_pending_observation():
    state = AnalysisState(user_id="user-1")
    observation = append_observation(state, _observation())
    provider = AgentSettingsProvider(state, SystemConfig(), global_client=None)

    dispatch = await provider.budget_text_dispatch(
        "pi",
        [{"role": "user", "content": "Review"}],
    )
    assert dispatch.pi_observation_ids == (observation.id,)
    assert observation.pi_delivered_at is None


@pytest.mark.asyncio
async def test_observation_persistence_failure_does_not_block_dispatch(monkeypatch):
    state = AnalysisState(id="budget-save-failure", user_id="user-1")
    monkeypatch.setitem(sessions, state.id, state)
    monkeypatch.setattr(
        "backend.services.session_store._async_save_session_under_lifecycle",
        AsyncMock(side_effect=OSError("disk unavailable")),
    )
    provider = AgentSettingsProvider(
        state,
        SystemConfig(context_windows={"literature": 8_192}),
        global_client=None,
        session_id=state.id,
    )

    dispatch = await provider.budget_text_dispatch(
        "literature",
        [{"role": "system", "content": "s" * 100_000}],
    )

    assert dispatch.messages
    assert dispatch.report.fits is False
    assert state.context_budget_observations


@pytest.mark.asyncio
async def test_budget_plumbing_failure_returns_the_original_dispatch(monkeypatch):
    state = AnalysisState(user_id="user-1")
    provider = AgentSettingsProvider(state, SystemConfig(), global_client=None)
    messages = [{"role": "user", "content": "Continue analysis"}]
    monkeypatch.setattr(
        provider,
        "_budget_text_dispatch",
        AsyncMock(side_effect=RuntimeError("budget unavailable")),
    )

    dispatch = await provider.budget_text_dispatch(
        "pi",
        messages,
        max_tokens=2048,
    )

    assert dispatch.messages == messages
    assert dispatch.max_tokens == 2048
    assert dispatch.report is None
    assert dispatch.pi_observation_ids == ()


@pytest.mark.asyncio
async def test_observation_acknowledgement_failure_is_non_blocking(monkeypatch):
    state = AnalysisState(user_id="user-1")
    observation = append_observation(state, _observation())
    provider = AgentSettingsProvider(state, SystemConfig(), global_client=None)
    dispatch = await provider.budget_text_dispatch(
        "pi",
        [{"role": "user", "content": "Review"}],
    )
    monkeypatch.setattr(
        "backend.utils.context_budget.mark_observations_delivered",
        AsyncMock(side_effect=RuntimeError("ack unavailable")),
    )

    await provider.acknowledge_text_dispatch("pi", dispatch)

    assert observation.pi_delivered_at is None
