"""Tests for AgentRoundCache model and AnalysisState.current_round_cache field."""
from backend.models.schemas import AgentRoundCache, AgentType, AnalysisState


def test_agent_round_cache_defaults():
    cache = AgentRoundCache(round_number=1)
    assert cache.round_number == 1
    assert cache.results == {}


def test_agent_round_cache_stores_dict_results():
    cache = AgentRoundCache(round_number=2)
    cache.results[AgentType.BIOCHEMISTRY.value] = {"findings": ["a", "b"], "status": "completed"}
    assert cache.results[AgentType.BIOCHEMISTRY.value]["status"] == "completed"


def test_analysis_state_has_current_round_cache_default_none():
    state = AnalysisState(user_id="test_user", initial_instruction="test")
    assert state.current_round_cache is None


def test_analysis_state_round_cache_round_trip():
    state = AnalysisState(user_id="test_user", initial_instruction="test")
    state.current_round_cache = AgentRoundCache(
        round_number=1,
        results={AgentType.PATHOLOGY.value: {"findings": ["x"], "status": "completed"}},
    )
    dumped = state.model_dump(mode="json")
    restored = AnalysisState.model_validate(dumped)
    assert restored.current_round_cache is not None
    assert restored.current_round_cache.round_number == 1
    assert restored.current_round_cache.results[AgentType.PATHOLOGY.value]["status"] == "completed"


from backend.workflow.helpers import cache_hit, cache_invalidate, cache_store


def _make_state(round_number=1):
    state = AnalysisState(user_id="test_user", initial_instruction="t")
    state.current_round = round_number
    return state


def test_cache_hit_returns_none_when_no_cache():
    state = _make_state(round_number=1)
    assert cache_hit(state, AgentType.BIOCHEMISTRY) is None


def test_cache_hit_returns_none_for_missing_agent():
    state = _make_state(round_number=1)
    state.current_round_cache = AgentRoundCache(round_number=1)
    state.current_round_cache.results[AgentType.PATHOLOGY.value] = {"findings": []}
    assert cache_hit(state, AgentType.BIOCHEMISTRY) is None


def test_cache_hit_returns_none_on_round_number_mismatch():
    state = _make_state(round_number=2)
    state.current_round_cache = AgentRoundCache(round_number=1)
    state.current_round_cache.results[AgentType.BIOCHEMISTRY.value] = {"findings": ["a"]}
    assert cache_hit(state, AgentType.BIOCHEMISTRY) is None


def test_cache_hit_returns_dict_on_match():
    state = _make_state(round_number=1)
    state.current_round_cache = AgentRoundCache(round_number=1)
    state.current_round_cache.results[AgentType.BIOCHEMISTRY.value] = {"findings": ["a"], "status": "completed"}
    hit = cache_hit(state, AgentType.BIOCHEMISTRY)
    assert hit == {"findings": ["a"], "status": "completed"}


def test_cache_store_creates_cache_when_none():
    state = _make_state(round_number=3)
    cache_store(state, AgentType.PATHOLOGY, {"findings": ["x"], "status": "completed"})
    assert state.current_round_cache is not None
    assert state.current_round_cache.round_number == 3
    assert state.current_round_cache.results[AgentType.PATHOLOGY.value]["status"] == "completed"


def test_cache_store_replaces_stale_cache_on_round_mismatch():
    state = _make_state(round_number=2)
    state.current_round_cache = AgentRoundCache(round_number=1)
    state.current_round_cache.results[AgentType.BIOCHEMISTRY.value] = {"old": True}
    cache_store(state, AgentType.BIOCHEMISTRY, {"new": True})
    assert state.current_round_cache.round_number == 2
    assert state.current_round_cache.results == {AgentType.BIOCHEMISTRY.value: {"new": True}}


def test_cache_store_appends_to_existing_cache():
    state = _make_state(round_number=1)
    cache_store(state, AgentType.BIOCHEMISTRY, {"findings": ["a"]})
    cache_store(state, AgentType.PATHOLOGY, {"findings": ["b"]})
    assert set(state.current_round_cache.results.keys()) == {AgentType.BIOCHEMISTRY.value, AgentType.PATHOLOGY.value}


def test_cache_invalidate_no_cache_is_noop():
    state = _make_state()
    cache_invalidate(state, AgentType.BIOCHEMISTRY)
    assert state.current_round_cache is None


def test_cache_invalidate_missing_agent_is_noop():
    state = _make_state()
    cache_store(state, AgentType.PATHOLOGY, {"findings": ["p"]})
    cache_invalidate(state, AgentType.BIOCHEMISTRY)
    assert AgentType.PATHOLOGY.value in state.current_round_cache.results


def test_cache_invalidate_removes_only_target_agent():
    state = _make_state()
    cache_store(state, AgentType.BIOCHEMISTRY, {"findings": ["b"]})
    cache_store(state, AgentType.PATHOLOGY, {"findings": ["p"]})
    cache_invalidate(state, AgentType.BIOCHEMISTRY)
    assert AgentType.BIOCHEMISTRY.value not in state.current_round_cache.results
    assert state.current_round_cache.results[AgentType.PATHOLOGY.value]["findings"] == ["p"]


def test_cache_invalidate_does_not_reset_cache_object():
    """Invalidating one entry must leave the cache (and round_number) intact,
    so other agents can still hit/store."""
    state = _make_state(round_number=5)
    cache_store(state, AgentType.BIOCHEMISTRY, {"findings": ["b"]})
    cache_invalidate(state, AgentType.BIOCHEMISTRY)
    assert state.current_round_cache is not None
    assert state.current_round_cache.round_number == 5
