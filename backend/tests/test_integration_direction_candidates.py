import inspect
from backend.agents.pi_agent import PIAgent


def test_integration_candidate_method_exists():
    assert hasattr(PIAgent, "generate_integration_direction_candidates")


def test_integration_scorer_method_exists():
    assert hasattr(PIAgent, "score_integration_directions_with_evidence")


def test_integration_prompt_is_cumulative_not_round1():
    src = inspect.getsource(PIAgent.generate_integration_direction_candidates)
    assert "Round 1 exploration results" not in src
    assert "cumulative" in src.lower() or "all innovation" in src.lower() or "integration" in src.lower()
    assert "user_chosen_directions" in src or "already" in src.lower()


def test_integration_scorer_returns_list_annotation():
    sig = inspect.signature(PIAgent.score_integration_directions_with_evidence)
    assert "List" in str(sig.return_annotation) or "list" in str(sig.return_annotation)
