"""literature_prompt must be a clean role-type system prompt (no unformatted placeholder, no task-level imperative)."""
from backend.config.agent_prompts import AgentPrompts


def test_no_unformatted_topic_placeholder():
    p = AgentPrompts().literature_prompt
    assert "{topic}" not in p


def test_is_role_definition_not_user_request():
    p = AgentPrompts().literature_prompt
    assert "Please help me conduct literature research" not in p
    assert "You are" in p


def test_enforces_english_and_nonempty():
    p = AgentPrompts().literature_prompt
    assert p.strip()
    assert "English" in p
