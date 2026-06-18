"""technique 与 painter 必须有非空系统提示词。"""
from backend.config.agent_prompts import AgentPrompts


def test_technique_prompt_present():
    p = AgentPrompts()
    assert getattr(p, "technique_prompt", "").strip()


def test_painter_prompt_present():
    p = AgentPrompts()
    assert p.painter_prompt.strip()
