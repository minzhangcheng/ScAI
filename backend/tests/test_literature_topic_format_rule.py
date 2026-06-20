"""PI 必须把 literature_topics 的纯文本格式约束作为可复用常量注入 innovation prompt。"""
import inspect
from backend.agents.pi_agent import PIAgent


def test_literature_topic_format_rule_constant_exists():
    rule = getattr(PIAgent, "_LITERATURE_TOPIC_FORMAT_RULE", "")
    assert rule.strip()
    assert "literature_topics" in rule
    assert "plain" in rule.lower()
    assert "dict" in rule.lower()


def test_innovation_instruction_references_the_rule():
    src = inspect.getsource(PIAgent.innovation_instruction)
    assert "_LITERATURE_TOPIC_FORMAT_RULE" in src
