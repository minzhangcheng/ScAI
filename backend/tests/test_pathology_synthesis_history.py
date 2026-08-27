"""Regression tests for history-free pathology synthesis calls."""

import inspect

from backend.agents.pathology_agent import PathologyAgent
from backend.agents.pathology_parts import image_analysis


def test_per_image_synthesis_does_not_use_message_history():
    source = inspect.getsource(image_analysis.analyze_image)

    assert "agent.chat_without_history(combined_prompt" in source
    assert "self.chat(combined_prompt" not in source


def test_cross_image_synthesis_does_not_use_message_history():
    source = inspect.getsource(PathologyAgent._synthesize_findings)

    assert "self.chat_without_history(prompt" in source
    assert "self.chat(prompt" not in source
