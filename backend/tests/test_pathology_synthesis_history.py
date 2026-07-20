"""Regression tests for history-free pathology synthesis calls."""

import inspect

from backend.agents.pathology_agent import PathologyAgent


def test_per_image_synthesis_does_not_use_message_history():
    source = inspect.getsource(PathologyAgent._analyze_image)

    assert "self.chat_without_history(combined_prompt" in source
    assert "self.chat(combined_prompt" not in source


def test_cross_image_synthesis_does_not_use_message_history():
    source = inspect.getsource(PathologyAgent._synthesize_findings)

    assert "self.chat_without_history(prompt" in source
    assert "self.chat(prompt" not in source
