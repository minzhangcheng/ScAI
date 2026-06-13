"""LiteratureAgent.resolve_model_label maps search_mode -> display label."""
import pytest


class _Agent:
    """Stand-in exposing resolve_model_label with controlled model attrs."""
    from backend.agents.literature_agent import LiteratureAgent
    resolve_model_label = LiteratureAgent.resolve_model_label

    def __init__(self, dr_model, lit_model):
        self.deep_research_model = dr_model
        self._lit_model = lit_model

    @property
    def model(self):
        return self._lit_model


def test_gemini_label():
    a = _Agent("openai/o4-deep-research", "google/gemini-flash")
    assert a.resolve_model_label("gemini_deep_research") == "Google Deep Research"


def test_deep_research_uses_dr_model():
    a = _Agent("openai/o4-deep-research", "google/gemini-flash")
    assert a.resolve_model_label("deep_research") == "openai/o4-deep-research"


def test_deep_research_empty_falls_back_to_lit_model():
    a = _Agent("", "google/gemini-flash")
    assert a.resolve_model_label("deep_research") == "google/gemini-flash"


def test_pubmed_and_fallback_use_lit_model():
    a = _Agent("openai/o4-deep-research", "google/gemini-flash")
    assert a.resolve_model_label("pubmed_llm") == "google/gemini-flash"
    assert a.resolve_model_label("pubmed_llm (fallback from deep_research)") == "google/gemini-flash"
    assert a.resolve_model_label("quick_search") == "google/gemini-flash"