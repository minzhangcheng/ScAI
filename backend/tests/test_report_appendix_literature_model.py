"""Saved appendix agent_models[literature] reflects the actual model used."""
from backend.models.schemas import AnalysisState
from backend.services.report.generator import build_agent_models_appendix


def test_actual_model_written_into_appendix():
    s = AnalysisState(id="x", name="n", initial_instruction="i", user_id="u1")
    s.custom_models = {"literature": "google/gemini-flash", "pi": "anthropic/x"}
    s.literature_actual_model = "openai/o4-mini-deep-research"
    out = build_agent_models_appendix(s)
    assert out["literature"] == "openai/o4-mini-deep-research"
    assert out["pi"] == "anthropic/x"


def test_no_actual_model_keeps_custom():
    s = AnalysisState(id="x", name="n", initial_instruction="i", user_id="u1")
    s.custom_models = {"literature": "google/gemini-flash"}
    out = build_agent_models_appendix(s)
    assert out["literature"] == "google/gemini-flash"