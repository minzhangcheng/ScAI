"""Report agent-config table shows literature_actual_model when set."""
from backend.models.schemas import AnalysisState
from backend.services.report.sections import resolve_literature_model


def _resolve_cell(state):
    """Mirror the Literature-row lookup sections.py performs."""
    return resolve_literature_model(
        state,
        default=state.custom_models.get("literature", "-"),
    )


def test_actual_model_overrides_default():
    s = AnalysisState(id="x", name="n", initial_instruction="i", user_id="u1")
    s.literature_actual_model = "Google Deep Research"
    assert _resolve_cell(s) == "Google Deep Research"


def test_falls_back_to_custom_model():
    s = AnalysisState(id="x", name="n", initial_instruction="i", user_id="u1")
    s.custom_models = {"literature": "google/gemini-flash"}
    assert _resolve_cell(s) == "google/gemini-flash"