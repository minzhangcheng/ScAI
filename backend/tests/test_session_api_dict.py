from backend.models.schemas import AnalysisState
from backend.utils.settings_provider import MASKED_SECRET_VALUE


def test_to_api_dict_includes_masked_literature_config():
    state = AnalysisState(id="api-dict-lit", name="API Dict Lit", user_id="user-1", initial_instruction="test")
    state.literature_config = {
        "mode": "deep_research",
        "deep_research_model": "o4-mini-deep-research",
        "deep_research_llm_base_url": "https://sg.uiuiapi.com/v1",
        "deep_research_api_key": "secret-key",
        "google_api_key": "google-secret",
        "pubmed_email": "user@example.com",
        "max_search_rounds": 5,
    }

    api = state.to_api_dict()

    assert api["literature_config"]["deep_research_api_key"] == MASKED_SECRET_VALUE
    assert api["literature_config"]["google_api_key"] == MASKED_SECRET_VALUE
    assert "secret-key" not in repr(api)
    assert "google-secret" not in repr(api)
