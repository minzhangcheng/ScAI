from backend.config._config import ModelConfig


def test_agent_model_defaults_match_agent_page_defaults():
    assert ModelConfig().model_dump() == {
        "pi_model": "gpt-5.6-sol",
        "pathology_model": "gemini-3.1-pro-preview",
        "bioinformatics_model": "gpt-5.6-sol",
        "biochemistry_model": "gemini-3.1-pro-preview",
        "literature_model": "gpt-5.4-mini",
        "painter_model": "gpt-image-2",
        "supervisor_model": "gpt-5.4-mini",
        "technique_model": "gpt-5.4-mini",
    }
