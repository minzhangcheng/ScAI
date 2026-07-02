"""Tests for temperature type correctness in config_io."""
from backend.models.config_schema import ConfigV1
from backend.services.config_io import build_agent_server_configs


def _minimal_config_dict() -> dict:
    return {
        "schema_version": "1.0",
        "analysis": {"name": "Test", "initial_instruction": "x", "round_wait_seconds": 60},
        "api_config": {
            "default_llm": {"llm_base_url": "https://test.api", "api_key": "sk-test", "model": "gpt-4o"},
            "literature": {"mode": "disabled"},
        },
        "data": {"groups": [], "files": [], "indicators": []},
    }


def test_temperature_is_float():
    config = ConfigV1.model_validate(_minimal_config_dict())
    cfg = build_agent_server_configs(config)
    for name, c in cfg.items():
        assert isinstance(c["temperature"], float), f"{name}.temperature is {type(c['temperature']).__name__}, expected float"


def test_temperature_with_override_is_float():
    d = _minimal_config_dict()
    d["agents"] = {"pi": {"temperature": 0.9}}
    config = ConfigV1.model_validate(d)
    cfg = build_agent_server_configs(config)
    assert cfg["pi"]["temperature"] == 0.9
    assert isinstance(cfg["pi"]["temperature"], float)


def test_temperature_default_is_float():
    d = _minimal_config_dict()
    d["agents"] = {"pi": {}}
    config = ConfigV1.model_validate(d)
    cfg = build_agent_server_configs(config)
    assert cfg["pi"]["temperature"] == 0.7
    assert isinstance(cfg["pi"]["temperature"], float)
