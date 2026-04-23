"""Verify BaseAgent._safe_parse_json emits a labeled warning on failure."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agents.base import BaseAgent


class _Stub(BaseAgent):
    """Minimal subclass that implements the abstract method."""
    def __init__(self):
        # Pass minimal args; we only need the inherited _safe_parse_json method.
        pass

    async def process(self, *args, **kwargs):
        raise NotImplementedError


def test_safe_parse_json_returns_data_on_success():
    agent = _Stub()
    out = agent._safe_parse_json('{"a": 1}')
    assert out == {"a": 1}


def test_safe_parse_json_logs_with_agent_label_on_failure(caplog):
    agent = _Stub()
    with caplog.at_level(logging.WARNING, logger="backend.agents.base"):
        result = agent._safe_parse_json("not json", agent_label="PathologyAgent")
    assert "raw_response" in result
    assert any("PathologyAgent" in rec.message for rec in caplog.records), (
        f"Expected agent label in warning, got: {[r.message for r in caplog.records]}"
    )


def test_safe_parse_json_no_label_no_log(caplog):
    agent = _Stub()
    with caplog.at_level(logging.WARNING, logger="backend.agents.base"):
        agent._safe_parse_json("not json")
    # Without an agent_label, base must stay quiet (preserve existing default).
    assert not any("JSON parse failed" in rec.message for rec in caplog.records)