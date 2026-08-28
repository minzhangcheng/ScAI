import pytest
from unittest.mock import MagicMock

from backend.agents.bioinformatics_agent import BioinformaticsAgent, _ensure_list
from backend.agents.pi_agent import PIAgent


class TestEnsureList:
    def test_list_passthrough(self):
        assert _ensure_list(["a", "b"]) == ["a", "b"]

    def test_string_wrapped(self):
        assert _ensure_list("cluster 0 is T cells") == ["cluster 0 is T cells"]

    def test_none_returns_empty(self):
        assert _ensure_list(None) == []

    def test_empty_string_returns_list_with_empty_string(self):
        assert _ensure_list("") == [""]

    def test_empty_list_passthrough(self):
        assert _ensure_list([]) == []


@pytest.mark.asyncio
async def test_report_findings_accepts_string_findings():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._last_findings = []
    out = await agent._handle_report_findings({"findings": "cluster 0 is T cells"})
    assert agent._last_findings == ["cluster 0 is T cells"]
    assert out == ["cluster 0 is T cells"]


@pytest.mark.asyncio
async def test_report_findings_accepts_list_findings():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._last_findings = []
    out = await agent._handle_report_findings({"findings": ["cluster 0 is T cells", "cluster 1 is macrophages"]})
    assert agent._last_findings == ["cluster 0 is T cells", "cluster 1 is macrophages"]


@pytest.mark.asyncio
async def test_report_findings_normalizes_structured_findings_to_text():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._last_findings = []
    out = await agent._handle_report_findings({
        "findings": [
            {"text": "bulk result", "modality": "bulk_rna_seq"},
            {"text": "single-cell localization", "modality": "single_cell"},
        ]
    })
    assert out == ["bulk result", "single-cell localization"]
    assert agent._last_findings == out


@pytest.mark.asyncio
async def test_report_findings_accepts_one_structured_finding_object():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._last_findings = []
    out = await agent._handle_report_findings({
        "findings": {
            "text": "bulk result",
            "modality": "bulk_rna_seq",
            "evidence_execution_ids": ["bulk-exec"],
        },
        "evidence_execution_ids": ["bulk-exec"],
    })

    assert out == ["bulk result"]
    assert agent._last_finding_records == [{
        "text": "bulk result",
        "modality": "bulk_rna_seq",
        "evidence_execution_ids": ["bulk-exec"],
    }]


@pytest.mark.asyncio
async def test_report_findings_missing_findings():
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._last_findings = []
    out = await agent._handle_report_findings({})
    assert agent._last_findings == []


def test_pi_direction_selection_missing_key_no_keyerror():
    result = PIAgent._select_directions([{"direction_id": 1}, {"direction": "cell death"}])
    assert "cell death" in result
    assert "" not in result


def test_pi_direction_selection_all_missing_keys():
    result = PIAgent._select_directions([{"direction_id": 1}, {"direction_id": 2}])
    assert result == []
