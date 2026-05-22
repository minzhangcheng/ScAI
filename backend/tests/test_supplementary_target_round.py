import pytest
from pydantic import ValidationError
from backend.models.api_requests import SupplementaryAnalysisRequest


def test_target_round_accepted():
    """P1-6: target='round' must be a valid value for SupplementaryAnalysisRequest."""
    req = SupplementaryAnalysisRequest(request_text="extra analysis", target="round")
    assert req.target == "round"


def test_target_annotation_still_accepted():
    req = SupplementaryAnalysisRequest(request_text="x", target="annotation")
    assert req.target == "annotation"


def test_target_direction_still_accepted():
    req = SupplementaryAnalysisRequest(request_text="x", target="direction")
    assert req.target == "direction"


def test_target_invalid_rejected():
    with pytest.raises(ValidationError):
        SupplementaryAnalysisRequest(request_text="x", target="invalid_value")
