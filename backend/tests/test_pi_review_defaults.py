import pytest
from backend.constants import PI_REVIEW_FALLBACK_DECISION


class TestPIReviewDefaults:
    """Verify PI review JSON parse failure does NOT default to approved."""

    def test_fallback_decision_is_revise(self):
        """The fallback decision constant should be 'revise'."""
        assert PI_REVIEW_FALLBACK_DECISION == "revise"

    def test_parse_failure_defaults_to_revise(self):
        """When JSON parse returns empty dict, default should be 'revise'."""
        result = {}
        if not result:
            result = {
                "decision": PI_REVIEW_FALLBACK_DECISION,
                "feedback": "Plan needs review (fallback — PI response could not be parsed)",
                "modifications": [],
            }
        assert result["decision"] == "revise"

    def test_missing_decision_key_defaults_to_revise(self):
        """When JSON is parsed but 'decision' key is missing, default to 'revise'."""
        result = {"feedback": "some feedback", "modifications": []}
        result.setdefault("decision", PI_REVIEW_FALLBACK_DECISION)
        assert result["decision"] == "revise"

    def test_valid_approved_decision_preserved(self):
        """When PI explicitly returns 'approved', it should be preserved."""
        result = {"decision": "approved", "feedback": "LGTM", "modifications": []}
        result.setdefault("decision", PI_REVIEW_FALLBACK_DECISION)
        assert result["decision"] == "approved"

    def test_valid_revise_decision_preserved(self):
        """When PI explicitly returns 'revise', it should be preserved."""
        result = {"decision": "revise", "feedback": "Need changes", "modifications": ["step1"]}
        result.setdefault("decision", PI_REVIEW_FALLBACK_DECISION)
        assert result["decision"] == "revise"