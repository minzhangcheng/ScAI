"""Tests for SynthesisMixin._exploration_round_succeeded gating predicate.

The predicate decides whether _pi_synthesis should generate the
exploration Report A and transition analysis_phase EXPLORATION ->
INNOVATION. When bioinformatics reports failed/error status, the
exploration round is NOT a success and that block must be skipped so
the user can retry stage 1 in the next round.
"""
import pytest

from backend.workflow.synthesis import SynthesisMixin


class _Probe(SynthesisMixin):
    """Minimal subclass to expose the predicate without spinning up the
    full workflow (which composes 11 mixins and needs a SystemConfig)."""
    def __init__(self):
        pass


class TestExplorationRoundSucceeded:
    def setup_method(self):
        self.probe = _Probe()

    def test_no_bio_result_blocks_advance(self):
        # P1-12: whitelist — missing status is NOT a confirmed completion.
        # Treat absence of bioinformatics_result as "not succeeded" so the
        # workflow doesn't advance phases on empty exploration.
        assert self.probe._exploration_round_succeeded({}) is False

    def test_completed_status_is_success(self):
        state = {"bioinformatics_result": {"status": "completed", "findings": ["x"]}}
        assert self.probe._exploration_round_succeeded(state) is True

    def test_completed_with_fallback_blocks_advance(self):
        # P1-12: whitelist — only literal "completed" advances. Fallback
        # extraction (max_steps_reached salvage) signals an incomplete loop
        # and must not promote to INNOVATION.
        state = {"bioinformatics_result": {"status": "completed_with_fallback", "findings": ["x"]}}
        assert self.probe._exploration_round_succeeded(state) is False

    def test_skipped_status_blocks_advance(self):
        # P1-12: whitelist — skipped is not a completion signal.
        state = {"bioinformatics_result": {"status": "skipped", "findings": []}}
        assert self.probe._exploration_round_succeeded(state) is False

    def test_failed_status_blocks_advance(self):
        state = {"bioinformatics_result": {"status": "failed", "findings": [], "error": "x"}}
        assert self.probe._exploration_round_succeeded(state) is False

    def test_error_status_blocks_advance(self):
        state = {"bioinformatics_result": {"status": "error", "findings": [], "error": "x"}}
        assert self.probe._exploration_round_succeeded(state) is False

    def test_max_steps_reached_no_findings_blocks_advance(self):
        state = {"bioinformatics_result": {"status": "max_steps_reached", "findings": []}}
        assert self.probe._exploration_round_succeeded(state) is False

    def test_max_steps_reached_with_findings_blocks_advance(self):
        # P1-12: whitelist — max_steps_reached means the loop terminated
        # without an explicit completion signal, even if some findings exist.
        state = {"bioinformatics_result": {"status": "max_steps_reached", "findings": ["x"]}}
        assert self.probe._exploration_round_succeeded(state) is False

    def test_failed_node_field_also_blocks(self):
        # @with_retry sets failed_node on the skip path. Honor that
        # signal even if a stale bioinformatics_result is present.
        state = {
            "failed_node": "bioinformatics_analysis",
            "bioinformatics_result": {"status": "completed", "findings": ["x"]},
        }
        assert self.probe._exploration_round_succeeded(state) is False
