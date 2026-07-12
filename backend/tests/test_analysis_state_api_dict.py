"""Round-trip tests for AnalysisState.to_api_dict() serializing to JSON."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from datetime import datetime

from backend.models.schemas import (
    AnalysisState,
    AnalysisPlan,
    PlanStep,
    TaskStatus,
    TreatmentGroup,
    DataFile,
    BiochemistryIndicator,
    AgentError,
    AgentType,
    AgentMessage,
    MessageRole,
    RoundResult,
    WebSocketMessage,
)


class TestAnalysisStateApiDictRoundTrip:
    """Verify to_api_dict() output is JSON-safe and field-complete."""

    def test_empty_state_api_dict_is_json_serializable(self):
        """Minimal state with no nested objects must round-trip through json."""
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        json.dumps(api)
        assert api["user_id"] == "test"
        assert api["status"] == "pending"

    def test_state_with_plan_api_dict_is_json_serializable(self):
        """State with a current_plan (contains datetime) must round-trip cleanly."""
        state = AnalysisState(user_id="test")
        plan = AnalysisPlan(
            phase="exploration",
            stage="full",
            overview="test plan",
            steps=[PlanStep(id=1, title="QC", method="scanpy")],
            status="approved",
            approved_at=datetime(2026, 5, 15, 3, 5, 2),
        )
        state.current_plan = plan
        api = state.to_api_dict()
        json.dumps(api)
        assert api["current_plan"]["status"] == "approved"
        assert "2026-05-15" in api["current_plan"]["approved_at"]

    def test_state_api_dict_preserves_all_top_level_keys(self):
        """to_api_dict() must return a dict with the same top-level keys
        regardless of whether nested lists are empty or populated."""
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        # These are all the keys the frontend load_session endpoint expects
        expected_keys = {
            "session_id", "user_id", "name", "status", "current_round",
            "max_rounds", "round_wait_seconds", "waiting_input_started_at",
            "created_at", "updated_at", "initial_instruction",
            "treatment_groups", "data_files", "biochemistry_indicators",
            "round_results", "final_summary",
            "exploration_report_url", "annotation_report_url", "detailed_report_url", "integration_report_url",
            "analysis_phase", "messages", "custom_prompts",
            "custom_models", "custom_temperatures", "deg_thresholds",
            "literature_config",
            "state_version",
            "last_sync_timestamp", "snapshot_count", "current_plan",
            "plan_dialogue_history", "agent_errors",
            "publication_figures", "publication_tables", "graphic_abstract",
            "integrated_report",
            "species", "species_resolved", "species_genes_validated",
            "species_input_info",
            "runtime_restore_status",
            "snapshot_failure",
            "snapshot_transaction",
            "pending_required_snapshot",
        }
        assert set(api.keys()) == expected_keys

    def test_state_with_nested_lists_api_dict_is_json_serializable(self):
        """State with populated treatment_groups, data_files, etc. must round-trip."""
        state = AnalysisState(user_id="test")
        state.treatment_groups = [
            TreatmentGroup(id="tg-1", name="Control", description="ctrl", sample_count=3, order=0),
        ]
        state.data_files = [
            DataFile(
                id="df-1", filename="test.h5ad", file_path="/tmp/test.h5ad",
                section="sequencing", data_type="scRNA-seq",
                group_id="tg-1", sample_number=1,
            ),
        ]
        state.biochemistry_indicators = [
            BiochemistryIndicator(
                id="bi-1", indicator="ALT", value=45.0, unit="U/L",
                reference_range="7-56", interpretation="normal",
                group_id="tg-1", sample_number=1,
            ),
        ]
        state.agent_errors = [
            AgentError(
                agent=AgentType.BIOINFORMATICS, error_type="execution_error",
                message="test error", details="detail", round_number=1,
            ),
        ]
        state.realtime_messages = [
            WebSocketMessage(
                type="message",
                agent=AgentType.PI, role=MessageRole.ASSISTANT,
                content="hello", metadata={"k": "v"},
            ),
        ]
        state.round_results = [
            RoundResult(
                round_number=1, pi_summary="summary",
                pathology_findings=["f1"], bioinformatics_findings=["f2"],
                biochemistry_findings=["f3"], literature_findings=["f4"],
                next_steps=["next"], conflicts=[{"agent": "bioinfo", "reason": "none"}],
            ),
        ]
        api = state.to_api_dict()
        json.dumps(api)
        # Spot-check a few nested fields
        assert len(api["treatment_groups"]) == 1
        assert api["treatment_groups"][0]["sampleCount"] == 3
        assert len(api["data_files"]) == 1
        assert api["data_files"][0]["dataType"] == "scRNA-seq"
        assert len(api["biochemistry_indicators"]) == 1
        assert api["biochemistry_indicators"][0]["referenceRange"] == "7-56"
        assert len(api["agent_errors"]) == 1
        assert api["agent_errors"][0]["agent"] == "bioinformatics"
        assert len(api["messages"]) == 1
        assert api["messages"][0]["agent"] == "pi"
        assert len(api["round_results"]) == 1
        assert api["round_results"][0]["round"] == 1

    def test_strip_message_base64_callable(self):
        """strip_message_base64 callable must be applied to message content."""
        state = AnalysisState(user_id="test")
        state.realtime_messages = [
            WebSocketMessage(
                type="message",
                agent=AgentType.PI, role=MessageRole.ASSISTANT,
                content="keep this", metadata={},
            ),
        ]

        def stripper(msg_type, content):
            return content.replace("keep", "stripped")

        api = state.to_api_dict(strip_message_base64=stripper)
        assert api["messages"][0]["content"] == "stripped this"
