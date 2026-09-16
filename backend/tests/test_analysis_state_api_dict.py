"""Round-trip tests for AnalysisState.to_api_dict() serializing to JSON."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base64
import hashlib
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
    PublicationFigure,
    PublicationTable,
    FigureReviewStatus,
)
from backend.tests.figure_image_helpers import valid_image_bytes


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
            "max_rounds", "round_wait_seconds", "max_directions_per_gate",
            "waiting_input_started_at",
            "created_at", "updated_at", "initial_instruction",
            "treatment_groups", "data_files", "biochemistry_indicators",
            "round_results", "final_summary",
            "exploration_report_url", "annotation_report_url", "detailed_report_url", "integration_report_url",
            "analysis_phase", "annotation_state", "messages", "custom_prompts",
            "custom_models", "custom_temperatures", "deg_thresholds",
            "literature_config",
            "state_version",
            "last_sync_timestamp", "current_plan",
            "plan_dialogue_history", "agent_errors",
            "publication_figures", "publication_tables", "graphic_abstract",
            "figure_revision_tasks",
            "bulk_input_profiles", "method_decisions",
            "integrated_report",
            "species", "species_resolved", "species_genes_validated",
            "species_input_info",
            "runtime_restore_status",
            "runtime_restore_warning",
            "runtime_persistence",
            "evidence_gate_degraded",
            "evidence_gate_findings",
            "detected_sample_counts",
            "termination_reason",
            "snapshot_failure",
            "snapshot_transaction",
            "pending_required_snapshot",
            "detailed_report_fallback_pending",
            "restore_branch_id",
            # Promoted onto the api dict in ver 0.34.7 and consumed by the
            # frontend (utils/transformLoadResponse.ts, hooks/api/useWebSocket.ts).
            "wait_input_reason",
            "llm_failure_info",
            "agent_disabled_info",
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

    def test_restored_state_exposes_branch_and_only_its_round_results(self):
        state = AnalysisState(
            user_id="test",
            restore_branch_id="restore/source-r003",
            round_results=[
                RoundResult(
                    round_number=3,
                    lineage_id="main",
                    pi_summary="source summary",
                ),
                RoundResult(
                    round_number=3,
                    lineage_id="restore/source-r003",
                    branch_id="restore/source-r003",
                    pi_summary="inherited branch summary",
                ),
            ],
        )

        api = state.to_api_dict()

        assert api["restore_branch_id"] == "restore/source-r003"
        assert [item["pi_summary"] for item in api["round_results"]] == [
            "inherited branch summary"
        ]

    def test_pre_fix_restore_without_branch_clones_keeps_unique_source_rounds(self):
        state = AnalysisState(
            user_id="test",
            restore_branch_id="restore/source-r003",
            last_restored_snapshot_id="source/r003-integration-g277",
            round_results=[
                RoundResult(
                    round_number=3,
                    lineage_id="main",
                    pi_summary="legacy inherited summary",
                ),
            ],
        )

        api = state.to_api_dict()

        assert [item["pi_summary"] for item in api["round_results"]] == [
            "legacy inherited summary"
        ]

    def test_pre_fix_restore_does_not_guess_an_ambiguous_source_lineage(self):
        state = AnalysisState(
            user_id="test",
            restore_branch_id="restore/source-r003",
            last_restored_snapshot_id="source/r003-integration-g277",
            round_results=[
                RoundResult(round_number=2, lineage_id="main"),
                RoundResult(round_number=3, lineage_id="branch-x"),
            ],
        )

        assert state.to_api_dict()["round_results"] == []

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

    def test_publication_figures_carry_gallery_badge_fields(self):
        """Results gallery badges must survive a reload.

        A reload rebuilds the gallery from this payload alone. Report-blocking
        renders remain visible with an explicit badge.
        """
        state = AnalysisState(user_id="test")
        preview = base64.b64encode(valid_image_bytes("PNG")).decode()
        preview_sha256 = hashlib.sha256(base64.b64decode(preview)).hexdigest()
        state.publication_figures = [
            PublicationFigure(
                id="f-blocked", name="umap_clusters", description="d",
                figure_type="umap", png_path="/x/umap.png", pdf_path="/x/umap.pdf",
                review_status=FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
                png_base64=preview,
                preview_sha256=preview_sha256,
                round_number=3,
                provenance_schema_version=2,
                producer_run_kind="figure_followup",
                producer_run_scope_id="figure-run-1",
                source_execution_id="source-execution-1",
                source_execution_run_kind="ordinary",
                source_execution_validated=True,
                source_preparation_id="preparation-1",
                figure_followup_task_id="figure-task-1",
                provenance_validation_reason="Source receipt is missing.",
                layout_violations=[{"kind": "text_over_text", "severity": "block"}],
            ),
            PublicationFigure(
                id="f-diagnostic", name="qc_panel", description="d",
                figure_type="qc", png_path="/x/qc.png", pdf_path="/x/qc.pdf",
                diagnostic=True,
                png_base64=preview,
                preview_sha256=preview_sha256,
                # Advisory only: a warning is not a blocking defect.
                layout_violations=[{"kind": "text_over_text", "severity": "warn"}],
            ),
        ]
        api = state.to_api_dict()
        json.dumps(api)
        assert [figure["id"] for figure in api["publication_figures"]] == [
            "f-blocked", "f-diagnostic",
        ]
        blocked, diagnostic = api["publication_figures"]
        assert blocked["round"] == 3
        assert blocked["layout_blocking"] is True
        assert blocked["provenance_schema_version"] == 2
        assert blocked["producer_run_kind"] == "figure_followup"
        assert blocked["producer_run_scope_id"] == "figure-run-1"
        assert blocked["source_execution_run_kind"] == "ordinary"
        assert blocked["source_execution_validated"] is True
        assert blocked["source_preparation_id"] == "preparation-1"
        assert blocked["figure_followup_task_id"] == "figure-task-1"
        assert blocked["provenance_validation_reason"] == (
            "Source receipt is missing."
        )
        # An unattributed figure reports no round rather than "round 0".
        assert diagnostic["round"] is None
        assert diagnostic["layout_blocking"] is False
        assert diagnostic["diagnostic"] is True

    def test_publication_figure_without_render_or_preview_is_not_serialized(self):
        state = AnalysisState(user_id="test")
        state.publication_figures = [
            PublicationFigure(
                id="broken",
                name="broken",
                description="Missing physical render",
                figure_type="other",
                png_path="/definitely/missing/broken.png",
                pdf_path="",
                review_status=FigureReviewStatus.APPROVED,
            )
        ]
        state.realtime_messages = [
            WebSocketMessage(
                type="figure_approved",
                content={
                    "figure": {
                        "id": "broken",
                        "name": "broken",
                        "is_publication": True,
                    }
                },
            ),
        ]

        api = state.to_api_dict()

        assert api["publication_figures"] == []
        assert api["messages"] == []

    def test_publication_tables_are_filtered_to_the_active_lineage(self):
        state = AnalysisState(
            user_id="test",
            restore_branch_id="restore/a",
            publication_tables=[
                PublicationTable(
                    id="main-table", name="main", description="main",
                    csv_path="/tmp/main.csv", lineage_id="main",
                ),
                PublicationTable(
                    id="branch-table", name="branch", description="branch",
                    csv_path="/tmp/branch.csv", lineage_id="restore/a",
                ),
            ],
        )

        api = state.to_api_dict()

        assert [table["id"] for table in api["publication_tables"]] == [
            "branch-table"
        ]

    def test_runtime_warning_is_serialized(self):
        """to_api_dict() must include runtime_restore_warning when set, default to None."""
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        # Default should be absent (None)
        assert api.get("runtime_restore_warning") is None

        state.runtime_restore_warning = "Live runtime intentionally kept"
        api = state.to_api_dict()
        assert api["runtime_restore_warning"] == "Live runtime intentionally kept"

        state.runtime_restore_warning = None
        api = state.to_api_dict()
        assert api.get("runtime_restore_warning") is None

    def test_invalidated_publication_figure_is_not_serialized(self):
        state = AnalysisState(user_id="test")
        state.publication_figures = [
            PublicationFigure(
                id="stale",
                name="stale",
                description="superseded",
                figure_type="other",
                png_path="/x/stale.jpg",
                pdf_path="/x/stale.pdf",
                is_current=False,
                invalidated_at=datetime.now(),
            )
        ]
        state.realtime_messages = [
            WebSocketMessage(
                type="figure_approved",
                content={"figure": {"id": "stale", "name": "stale"}},
            ),
            WebSocketMessage(
                type="integrated_report",
                content={
                    "report": "old highlight reel",
                    "figures": [
                        {"id": "stale", "name": "stale", "is_publication": True},
                        {
                            "id": "workspace",
                            "name": "workspace_context",
                            "is_publication": False,
                        },
                    ],
                },
            ),
        ]

        api = state.to_api_dict()
        assert api["publication_figures"] == []
        assert [message["type"] for message in api["messages"]] == [
            "integrated_report"
        ]
        assert [
            figure["name"] for figure in api["messages"][0]["content"]["figures"]
        ] == ["workspace_context"]
