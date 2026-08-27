"""Test that to_api_dict includes all AnalysisState fields unless explicitly excluded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from backend.models.schemas import AnalysisState


_EXCLUDED_FIELDS = {
    # Host-internal bound on the execution-driven replan cycle. The client
    # learns the outcome from the round's status and its gate observations;
    # the raw attempt counter is bookkeeping.
    "execution_revision_attempts",
    "original_max_rounds",
    "pre_pause_status",
    # One-shot marker set by POST /run when the run exists to rebuild a kernel
    # that is gone. Read only by the bioinformatics data context and cleared by
    # the bioinformatics node; the client already learns the situation from
    # runtime_restore_status / runtime_restore_warning.
    "runtime_recompute_pending",
    # Backend-internal artifact-invalidation bookkeeping (expression_contract,
    # graphic_abstract, report generator). No frontend consumer — verified by
    # grepping frontend/src — so it stays off the API surface.
    "invalidated_artifact_stems",
    # annotation_state is deliberately exposed: the client rebuilds the
    # annotation gate from it after a reload.
    "exploration_report",
    "user_chosen_direction",
    "current_round_direction",
    "user_chosen_directions",
    "directions_queue",
    "multi_direction_correlation",
    "detailed_report_segments",
    "detailed_report_html_path",
    "exploration_report_html_path",
    "annotation_report_html_path",
    "integration_report_html_path",
    "snapshots",
    "last_restored_snapshot_id",
    "restore_provenance_snapshot_id",
    "last_snapshot_id",
    # One-shot marker armed and consumed server-side by the snapshot failure
    # gate (services/snapshot_v2/failure_gate.py); the client never reads it.
    "skip_required_snapshot_phase",
    # Likewise one-shot: armed by the Proceed action on a memory-preflight gate
    # and consumed by the next runtime capture. Server-side only.
    "force_snapshot_memory_override",
    "code_history",
    "analysis_log",
    "literature_reports",
    "literature_actual_model",
    "sandbox_state",
    "config_snapshot",
    "pending_human_input",
    "pending_uploaded_files",
    "human_input_requested",
    "human_messages",
    "agent_server_configs",
    "agent_vision_settings",
    "chat_messages",
    "score_definitions",
    # Methodology records: these reach the user through the report, not the
    # session API, exactly like score_definitions above.
    "scoring_plans",
    "data_driven_mechanism_programs",
    # Plan-audit / supplementary-pass records. Live progress is pushed as a
    # `post_execution_review` websocket message and the durable form reaches the
    # user through the report; no frontend consumer reads it back off the session
    # API — verified by grepping frontend/src — and it is per-execution, so
    # keeping it off the load response keeps that payload bounded.
    "post_execution_reviews",
    "current_round_cache",
    # Server-side bounded caches used to avoid repeated provider/search work;
    # their rendered guidance/results reach clients through normal outputs.
    "literature_search_cache",
    "environment_api_notes",
    "messages",
    "revision_history",
    "all_candidates",
    "selected_index",
    "plan_review_wait_seconds",
    "auto_approve_plans",
    "allow_partial_report",
    "partial_report_requested_at",
    "restore_resume_phase",
    "restore_resume_kind",
    "restore_resume_node",
    "restore_resume_gate",
    "restore_resume_workflow_state",
    "restore_start_intent",
    "restore_branch_id",
    "restore_branch_label",
    "integration_cycle",
    "max_integration_cycles",
    "pending_integration",
    "active_direction_batch",
    "last_integrated_round",
    "integration_gate_candidates",
    "direction_selection_records",
    "annotation_record",
    "annotation_supplementary_requests",
    "integration_gate_literature_evidence",
    "integration_reports",
    "final_report_requested",
    "report_generation_id",
    "recovery_blocked",
    # Server-side report artifact inventory; clients consume the dedicated
    # report and gallery endpoints instead of the raw registry records.
    "artifact_registry_version",
    "legacy_artifacts",
    # Persistent server-side incarnation used to reject stale writes/deletes;
    # clients use state_version and never submit a lifecycle generation.
    "lifecycle_generation",
}

_COMPUTED_FIELDS = {
    "exploration_report_url",
    "annotation_report_url",
    "integration_report_url",
    "snapshot_count",
    "messages",
}


class TestToApiDictCoverage:
    def test_pending_hitl_gate_is_exposed_for_client_reload(self):
        state = AnalysisState(
            user_id="test",
            wait_input_reason="llm_failure",
            llm_failure_info={
                "agent": "bioinformatics",
                "model": "test-model",
                "llm_server": "https://llm.example/v1",
                "error": "upstream exhausted",
                "node_name": "bioinformatics_analysis",
                "generation": 4,
            },
        )

        api = state.to_api_dict()

        assert api["wait_input_reason"] == "llm_failure"
        assert api["llm_failure_info"]["node_name"] == "bioinformatics_analysis"
        assert api["llm_failure_info"]["generation"] == 4
        assert api["agent_disabled_info"] is None

    def test_all_non_excluded_fields_are_present(self):
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        json.dumps(api)

        model_fields = set(AnalysisState.model_fields.keys())
        included_model_fields = model_fields - _EXCLUDED_FIELDS

        for field_name in included_model_fields:
            assert field_name in state._API_KEY_MAP or field_name in _COMPUTED_FIELDS, (
                f"Field '{field_name}' is in AnalysisState but not in _API_KEY_MAP "
                f"and not in _EXCLUDED_FIELDS or _COMPUTED_FIELDS. "
                f"If it should be excluded from the API, add it to _EXCLUDED_FIELDS in this test."
            )

    def test_all_api_keys_are_json_serializable(self):
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        json.dumps(api)

    def test_excluded_fields_not_in_output(self):
        state = AnalysisState(user_id="test")
        api = state.to_api_dict()
        json.dumps(api)
        api_keys = set(api.keys())
        for excluded in _EXCLUDED_FIELDS:
            camel = "".join(
                word.capitalize() if i > 0 else word
                for i, word in enumerate(excluded.split("_"))
            )
            if excluded in state._API_KEY_MAP or excluded in state._API_KEY_MAP.values():
                continue
            assert excluded not in api_keys, f"Excluded field '{excluded}' appeared in API output"
            assert camel not in api_keys, f"Excluded field camelCase '{camel}' appeared in API output"
