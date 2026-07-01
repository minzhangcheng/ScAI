"""Test that to_api_dict includes all AnalysisState fields unless explicitly excluded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pytest
from backend.models.schemas import AnalysisState


_EXCLUDED_FIELDS = {
    "original_max_rounds",
    "pre_pause_status",
    "annotation_state",
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
    "snapshots",
    "last_restored_snapshot_id",
    "last_snapshot_id",
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
    "wait_input_reason",
    "llm_failure_info",
    "agent_disabled_info",
    "agent_server_configs",
    "agent_vision_settings",
    "chat_messages",
    "score_definitions",
    "data_driven_mechanism_programs",
    "current_round_cache",
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
    "restore_branch_id",
    "restore_branch_label",
}

_COMPUTED_FIELDS = {
    "exploration_report_url",
    "annotation_report_url",
    "snapshot_count",
    "messages",
}


class TestToApiDictCoverage:
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
