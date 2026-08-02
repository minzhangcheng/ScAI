"""Unit tests for pure helper functions extracted from workflow."""
from backend.workflow.helpers import (
    _retry_backoff_delay,
    _diff_plan_steps,
    _get_task_text,
    _normalize_tasks,
    STAGE1_KEYWORDS,
    _DIFF_LABELS,
)


class TestRetryBackoffDelay:
    def test_first_attempt_returns_base_delay(self):
        d = _retry_backoff_delay(0)
        # 2^0 + jitter(0,1) = 1 + [0,1] = [1, 2]
        assert 1 <= d <= 2

    def test_delay_grows_with_attempt(self):
        d0 = _retry_backoff_delay(0)
        d3 = _retry_backoff_delay(3)
        assert d3 >= d0

    def test_delay_capped_at_30(self):
        d = _retry_backoff_delay(100)
        assert d <= 30


class TestDiffPlanSteps:
    def test_no_change_returns_empty_diff(self):
        steps = [{"id": 1, "title": "QC", "method": "scanpy"}]
        diff = _diff_plan_steps(steps, steps)
        assert diff[1] == "unchanged"

    def test_added_step_marked_added(self):
        prev = [{"id": 1, "title": "QC", "method": "scanpy"}]
        curr = prev + [{"id": 2, "title": "Cluster", "method": "leiden"}]
        diff = _diff_plan_steps(prev, curr)
        assert diff[2] == "added"

    def test_removed_step_marked_removed(self):
        prev = [{"id": 1, "title": "QC", "method": "scanpy"},
                {"id": 2, "title": "Cluster", "method": "leiden"}]
        curr = [{"id": 1, "title": "QC", "method": "scanpy"}]
        diff = _diff_plan_steps(prev, curr)
        assert diff[2] == "removed"

    def test_modified_step_marked_modified(self):
        prev = [{"id": 1, "title": "QC", "method": "scanpy"}]
        curr = [{"id": 1, "title": "QC", "method": "scanpy+harmony"}]
        diff = _diff_plan_steps(prev, curr)
        assert diff[1] == "modified"

    def test_unchanged_step(self):
        prev = [{"id": 1, "title": "QC", "method": "scanpy"}]
        curr = [{"id": 1, "title": "QC", "method": "scanpy"}]
        diff = _diff_plan_steps(prev, curr)
        assert diff[1] == "unchanged"

    def test_changed_figure_contract_is_modified(self):
        prev = [{"id": 1, "title": "DE", "method": "scanpy"}]
        curr = [{
            **prev[0],
            "figure_requirements": [{"figure_type": "volcano", "minimum_count": 1}],
        }]
        assert _diff_plan_steps(prev, curr)[1] == "modified"

    def test_changed_text_only_exception_is_modified(self):
        prev = [{"id": 1, "title": "Integration", "method": "synthesis"}]
        curr = [{**prev[0], "allow_text_only": True}]
        assert _diff_plan_steps(prev, curr)[1] == "modified"

    def test_changed_text_only_reason_is_modified(self):
        prev = [{
            "id": 1,
            "title": "Integration",
            "method": "synthesis",
            "allow_text_only": True,
            "text_only_reason": "Narrative synthesis",
        }]
        curr = [{**prev[0], "text_only_reason": "Workflow integration checkpoint"}]
        assert _diff_plan_steps(prev, curr)[1] == "modified"


class TestGetTaskText:
    def test_string_returned_as_lower(self):
        assert _get_task_text("Do QC") == "do qc"

    def test_dict_values_joined_lower(self):
        result = _get_task_text({"title": "QC", "method": "Scanpy"})
        assert "qc" in result
        assert "scanpy" in result


class TestNormalizeTasks:
    def test_list_of_strings_passthrough(self):
        assert _normalize_tasks(["a", "b"]) == ["a", "b"]

    def test_list_of_dicts_extracts_task_key(self):
        result = _normalize_tasks([{"task": "QC", "method": "scanpy"}])
        assert len(result) == 1
        assert "QC" in result[0]

    def test_dict_with_description_key(self):
        result = _normalize_tasks([{"description": "Run clustering"}])
        assert "Run clustering" in result[0]

    def test_dict_with_rationale(self):
        result = _normalize_tasks([{"task": "QC", "rationale": "needed"}])
        assert "Rationale: needed" in result[0]

    def test_empty_list(self):
        assert _normalize_tasks([]) == []

    def test_non_string_non_dict_to_string(self):
        result = _normalize_tasks([42])
        assert result == ["42"]

    def test_dict_repr_string_is_parsed(self):
        result = _normalize_tasks(["{'Topic': 'Copper deficiency in hepatocytes', 'Rationale': 'mito'}"])
        assert len(result) == 1
        assert "Copper deficiency in hepatocytes" in result[0]
        assert "Rationale: mito" in result[0]
        assert "{'Topic'" not in result[0]

    def test_dict_with_capitalized_keys(self):
        result = _normalize_tasks([{"Topic": "X pathway", "Rationale": "Y"}])
        assert "X pathway" in result[0]
        assert "Rationale: Y" in result[0]

    def test_plain_brace_string_not_a_dict_passthrough(self):
        result = _normalize_tasks(["{not a dict}"])
        assert result == ["{not a dict}"]


class TestConstants:
    def test_stage1_keywords_is_frozenset(self):
        assert isinstance(STAGE1_KEYWORDS, frozenset)
        assert "qc" in STAGE1_KEYWORDS

    def test_diff_labels_has_expected_keys(self):
        for key in ("added", "removed", "modified", "unchanged"):
            assert key in _DIFF_LABELS
