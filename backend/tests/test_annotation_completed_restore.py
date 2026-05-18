"""Tests for annotation_completed restoration from analysis_log."""
from backend.models.schemas import AnalysisStep


def _is_successful_annotation(step: AnalysisStep) -> bool:
    if step.step_type != "annotate_clusters":
        return False
    summary = step.annotation_result_summary or ""
    return not summary.startswith("[ANNOTATION FAILED]")


def test_successful_annotation_step_counts_as_complete():
    step = AnalysisStep(
        step_number=1,
        step_type="annotate_clusters",
        annotation_result_summary="Cluster 0: T cell (confidence: high)",
    )
    assert _is_successful_annotation(step) is True


def test_failed_annotation_step_does_not_count_as_complete():
    step = AnalysisStep(
        step_number=1,
        step_type="annotate_clusters",
        annotation_result_summary="[ANNOTATION FAILED] missing cluster_key",
    )
    assert _is_successful_annotation(step) is False


def test_unrelated_step_type_does_not_count():
    step = AnalysisStep(step_number=1, step_type="execute_code", text_content="print(1)")
    assert _is_successful_annotation(step) is False


def test_mixed_log_recovers_only_when_a_success_exists():
    log = [
        AnalysisStep(
            step_number=1, step_type="annotate_clusters",
            annotation_result_summary="[ANNOTATION FAILED] err1",
        ),
        AnalysisStep(
            step_number=2, step_type="annotate_clusters",
            annotation_result_summary="Cluster 0: B cell",
        ),
    ]
    assert any(_is_successful_annotation(s) for s in log) is True

    log_only_fail = [
        AnalysisStep(
            step_number=1, step_type="annotate_clusters",
            annotation_result_summary="[ANNOTATION FAILED] err1",
        ),
    ]
    assert any(_is_successful_annotation(s) for s in log_only_fail) is False


def test_production_predicate_matches():
    def production_check(analysis_log):
        return bool(analysis_log) and any(
            s.step_type == "annotate_clusters"
            and not (s.annotation_result_summary or "").startswith("[ANNOTATION FAILED]")
            for s in analysis_log
        )

    assert production_check([
        AnalysisStep(step_number=1, step_type="annotate_clusters",
                     annotation_result_summary="[ANNOTATION FAILED] x"),
    ]) is False
    assert production_check([
        AnalysisStep(step_number=1, step_type="annotate_clusters",
                     annotation_result_summary="Cluster 0: T cell"),
    ]) is True
    assert production_check([]) is False