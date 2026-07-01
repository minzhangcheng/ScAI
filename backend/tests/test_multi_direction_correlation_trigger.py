"""复现并锁定多方向关联分析触发 bug：最后一个方向轮完成时必须触发。"""
from backend.models.schemas import AnalysisState, AnalysisPhase, RoundResult
from backend.workflow.synthesis import (
    _build_directions_with_results_for_correlation,
    _multi_direction_analysis_completed,
    _should_force_next_round,
    _should_run_direction_correlation,
)


def _state_with_two_directions():
    """构造 round 3（第二个 innovation 轮）synthesis 时刻的状态快照。

    复现 session d2c3dddc 的终止状态：两个方向都已跑完（两条 innovation
    round_result 已 append），但因 P1-1 peek-commit，最后一个方向仍留在队列里。
    """
    dir1 = {"direction": "Direction one", "notes": ""}
    dir2 = {"direction": "Direction two", "notes": ""}
    return AnalysisState(
        session_id="t",
        user_id="test",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[dir1, dir2],
        directions_queue=[dir2],  # 最后一个方向未出队（终止轮，无下一轮 init）
        multi_direction_correlation=None,
        round_results=[
            RoundResult(round_number=1, phase="exploration"),
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction one",
                pi_summary="Direction one summary",
                bioinformatics_findings=["Direction one finding"],
                evidence_status="substantive",
            ),
            RoundResult(
                round_number=3,
                phase="innovation",
                direction="Direction two",
                pi_summary="Direction two summary",
                bioinformatics_findings=["Direction two finding"],
                evidence_status="substantive",
            ),
        ],
    )


def test_triggers_when_all_directions_explored_despite_nonempty_queue():
    """Bug 复现：所有方向都跑完后，即便队列因 P1-1 仍非空，也必须触发关联分析。"""
    state = _state_with_two_directions()
    assert _should_run_direction_correlation(state) is True


def test_no_trigger_before_last_direction_completes():
    """只跑完第一个方向（round 2）时不得提前触发。"""
    dir1 = {"direction": "Direction one", "notes": ""}
    dir2 = {"direction": "Direction two", "notes": ""}
    state = AnalysisState(
        session_id="t",
        user_id="test",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[dir1, dir2],
        directions_queue=[dir1, dir2],
        multi_direction_correlation=None,
        round_results=[
            RoundResult(round_number=1, phase="exploration"),
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction one",
                pi_summary="Direction one summary",
                bioinformatics_findings=["Direction one finding"],
                evidence_status="substantive",
            ),
        ],
    )
    assert _should_run_direction_correlation(state) is False


def test_no_trigger_for_single_direction():
    """单方向模式（len<=1）不做关联分析。"""
    state = AnalysisState(
        session_id="t",
        user_id="test",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[{"direction": "only", "notes": ""}],
        directions_queue=[],
        multi_direction_correlation=None,
        round_results=[
            RoundResult(round_number=1, phase="exploration"),
            RoundResult(round_number=2, phase="innovation", direction="only"),
        ],
    )
    assert _should_run_direction_correlation(state) is False


def test_no_retrigger_when_already_computed():
    """已算过关联（非 None）则不重复触发，避免重复 LLM 调用。"""
    state = _state_with_two_directions()
    state.multi_direction_correlation = "already done"
    assert _should_run_direction_correlation(state) is False


def test_no_trigger_outside_innovation_phase():
    """非 innovation 阶段不触发。"""
    state = _state_with_two_directions()
    state.analysis_phase = AnalysisPhase.EXPLORATION
    assert _should_run_direction_correlation(state) is False


def test_multi_direction_completed_when_correlation_exists_despite_stale_queue():
    """Terminal state: all selected directions are done and correlation exists."""
    state = _state_with_two_directions()
    state.multi_direction_correlation = "correlation summary"

    assert _multi_direction_analysis_completed(state) is True


def test_multi_direction_not_completed_before_correlation_exists():
    """All directions may be done, but terminal handling waits for correlation."""
    state = _state_with_two_directions()

    assert _multi_direction_analysis_completed(state) is False


def test_force_next_round_ignores_stale_queue_after_multi_direction_completion():
    """P1-1 leaves the final direction in the queue on terminal rounds; do not loop."""
    state = _state_with_two_directions()
    state.multi_direction_correlation = "correlation summary"

    assert _should_force_next_round(state) is False


def test_force_next_round_keeps_pending_queue_before_multi_direction_completion():
    """Before all selected directions finish, queued directions still force the next round."""
    dir1 = {"direction": "Direction one", "notes": ""}
    dir2 = {"direction": "Direction two", "notes": ""}
    state = AnalysisState(
        session_id="t",
        user_id="test",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[dir1, dir2],
        directions_queue=[dir2],
        multi_direction_correlation=None,
        round_results=[
            RoundResult(round_number=1, phase="exploration"),
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction one",
                pi_summary="Direction one summary",
                bioinformatics_findings=["Direction one finding"],
                evidence_status="substantive",
            ),
        ],
    )

    assert _should_force_next_round(state) is True


def _state_with_selected_a_b_c_and_results(completed: list[str]) -> AnalysisState:
    """Build an INNOVATION state with three chosen directions (A, B, C) and
    completed round_results for the given completed directions.
    """
    results: list[RoundResult] = []
    for i, direction in enumerate(completed):
        if not direction:
            continue
        results.append(
            RoundResult(
                round_number=i + 2,
                phase="innovation",
                direction=direction,
                pi_summary=f"{direction} summary",
                bioinformatics_findings=[f"{direction} finding"],
                evidence_status="substantive",
            )
        )
    return AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        user_chosen_directions=[
            {"direction": "A", "notes": ""},
            {"direction": "B", "notes": ""},
            {"direction": "C", "notes": ""},
        ],
        round_results=[RoundResult(round_number=1, phase="exploration")] + results,
    )


def test_correlation_does_not_run_for_duplicate_direction_coverage():
    state = _state_with_selected_a_b_c_and_results(["A", "B", "B"])
    assert _should_run_direction_correlation(state) is False


def test_correlation_does_not_run_for_failed_non_empty_direction_summary():
    state = _state_with_selected_a_b_c_and_results(["A", "B"])
    state.round_results.append(
        RoundResult(
            round_number=4,
            phase="innovation",
            direction="C",
            pi_summary="Analysis failed after retries.",
            bioinformatics_findings=[],
            round_status="failed",
            evidence_status="failed",
        )
    )
    assert _should_run_direction_correlation(state) is False


def test_force_next_round_keeps_queue_when_coverage_incomplete_despite_duplicate_round_count():
    state = _state_with_selected_a_b_c_and_results(["A", "B", ""])
    state.multi_direction_correlation = "stale correlation should not satisfy coverage"
    state.directions_queue = [{"direction": "C", "notes": ""}]
    assert _multi_direction_analysis_completed(state) is False
    assert _should_force_next_round(state) is True


def test_direction_correlation_input_uses_target_lineage_summary_only():
    state = AnalysisState(
        id="s",
        user_id="u",
        analysis_phase=AnalysisPhase.INNOVATION,
        restore_branch_id="branch-a",
        user_chosen_directions=[{"direction": "Direction A", "notes": ""}],
        round_results=[
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction A",
                lineage_id="main",
                pi_summary="main summary must not be used",
                bioinformatics_findings=["main finding"],
                evidence_status="substantive",
            ),
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Direction A",
                lineage_id="branch-a",
                pi_summary="branch summary",
                bioinformatics_findings=["branch finding"],
                evidence_status="substantive",
            ),
        ],
    )

    assert _build_directions_with_results_for_correlation(state)[0]["round_summary"] == "branch summary"
