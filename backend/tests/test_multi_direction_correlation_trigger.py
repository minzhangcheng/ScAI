"""复现并锁定多方向关联分析触发 bug：最后一个方向轮完成时必须触发。"""
from backend.models.schemas import AnalysisState, AnalysisPhase, RoundResult
from backend.workflow.synthesis import _should_run_direction_correlation


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
            RoundResult(round_number=2, phase="innovation", direction="Direction one"),
            RoundResult(round_number=3, phase="innovation", direction="Direction two"),
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
            RoundResult(round_number=2, phase="innovation", direction="Direction one"),
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
