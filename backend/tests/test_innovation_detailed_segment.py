"""锁定创新轮次详细报告分段：每个创新轮恰好生成一段，且节点重试不重复。

边界回归：探索轮(round 1) synthesis 末尾 analysis_phase 已被提前置为
INNOVATION，但 round_result.phase 仍为 'exploration'，不得误生成
innovation_round_1 分段。
"""
from backend.models.schemas import AnalysisState, AnalysisPhase, RoundResult, DetailedReportSegment
from backend.workflow.synthesis import _should_build_innovation_segment


def _state(current_round, existing_phases):
    s = AnalysisState(
        id="t",
        user_id="test",
        analysis_phase=AnalysisPhase.INNOVATION,
        current_round=current_round,
    )
    s.detailed_report_segments = [
        DetailedReportSegment(phase=p, title="x", content_html="", round_number=current_round)
        for p in existing_phases
    ]
    return s


def test_builds_segment_for_new_innovation_round():
    s = _state(2, existing_phases=["pre_annotation", "exploration_complete"])
    rr = RoundResult(round_number=2, phase="innovation")
    assert _should_build_innovation_segment(s, rr) is True


def test_does_not_duplicate_existing_innovation_segment():
    s = _state(2, existing_phases=["pre_annotation", "innovation_round_2"])
    rr = RoundResult(round_number=2, phase="innovation")
    assert _should_build_innovation_segment(s, rr) is False


def test_no_innovation_segment_for_exploration_round_despite_phase_flip():
    """回归：探索轮末尾 analysis_phase 已变 INNOVATION，但 round_result.phase
    仍为 exploration，必须返回 False（否则会误生成 innovation_round_1）。"""
    s = _state(1, existing_phases=["pre_annotation", "exploration_complete"])
    rr = RoundResult(round_number=1, phase="exploration")
    assert _should_build_innovation_segment(s, rr) is False
