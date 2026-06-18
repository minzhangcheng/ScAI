"""round2 进入时：上一轮探索成功 → INNOVATION；探索未成功（无 exploration_report）
→ 保持 EXPLORATION 以便用户重试 stage1。"""
from backend.workflow.planning import resolve_round_phase


def test_advances_to_innovation_when_exploration_succeeded():
    assert resolve_round_phase(current_round=2, current_phase="exploration",
                               has_exploration_report=True) == "innovation"


def test_stays_exploration_when_no_report():
    assert resolve_round_phase(current_round=2, current_phase="exploration",
                               has_exploration_report=False) == "exploration"


def test_innovation_phase_stays_innovation():
    assert resolve_round_phase(current_round=3, current_phase="innovation",
                               has_exploration_report=True) == "innovation"


def test_exploration_failed_is_a_valid_ws_message_type():
    from backend.models.schemas import WebSocketMessage
    msg = WebSocketMessage(type="exploration_failed", content={"reason": "x"})
    assert msg.type == "exploration_failed"
