import pytest
from datetime import datetime
from backend.agents.supervisor_agent import SupervisorAgent, SupervisorRecord


def test_full_round_number_roundtrip():
    """End-to-end: get_supervision_log includes round_number -> serialize -> restore_log_from_saved -> get_summary_stats correct."""
    supervisor = SupervisorAgent(settings_provider=None, session_id="test")
    
    supervisor.all_rounds_log = [
        [SupervisorRecord(step_number=1, step_type="execute_code", timestamp=datetime.now())],
        [SupervisorRecord(step_number=2, step_type="execute_code", timestamp=datetime.now()),
         SupervisorRecord(step_number=3, step_type="execute_code", timestamp=datetime.now())],
    ]
    supervisor.supervision_log = [
        SupervisorRecord(step_number=4, step_type="execute_code", timestamp=datetime.now()),
    ]
    
    log = supervisor.get_supervision_log()
    assert log[0]["round_number"] == 1
    assert log[1]["round_number"] == 2
    assert log[2]["round_number"] == 2
    assert log[3]["round_number"] == 3
    
    new_supervisor = SupervisorAgent(settings_provider=None, session_id="test2")
    new_supervisor.restore_log_from_saved(log)
    
    assert len(new_supervisor.all_rounds_log) == 2
    assert len(new_supervisor.all_rounds_log[0]) == 1
    assert len(new_supervisor.all_rounds_log[1]) == 2
    assert len(new_supervisor.supervision_log) == 1
    
    stats = new_supervisor.get_summary_stats()
    assert stats["rounds_completed"] == 2
    assert stats["total_steps_monitored"] == 4


def test_post_restart_regeneration_preserves_rounds():
    """After restarting (loading state from snapshot), regeneration should preserve round_number."""
    supervisor = SupervisorAgent(settings_provider=None, session_id="test")
    records = [
        {"step_number": 1, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
        {"step_number": 2, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
        {"step_number": 3, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 2},
        {"step_number": 4, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 2},
        {"step_number": 5, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 3},
    ]
    
    supervisor.restore_log_from_saved(records)
    
    assert len(supervisor.all_rounds_log) == 2
    assert len(supervisor.supervision_log) == 1
    
    stats = supervisor.get_summary_stats()
    assert stats["rounds_completed"] == 2
    assert stats["total_steps_monitored"] == 5