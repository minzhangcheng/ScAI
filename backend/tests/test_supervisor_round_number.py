import pytest
from datetime import datetime
from dataclasses import fields as dataclass_fields
from backend.agents.supervisor_agent import SupervisorAgent, SupervisorRecord

@pytest.fixture
def supervisor():
    return SupervisorAgent(settings_provider=None, session_id="test")

def test_get_supervision_log_includes_round_number(supervisor):
    r1 = SupervisorRecord(step_number=1, step_type="execute_code", timestamp=datetime.now())
    r2 = SupervisorRecord(step_number=2, step_type="execute_code", timestamp=datetime.now())
    r3 = SupervisorRecord(step_number=3, step_type="execute_code", timestamp=datetime.now())

    supervisor.all_rounds_log = [[r1], [r2]]
    supervisor.supervision_log = [r3]

    log = supervisor.get_supervision_log()
    assert len(log) == 3
    assert log[0]["round_number"] == 1
    assert log[1]["round_number"] == 2
    assert log[2]["round_number"] == 3

def test_get_supervision_log_empty(supervisor):
    log = supervisor.get_supervision_log()
    assert log == []

def test_restore_log_from_saved_groups_by_round_number(supervisor):
    records = [
        {"step_number": 1, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
        {"step_number": 2, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
        {"step_number": 3, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 2},
        {"step_number": 4, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 2},
    ]
    supervisor.restore_log_from_saved(records)
    assert len(supervisor.all_rounds_log) == 1
    assert len(supervisor.all_rounds_log[0]) == 2
    assert len(supervisor.supervision_log) == 2

def test_restore_log_from_saved_single_round(supervisor):
    records = [
        {"step_number": 1, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
        {"step_number": 2, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
    ]
    supervisor.restore_log_from_saved(records)
    assert len(supervisor.all_rounds_log) == 0
    assert len(supervisor.supervision_log) == 2

def test_restore_log_from_saved_old_format_no_round_number(supervisor):
    records = [
        {"step_number": 1, "step_type": "execute_code", "timestamp": datetime.now().isoformat()},
        {"step_number": 2, "step_type": "execute_code", "timestamp": datetime.now().isoformat()},
    ]
    supervisor.restore_log_from_saved(records)
    assert len(supervisor.all_rounds_log) == 0
    assert len(supervisor.supervision_log) == 2

def test_restore_log_from_saved_empty_records(supervisor):
    supervisor.restore_log_from_saved([])
    assert len(supervisor.all_rounds_log) == 0
    assert len(supervisor.supervision_log) == 0

def test_rounds_completed_after_restore(supervisor):
    records = [
        {"step_number": 1, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 1},
        {"step_number": 2, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 2},
        {"step_number": 3, "step_type": "execute_code", "timestamp": datetime.now().isoformat(), "round_number": 2},
    ]
    supervisor.restore_log_from_saved(records)
    stats = supervisor.get_summary_stats()
    assert stats["rounds_completed"] == 1

def test_supervisor_record_restore_field_coverage():
    restored_keys = {
        "step_number", "step_type", "timestamp",
        "technical_issues", "checkpoint_type", "llm_severity",
        "pi_escalated", "pi_response", "feedback_injected",
        "literature_topic", "literature_result",
        "code", "code_output", "code_error",
        "saved_figure_filenames", "agent_reasoning",
    }
    # timestamp is parsed specially, so it's in restored_keys but expected_keys excludes it
    expected_keys = {f.name for f in dataclass_fields(SupervisorRecord)}

    # All restored fields must exist on SupervisorRecord
    missing_from_record = restored_keys - expected_keys
    # All SupervisorRecord fields (except timestamp which is handled specially) must be in restore
    missing_from_restore = expected_keys - restored_keys

    assert not missing_from_record, f"Fields in restore but not on SupervisorRecord: {missing_from_record}"
    assert not missing_from_restore, f"SupervisorRecord fields missing from restore_log_from_saved: {missing_from_restore}"