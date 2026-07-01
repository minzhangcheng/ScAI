from backend.workflow.callbacks import parse_plan_revision_signal


def test_normal_not_applicable_text_does_not_trigger_revision():
    text = "This pathway is not applicable to the current result interpretation."
    assert parse_plan_revision_signal(text) is False


def test_structured_sentinel_triggers_revision():
    text = "Analysis cannot continue safely. [__PLAN_REVISION_NEEDED__]"
    assert parse_plan_revision_signal(text) is True


def test_json_signal_triggers_revision():
    text = '{"plan_revision_needed": true, "reason": "required input missing"}'
    assert parse_plan_revision_signal(text) is True
