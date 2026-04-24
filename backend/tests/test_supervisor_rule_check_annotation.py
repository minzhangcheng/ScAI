"""Supervisor should surface annotate_clusters failures as rule-based issues."""
from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import AnalysisStep


def _mk_supervisor() -> SupervisorAgent:
    sup = SupervisorAgent.__new__(SupervisorAgent)
    sup._recent_errors = []
    sup._recent_rule_feedback_hashes = set()
    sup._step_count = 0
    return sup


def test_rule_check_flags_annotate_failure():
    sup = _mk_supervisor()
    step = AnalysisStep(
        step_number=40,
        step_type="annotate_clusters",
        annotation_result_summary=(
            "Error: Could not find an AnnData object with a clustering column "
            "(leiden/louvain/cluster) in the sandbox."
        ),
    )
    issues = sup._rule_check(step)
    assert any("annotate_clusters failed" in i.lower() for i in issues), issues


def test_rule_check_silent_on_successful_annotation():
    sup = _mk_supervisor()
    step = AnalysisStep(
        step_number=40,
        step_type="annotate_clusters",
        annotation_result_summary="Cluster annotations:\n0: Hepatocyte (high)\n",
    )
    issues = sup._rule_check(step)
    assert not any("annotate_clusters failed" in i.lower() for i in issues)


def test_rule_check_silent_on_non_annotate_steps():
    sup = _mk_supervisor()
    step = AnalysisStep(step_number=1, step_type="execute_code", code_output="ok")
    issues = sup._rule_check(step)
    assert not any("annotate_clusters" in i.lower() for i in issues)