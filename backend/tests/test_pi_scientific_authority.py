"""Scientific plan and result acceptance belongs to the PI, not host vetoes."""

import inspect

from backend.agents.bioinformatics import (
    executor_execute,
    executor_report,
    executor_report_gates,
    supplementary_support,
)
from backend.workflow.planning_parts import dialogue, review
from backend.workflow.reporting_parts import final_report
from backend.workflow.agent_nodes import shared as agent_node_shared
from backend.agents.pi import shared as pi_shared


def test_host_does_not_override_pi_plan_decisions():
    source = "\n".join((
        inspect.getsource(dialogue.run_planning_dialogue_rounds),
        inspect.getsource(dialogue.finalize_planning_dialogue),
        inspect.getsource(review._user_review_gate),
    ))

    forbidden = (
        "plan_sample_design_violations",
        "enforce_sample_design_review",
        "Final plan violates the measured sequencing sample design",
        "deterministic sample-design",
        "auto-accepting plan",
    )
    assert not [token for token in forbidden if token in source]
    assert "review[\"decision\"]" in source
    assert "generate_final_plan_from_dialogue" in source


def test_execution_and_result_review_have_no_scientific_veto_path():
    execution_source = inspect.getsource(executor_execute.handle_execute_code)
    report_source = "\n".join((
        inspect.getsource(executor_report.handle_report_findings),
        inspect.getsource(executor_report_gates.prepare_report_attempt),
        inspect.getsource(executor_report_gates.attach_report_observations),
        inspect.getsource(supplementary_support._validated_supplementary_findings),
    ))

    assert "reject_sample_design_code" not in execution_source
    assert "code_sample_design_violation" not in execution_source
    assert "stats_gate_reject_count" not in report_source
    assert "mechanism_gate_reject_count" not in report_source
    assert "Discarded supplementary finding" not in report_source
    assert "PI review observation" in report_source


def test_direction_coverage_cannot_block_report_acceptance():
    source = inspect.getsource(final_report.FinalReportMixin._generate_final_report)
    assert "validate_complete_report_allowed" not in source
    assert "report_coverage_observation" in source
    assert '"report_blocked": True' not in source
    assert "allow_partial_report = True" not in source
    assert not hasattr(final_report.FinalReportMixin, "_after_final_report")


def test_measured_design_does_not_override_pi_execution_plan():
    source = "\n".join((
        inspect.getsource(agent_node_shared._build_exploration_instruction),
        inspect.getsource(agent_node_shared._build_stage2_instruction),
        inspect.getsource(agent_node_shared.build_phase_instruction),
        inspect.getsource(pi_shared._build_default_bio_tasks),
    ))
    assert "DESIGN OVERRIDE" not in source
    assert "do NOT use Harmony" not in source
    assert "if scrna_replicated" not in source
    assert "if bulk_replicated" not in source
