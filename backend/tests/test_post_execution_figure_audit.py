"""Artifact-completeness checks for approved plan visualization deliverables."""



from backend.models.schemas import (
    AnalysisPlan,
    CodeExecution,
    FigureRequirement,
    PlanStep,
)
from backend.workflow.post_execution_review import (
    audit_plan_execution,
)


def _plan(*steps: PlanStep) -> AnalysisPlan:
    return AnalysisPlan(phase="innovation", stage="stage2", steps=list(steps))


def test_host_linkage_mode_reports_missing_figures_for_pi_review():
    """A missing required artifact must reach the PI in host-linkage mode.

    Innovation rounds run without Supervisor semantic reconciliation, so every
    step is left `awaiting_pi_decision` and `verifiable=False` — the scientific
    completion call belongs to the PI. Whether the round produced the artifact
    a step asked for is a technical fact about artifact linkage, not that
    scientific call, so it must still be summarized rather than dropped along
    with the verdict.
    """
    plan = _plan(PlanStep(
        id=1,
        title="Condition-level expression heatmap",
        method="Plot the condition-level expression heatmap",
        expected_output="Publication-quality heatmap",
        figure_requirements=[FigureRequirement(figure_type="heatmap")],
    ))
    receipt = CodeExecution(
        id="host-linkage-receipt",
        code="run_condition_heatmap_step()",
        round_number=2,
        evidence_schema_version=2,
        plan_id=plan.id,
        plan_step_ids=[1],
        execution_purpose="planned",
        execution_status="succeeded",
        lineage_id="main",
    )

    summary = audit_plan_execution(
        plan,
        [receipt],
        round_number=2,
        publication_figures=[],
    ).summary()

    assert summary["evidence_mode"] == "host_execution_linkage"
    assert summary["n_awaiting_pi_decision"] == 1
    assert summary["n_figure_steps"] == 1
    assert summary["n_figure_unmet"] == 1
    assert summary["unmet_figure_kinds"] == ["heatmap"]
