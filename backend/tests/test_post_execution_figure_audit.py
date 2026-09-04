"""Artifact-completeness checks for approved plan visualization deliverables."""

import base64
import hashlib
import itertools
from datetime import timedelta
from pathlib import Path

import pytest

from backend.models.schemas import (
    AnalysisPlan,
    CodeExecution,
    FigureRequirement,
    FigureReviewStatus,
    PlanStep,
    PublicationFigure,
    PublicationTable,
)
from backend.workflow.post_execution_review import (
    audit_plan_execution,
    eligible_current_round_figures,
    extract_planned_figure_kinds,
    should_run_post_execution_review,
)
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes


_ARTIFACT_ROOT: Path | None = None
_ARTIFACT_SEQUENCE = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def _frozen_figure_artifact_root(tmp_path_factory):
    global _ARTIFACT_ROOT
    _ARTIFACT_ROOT = tmp_path_factory.mktemp("post-execution-figure-artifacts")
    yield
    _ARTIFACT_ROOT = None


def _figure(
    name: str,
    figure_type: str,
    *,
    round_number: int = 2,
    status: FigureReviewStatus = FigureReviewStatus.APPROVED,
    **overrides,
) -> PublicationFigure:
    assert _ARTIFACT_ROOT is not None
    artifact_id = next(_ARTIFACT_SEQUENCE)
    png_path = _ARTIFACT_ROOT / f"{artifact_id}-{name}.png"
    pdf_path = _ARTIFACT_ROOT / f"{artifact_id}-{name}.pdf"
    render = valid_image_bytes("PNG", min_bytes=2200)
    pdf = valid_pdf_bytes()
    png_path.write_bytes(render)
    pdf_path.write_bytes(pdf)
    values = {
        "name": name,
        "description": f"Publication-ready {figure_type} result",
        "figure_type": figure_type,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "png_base64": base64.b64encode(render).decode(),
        "render_sha256": hashlib.sha256(render).hexdigest(),
        "preview_sha256": hashlib.sha256(render).hexdigest(),
        "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
        "round_number": round_number,
        "review_status": status,
    }
    values.update(overrides)
    return PublicationFigure(**values)


def _plan(*steps: PlanStep) -> AnalysisPlan:
    return AnalysisPlan(phase="innovation", stage="stage2", steps=list(steps))


def _audit_current(
    plan: AnalysisPlan,
    *,
    code_history=(),
    round_number: int = 2,
    publication_figures=(),
    target_lineage_id: str = "main",
    inherited_plan_ids_by_step=None,
):
    """Audit figure completeness after Supervisor confirms the execution."""
    receipt = CodeExecution(
        id="current-receipt",
        code="run_current_plan_step()",
        round_number=round_number,
        evidence_schema_version=2,
        plan_id=plan.id,
        plan_step_ids=[int(step.id) for step in plan.steps],
        execution_purpose="planned",
        plan_binding_status="confirmed",
        execution_status="succeeded",
        lineage_id=target_lineage_id,
    )
    reconciliation = {
        "plan_id": plan.id,
        "round_number": round_number,
        "lineage_id": target_lineage_id,
        "links": [
            {
                "plan_step_id": int(step.id),
                "status": "evidenced",
                "execution_ids": [receipt.id],
            }
            for step in plan.steps
        ],
    }
    return audit_plan_execution(
        plan,
        [*code_history, receipt],
        round_number=round_number,
        publication_figures=publication_figures,
        inherited_plan_ids_by_step=inherited_plan_ids_by_step,
        target_lineage_id=target_lineage_id,
        semantic_reconciliation=reconciliation,
    )


def test_plot_only_step_is_verifiable_and_unmet_without_an_artifact():
    plan = _plan(PlanStep(
        id=1,
        title="Annotated embedding",
        method="Generate a cell-label UMAP",
        expected_output="Publication-quality annotated UMAP",
    ))

    audit = _audit_current(
        plan,
        code_history=[CodeExecution(
            code="sc.pl.umap(adata); plt.savefig('umap.png')", round_number=2
        )],
        publication_figures=[],
    )
    step = audit.steps[0]

    assert step.status == "pending"
    assert step.verifiable is True
    assert step.planned_families == []
    assert step.planned_figure_kinds == ["umap"]
    assert step.missing_figure_kinds == ["umap"]
    assert audit.summary()["n_unverifiable"] == 0
    assert audit.summary()["n_figure_unmet"] == 1

    run, reason = should_run_post_execution_review(audit, [], "completed", False)
    assert run is True
    assert "figure:umap" in reason


def test_plot_only_step_completes_from_a_current_report_eligible_figure():
    plan = _plan(PlanStep(
        id=1,
        title="Annotated embedding",
        method="Render a UMAP",
    ))
    figure = _figure("annotated_umap", "umap")

    step = _audit_current(
        plan, publication_figures=[figure]
    ).steps[0]

    assert step.status == "completed"
    assert step.missing_figure_kinds == []


def test_mixed_step_requires_method_and_each_named_figure_kind():
    plan = _plan(PlanStep(
        id=1,
        title="Marker detection and annotation display",
        method="Run sc.tl.rank_genes_groups and generate figures",
        expected_output="Marker table, annotated UMAP, and marker dot plot",
    ))
    history = [CodeExecution(
        code="sc.tl.rank_genes_groups(adata, 'leiden')", round_number=2
    )]

    only_umap = _audit_current(
        plan,
        code_history=history,
        publication_figures=[_figure("annotated_umap", "umap")],
    ).steps[0]
    assert only_umap.status == "pending"
    assert only_umap.missing_families == []
    assert only_umap.planned_figure_kinds == ["dotplot", "umap"]
    assert only_umap.missing_figure_kinds == ["dotplot"]

    complete = _audit_current(
        plan,
        code_history=history,
        publication_figures=[
            _figure("annotated_umap", "umap"),
            _figure("marker_dotplot", "dotplot"),
        ],
    ).steps[0]
    assert complete.status == "completed"
    assert complete.missing_families == []
    assert complete.missing_figure_kinds == []


def test_only_current_round_report_eligible_figures_count():
    plan = _plan(PlanStep(id=1, title="Result display", method="Create a heatmap"))
    candidates = [
        _figure("old_heatmap", "heatmap", round_number=1),
        _figure(
            "pending_heatmap",
            "heatmap",
            status=FigureReviewStatus.PENDING,
        ),
        _figure("stale_heatmap", "heatmap", is_current=False),
        _figure(
            "blocked_heatmap",
            "heatmap",
            layout_violations=[{"kind": "text_over_text", "severity": "block"}],
        ),
    ]

    assert eligible_current_round_figures(candidates, 2) == []
    step = _audit_current(
        plan, publication_figures=candidates
    ).steps[0]
    assert step.status == "pending"
    assert step.missing_figure_kinds == ["heatmap"]


def test_only_target_restore_lineage_figures_count_as_execution_evidence():
    plan = _plan(PlanStep(id=1, title="Result display", method="Create a heatmap"))
    main = _figure(
        "main_heatmap", "heatmap", plan_id=plan.id, plan_step_id=1
    )
    branch = _figure(
        "branch_heatmap",
        "heatmap",
        plan_id=plan.id,
        plan_step_id=1,
        lineage_id="restore-branch",
    )

    assert eligible_current_round_figures([main, branch], 2) == [main]
    assert eligible_current_round_figures(
        [main, branch], 2, target_lineage_id="restore-branch"
    ) == [branch]
    verdict = _audit_current(
        plan,
        publication_figures=[main, branch],
        target_lineage_id="restore-branch",
    ).steps[0]
    assert verdict.status == "completed"
    assert verdict.missing_figure_kinds == []


def test_report_eligible_tables_cannot_satisfy_a_figure_requirement():
    table = PublicationTable(
        name="heatmap_values",
        description="Values used to construct the heatmap",
        csv_path="/tmp/heatmap_values.csv",
        round_number=2,
        review_status=FigureReviewStatus.APPROVED,
    )
    plan = _plan(PlanStep(id=1, title="Result display", method="Create a heatmap"))

    assert eligible_current_round_figures([table], 2) == []
    step = _audit_current(
        plan, publication_figures=[table]
    ).steps[0]
    assert step.status == "pending"


def test_legacy_dict_figures_are_diagnostic_but_cannot_grant_plan_credit():
    plan = _plan(PlanStep(id=1, title="Result display", method="Create a heatmap"))
    frozen = _figure("legacy_heatmap", "heatmap")
    legacy_current = {
        "id": "legacy-current",
        "name": frozen.name,
        "description": "Result heat map",
        "figure_type": "heatmap",
        "png_path": frozen.png_path,
        "pdf_path": frozen.pdf_path,
        "png_base64": frozen.png_base64,
        "render_sha256": frozen.render_sha256,
        "preview_sha256": frozen.preview_sha256,
        "pdf_sha256": frozen.pdf_sha256,
        "review_status": "approved",
        "round_number": 2,
    }
    legacy_unattributed = {
        **legacy_current,
        "id": "legacy-unattributed",
        "name": "unattributed_heatmap",
        "round_number": 0,
    }

    current = audit_plan_execution(
        plan, [], round_number=2, publication_figures=[legacy_current]
    ).steps[0]
    assert current.status == "pending"
    assert current.verifiable is False
    assert current.missing_figure_kinds == []

    unattributed = audit_plan_execution(
        plan, [], round_number=2, publication_figures=[legacy_unattributed]
    ).steps[0]
    assert unattributed.status == "pending"


def test_generic_visualization_accepts_any_eligible_publication_figure():
    plan = _plan(PlanStep(
        id=1,
        title="Visual summary",
        method="Create a publication-ready visualization of the main result",
    ))
    figure = _figure("result_scatter", "other", description="Effect-size scatter plot")

    step = _audit_current(
        plan, publication_figures=[figure]
    ).steps[0]
    assert step.planned_figure_kinds == ["figure"]
    assert step.status == "completed"


def test_generic_legacy_figure_must_not_predate_the_original_plan():
    plan = _plan(PlanStep(
        id=1,
        title="Visual summary",
        method="Create a publication-ready visualization of the main result",
    ))
    old_unattributed = _figure("old_result_scatter", "other")
    old_unattributed.created_at = plan.created_at - timedelta(seconds=1)

    verdict = _audit_current(
        plan, publication_figures=[old_unattributed]
    ).steps[0]

    assert verdict.status == "pending"
    assert verdict.missing_figure_kinds == ["figure"]


def test_generic_visualization_does_not_steal_another_steps_attributed_figure():
    plan = _plan(PlanStep(
        id=1,
        title="Visual summary",
        method="Create a publication-ready visualization of the main result",
    ))
    another_step = _figure("step_two_plot", "other", plan_step_id=2)
    unattributed = _figure("legacy_plot", "other")

    # Once this round contains attributed figures, an unattributed figure cannot
    # be used as a wildcard beside one explicitly owned by another step.
    missing = _audit_current(
        plan,
        publication_figures=[another_step, unattributed],
    ).steps[0]
    assert missing.status == "pending"
    assert missing.missing_figure_kinds == ["figure"]

    matched = _audit_current(
        plan,
        publication_figures=[another_step, _figure("step_one_plot", "other", plan_step_id=1)],
    ).steps[0]
    assert matched.status == "completed"


def test_generic_visualization_uses_the_same_plan_lineage_rules_as_typed_figures():
    step = PlanStep(
        id=1,
        title="Visual summary",
        method="Create a publication-ready visualization of the main result",
    )
    parent = AnalysisPlan(
        id="generic-parent", phase="innovation", stage="stage2", steps=[step]
    )
    revised = AnalysisPlan(
        id="generic-revised",
        revision_of=parent.id,
        phase="innovation",
        stage="stage2",
        steps=[step.model_copy(deep=True)],
    )
    current = _figure(
        "current_scatter", "other", plan_step_id=1, plan_id=revised.id
    )
    inherited = _figure(
        "parent_scatter", "other", plan_step_id=1, plan_id=parent.id
    )

    assert _audit_current(
        revised, publication_figures=[current]
    ).steps[0].status == "completed"

    stale = _audit_current(
        revised, publication_figures=[inherited]
    ).steps[0]
    assert stale.status == "pending"
    assert stale.missing_figure_kinds == ["figure"]

    accepted_inheritance = _audit_current(
        revised,
        publication_figures=[inherited],
        inherited_plan_ids_by_step={1: (parent.id,)},
    ).steps[0]
    assert accepted_inheritance.status == "completed"


def test_unavailable_method_keyword_diagnostic_cannot_skip_the_plan_step():
    plan = _plan(PlanStep(
        id=1,
        title="Composition analysis and display",
        method="Run scCODA and create a heatmap",
    ))
    unavailable = [CodeExecution(
        code="import sccoda",
        error="ModuleNotFoundError: No module named 'sccoda'",
        round_number=2,
    )]

    missing_both = audit_plan_execution(
        plan, unavailable, round_number=2, publication_figures=[]
    ).steps[0]
    assert missing_both.status == "pending"
    assert missing_both.verifiable is False
    assert missing_both.missing_families == ["composition_model"]
    assert missing_both.missing_figure_kinds == ["heatmap"]

    figure_complete = audit_plan_execution(
        plan,
        unavailable,
        round_number=2,
        publication_figures=[_figure("composition_heatmap", "heatmap")],
    ).steps[0]
    assert figure_complete.status == "pending"
    assert figure_complete.verifiable is False
    assert figure_complete.missing_figure_kinds == []


def test_legacy_snapshot_opt_out_waives_unstructured_result_figures_too():
    steps = [
        PlanStep(
            id=1,
            title="Narrative interpretation",
            method="Discuss how the heatmap would be interpreted",
            expected_output="Text-only narrative; no figure",
        ),
        PlanStep(
            id=2,
            title="Optional display",
            method="Optionally create a volcano plot",
        ),
        PlanStep(
            id=3,
            title="Marker analysis",
            method="Run sc.tl.rank_genes_groups and optionally create a volcano plot",
        ),
        PlanStep(
            id=4,
            title="Optional embedding",
            method="Create an annotated UMAP if useful",
        ),
        PlanStep(
            id=5,
            title="Rationale-scoped exception",
            method="Create an annotated UMAP",
            rationale="Optional visualization; a narrative is sufficient.",
        ),
    ]
    assert extract_planned_figure_kinds(steps[0]) == set()
    assert extract_planned_figure_kinds(steps[1]) == set()
    assert extract_planned_figure_kinds(steps[2]) == set()
    assert extract_planned_figure_kinds(steps[3]) == set()
    assert extract_planned_figure_kinds(steps[4]) == set()
    audit = audit_plan_execution(
        _plan(*steps),
        [CodeExecution(code="sc.tl.rank_genes_groups(adata)", round_number=2)],
        round_number=2,
        publication_figures=[],
    )
    assert audit.steps[0].verifiable is False
    assert audit.steps[1].verifiable is False
    assert audit.steps[2].status == "pending"
    assert audit.steps[2].verifiable is False
    assert audit.steps[2].planned_families == ["marker_de"]
    assert audit.steps[2].missing_figure_kinds == []
    assert audit.steps[3].verifiable is False
    assert audit.steps[4].verifiable is False

    missing_method = audit_plan_execution(
        _plan(steps[2]), [], round_number=2, publication_figures=[]
    ).steps[0]
    assert missing_method.status == "pending"
    assert missing_method.missing_families == ["marker_de"]


def test_optional_qualifier_cannot_cross_a_sentence_boundary():
    step = PlanStep(
        id=1,
        title="Embedding visualization",
        method="Plot the UMAP. If useful, also run pseudotime.",
    )
    assert extract_planned_figure_kinds(step) == {"umap"}


def test_new_plan_cannot_reintroduce_the_legacy_no_figure_prose_escape():
    step = PlanStep(
        id=6,
        title="Differential expression",
        method="Run pydeseq2",
        expected_output="Tabular result; no figure",
        figure_requirements=[],
        allow_text_only=False,
        text_only_reason="",
    )
    execution = CodeExecution(code="import pydeseq2", round_number=2)

    verdict = _audit_current(
        _plan(step),
        code_history=[execution],
        publication_figures=[],
    ).steps[0]

    assert verdict.status == "pending"
    assert verdict.missing_families == []
    assert verdict.planned_figure_kinds == ["result"]
    assert verdict.missing_figure_kinds == ["result"]


def test_clean_diagnostic_figure_uses_the_shared_report_eligibility_rule():
    plan = _plan(PlanStep(id=1, title="Embedding display", method="Create a UMAP"))
    diagnostic = _figure(
        "stage1_cell_type_umap",
        "umap",
        status=FigureReviewStatus.PENDING,
        diagnostic=True,
        layout_violations=[],
    )

    step = _audit_current(
        plan, publication_figures=[diagnostic]
    ).steps[0]
    assert step.status == "pending"
    assert step.missing_figure_kinds == ["umap"]


def test_structured_requirements_use_count_and_plan_step_attribution():
    plan = _plan(PlanStep(
        id=7,
        title="Two complementary views",
        method="Summarize the result",
        figure_requirements=[FigureRequirement(
            figure_type="umap",
            description="Two independently colored embeddings",
            minimum_count=2,
        )],
    ))
    wrong_step = _figure("wrong_step", "umap", plan_step_id=8, plan_id=plan.id)
    right_step = _figure("right_step", "umap", plan_step_id=7, plan_id=plan.id)

    one = _audit_current(
        plan,
        publication_figures=[wrong_step, right_step],
    ).steps[0]
    assert one.status == "pending"
    assert one.planned_figure_kinds == ["umap"]
    assert one.missing_figure_kinds == ["umap"]

    two = _audit_current(
        plan,
        publication_figures=[
            right_step,
            _figure(
                "right_step_second", "umap", plan_step_id=7, plan_id=plan.id
            ),
        ],
    ).steps[0]
    assert two.status == "completed"


def test_revised_plan_cannot_reuse_a_parent_plans_attributed_result():
    parent = _plan(PlanStep(
        id=7,
        title="Result view",
        method="Summarize the result",
        figure_requirements=[FigureRequirement(figure_type="heatmap")],
    ))
    revised = _plan(PlanStep(
        id=7,
        title="Revised result view",
        method="Summarize the revised result",
        figure_requirements=[FigureRequirement(figure_type="heatmap")],
    ))
    revised.revision_of = parent.id
    old = _figure(
        "parent_heatmap",
        "heatmap",
        plan_step_id=7,
        plan_id=parent.id,
    )

    verdict = _audit_current(
        revised, publication_figures=[old]
    ).steps[0]

    assert verdict.status == "pending"
    assert verdict.missing_figure_kinds == ["heatmap"]


def test_structured_requirement_cannot_be_waived_by_text_only_or_optional_prose():
    step = PlanStep(
        id=3,
        title="Narrative-only special case (figure optional)",
        method="Summarize the result in prose",
        figure_requirements=[FigureRequirement(
            figure_type="heatmap", description="Not required for this exception"
        )],
        allow_text_only=True,
        text_only_reason="Synthesis only",
    )

    assert extract_planned_figure_kinds(step) == {"heatmap"}
    verdict = _audit_current(
        _plan(step), publication_figures=[]
    ).steps[0]
    assert verdict.verifiable is True
    assert verdict.status == "pending"


def test_explained_text_only_step_without_requirement_is_waived():
    step = PlanStep(
        id=4,
        title="Narrative-only special case",
        method="Summarize prior evidence in prose",
        allow_text_only=True,
        text_only_reason="No new data result is computed in this synthesis step.",
    )
    assert extract_planned_figure_kinds(step) == set()
    verdict = audit_plan_execution(
        _plan(step), [], round_number=2, publication_figures=[]
    ).steps[0]
    assert verdict.verifiable is False


def test_integration_phase_text_only_does_not_waive_a_mixed_quantitative_step():
    synthesis = AnalysisPlan(
        phase="integration",
        stage="full",
        steps=[PlanStep(
            id=5,
            title="Cross-direction evidence assembly",
            method=(
                "Reconcile previously reported differential expression and pathway "
                "enrichment evidence in a network diagram description"
            ),
            figure_requirements=[],
            allow_text_only=True,
            text_only_reason=(
                "This step only reconciles existing direction-level evidence and "
                "does not compute a new result."
            ),
        )],
    )
    synthesis_verdict = audit_plan_execution(
        synthesis, [], round_number=2, publication_figures=[]
    ).steps[0]
    assert synthesis_verdict.verifiable is False

    mixed = AnalysisPlan(
        phase="integration",
        stage="full",
        steps=[PlanStep(
            id=6,
            title="Integrated quantitative association",
            method="Run correlation analysis and synthesize the integrated findings",
            figure_requirements=[],
            allow_text_only=True,
            text_only_reason=(
                "The narrative will explain how the newly computed association "
                "relates to the prior evidence."
            ),
        )],
    )
    mixed_verdict = audit_plan_execution(
        mixed, [], round_number=2, publication_figures=[]
    ).steps[0]
    assert mixed_verdict.status == "pending"
    assert mixed_verdict.planned_figure_kinds == ["result"]
    assert mixed_verdict.missing_figure_kinds == ["result"]


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


def test_host_linkage_figure_report_stays_silent_for_a_waived_step():
    """A step that legitimately produces no figure is not reported as missing.

    The counterpart to the report above: the observation must distinguish "the
    plan asked for an artifact and the round did not produce it" from "this
    step was never going to produce one". A step with no declared requirement
    whose text-only exception explains that it computes no new result is the
    latter, and must not inflate the observation the PI reviews.
    """
    plan = _plan(
        PlanStep(
            id=1,
            title="Condition-level expression heatmap",
            method="Plot the condition-level expression heatmap",
            expected_output="Publication-quality heatmap",
            figure_requirements=[FigureRequirement(figure_type="heatmap")],
        ),
        PlanStep(
            id=2,
            title="Narrative synthesis of prior evidence",
            method="Summarize prior evidence in prose",
            allow_text_only=True,
            text_only_reason=(
                "No new data result is computed in this synthesis step."
            ),
        ),
    )
    receipt = CodeExecution(
        id="host-linkage-receipt-two-steps",
        code="run_plan_steps()",
        round_number=2,
        evidence_schema_version=2,
        plan_id=plan.id,
        plan_step_ids=[1, 2],
        execution_purpose="planned",
        execution_status="succeeded",
        lineage_id="main",
    )

    audit = audit_plan_execution(
        plan, [receipt], round_number=2, publication_figures=[]
    )
    summary = audit.summary()

    assert summary["evidence_mode"] == "host_execution_linkage"
    assert summary["n_steps"] == 2
    assert summary["n_figure_steps"] == 1
    assert summary["n_figure_unmet"] == 1
    assert summary["unmet_figure_kinds"] == ["heatmap"]
    assert [step.step_id for step in audit.figure_unmet] == [1]
