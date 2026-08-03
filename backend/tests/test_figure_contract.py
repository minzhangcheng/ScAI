import base64
import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import (
    AnalysisPlan,
    FigureReviewStatus,
    PlanStep,
    PublicationFigure,
)
from backend.services.figure_contract import (
    audit_figure_contract,
    collect_plan_figure_requirements,
    record_plan_figure_inheritance,
    visual_plan_inheritance_for_plan,
)
from backend.agents.bioinformatics_agent import TOOLS
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes


_CURRENT_PLAN = "__current_plan__"
_ARTIFACT_ROOT: Path | None = None
_ARTIFACT_SEQUENCE = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def _frozen_figure_artifact_root(tmp_path_factory):
    global _ARTIFACT_ROOT
    _ARTIFACT_ROOT = tmp_path_factory.mktemp("figure-contract-artifacts")
    yield
    _ARTIFACT_ROOT = None


def _figure(
    name: str,
    figure_type: str,
    *,
    step_id=None,
    description="",
    status=FigureReviewStatus.APPROVED,
    diagnostic=False,
    round_number=2,
    plan_id=_CURRENT_PLAN,
    purpose="result",
):
    assert _ARTIFACT_ROOT is not None
    artifact_id = next(_ARTIFACT_SEQUENCE)
    png_path = _ARTIFACT_ROOT / f"{artifact_id}-{name}.jpg"
    pdf_path = _ARTIFACT_ROOT / f"{artifact_id}-{name}.pdf"
    render = valid_image_bytes("JPEG", min_bytes=2200)
    pdf = valid_pdf_bytes()
    png_path.write_bytes(render)
    pdf_path.write_bytes(pdf)
    return PublicationFigure(
        name=name,
        description=description,
        figure_type=figure_type,
        png_path=str(png_path),
        pdf_path=str(pdf_path),
        png_base64=base64.b64encode(render).decode(),
        render_sha256=hashlib.sha256(render).hexdigest(),
        preview_sha256=hashlib.sha256(render).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf).hexdigest(),
        review_status=status,
        plan_step_id=step_id,
        plan_id=plan_id,
        purpose=purpose,
        diagnostic=diagnostic,
        round_number=round_number,
    )


def _state(plan, figures, *, round_number=2, has_scrna=True):
    for figure in figures:
        if figure.plan_id == _CURRENT_PLAN:
            figure.plan_id = plan.id
    return SimpleNamespace(
        current_plan=plan,
        publication_figures=figures,
        current_round=round_number,
        sandbox_state={},
        has_scrnaseq=lambda: has_scrna,
    )


def test_structured_requirements_need_linked_report_eligible_figures():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=7,
            title="Score response programs",
            method="scanpy.tl.score_genes",
            figure_requirements=[
                {"figure_type": "umap", "description": "Score on embedding"},
                {"figure_type": "violin", "description": "Score by group"},
            ],
        )],
    )
    state = _state(plan, [
        _figure("score_umap", "umap", step_id=7),
        _figure("score_violin", "violin", step_id=7),
    ])
    assert audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False
    ).passed


def test_exact_plan_id_allows_transitional_figure_without_step_id():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=7,
            title="Embedding",
            method="Plot UMAP",
            figure_requirements=[{"figure_type": "umap"}],
        )],
    )
    transitional = _figure("legacy_plan_umap", "umap", step_id=None)
    state = _state(plan, [transitional])

    assert audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False,
    ).passed

    transitional.plan_id = "another-plan"
    assert not audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False,
    ).passed


def test_revision_requested_figure_does_not_close_contract():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=2,
            title="Differential expression",
            method="pydeseq2",
            figure_requirements=[{"figure_type": "volcano"}],
        )],
    )
    state = _state(plan, [
        _figure(
            "de_volcano",
            "volcano",
            step_id=2,
            status=FigureReviewStatus.REVISION_REQUESTED,
        )
    ])
    audit = audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False
    )
    assert not audit.passed
    assert audit.missing[0].requirement.figure_type == "volcano"


def test_structured_requirement_cannot_be_satisfied_by_unlinked_same_type():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=9,
            title="Visualize a registered score",
            method="scanpy.tl.score_genes",
            figure_requirements=[{"figure_type": "umap"}],
        )],
    )
    state = _state(
        plan, [_figure("unlinked_umap", "umap", plan_id=None)]
    )
    assert not audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False
    ).passed


def test_text_only_exception_requires_reason_and_cannot_override_requirement():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=4,
            title="Narrative integration",
            method="Evidence synthesis",
            allow_text_only=True,
            text_only_reason="This step synthesizes existing evidence without a new data result.",
            figure_requirements=[{"figure_type": "network"}],
        )],
    )
    assert [item.figure_type for item in collect_plan_figure_requirements(plan)] == ["network"]
    assert not audit_figure_contract(
        _state(plan, []), planning_stage="full", include_stage_baseline=False
    ).passed


def test_explained_text_only_exception_waives_only_inferred_visual_prose():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=4,
            title="Narrative integration",
            method=(
                "Summarize previously reported differential expression evidence "
                "without recomputing a network plot"
            ),
            allow_text_only=True,
            text_only_reason="No new numerical result is produced by this synthesis step.",
        )],
    )
    assert collect_plan_figure_requirements(plan) == []


def test_unexplained_text_only_flag_does_not_waive_named_plot():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=4,
            title="Narrative integration with network plot",
            method="Evidence synthesis",
            allow_text_only=True,
        )],
    )
    assert [item.figure_type for item in collect_plan_figure_requirements(plan)] == ["network"]


def test_arbitrary_reason_cannot_turn_an_analytical_result_text_only():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=5,
            title="Differential expression volcano plot",
            method="pydeseq2 differential expression",
            allow_text_only=True,
            text_only_reason="Skip the figure to save time during this analysis.",
        )],
    )
    assert [item.figure_type for item in collect_plan_figure_requirements(plan)] == ["volcano"]


def test_integration_text_only_is_limited_to_non_result_synthesis_steps():
    synthesis = AnalysisPlan(
        phase="integration",
        stage="full",
        steps=[PlanStep(
            id=6,
            title="Cross-direction evidence assembly",
            method=(
                "Reconcile previously reported differential expression and pathway "
                "enrichment evidence in a network diagram description"
            ),
            figure_requirements=[],
            allow_text_only=True,
            text_only_reason=(
                "This phase only reconciles prior direction-level evidence and "
                "computes no new analytical result."
            ),
        )],
    )
    assert collect_plan_figure_requirements(synthesis) == []

    mixed_result = AnalysisPlan(
        phase="integration",
        stage="full",
        steps=[PlanStep(
            id=7,
            title="Correlation analysis and integrated findings",
            method="Run correlation analysis, then synthesize the integrated findings",
            figure_requirements=[],
            allow_text_only=True,
            text_only_reason=(
                "The written synthesis will explain how the newly computed "
                "quantitative result relates to prior evidence."
            ),
        )],
    )
    requirements = collect_plan_figure_requirements(mixed_result)
    assert [item.figure_type for item in requirements] == ["result"]
    assert requirements[0].wildcard_result_type is True


def test_prose_figure_opt_out_is_legacy_snapshot_compatibility_only():
    old_step = PlanStep.model_validate({
        "id": 8,
        "title": "Differential expression",
        "method": "Run pydeseq2",
        "expected_output": "Tabular result; no figure",
    })
    assert {
        "figure_requirements", "allow_text_only", "text_only_reason",
    }.isdisjoint(old_step.model_fields_set)
    old_plan = AnalysisPlan(phase="innovation", stage="full", steps=[old_step])
    assert collect_plan_figure_requirements(old_plan) == []

    new_step = PlanStep(
        id=8,
        title="Differential expression",
        method="Run pydeseq2",
        expected_output="Tabular result; no figure",
        figure_requirements=[],
        allow_text_only=False,
        text_only_reason="",
    )
    assert {
        "figure_requirements", "allow_text_only", "text_only_reason",
    }.issubset(new_step.model_fields_set)
    new_plan = AnalysisPlan(phase="innovation", stage="full", steps=[new_step])
    requirements = collect_plan_figure_requirements(new_plan)
    assert [item.figure_type for item in requirements] == ["result"]


def test_legacy_explicit_plot_promise_is_machine_checkable():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=3,
            title="Show differential expression with a volcano plot and heatmap",
            method="rank_genes_groups",
        )],
    )
    requirements = collect_plan_figure_requirements(plan)
    assert {item.figure_type for item in requirements} == {"volcano", "heatmap"}
    state = _state(plan, [
        _figure("legacy_volcano", "volcano", plan_id=None),
        _figure("legacy_heatmap", "heatmap", plan_id=None),
    ])
    assert audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False
    ).passed


def test_stage1_baseline_requires_annotation_semantics_not_any_umap():
    plan = AnalysisPlan(phase="exploration", stage="stage1", steps=[])
    unrelated = _figure(
        "condition_score_umap",
        "umap",
        description="Condition score embedding",
        diagnostic=True,
        round_number=1,
    )
    state = _state(plan, [unrelated], round_number=1)
    assert not audit_figure_contract(state, planning_stage="stage1").passed

    annotation = _figure(
        "stage1_cell_type_umap",
        "umap",
        description="Accepted cell-type annotation baseline",
        diagnostic=True,
        round_number=1,
    )
    state.publication_figures.append(annotation)
    assert audit_figure_contract(state, planning_stage="stage1").passed


def test_stage1_platform_baseline_can_fulfil_matching_structured_requirement():
    plan = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        steps=[PlanStep(
            id=8,
            title="Cell annotation",
            method="annotate_clusters",
            figure_requirements=[{
                "figure_type": "umap",
                "description": "Cell-type annotation UMAP",
                "platform_baseline": "annotation",
            }],
        )],
    )
    baseline = _figure(
        "stage1_cell_type_umap",
        "umap",
        description="Accepted cell-type annotation baseline",
        diagnostic=True,
        round_number=1,
    )
    baseline.purpose = "baseline"
    state = _state(plan, [baseline], round_number=1)
    assert audit_figure_contract(state, planning_stage="stage1").passed


def test_executor_defers_only_platform_owned_stage1_baseline_until_workflow_render():
    plan = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        steps=[PlanStep(
            id=8,
            title="Cell annotation",
            method="annotate_clusters",
            figure_requirements=[{
                "figure_type": "umap",
                "description": "Cell-type annotation UMAP",
                "platform_baseline": "annotation",
            }],
        )],
    )
    state = _state(plan, [], round_number=1)

    assert audit_figure_contract(
        state,
        planning_stage="stage1",
        include_stage_baseline=False,
        defer_platform_baseline=True,
    ).passed
    assert not audit_figure_contract(
        state, planning_stage="stage1", include_stage_baseline=False
    ).passed


def test_stage1_exceptional_comparison_still_needs_plan_linked_figure():
    plan = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        steps=[PlanStep(
            id=6,
            title="Batch correction decision",
            method="Compare representations",
            figure_requirements=[{
                "figure_type": "umap",
                "description": "Before versus after batch-correction comparison",
            }],
        )],
    )
    platform_batch = _figure(
        "stage1_batch_umap",
        "umap",
        description="Cells colored by batch baseline",
        diagnostic=True,
        round_number=1,
    )
    platform_batch.purpose = "baseline"
    state = _state(plan, [platform_batch], round_number=1)
    assert not audit_figure_contract(state, planning_stage="stage1").passed
    assert not audit_figure_contract(
        state,
        planning_stage="stage1",
        include_stage_baseline=False,
        defer_platform_baseline=True,
    ).passed


def test_stage1_subcluster_umap_is_not_misclassified_as_platform_cluster_baseline():
    plan = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        steps=[PlanStep(
            id=11,
            title="Focused subclustering",
            method="leiden on selected cells",
            figure_requirements=[{
                "figure_type": "umap",
                "description": "Focused subcluster UMAP for the selected population",
            }],
        )],
    )
    platform_cluster = _figure(
        "stage1_cluster_umap", "umap",
        description="Platform-generated cluster UMAP baseline",
        diagnostic=True, round_number=1,
    )
    platform_cluster.purpose = "baseline"

    assert not audit_figure_contract(
        _state(plan, [platform_cluster], round_number=1),
        planning_stage="stage1",
        include_stage_baseline=False,
    ).passed


def test_missing_image_bytes_cannot_close_contract():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=2,
            title="Result heatmap",
            method="Render heatmap",
            figure_requirements=[{"figure_type": "heatmap"}],
        )],
    )
    missing = _figure("missing_render", "heatmap", step_id=2)
    Path(missing.png_path).unlink()
    missing.png_base64 = None
    assert not audit_figure_contract(
        _state(plan, [missing]), planning_stage="full", include_stage_baseline=False
    ).passed


def test_bulk_only_stage1_does_not_require_cell_annotation_umap():
    plan = AnalysisPlan(phase="exploration", stage="stage1", steps=[])
    state = _state(plan, [], round_number=1, has_scrna=False)
    assert audit_figure_contract(state, planning_stage="stage1").passed


def test_prior_round_figure_cannot_satisfy_current_plan():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=5,
            title="Pathway activity",
            method="decoupler",
            figure_requirements=[{"figure_type": "heatmap"}],
        )],
    )
    state = _state(plan, [
        _figure("old_heatmap", "heatmap", step_id=5, round_number=1)
    ], round_number=2)
    assert not audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False
    ).passed


def test_unattributed_round_zero_figure_cannot_satisfy_current_plan():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=5,
            title="Pathway activity",
            method="decoupler",
            figure_requirements=[{"figure_type": "heatmap"}],
        )],
    )
    state = _state(plan, [
        _figure("legacy_heatmap", "heatmap", step_id=5, round_number=0)
    ], round_number=2)
    assert not audit_figure_contract(
        state, planning_stage="full", include_stage_baseline=False
    ).passed


def test_revised_plan_cannot_inherit_parent_plan_figure_with_same_step_and_type():
    parent = AnalysisPlan(
        id="plan-parent",
        phase="innovation",
        steps=[PlanStep(
            id=5,
            title="Pathway activity",
            method="decoupler",
            figure_requirements=[{"figure_type": "heatmap"}],
            status="completed",
        )],
    )
    old_figure = _figure(
        "parent_heatmap", "heatmap", step_id=5, plan_id=parent.id,
    )
    revised = AnalysisPlan(
        id="plan-revised",
        revision_of=parent.id,
        phase="innovation",
        steps=[PlanStep(
            id=5,
            title="Revised pathway activity",
            method="decoupler",
            figure_requirements=[{"figure_type": "heatmap"}],
            status="completed",
        )],
    )

    state = _state(revised, [old_figure])
    record_plan_figure_inheritance(state, parent)
    record_plan_figure_inheritance(state, revised)
    audit = audit_figure_contract(
        state,
        planning_stage="full",
        include_stage_baseline=False,
    )
    assert not audit.passed
    assert audit.missing[0].requirement.plan_id == revised.id


def test_unchanged_completed_step_transitively_inherits_reviewed_parent_figure():
    def completed_step():
        return PlanStep(
            id=5,
            title="Pathway activity",
            method="decoupler",
            figure_requirements=[{"figure_type": "heatmap"}],
            status="completed",
        )

    parent = AnalysisPlan(
        id="plan-parent", phase="innovation", steps=[completed_step()]
    )
    revision_one = AnalysisPlan(
        id="plan-revision-one",
        revision_of=parent.id,
        phase="innovation",
        steps=[completed_step()],
    )
    revision_two = AnalysisPlan(
        id="plan-revision-two",
        revision_of=revision_one.id,
        phase="innovation",
        steps=[completed_step()],
    )
    parent_figure = _figure(
        "parent_heatmap", "heatmap", step_id=5, plan_id=parent.id,
    )
    state = _state(revision_two, [parent_figure])
    for plan in (parent, revision_one, revision_two):
        record_plan_figure_inheritance(state, plan)

    assert visual_plan_inheritance_for_plan(state, revision_two) == {
        5: (revision_one.id, parent.id),
    }
    assert audit_figure_contract(
        state,
        planning_stage="full",
        include_stage_baseline=False,
    ).passed


def test_original_legacy_plan_can_use_post_plan_unattributed_legacy_figure():
    plan = AnalysisPlan(
        id="legacy-original",
        phase="innovation",
        steps=[PlanStep(
            id=3,
            title="Show differential expression with a volcano plot",
            method="rank_genes_groups",
        )],
    )
    restored = _figure(
        "legacy_volcano", "volcano", step_id=3, plan_id=None,
    )
    assert restored.created_at >= plan.created_at
    assert audit_figure_contract(
        _state(plan, [restored]),
        planning_stage="full",
        include_stage_baseline=False,
    ).passed


def test_revised_plan_rejects_unattributed_parent_era_figure():
    old_plan = AnalysisPlan(id="old", phase="innovation", steps=[])
    old_figure = _figure(
        "legacy_heatmap", "heatmap", step_id=2, plan_id=None,
    )
    revised = AnalysisPlan(
        id="new",
        revision_of=old_plan.id,
        phase="innovation",
        steps=[PlanStep(
            id=2,
            title="Revised heatmap",
            method="Render heatmap",
        )],
    )
    assert old_figure.created_at <= revised.created_at
    assert not audit_figure_contract(
        _state(revised, [old_figure]),
        planning_stage="full",
        include_stage_baseline=False,
    ).passed


def test_diagnostic_and_baseline_figures_do_not_satisfy_result_requirements():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=6,
            title="Differential expression",
            method="pydeseq2",
            figure_requirements=[{"figure_type": "volcano"}],
        )],
    )
    diagnostic = _figure(
        "diagnostic_volcano", "volcano", step_id=6, diagnostic=True,
        purpose="diagnostic",
    )
    baseline = _figure(
        "baseline_volcano", "volcano", step_id=6, purpose="baseline",
    )
    for supporting in (diagnostic, baseline):
        assert not audit_figure_contract(
            _state(plan, [supporting]),
            planning_stage="full",
            include_stage_baseline=False,
        ).passed


def test_current_round_match_is_strict_even_when_current_round_is_zero():
    plan = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=1,
            title="Result heatmap",
            method="Render heatmap",
        )],
    )
    future = _figure("future_heatmap", "heatmap", round_number=1, plan_id=None)
    assert not audit_figure_contract(
        _state(plan, [future], round_number=0),
        planning_stage="full",
        include_stage_baseline=False,
    ).passed


def test_result_families_without_structured_policy_get_wildcard_requirement():
    methods = (
        "pydeseq2 differential expression",
        "gseapy pathway enrichment",
        "scanpy.tl.score_genes module score",
        "sc.tl.dpt pseudotime",
        "Milo differential abundance",
        "LIANA cell-cell communication",
        "regulatory network analysis",
    )
    for index, method in enumerate(methods, start=1):
        plan = AnalysisPlan(
            phase="innovation",
            stage="stage2",
            steps=[PlanStep(id=index, title="Analytical result", method=method)],
        )
        requirements = collect_plan_figure_requirements(plan)
        assert len(requirements) == 1, method
        assert requirements[0].figure_type == "result", method
        assert requirements[0].wildcard_result_type, method


def test_wildcard_result_accepts_any_linked_result_type_but_not_supporting_views():
    plan = AnalysisPlan(
        phase="innovation",
        stage="stage2",
        steps=[PlanStep(
            id=4,
            title="Differential abundance",
            method="Milo differential abundance",
        )],
    )
    heatmap = _figure("abundance_summary", "heatmap", step_id=4)
    assert audit_figure_contract(
        _state(plan, [heatmap]),
        planning_stage="stage2",
        include_stage_baseline=False,
    ).passed

    diagnostic = _figure(
        "abundance_qc", "heatmap", step_id=4,
        diagnostic=True, purpose="diagnostic",
    )
    baseline = _figure(
        "abundance_baseline", "heatmap", step_id=4, purpose="baseline",
    )
    for supporting in (diagnostic, baseline):
        assert not audit_figure_contract(
            _state(plan, [supporting]),
            planning_stage="stage2",
            include_stage_baseline=False,
        ).passed


def test_result_fallback_skips_technical_setup_and_stage1_baseline_work():
    technical = AnalysisPlan(
        phase="innovation",
        stage="stage2",
        steps=[PlanStep(
            id=1,
            title="Load data",
            method="Read the matrix and validate metadata columns",
        )],
    )
    stage1 = AnalysisPlan(
        phase="exploration",
        stage="stage1",
        steps=[PlanStep(
            id=2,
            title="Marker ranking for annotation",
            method="scanpy.tl.rank_genes_groups",
        )],
    )
    assert collect_plan_figure_requirements(technical) == []
    assert collect_plan_figure_requirements(stage1) == []


def test_legacy_optional_or_no_figure_prose_waives_only_unstructured_visuals():
    optional = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=1,
            title="Marker analysis",
            method="Run rank_genes_groups and optionally create a volcano plot",
        )],
    )
    no_figure = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=2,
            title="Differential expression",
            method="Run pydeseq2",
            expected_output="Tabular result; no figure",
        )],
    )
    structured = AnalysisPlan(
        phase="innovation",
        steps=[PlanStep(
            id=3,
            title="Differential expression",
            method="Run pydeseq2 and optionally plot",
            figure_requirements=[{"figure_type": "volcano"}],
        )],
    )
    assert collect_plan_figure_requirements(optional) == []
    assert collect_plan_figure_requirements(no_figure) == []
    assert [req.figure_type for req in collect_plan_figure_requirements(structured)] == [
        "volcano"
    ]


def test_unknown_later_stage_method_gets_a_default_result_figure_contract():
    plan = AnalysisPlan(
        phase="innovation",
        stage="stage2",
        steps=[PlanStep(
            id=11,
            title="Discover latent response geometry",
            method="Run the custom manifold contrast and quantify its result",
        )],
    )

    requirements = collect_plan_figure_requirements(plan)

    assert [req.figure_type for req in requirements] == ["result"]
    assert requirements[0].wildcard_result_type is True


def test_legacy_text_only_opt_out_remains_for_pure_integration_synthesis():
    step = PlanStep.model_validate({
        "id": 12,
        "title": "Cross-direction synthesis",
        "method": "Reconcile previously reported correlation analysis evidence",
        "expected_output": "Narrative only; no figure",
    })
    plan = AnalysisPlan(phase="integration", stage="full", steps=[step])

    assert collect_plan_figure_requirements(plan) == []


def test_publication_figure_tool_uses_open_type_vocabulary():
    tool = next(
        item for item in TOOLS
        if item["function"]["name"] == "generate_publication_figure"
    )
    figure_type = tool["function"]["parameters"]["properties"]["figure_type"]
    assert "enum" not in figure_type
    assert "exact figure_type" in figure_type["description"]
    properties = tool["function"]["parameters"]["properties"]
    required = tool["function"]["parameters"]["required"]
    assert "logical_figure_id" in properties
    assert "logical_figure_id" in required
    assert "Filename suffixes are not revision metadata" in properties[
        "logical_figure_id"
    ]["description"]
