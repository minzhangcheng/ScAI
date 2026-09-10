"""Tests for Host evidence identity and PI-owned plan review."""
import base64
import hashlib
import itertools
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import (
    AnalysisPlan,
    AnalysisStep,
    AnalysisState,
    CodeExecution,
    FigureReviewStatus,
    PlanStep,
    PublicationFigure,
)
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes
from backend.workflow.post_execution_review import (
    PlanAudit,
    StepAudit,
    apply_audit_to_plan,
    audit_plan_execution,
    collect_failure_admissions,
    compute_review_attempt_id,
    empty_review_meta,
    executed_method_families,
    extract_planned_methods,
    sanitize_review_plan,
    sanitize_review_result,
    should_run_post_execution_review,
    step_is_conditional,
    unavailable_method_families,
)
from backend.workflow.post_execution_review_parts.reconciliation import (
    _apply_links,
    reconcile_plan_evidence,
)


_ARTIFACT_ROOT: Path | None = None
_ARTIFACT_SEQUENCE = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def _frozen_figure_artifact_root(tmp_path_factory):
    global _ARTIFACT_ROOT
    _ARTIFACT_ROOT = tmp_path_factory.mktemp("post-execution-review-artifacts")
    yield
    _ARTIFACT_ROOT = None


def _audited_plan():
    """A plan whose five steps exercise all five audit verdicts in order."""
    return AnalysisPlan(
        phase="innovation",
        stage="stage2",
        steps=[
            PlanStep(id=1, title="Cluster marker detection",
                     method="Run sc.tl.rank_genes_groups on the leiden clusters"),
            PlanStep(id=2, title="Compositional shift test",
                     method="Run scCODA on the per-cluster proportions"),
            PlanStep(id=3, title="Trajectory ordering",
                     method="Run slingshot pseudotime if the data support a continuum"),
            PlanStep(id=4, title="Cell-cell communication",
                     method="Run liana rank_aggregate on the annotated object"),
            PlanStep(id=5, title="Interpret the observed group differences",
                     method="Summarize the biological meaning of the comparison in prose"),
        ],
    )


def _audited_history():
    """Evidence matching `_audited_plan`: two families ran, one is uninstallable."""
    return [
        CodeExecution(code="sc.tl.rank_genes_groups(adata, 'leiden')", round_number=2),
        CodeExecution(code="import gseapy\ngseapy.prerank(rnk=rnk)", round_number=2),
        CodeExecution(code="import sccoda", round_number=2,
                      error="ModuleNotFoundError: No module named 'sccoda'"),
    ]


def _result_figure(plan: AnalysisPlan, step_id: int) -> PublicationFigure:
    assert _ARTIFACT_ROOT is not None
    artifact_id = next(_ARTIFACT_SEQUENCE)
    png_path = _ARTIFACT_ROOT / f"{artifact_id}-step-{step_id}.png"
    pdf_path = _ARTIFACT_ROOT / f"{artifact_id}-step-{step_id}.pdf"
    render = valid_image_bytes("PNG", min_bytes=2200)
    pdf = valid_pdf_bytes()
    png_path.write_bytes(render)
    pdf_path.write_bytes(pdf)
    return PublicationFigure(
        name=f"step_{step_id}_result",
        description=f"Reviewed visual result for step {step_id}",
        figure_type="other",
        png_path=str(png_path),
        pdf_path=str(pdf_path),
        png_base64=base64.b64encode(render).decode(),
        render_sha256=hashlib.sha256(render).hexdigest(),
        preview_sha256=hashlib.sha256(render).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        plan_step_id=step_id,
        plan_id=plan.id,
        purpose="result",
        round_number=2,
    )


def _audited_figures(plan: AnalysisPlan):
    # Step 1 ran and step 2 is unavailable after having produced its planned
    # context view. Step 3 is explicitly conditional, step 4 remains unmet, and
    # step 5 is prose-only legacy work.
    return [_result_figure(plan, 1), _result_figure(plan, 2)]


def _clean_audit():
    return PlanAudit(
        steps=[StepAudit(step_id=1, title="Marker detection", status="completed",
                         reason="evidenced", planned_families=["marker_de"])],
        executed_families=["marker_de"],
    )


def _unmet_audit():
    return PlanAudit(
        steps=[StepAudit(step_id=1, title="Cell-cell communication", status="pending",
                         reason="no evidence", planned_families=["cell_communication"],
                         missing_families=["cell_communication"])],
    )


def _receipt(plan: AnalysisPlan, **overrides) -> CodeExecution:
    values = {
        "id": "receipt-1",
        "code": "run_current_plan_step()",
        "round_number": 2,
        "evidence_schema_version": 2,
        "plan_id": plan.id,
        "plan_step_ids": [1],
        "execution_purpose": "planned",
        "plan_binding_status": "bound",
        "execution_status": "succeeded",
        "lineage_id": "main",
        "cohort_digest": "a" * 64,
    }
    values.update(overrides)
    return CodeExecution(**values)


def _semantic_reconciliation(plan: AnalysisPlan, links):
    return {
        "plan_id": plan.id,
        "round_number": 2,
        "lineage_id": "main",
        "cohort_digest": "a" * 64,
        "links": links,
    }


def test_supervisor_confirmation_completes_only_its_semantic_plan_step():
    plan = AnalysisPlan(
        id="receipt-plan",
        phase="innovation",
        steps=[
            PlanStep(id=1, title="First", method="Run custom method"),
            PlanStep(id=2, title="Second", method="Run another method"),
        ],
    )

    audit = audit_plan_execution(
        plan,
        [_receipt(plan)],
        round_number=2,
        target_cohort_digest="a" * 64,
        semantic_reconciliation=_semantic_reconciliation(plan, [{
            "plan_step_id": 1,
            "status": "evidenced",
            "execution_ids": ["receipt-1"],
        }]),
    )

    assert audit.evidence_mode == "supervisor_semantic"
    assert [step.status for step in audit.steps] == ["completed", "pending"]
    assert "receipt-1" in audit.steps[0].reason
    observations = audit.summary()["step_observations"]
    assert observations[0]["plan_step_id"] == 1
    assert observations[0]["technical_linkage"] == "evidenced"
    assert observations[1]["technical_linkage"] == "unverified"


def test_host_execution_linkage_is_observation_not_plan_completion():
    plan = AnalysisPlan(
        id="receipt-plan",
        phase="innovation",
        steps=[
            PlanStep(id=1, title="First", method="Run custom method"),
            PlanStep(id=2, title="Second", method="Run another method"),
        ],
    )

    audit = audit_plan_execution(
        plan,
        [_receipt(plan)],
        round_number=2,
        target_cohort_digest="a" * 64,
    )

    assert audit.evidence_mode == "host_execution_linkage"
    assert [step.status for step in audit.steps] == ["pending", "pending"]
    assert all(step.verifiable is False for step in audit.steps)
    observations = audit.summary()["step_observations"]
    assert observations[0]["technical_linkage"] == "execution_linked"
    assert observations[1]["technical_linkage"] == "not_linked"
    assert "receipt-1" in observations[0]["reason"]
    assert "PI review determines scientific completion" in observations[0]["reason"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"execution_purpose": "supplementary", "plan_binding_status": "supplementary"},
        {"execution_status": "failed", "error": "boom"},
        {"code": ""},
        {"cohort_digest": "b" * 64},
    ],
)
def test_ineligible_v2_receipt_never_falls_back_to_keyword_inference(overrides):
    plan = AnalysisPlan(
        id="strict-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Markers", method="sc.tl.rank_genes_groups")],
    )
    receipt_values = {
        "code": "sc.tl.rank_genes_groups(adata, 'cluster')",
        **overrides,
    }
    receipt = _receipt(plan, **receipt_values)

    audit = audit_plan_execution(
        plan,
        [receipt],
        round_number=2,
        target_cohort_digest="a" * 64,
    )

    assert audit.evidence_mode == "host_execution_linkage"
    assert audit.steps[0].status == "pending"
    assert audit.steps[0].verifiable is False


def test_semantic_reconciliation_cannot_cross_cohort_generations():
    plan = AnalysisPlan(
        id="generation-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    reconciliation = _semantic_reconciliation(plan, [{
        "plan_step_id": 1,
        "status": "evidenced",
        "execution_ids": ["receipt-1"],
    }])
    reconciliation["cohort_generation"] = 1

    audit = audit_plan_execution(
        plan,
        [_receipt(plan, cohort_generation=1)],
        round_number=2,
        target_cohort_digest="a" * 64,
        target_cohort_generation=2,
        semantic_reconciliation=reconciliation,
    )

    assert audit.steps[0].status == "pending"
    assert audit.steps[0].verifiable is False


@pytest.mark.asyncio
async def test_runtime_linkage_skips_supervisor_and_verifies_current_receipts():
    plan = AnalysisPlan(
        id="runtime-link-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    execution = _receipt(plan, cohort_generation=1)
    finding = {
        "id": "finding-1",
        "text": "A current execution-linked observation.",
        "round_number": 2,
        "lineage_id": "main",
        "plan_id": plan.id,
        "cohort_digest": "a" * 64,
        "cohort_generation": 1,
        "candidate_evidence_execution_ids": [execution.id],
    }
    state = AnalysisState(
        id="runtime-link-state",
        user_id="u",
        current_round=2,
        current_plan=plan,
        code_history=[execution],
        sandbox_state={
            "cohort_identity_v1": {
                "cohort_digest": "a" * 64,
                "cohort_generation": 1,
            }
        },
    )

    class ForbiddenSupervisor:
        async def review_plan_evidence(self, *_args, **_kwargs):
            raise AssertionError("Supervisor reconciliation must not be called")

    result = {
        "finding_records": [finding],
        "evidence_gate_degraded": False,
        "evidence_gate_findings": [{"gate": "supervisor-plan-evidence"}],
    }
    reconciliation = await reconcile_plan_evidence(
        SimpleNamespace(supervisor_agent=ForbiddenSupervisor(), session_id="s"),
        state,
        result,
    )

    assert reconciliation == {}
    assert state.plan_evidence_reconciliations == []
    assert result["evidence_verified"] is True
    assert result["evidence_execution_ids"] == [execution.id]
    assert result["evidence_gate_findings"] == []
    assert finding["evidence_status"] == "verified"
    assert finding["semantic_status"] == "unverified"
    assert finding["confirmed_plan_step_ids"] == []


@pytest.mark.asyncio
async def test_runtime_linkage_uses_full_current_execution_history():
    plan = AnalysisPlan(
        id="long-runtime-link-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    target = _receipt(plan, id="oldest-current-receipt", cohort_generation=1)
    later = [
        _receipt(plan, id=f"later-current-receipt-{index}", cohort_generation=1)
        for index in range(61)
    ]
    finding = {
        "id": "finding-oldest-receipt",
        "text": "A finding linked to an earlier current-plan execution.",
        "round_number": 2,
        "lineage_id": "main",
        "plan_id": plan.id,
        "cohort_digest": "a" * 64,
        "cohort_generation": 1,
        "candidate_evidence_execution_ids": [target.id],
    }
    state = AnalysisState(
        id="long-runtime-link-state",
        user_id="u",
        current_round=2,
        current_plan=plan,
        code_history=[target, *later],
        sandbox_state={"cohort_identity_v1": {
            "cohort_digest": "a" * 64,
            "cohort_generation": 1,
        }},
    )
    result = {"finding_records": [finding], "evidence_gate_findings": []}

    await reconcile_plan_evidence(
        SimpleNamespace(supervisor_agent=None, session_id="s"), state, result,
    )

    assert result["evidence_verified"] is True
    assert result["evidence_execution_ids"] == [target.id]


def test_host_execution_linkage_is_awaiting_pi_not_unverifiable():
    plan = AnalysisPlan(
        id="host-pi-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )

    summary = audit_plan_execution(
        plan,
        [_receipt(plan)],
        round_number=2,
        target_cohort_digest="a" * 64,
    ).summary()

    assert summary["n_unverifiable"] == 0
    assert summary["n_awaiting_pi_decision"] == 1
    assert summary["step_observations"][0]["awaiting_pi_decision"] is True


@pytest.mark.asyncio
async def test_runtime_linkage_does_not_pass_a_missing_execution_receipt():
    plan = AnalysisPlan(
        id="runtime-missing-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    finding = {
        "id": "finding-missing",
        "text": "An observation with a nonexistent receipt.",
        "round_number": 2,
        "lineage_id": "main",
        "plan_id": plan.id,
        "cohort_digest": "a" * 64,
        "cohort_generation": 1,
        "candidate_evidence_execution_ids": ["missing-execution"],
    }
    state = AnalysisState(
        id="runtime-missing-state",
        user_id="u",
        current_round=2,
        current_plan=plan,
        sandbox_state={
            "cohort_identity_v1": {
                "cohort_digest": "a" * 64,
                "cohort_generation": 1,
            }
        },
    )
    result = {
        "finding_records": [finding],
        "evidence_gate_degraded": False,
        "evidence_gate_findings": [],
    }

    await reconcile_plan_evidence(
        SimpleNamespace(supervisor_agent=None, session_id="s"),
        state,
        result,
    )

    assert result["evidence_verified"] is False
    assert result["evidence_execution_ids"] == []
    assert finding["evidence_status"] == "unverified"
    assert result["evidence_gate_degraded"] is False
    assert [item["gate"] for item in result["evidence_gate_findings"]] == [
        "execution-evidence-linkage"
    ]


@pytest.mark.asyncio
async def test_runtime_linkage_rejects_stale_analysis_generation():
    plan = AnalysisPlan(
        id="runtime-generation-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    execution = _receipt(plan, analysis_generation=0, cohort_generation=1)
    finding = {
        "id": "finding-stale-generation",
        "text": "An observation linked to stale analysis output.",
        "round_number": 2,
        "lineage_id": "main",
        "plan_id": plan.id,
        "analysis_generation": 1,
        "cohort_digest": "a" * 64,
        "cohort_generation": 1,
        "candidate_evidence_execution_ids": [execution.id],
    }
    state = AnalysisState(
        id="runtime-generation-state",
        user_id="u",
        current_round=2,
        current_plan=plan,
        code_history=[execution],
        sandbox_state={
            "analysis_generation": 1,
            "cohort_identity_v1": {
                "cohort_digest": "a" * 64,
                "cohort_generation": 1,
            },
        },
    )
    result = {"finding_records": [finding], "evidence_gate_findings": []}

    await reconcile_plan_evidence(
        SimpleNamespace(supervisor_agent=None, session_id="s"), state, result,
    )

    assert result["evidence_verified"] is False
    assert result["evidence_execution_ids"] == []


@pytest.mark.asyncio
async def test_runtime_linkage_rejects_explicit_failed_validation_output():
    plan = AnalysisPlan(
        id="runtime-validation-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    execution = _receipt(
        plan,
        output="Validation failed: required result was not produced.",
        cohort_generation=1,
    )
    finding = {
        "id": "finding-failed-validation",
        "text": "An observation linked to rejected output.",
        "round_number": 2,
        "lineage_id": "main",
        "plan_id": plan.id,
        "cohort_digest": "a" * 64,
        "cohort_generation": 1,
        "candidate_evidence_execution_ids": [execution.id],
    }
    state = AnalysisState(
        id="runtime-validation-state",
        user_id="u",
        current_round=2,
        current_plan=plan,
        code_history=[execution],
        sandbox_state={"cohort_identity_v1": {
            "cohort_digest": "a" * 64,
            "cohort_generation": 1,
        }},
    )
    result = {"finding_records": [finding], "evidence_gate_findings": []}

    await reconcile_plan_evidence(
        SimpleNamespace(supervisor_agent=None, session_id="s"), state, result,
    )

    assert result["evidence_verified"] is False
    assert result["evidence_execution_ids"] == []


@pytest.mark.asyncio
async def test_runtime_linkage_without_structured_findings_is_unknown():
    plan = AnalysisPlan(
        id="runtime-no-findings-plan",
        phase="innovation",
        steps=[PlanStep(id=1, title="Current", method="Run current analysis")],
    )
    state = AnalysisState(
        id="runtime-no-findings-state",
        user_id="u",
        current_round=2,
        current_plan=plan,
    )
    result = {"finding_records": [], "evidence_gate_findings": []}

    await reconcile_plan_evidence(
        SimpleNamespace(supervisor_agent=None, session_id="s"), state, result,
    )

    assert result["evidence_verified"] is None
    assert result["evidence_gate_findings"] == []


def test_unverified_reconciliation_is_advisory_not_degraded():
    state = type("State", (), {"analysis_log": []})()
    finding = {
        "id": "finding-1",
        "text": "A retained scientific observation.",
        "candidate_evidence_execution_ids": [],
    }
    result = {
        "finding_records": [finding],
        "evidence_gate_degraded": False,
        "evidence_gate_findings": [],
    }

    _apply_links(
        state,
        result,
        {
            "executions": [],
            "figures": [],
            "tables": [],
            "finding_records": [finding],
        },
        [{
            "plan_step_id": 1,
            "status": "unverified",
            "execution_ids": [],
            "finding_ids": ["finding-1"],
            "figure_ids": [],
            "table_ids": [],
            "reason": "Supervisor reconciliation was unavailable.",
        }],
        "plan-1",
    )

    assert result["evidence_verified"] is False
    assert result["evidence_gate_degraded"] is False
    assert result["evidence_gate_findings"] == [{
        "gate": "supervisor-plan-evidence",
        "rule_id": "scai-native/supervisor-plan-evidence",
        "detail": (
            "One or more findings could not be semantically reconciled "
            "to current execution evidence. They remain retained as "
            "unverified observations for PI and report disclosure."
        ),
        "subject": "plan-1",
        "severity": "advise",
        "rejection_count": 0,
        "rejection_limit": 0,
        "forced_release": False,
    }]


def test_attempt_id_is_stable_under_evidence_reordering():
    a = compute_review_attempt_id(3, "plan-1", "stage2", ["e2", "e1", "e3"], "completed")
    b = compute_review_attempt_id(3, "plan-1", "stage2", ["e3", "e1", "e2"], "completed")
    assert a == b
    assert isinstance(a, str) and len(a) == 64


def test_attempt_id_changes_with_inputs():
    base = compute_review_attempt_id(3, "plan-1", "stage2", ["e1"], "completed")
    assert compute_review_attempt_id(4, "plan-1", "stage2", ["e1"], "completed") != base
    assert compute_review_attempt_id(3, "plan-2", "stage2", ["e1"], "completed") != base
    assert compute_review_attempt_id(3, "plan-1", "full", ["e1"], "completed") != base
    assert compute_review_attempt_id(3, "plan-1", "stage2", ["e1"], "error") != base
    assert compute_review_attempt_id(3, "plan-1", "stage2", ["e1", "e2"], "completed") != base


def test_planned_methods_and_conditional_wording():
    # "regulons" here is prose describing decoupler's input, not a pySCENIC ask.
    # Matching the bare word made this step demand two families and report a
    # decoupler-only execution as unmet, so only tool identifiers may match.
    named = PlanStep(id=1, title="Pathway activity",
                     method="Run decoupler run_ulm against the collected regulons")
    assert extract_planned_methods(named) == {"pathway_activity"}
    assert step_is_conditional(named) is False

    scenic = PlanStep(id=1, title="Regulon activity",
                      method="Run pyscenic aucell to score regulon activity")
    assert extract_planned_methods(scenic) == {"regulon"}

    prose = PlanStep(id=2, title="Interpret the observed group differences",
                     method="Summarize the biological meaning of the comparison in prose")
    assert extract_planned_methods(prose) == set()

    hedged = PlanStep(id=3, title="Compositional shift",
                      method="Run scCODA if available",
                      expected_output="Credible effects if assumptions are reasonable")
    assert step_is_conditional(hedged) is True
    # The hedge lives in `expected_output` too, not only in `method`.
    assert step_is_conditional(PlanStep(id=4, title="t", method="Run scCODA",
                                        expected_output="proportions, if available")) is True


def test_later_required_method_occurrence_is_not_hidden_by_optional_first_use():
    step = PlanStep(
        id=5,
        title="Differential expression",
        method="Optionally run limma for QC, then run limma for DE.",
    )
    audit = audit_plan_execution(
        AnalysisPlan(phase="innovation", steps=[step]),
        [],
        round_number=2,
    )
    assert audit.steps[0].planned_families == ["bulk_de"]
    assert audit.steps[0].status == "pending"
    assert audit.steps[0].missing_families == ["bulk_de"]


def test_ordinary_plan_prose_commits_the_plan_to_nothing():
    """Only an identifier commits a step to a family; English never does.

    Bare "ingest" was both scanpy's label-transfer call and the verb every plan
    uses for loading data, and it was the only token two families claimed. So the
    first step of nearly every analysis silently owed both batch integration and
    reference mapping, audited unmet, and sent a supplementary pass after work
    nobody planned -- with `unmet_families` telling the PI to prioritise it.
    """
    for text in ("Ingest the filtered matrix and normalise",
                 "Data ingestion and quality control",
                 "Compare each cluster in isolation"):
        step = PlanStep(id=1, title=text, method="sc.pp.filter_cells(adata, min_genes=200)")
        assert extract_planned_methods(step) == set(), text


def test_reference_mapping_is_still_matched_when_named_as_a_call():
    step = PlanStep(id=1, title="Label transfer",
                    method="Run sc.tl.ingest against the reference atlas")
    assert extract_planned_methods(step) == {"reference_mapping"}
    assert executed_method_families(
        [CodeExecution(code="sc.tl.ingest(adata, adata_ref, obs='cell_type')",
                       round_number=1)], 1) == {"reference_mapping"}


def test_dotted_identifiers_survive_in_both_prose_and_code():
    """These carry punctuation and no underscore, so they are reachable only by
    substring. Tokenising the table -- the obvious repair when a bare word
    over-matches -- drops all six silently, and nothing else covers them."""
    for text, family in (("sc.tl.paga(adata)", "trajectory"),
                         ("sc.tl.dpt(adata)", "trajectory"),
                         ("milo.count_nhoods(adata)", "composition_model"),
                         ("sc.external.pp.harmony_integrate(adata)", "integration"),
                         ("dc.run_ulm(mat=mat, net=net)", "pathway_activity"),
                         ("li.mt.rank_aggregate(adata)", "cell_communication")):
        assert family in extract_planned_methods(
            PlanStep(id=1, title="t", method=text)), text
        assert family in executed_method_families(
            [CodeExecution(code=text, round_number=1)], 1), text


def test_executed_families_require_success_in_this_round():
    history = [
        CodeExecution(code="sc.tl.rank_genes_groups(adata, 'leiden')", round_number=2),
        CodeExecution(code="import pyscenic\nauc = aucell(ex_matrix, regulons)", round_number=2,
                      error="ValueError: regulon set is empty"),
        CodeExecution(code="import scvelo as scv", round_number=1),
    ]
    assert executed_method_families(history, 2) == {"marker_de"}
    # The round-1 execution vouches for round 1, and an unfiltered call sees both.
    assert "rna_velocity" in executed_method_families(history, 1)
    assert executed_method_families(history, 0) == {"marker_de", "rna_velocity"}


def test_unavailable_families_come_from_import_errors_across_rounds():
    history = [
        CodeExecution(code="import cell2location", round_number=1,
                      error="ModuleNotFoundError: No module named 'cell2location'"),
        CodeExecution(code="from infercnvpy import cnv", round_number=4,
                      error="ImportError: cannot import name 'cnv'"),
    ]
    assert unavailable_method_families(history) == {"deconvolution", "copy_number"}
    # A runtime failure is not an availability signal.
    assert unavailable_method_families(
        [CodeExecution(code="import pyscenic", error="ValueError: boom")]) == set()

    # A round-1 import error remains diagnostic, but cannot skip a plan step.
    plan = AnalysisPlan(phase="innovation", steps=[
        PlanStep(id=1, title="Deconvolve bulk profiles",
                 method="Run cell2location against the reference signatures")])
    audit = audit_plan_execution(plan, history, round_number=9)
    assert audit.steps[0].status == "pending"
    assert audit.steps[0].verifiable is False
    assert "legacy" in audit.steps[0].reason.lower()


def test_one_import_failure_does_not_condemn_its_neighbours():
    """The verdict is permanent, so it must be about what actually failed.

    A first cell that imports several packages before doing anything is the
    ordinary shape. Matching the whole cell source made one ImportError retire
    every family named anywhere in it for the rest of the analysis -- and the PI
    is instructed never to propose a method the audit lists as unavailable.
    """
    history = [CodeExecution(
        code="import gseapy\nimport liana\nimport cell2location\n"
             "sc.tl.rank_genes_groups(adata, 'leiden')",
        error="ModuleNotFoundError: No module named 'cell2location'",
    )]
    assert unavailable_method_families(history) == {"deconvolution"}


def test_a_failure_naming_no_module_falls_back_to_the_import_lines():
    """Not every ImportError names its module, and the signal is still real."""
    history = [CodeExecution(
        code="x = 1\nfrom infercnvpy import cnv\nresult = cnv.tl.infercnv(adata)",
        error="ImportError: cannot import name 'cnv'",
    )]
    assert unavailable_method_families(history) == {"copy_number"}


def test_the_fallback_does_not_reach_past_the_import_lines():
    """Only an import statement can be the subject of an import failure."""
    history = [CodeExecution(
        code="from infercnvpy import cnv\n# then hand the ranking to gseapy.prerank",
        error="ImportError: cannot import name 'cnv'",
    )]
    assert unavailable_method_families(history) == {"copy_number"}


def test_legacy_keyword_diagnostics_never_grant_or_deny_plan_credit():
    plan = _audited_plan()
    audit = audit_plan_execution(
        plan,
        _audited_history(),
        round_number=2,
        publication_figures=_audited_figures(plan),
    )
    by_id = {s.step_id: s for s in audit.steps}

    assert all(item.status == "pending" for item in by_id.values())
    assert all(item.verifiable is False for item in by_id.values())
    assert audit.unmet == []
    assert [s.step_id for s in audit.unverifiable] == [1, 2, 3, 4, 5]
    assert audit.executed_families == ["enrichment", "marker_de"]
    assert audit.unavailable_families == ["composition_model"]


def test_audit_summary_counts_agree():
    plan = _audited_plan()
    audit = audit_plan_execution(
        plan,
        _audited_history(),
        round_number=2,
        publication_figures=_audited_figures(plan),
    )
    summary = audit.summary()
    assert summary["n_steps"] == 5
    assert summary["n_completed"] == 0
    assert summary["n_skipped"] == 0
    assert summary["n_unmet"] == 0
    assert summary["n_unverifiable"] == 5
    assert (summary["n_completed"] + summary["n_skipped"]
            + summary["n_unmet"] + summary["n_unverifiable"]) == summary["n_steps"]
    assert summary["unmet_titles"] == []
    assert summary["unmet_families"] == []
    assert summary["executed_families"] == ["enrichment", "marker_de"]
    assert summary["unavailable_families"] == ["composition_model"]
    assert all(
        item["technical_linkage"] == "unverified"
        for item in summary["step_observations"]
    )


def test_structured_keyword_family_is_diagnostic_not_plan_credit():
    plan = AnalysisPlan(phase="innovation", steps=[
        PlanStep(id=1, title="Signature scoring",
                 method="Compute a module score with sc.tl.score_genes")])
    assert audit_plan_execution(plan, [], round_number=2).steps[0].status == "pending"
    vouched = audit_plan_execution(
        plan,
        [],
        round_number=2,
        structured_families={"gene_scoring"},
        publication_figures=[_result_figure(plan, 1)],
    )
    assert vouched.steps[0].status == "pending"
    assert vouched.steps[0].verifiable is False
    assert vouched.executed_families == ["gene_scoring"]


def test_audit_of_empty_plan_is_empty():
    assert audit_plan_execution(None, _audited_history()).steps == []
    assert audit_plan_execution(AnalysisPlan(phase="innovation"), _audited_history()).steps == []


def test_apply_audit_writes_status_and_stamps_only_completed():
    plan = AnalysisPlan(
        id="apply-plan",
        phase="innovation",
        steps=[
            PlanStep(id=1, title="First", method="Run current analysis"),
            PlanStep(id=2, title="Second", method="Run later analysis"),
        ],
    )
    audit = audit_plan_execution(
        plan,
        [_receipt(plan)],
        round_number=2,
        target_cohort_digest="a" * 64,
        semantic_reconciliation=_semantic_reconciliation(plan, [{
            "plan_step_id": 1,
            "status": "evidenced",
            "execution_ids": ["receipt-1"],
        }]),
    )
    updated = apply_audit_to_plan(plan, audit, now=datetime(2026, 1, 1, 12, 0, 0))

    assert updated == 2
    assert [s.status for s in plan.steps] == ["completed", "pending"]
    assert plan.steps[0].executed_at == datetime(2026, 1, 1, 12, 0, 0)
    assert all(step.executed_at is None for step in plan.steps[1:])


def test_apply_audit_is_idempotent():
    plan = AnalysisPlan(
        id="idempotent-plan",
        phase="innovation",
        steps=[
            PlanStep(id=1, title="First", method="Run current analysis"),
            PlanStep(id=2, title="Second", method="Run later analysis"),
        ],
    )
    audit = audit_plan_execution(
        plan,
        [_receipt(plan)],
        round_number=2,
        target_cohort_digest="a" * 64,
        semantic_reconciliation=_semantic_reconciliation(plan, [{
            "plan_step_id": 1,
            "status": "evidenced",
            "execution_ids": ["receipt-1"],
        }]),
    )
    first = datetime(2026, 1, 1, 12, 0, 0)
    apply_audit_to_plan(plan, audit, now=first)

    again = apply_audit_to_plan(plan, audit, now=datetime(2026, 6, 30, 8, 0, 0))
    assert again == 2
    assert plan.steps[0].executed_at == first
    assert [s.status for s in plan.steps] == ["completed", "pending"]


def test_apply_audit_without_verdicts_updates_nothing():
    plan = _audited_plan()
    assert apply_audit_to_plan(plan, PlanAudit()) == 0
    assert all(step.status == "pending" for step in plan.steps)


def test_admissions_from_findings_and_report_findings_log():
    findings = [
        "scCODA could not be run because the package is missing.",
        "Cluster 3 shows elevated signalling relative to cluster 5.",
        "scCODA could not be run because the package is missing.",
    ]
    log = [
        {"step_type": "report_findings",
         "findings_summary": "Trajectory inference was skipped for this round. "
                             "Marker detection completed cleanly."},
        {"step_type": "execute_code",
         "findings_summary": "This entry could not be run and must be ignored."},
    ]
    admissions = collect_failure_admissions(findings, log)
    assert admissions == ["scCODA could not be run because the package is missing.",
                          "Trajectory inference was skipped for this round."]

    # The helper also accepts AnalysisStep objects, which is the in-memory shape.
    step = AnalysisStep(step_number=7, step_type="report_findings",
                        findings_summary="We fell back to a simpler ranking "
                                         "because the model failed to converge.")
    assert collect_failure_admissions([], [step]) == [
        "We fell back to a simpler ranking because the model failed to converge."]
    assert collect_failure_admissions(
        [], [AnalysisStep(step_number=8, step_type="consult_pi",
                          findings_summary="Nothing could not be run here.")]) == []


def test_admissions_ignore_ordinary_prose_and_respect_the_cap():
    assert collect_failure_admissions(
        ["We clustered the cells and annotated every population.",
         "Marker genes were ranked per cluster and the tables were saved."], []) == []

    many = [f"Analysis module {i} could not be run in this session." for i in range(30)]
    assert len(collect_failure_admissions(many, [])) == 20


def test_review_fires_on_hard_structural_signals():
    for status in ("error", "timeout", "max_steps_reached", "completed_with_fallback"):
        run, reason = should_run_post_execution_review(_clean_audit(), [], status, False)
        assert run is True and status in reason

    run, reason = should_run_post_execution_review(
        _clean_audit(), ["scCODA could not be run."], "completed", False)
    assert run is True and "admission" in reason.lower()

    run, reason = should_run_post_execution_review(_unmet_audit(), [], "completed", False)
    assert run is True and "cell_communication" in reason

    # Signal 4 is "last direction of the batch", not "terminal round": whether an
    # analysis round follows depends on the integration cycle and the PI's next
    # direction choice, neither of which has happened when this is decided.
    run, reason = should_run_post_execution_review(_clean_audit(), [], "completed", True)
    assert run is True and "last direction" in reason.lower()


def test_clean_audit_is_still_shown_to_pi_for_completion_decision():
    run, reason = should_run_post_execution_review(_clean_audit(), [], "completed", False)
    assert run is True
    assert "pi review" in reason.lower()


def test_review_without_runtime_remains_observational():
    for audit, admissions, status, terminal in (
        (_clean_audit(), [], "completed", False),
        (_unmet_audit(), ["could not be run"], "error", True),
    ):
        run, reason = should_run_post_execution_review(
            audit, admissions, status, terminal, has_runtime=False)
        assert run is True
        assert "runtime" in reason.lower()


def test_review_gate_always_explains_itself():
    cases = [
        (_clean_audit(), [], "completed", False, True),
        (_clean_audit(), [], "error", False, True),
        (_clean_audit(), ["it could not be run"], "completed", False, True),
        (_unmet_audit(), [], "completed", False, True),
        (_clean_audit(), [], "completed", True, True),
        (_clean_audit(), [], "completed", False, False),
    ]
    for audit, admissions, status, terminal, has_runtime in cases:
        _, reason = should_run_post_execution_review(
            audit, admissions, status, terminal, has_runtime=has_runtime)
        assert reason.strip()


def test_sanitize_review_plan_caps_tasks_and_drops_blanks():
    plan = {
        "gaps": [f"gap {i}" for i in range(30)],
        "supplementary_tasks": ["", "   ", "Run the compositional test"]
                               + [f"task {i}" for i in range(10)],
        "out_of_scope": ["needs new data"],
        "verdict": "supplement",
        "rationale": "x" * 5000,
    }
    out = sanitize_review_plan(plan, 3)
    # Blanks are dropped BEFORE the cap applies, so the budget buys three real
    # tasks — an empty string the model emitted must not consume a slot.
    assert out["supplementary_tasks"] == ["Run the compositional test", "task 0", "task 1"]
    assert len(out["gaps"]) == 12
    assert out["verdict"] == "supplement"
    assert len(out["rationale"]) <= 2000


def test_sanitize_review_plan_blanks_do_not_consume_the_cap():
    out = sanitize_review_plan(
        {"supplementary_tasks": ["", "  ", "only real task"], "verdict": "supplement"}, 2)
    assert out["supplementary_tasks"] == ["only real task"]


def test_sanitize_review_plan_zero_budget_admits_no_task():
    out = sanitize_review_plan(
        {"supplementary_tasks": ["must not run"], "verdict": "supplement"}, 0
    )
    assert out["supplementary_tasks"] == []


def test_sanitize_review_plan_keeps_only_current_step_decisions():
    out = sanitize_review_plan({
        "step_decisions": [
            {"plan_step_id": 1, "status": "completed", "reason": "PI accepted"},
            {"plan_step_id": 9, "status": "completed", "reason": "foreign"},
            {"plan_step_id": 2, "status": "invalid", "reason": "bad syntax"},
        ],
    }, 0, valid_step_ids={1, 2})

    assert out["step_decisions"] == [{
        "plan_step_id": 1,
        "status": "completed",
        "reason": "PI accepted",
    }]


def test_sanitize_review_plan_keeps_all_current_step_decisions():
    decisions = [
        {"plan_step_id": step_id, "status": "completed", "reason": "PI accepted"}
        for step_id in range(1, 22)
    ]

    out = sanitize_review_plan(
        {"step_decisions": decisions, "verdict": "no_action"},
        0,
        valid_step_ids=set(range(1, 22)),
    )

    assert [item["plan_step_id"] for item in out["step_decisions"]] == list(
        range(1, 22)
    )


def test_sanitize_review_plan_requires_explicit_supplement_verdict():
    assert sanitize_review_plan({"supplementary_tasks": ["do it"], "verdict": "maybe"},
                                5)["verdict"] == "no_action"
    assert sanitize_review_plan({"supplementary_tasks": []}, 5)["verdict"] == "no_action"

    for bad in (None, "not a plan", [1, 2]):
        out = sanitize_review_plan(bad, 5)
        assert out["verdict"] == "no_action"
        assert out["supplementary_tasks"] == [] and out["gaps"] == []


def test_pi_execution_audit_formats_every_plan_step():
    from backend.agents.pi.execution_review import ExecutionReviewMixin

    observations = [
        {
            "plan_step_id": step_id,
            "technical_linkage": "unverified",
            "verifiable": False,
            "title": f"Step {step_id}",
            "reason": "Reconciliation unavailable",
        }
        for step_id in range(1, 22)
    ]

    rendered = ExecutionReviewMixin._format_execution_audit(
        {"n_steps": 21, "n_unverifiable": 21, "step_observations": observations},
        [],
    )

    assert "plan_step_id=21" in rendered
    assert "awaiting PI completion decision 21" in rendered


def test_pi_execution_audit_reports_the_scale_of_missing_figures():
    """The PI must see how many declared figures the round did not produce.

    `unmet_figure_kinds` names the types but not the scale, and the headline
    counters describe method evidence only. A round where 17 approved steps
    declared a figure and 15 produced none is exactly the observation the PI
    needs in order to decide whether to ask for that work again. It is reported
    as a fact, not enforced: the decision stays with the PI.
    """
    from backend.agents.pi.execution_review import ExecutionReviewMixin

    rendered = ExecutionReviewMixin._format_execution_audit(
        {
            "n_steps": 19,
            "n_awaiting_pi_decision": 19,
            "n_figure_steps": 17,
            "n_figure_unmet": 15,
            "unmet_figure_kinds": ["heatmap", "umap"],
            "step_observations": [],
        },
        [],
    )

    assert "Required figures: 17 approved step(s) declared one" in rendered
    assert "15 produced no linked figure this round" in rendered


def test_pi_execution_audit_omits_the_figure_line_when_none_were_declared():
    """A plan that declared no figure deliverable gets no figure observation."""
    from backend.agents.pi.execution_review import ExecutionReviewMixin

    rendered = ExecutionReviewMixin._format_execution_audit(
        {
            "n_steps": 3,
            "n_awaiting_pi_decision": 3,
            "n_figure_steps": 0,
            "n_figure_unmet": 0,
            "step_observations": [],
        },
        [],
    )

    assert "Required figures:" not in rendered


def test_sanitize_review_result_bounds_lists_and_survives_non_dict():
    out = sanitize_review_result({
        "status": "completed",
        "summary": "y" * 9000,
        "findings": [f"finding {i}" for i in range(60)],
        "figures": [f"fig{i}.png" for i in range(30)],
        "warnings": [f"warn {i}" for i in range(30)],
    })
    assert len(out["summary"]) <= 4000
    assert len(out["findings"]) == 40
    assert out["figures"] == []
    assert len(out["warnings"]) == 12
    assert any("discarded" in warning.lower() for warning in out["warnings"])
    assert out["evidence_records"] == []

    bad = sanitize_review_result(None)
    assert bad["status"] == "failed"
    assert bad["warnings"] == ["non-dict result"]
    assert bad["findings"] == [] and bad["figures"] == []
    assert bad["evidence_records"] == []


def _long_review_diagnostic_traceback():
    frames = "\n".join(
        f'  File "/workspace/project/frame_{index}.py", line {index}, in helper\n'
        f"    frame_{index}()"
        for index in range(120)
    )
    return (
        "Traceback (most recent call last):\n"
        f"{frames}\n"
        "TypeError: boolean value of NA is ambiguous"
    )


def test_sanitize_review_result_bounds_evidence_records_and_drops_raw_values():
    from backend.models.schemas import CodeExecution

    stdout = "stdout-head-" + ("x" * 2200) + "-stdout-tail"
    records = [
        "malformed",
        CodeExecution(code="print('raw')", output="raw output"),
        {
            "execution_id": "e" * 140,
            "status": "completed",
            "stdout_excerpt": stdout,
            "error": _long_review_diagnostic_traceback(),
            "code": "print('must not cross the review boundary')",
        },
    ]
    records.extend({
        "execution_id": f"execution-{index}",
        "status": "failed",
        "stdout_excerpt": "failed output",
        "error": "RuntimeError: failed",
    } for index in range(20))

    out = sanitize_review_result({
        "evidence_records": records,
        "code_history": [CodeExecution(code="raw")],
        "analysis_log": [{"code": "raw"}],
    })

    evidence = out["evidence_records"]
    assert len(evidence) == 12
    first = evidence[0]
    assert first["execution_id"] == "e" * 100
    assert first["status"] == "completed"
    assert len(first["stdout_excerpt"]) <= 2000
    assert first["stdout_excerpt"].endswith("-stdout-tail")
    assert len(first["error"]) <= 1000
    assert "TypeError: boolean value of NA is ambiguous" in first["error"]
    assert evidence[1]["execution_id"] == "execution-0"
    assert evidence[1]["status"] == "failed"
    assert "code_history" not in out and "analysis_log" not in out
    assert all(set(record) == {
        "execution_id", "status", "stdout_excerpt", "error"
    } for record in evidence)
    import json
    json.dumps(out)


def test_sanitize_review_result_drops_non_terminal_evidence_statuses():
    out = sanitize_review_result({
        "evidence_records": [
            {"execution_id": "running", "status": "running"},
            {"execution_id": "success", "status": "success"},
            {"execution_id": "empty", "status": ""},
            {"execution_id": "completed", "status": "completed"},
            {"execution_id": "failed", "status": "failed"},
        ],
    })

    assert out["evidence_records"] == [
        {"execution_id": "completed", "status": "completed", "stdout_excerpt": "", "error": ""},
        {"execution_id": "failed", "status": "failed", "stdout_excerpt": "", "error": ""},
    ]


def test_supplementary_warning_retains_the_exception():
    out = sanitize_review_result({
        "warnings": [_long_review_diagnostic_traceback()],
    })

    assert len(out["warnings"][0]) <= 500
    assert "TypeError: boolean value of NA is ambiguous" in out["warnings"][0]


def test_supplementary_warning_survives_the_discarded_figure_prefix():
    out = sanitize_review_result({
        "figures": ["supplementary.png"],
        "warnings": [_long_review_diagnostic_traceback()],
    })

    assert any("discarded" in warning.lower() for warning in out["warnings"])
    assert any(
        "TypeError: boolean value of NA is ambiguous" in warning
        for warning in out["warnings"]
    )


def test_plan_prose_is_still_head_sliced():
    rationale = "plan-head-" + ("x" * 2200) + "-plan-tail"

    out = sanitize_review_plan(
        {
            "rationale": rationale,
            "supplementary_tasks": [rationale],
            "gaps": [rationale],
            "out_of_scope": [rationale],
        },
        1,
    )

    assert out["rationale"] == rationale[:2000]
    assert out["rationale"].startswith("plan-head-")
    assert "plan-tail" not in out["rationale"]
    assert out["supplementary_tasks"][0] == rationale[:600]
    assert out["gaps"][0] == rationale[:500]
    assert out["out_of_scope"][0] == rationale[:500]


def test_evidence_gap_labels_describe_vocabulary_not_feasibility():
    """The labels say what the text mentions. Measured on all 542 real items, a
    feasibility reading is wrong most of the time, so the two things a caller may
    rely on are: measurement vocabulary wins over computational vocabulary, and a
    "replicate-aware" request counts as new measurement even when it also names a
    computation."""
    from backend.workflow.post_execution_review import (
        GAP_MENTIONS_COMPUTATION,
        GAP_MENTIONS_NEW_MEASUREMENT,
        GAP_UNLABELLED,
        label_evidence_gap,
    )

    assert label_evidence_gap("Bulk concordance for canonical marker genes") == GAP_MENTIONS_COMPUTATION
    assert label_evidence_gap("Western blot validation of the protein") == GAP_MENTIONS_NEW_MEASUREMENT
    # Measurement wins even when a computation is named in the same sentence.
    assert label_evidence_gap(
        "Orthogonal protein validation of the correlation") == GAP_MENTIONS_NEW_MEASUREMENT
    # The dominant real-world false positive: separated words, plus a computation.
    assert label_evidence_gap(
        "Replicated snRNA libraries per condition for formal composition testing"
    ) == GAP_MENTIONS_NEW_MEASUREMENT
    assert label_evidence_gap("Something entirely unrelated") == GAP_UNLABELLED
    assert label_evidence_gap("") == GAP_UNLABELLED


def test_collect_gaps_accepts_dicts_and_objects_and_bounds_itself():
    from types import SimpleNamespace

    from backend.workflow.post_execution_review import collect_structured_evidence_gaps

    programs = [
        {"label": "dict-form", "evidence_execution_ids": ["e1"],
         "missing_evidence": ["Bulk concordance check", "  ", "qPCR validation"]},
        SimpleNamespace(label="object-form", evidence_execution_ids=[],
                        missing_evidence=["Ambient RNA sensitivity analysis"]),
    ]
    out = collect_structured_evidence_gaps(programs)

    assert out["programs_total"] == 2
    assert out["programs_with_gaps"] == 2
    assert out["programs_without_evidence_ids"] == ["object-form"]
    assert len(out["items"]) == 3, "blank gap text is dropped"
    assert sum(out["counts"].values()) == 3

    many = [{"label": f"p{i}", "evidence_execution_ids": ["e"],
             "missing_evidence": [f"gap {j}" for j in range(20)]} for i in range(10)]
    assert len(collect_structured_evidence_gaps(many)["items"]) <= 60

    assert collect_structured_evidence_gaps([])["programs_total"] == 0


def test_empty_review_meta_shape():
    meta = empty_review_meta("skipped", "nothing to do", attempt_id="abc")
    assert meta["review_attempt_id"] == "abc"
    assert meta["review_status"] == "skipped"
    assert meta["review_reason"] == "nothing to do"
    assert meta["review_plan"] == {} and meta["review_result"] == {}
    assert meta["plan_audit"] == {} and meta["supplementary_findings"] == []
