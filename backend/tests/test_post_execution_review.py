"""Tests for the deterministic layer of the post-execution review round.

The audit is the only writer of `PlanStep.status`, and a false "unmet" verdict
sends a supplementary pass after work that already ran. The verdict tests below
therefore pin each of the five outcomes -- especially the two exemptions
(conditional wording, missing tooling) and the unverifiable case, which must
never be counted as a gap.
"""
from datetime import datetime

from backend.models.schemas import AnalysisPlan, AnalysisStep, CodeExecution, PlanStep
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

    # A round-1 import error still explains a round-9 step's absence.
    plan = AnalysisPlan(phase="innovation", steps=[
        PlanStep(id=1, title="Deconvolve bulk profiles",
                 method="Run cell2location against the reference signatures")])
    audit = audit_plan_execution(plan, history, round_number=9)
    assert audit.steps[0].status == "skipped"
    assert "unavailable" in audit.steps[0].reason.lower()


def test_audit_assigns_all_five_verdicts():
    audit = audit_plan_execution(_audited_plan(), _audited_history(), round_number=2)
    by_id = {s.step_id: s for s in audit.steps}

    assert by_id[1].status == "completed"
    assert by_id[1].missing_families == []

    assert by_id[2].status == "skipped"
    assert "tooling unavailable" in by_id[2].reason.lower()

    assert by_id[3].status == "skipped"
    assert "conditional" in by_id[3].reason.lower()

    assert by_id[4].status == "pending"
    assert by_id[4].verifiable is True
    assert by_id[4].missing_families == ["cell_communication"]

    assert by_id[5].status == "pending"
    assert by_id[5].verifiable is False

    # Only step 4 is a gap: the unverifiable step must never be reported as one.
    assert [s.step_id for s in audit.unmet] == [4]
    assert [s.step_id for s in audit.unverifiable] == [5]


def test_audit_summary_counts_agree():
    audit = audit_plan_execution(_audited_plan(), _audited_history(), round_number=2)
    summary = audit.summary()
    assert summary["n_steps"] == 5
    assert summary["n_completed"] == 1
    assert summary["n_skipped"] == 2
    assert summary["n_unmet"] == 1
    assert summary["n_unverifiable"] == 1
    assert (summary["n_completed"] + summary["n_skipped"]
            + summary["n_unmet"] + summary["n_unverifiable"]) == summary["n_steps"]
    assert summary["unmet_titles"] == ["Cell-cell communication"]
    assert summary["unmet_families"] == ["cell_communication"]
    assert summary["executed_families"] == ["enrichment", "marker_de"]
    assert summary["unavailable_families"] == ["composition_model"]


def test_audit_accepts_structured_evidence():
    plan = AnalysisPlan(phase="innovation", steps=[
        PlanStep(id=1, title="Signature scoring",
                 method="Compute a module score with sc.tl.score_genes")])
    assert audit_plan_execution(plan, [], round_number=2).steps[0].status == "pending"
    vouched = audit_plan_execution(plan, [], round_number=2,
                                   structured_families={"gene_scoring"})
    assert vouched.steps[0].status == "completed"


def test_audit_of_empty_plan_is_empty():
    assert audit_plan_execution(None, _audited_history()).steps == []
    assert audit_plan_execution(AnalysisPlan(phase="innovation"), _audited_history()).steps == []


def test_apply_audit_writes_status_and_stamps_only_completed():
    plan = _audited_plan()
    audit = audit_plan_execution(plan, _audited_history(), round_number=2)
    updated = apply_audit_to_plan(plan, audit, now=datetime(2026, 1, 1, 12, 0, 0))

    assert updated == 5
    assert [s.status for s in plan.steps] == ["completed", "skipped", "skipped",
                                              "pending", "pending"]
    assert plan.steps[0].executed_at == datetime(2026, 1, 1, 12, 0, 0)
    assert all(step.executed_at is None for step in plan.steps[1:])


def test_apply_audit_is_idempotent():
    plan = _audited_plan()
    audit = audit_plan_execution(plan, _audited_history(), round_number=2)
    first = datetime(2026, 1, 1, 12, 0, 0)
    apply_audit_to_plan(plan, audit, now=first)

    again = apply_audit_to_plan(plan, audit, now=datetime(2026, 6, 30, 8, 0, 0))
    assert again == 5
    assert plan.steps[0].executed_at == first
    assert [s.status for s in plan.steps] == ["completed", "skipped", "skipped",
                                              "pending", "pending"]


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


def test_review_declines_when_clean_and_a_later_round_remains():
    run, reason = should_run_post_execution_review(_clean_audit(), [], "completed", False)
    assert run is False
    assert reason


def test_review_never_fires_without_a_runtime():
    for audit, admissions, status, terminal in (
        (_clean_audit(), [], "completed", False),
        (_unmet_audit(), ["could not be run"], "error", True),
    ):
        run, reason = should_run_post_execution_review(
            audit, admissions, status, terminal, has_runtime=False)
        assert run is False
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


def test_sanitize_review_plan_coerces_verdict_and_survives_non_dict():
    assert sanitize_review_plan({"supplementary_tasks": ["do it"], "verdict": "maybe"},
                                5)["verdict"] == "supplement"
    assert sanitize_review_plan({"supplementary_tasks": []}, 5)["verdict"] == "no_action"

    for bad in (None, "not a plan", [1, 2]):
        out = sanitize_review_plan(bad, 5)
        assert out["verdict"] == "no_action"
        assert out["supplementary_tasks"] == [] and out["gaps"] == []


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
    assert len(out["figures"]) == 12
    assert len(out["warnings"]) == 12

    bad = sanitize_review_result(None)
    assert bad["status"] == "failed"
    assert bad["warnings"] == ["non-dict result"]
    assert bad["findings"] == [] and bad["figures"] == []


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
