"""Post-execution review orchestration.

Covers the three contracts the round depends on: the plan audit is the writer of
`PlanStep.status`, the supplementary pass is paid for at most once per execution,
and neither can fail the round it reviews.
"""
import base64
import hashlib
import pytest
from types import SimpleNamespace

from backend.models.schemas import (
    AnalysisStep,
    CodeExecution,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.models.schema_parts.session import AnalysisPlan, PlanStep
from backend.workflow.post_execution import PostExecutionReviewMixin
from backend.tests.figure_image_helpers import valid_image_bytes


_RENDER_PREVIEW = base64.b64encode(valid_image_bytes("PNG")).decode()
_RENDER_PREVIEW_SHA256 = hashlib.sha256(
    base64.b64decode(_RENDER_PREVIEW)
).hexdigest()
_PDF_SHA256 = "0" * 64


class _Harness(PostExecutionReviewMixin):
    """Minimal host for the mixin: only the collaborators it actually reaches."""

    def __init__(self, pi=None, bio=None, config=None, state=None):
        self.session_id = "sess-test"
        self.pi_agent = pi
        self.bioinformatics_agent = bio
        self.config = config or SimpleNamespace(
            enable_post_execution_review=True,
            post_execution_review_max_tasks=5,
            post_execution_review_max_iterations=5,
            post_execution_review_timeout_seconds=900,
            max_rounds=3,
        )
        self.sent = []
        self._current_analysis_state = state

    async def _send_message(self, kind, payload, agent=None):
        self.sent.append((kind, payload, agent))


def _plan(steps):
    return AnalysisPlan(phase="innovation", steps=steps)


def _state(plan=None, **kwargs):
    defaults = dict(
        current_round=2,
        max_rounds=3,
        current_plan=plan,
        current_round_direction="test direction",
        code_history=[],
        analysis_log=[],
        score_definitions=[],
        directions_queue=[],
        pending_integration=False,
        integration_cycle=0,
        max_integration_cycles=1,
        post_execution_reviews=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _execution(code, error=None, round_number=2, exec_id=None, modalities=None):
    kwargs = {"code": code, "error": error, "round_number": round_number}
    if exec_id:
        kwargs["id"] = exec_id
    if modalities is not None:
        kwargs["analysis_modalities"] = modalities
    return CodeExecution(**kwargs)


def _result_figure(
    plan,
    step_id,
    *,
    round_number=2,
    name=None,
    figure_type="other",
):
    name = name or f"step_{step_id}_result"
    return PublicationFigure(
        name=name,
        description=f"Reviewed result for plan step {step_id}",
        figure_type=figure_type,
        png_path=f"/missing/{name}.png",
        pdf_path=f"/missing/{name}.pdf",
        png_base64=_RENDER_PREVIEW,
        preview_sha256=_RENDER_PREVIEW_SHA256,
        pdf_sha256=_PDF_SHA256,
        review_status=FigureReviewStatus.APPROVED,
        plan_step_id=step_id,
        plan_id=plan.id,
        purpose="result",
        round_number=round_number,
    )


# ---------------------------------------------------------------------------
# Feature A — the audit writes plan step status
# ---------------------------------------------------------------------------

def test_audit_writes_step_status_and_stamps_only_completed():
    plan = _plan([
        PlanStep(id=1, title="Cell communication", method="Run liana rank_aggregate"),
        PlanStep(id=2, title="Pathway activity", method="Run decoupler run_ulm"),
    ])
    state = _state(
        plan,
        code_history=[_execution("import liana as li; li.mt.rank_aggregate(adata)")],
        publication_figures=[_result_figure(plan, 1)],
    )

    summary = _Harness()._audit_round_plan_execution(state, {})

    assert summary["n_steps"] == 2
    assert summary["n_completed"] == 1
    assert summary["n_unmet"] == 1
    by_id = {s.id: s for s in plan.steps}
    assert by_id[1].status == "completed" and by_id[1].executed_at is not None
    assert by_id[2].status == "pending" and by_id[2].executed_at is None


def test_audit_wiring_passes_current_publication_figures():
    plan = _plan([
        PlanStep(id=1, title="Annotated embedding", method="Create a UMAP"),
    ])
    figure = PublicationFigure(
        name="annotated_umap",
        description="Annotated cell-type UMAP",
        figure_type="umap",
        png_path="/tmp/annotated_umap.png",
        pdf_path="/tmp/annotated_umap.pdf",
        png_base64=_RENDER_PREVIEW,
        preview_sha256=_RENDER_PREVIEW_SHA256,
        pdf_sha256=_PDF_SHA256,
        review_status=FigureReviewStatus.APPROVED,
        round_number=2,
    )
    state = _state(plan, publication_figures=[figure])

    summary = _Harness()._audit_round_plan_execution(state, {})

    assert summary["n_figure_steps"] == 1
    assert summary["n_figure_unmet"] == 0
    assert summary["eligible_figures"] == ["annotated_umap"]
    assert plan.steps[0].status == "completed"


def test_audit_is_noop_without_a_plan():
    harness = _Harness()
    assert harness._audit_round_plan_execution(_state(None), {}) == {}
    assert harness._audit_round_plan_execution(_state(_plan([])), {}) == {}


def test_audit_never_raises_on_a_broken_plan():
    class _Exploding:
        @property
        def steps(self):
            raise RuntimeError("boom")

    # A plan that raises on access must degrade to "no audit", not fail the round.
    state = _state(None)
    state.current_plan = _plan([PlanStep(id=1, title="t", method="run liana")])
    harness = _Harness()
    harness._structured_method_families = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert harness._audit_round_plan_execution(state, {}) == {}


def test_evidence_gaps_are_recorded_but_never_shown_to_the_pi():
    """Step 1 is measurement: recording gaps must not change what the round
    concludes, so the observation must not enter the PI's synthesis context."""
    import re
    import backend.workflow.synthesis as syn_mod

    plan = _plan([PlanStep(id=1, title="Communication", method="Run liana rank_aggregate")])
    state = _state(plan, code_history=[_execution("li.mt.rank_aggregate(adata)")])
    result = {
        "findings": [],
        "mechanism_programs": [
            {"label": "P1", "evidence_execution_ids": ["e1"],
             "missing_evidence": ["Bulk concordance for the reported signature",
                                  "Western blot validation of the protein"]},
            {"label": "P2", "evidence_execution_ids": [], "missing_evidence": []},
        ],
    }

    summary = _Harness()._audit_round_plan_execution(state, result)
    gaps = summary["evidence_gaps"]

    assert gaps["programs_total"] == 2
    assert gaps["programs_with_gaps"] == 1
    assert gaps["programs_without_evidence_ids"] == ["P2"]
    assert gaps["counts"]["mentions_computation"] == 1
    assert gaps["counts"]["mentions_new_measurement"] == 1
    # Verbatim text is kept: the labels are not a feasibility verdict, so the item
    # itself is the part a later calibration needs.
    assert any(i["gap"].startswith("Bulk concordance") for i in gaps["items"])

    # The PI's bioinformatics context enumerates its keys explicitly; evidence_gaps
    # must not be among them.
    source = open(syn_mod.__file__).read()
    fn = re.search(r"async def _pi_synthesis\b.*?(?=\n    (?:async )?def )", source, re.DOTALL)
    assert "evidence_gaps" not in fn.group(0), "observation leaked into the PI prompt"


def test_no_mechanism_programs_records_nothing():
    state = _state(_plan([PlanStep(id=1, title="t", method="Run liana rank_aggregate")]))
    summary = _Harness()._audit_round_plan_execution(state, {"mechanism_programs": []})
    assert "evidence_gaps" not in summary


def test_registered_score_vouches_for_gene_scoring():
    plan = _plan([PlanStep(id=1, title="Module scores", method="Use score_genes for the panel")])
    state = _state(
        plan,
        score_definitions=[SimpleNamespace(round_number=2)],
        publication_figures=[_result_figure(plan, 1)],
    )

    summary = _Harness()._audit_round_plan_execution(state, {})

    assert summary["n_completed"] == 1
    assert plan.steps[0].status == "completed"


# ---------------------------------------------------------------------------
# Evidence binding must survive the node's result dict
# ---------------------------------------------------------------------------

def test_final_result_carries_evidence_binding_keys():
    """The fixed-key result dict is the only writer of workflow-state
    `bioinformatics_result`. Three consumers test `evidence_verified is False`, so
    dropping the key made every round read as verified."""
    import re
    import backend.workflow.agents as agents_mod

    source = open(agents_mod.__file__).read()
    block = re.search(r"final_result = \{.*?\n        \}", source, re.DOTALL)
    assert block, "could not locate the final_result literal"
    body = block.group(0)
    assert '"evidence_verified"' in body
    assert '"evidence_gate_degraded"' in body
    assert '"evidence_execution_ids"' in body
    assert '"evidence_gate_findings"' in body


def _healthy_round(evidence_verified):
    """A round where every other completion condition is satisfied, so the only
    variable is whether the findings were bound to successful executions."""
    from backend.models.schemas import RoundResult

    state = {
        "bioinformatics_result": {"findings": ["a finding"], "status": "completed",
                                  "evidence_verified": evidence_verified},
        "literature_result": {"findings": ["a citation"], "status": "completed"},
    }
    result = RoundResult(round_number=2, bioinformatics_findings=["a finding"],
                         literature_findings=["a citation"])
    return state, result


def test_evidence_gate_fires_when_binding_failed():
    """The completeness gate withholds direction-coverage eligibility from a round
    whose findings are not bound to successful executions. It could not fire at all
    while the key was being dropped from the node's result dict."""
    from backend.workflow.synthesis import derive_round_completion_status

    result = derive_round_completion_status(*_healthy_round(False))

    assert result.round_status == "partial"
    assert result.direction_coverage_eligible is False


def test_evidence_gate_does_not_fire_when_binding_succeeded():
    from backend.workflow.synthesis import derive_round_completion_status

    result = derive_round_completion_status(*_healthy_round(True))

    assert result.round_status == "completed"
    assert result.direction_coverage_eligible is True


def test_unknown_evidence_binding_is_not_reported_as_verified():
    """`None` means the binding was never recorded. The note handed to the PI must
    not claim the findings are linked to successful executions."""
    import re
    import backend.workflow.synthesis as syn_mod

    source = open(syn_mod.__file__).read()
    fn = re.search(r"async def _pi_synthesis\b.*?(?=\n    (?:async )?def )", source, re.DOTALL)
    assert fn, "could not locate _pi_synthesis"
    body = fn.group(0)
    # A two-branch form on `is False` is exactly the bug: it maps None to verified.
    assert "elif evidence_verified is False:" in body, "the note must handle all three states"
    assert "unknown" in body.lower()


# ---------------------------------------------------------------------------
# Budget derivation
# ---------------------------------------------------------------------------

def _limits_for(**cfg):
    harness = _Harness()
    harness.config = SimpleNamespace(**cfg)
    return harness._post_execution_review_limits()


def test_step_budget_is_derived_so_it_never_cuts_off_a_single_analysis():
    """A supplementary step can legitimately be a long analysis, and the sandbox —
    not this budget — is what bounds and interrupts the cell. The derived per-step
    value must therefore exceed sandbox_timeout, never sit under it."""
    iterations, step, total, _ = _limits_for(
        post_execution_review_max_iterations=5, post_execution_review_max_tasks=5,
        post_execution_review_timeout_seconds=0, sandbox_timeout=5400)
    assert iterations == 5
    assert step == 5400 + 120
    assert step > 5400, "the ceiling must not be the thing that cuts a cell off"
    assert total == step * 5, "the total defaults to the sum of the step budgets"

    # Derivation follows sandbox_timeout rather than a constant of its own.
    _, small_step, _, _ = _limits_for(
        post_execution_review_max_iterations=2, post_execution_review_max_tasks=5,
        post_execution_review_timeout_seconds=0, sandbox_timeout=600)
    assert small_step == 600 + 120


def test_explicit_budgets_override_the_derivation():
    _, step, total, _ = _limits_for(
        post_execution_review_max_iterations=3, post_execution_review_max_tasks=5,
        post_execution_review_timeout_seconds=1200,
        post_execution_review_total_timeout_seconds=2000, sandbox_timeout=5400)
    assert (step, total) == (1200, 2000)


def test_total_is_floored_at_one_step_budget():
    """A total under one step re-creates the bug this split fixes: the pass ceiling
    aborting a single legitimate analysis."""
    _, step, total, _ = _limits_for(
        post_execution_review_max_iterations=5, post_execution_review_max_tasks=5,
        post_execution_review_timeout_seconds=0,
        post_execution_review_total_timeout_seconds=60, sandbox_timeout=5400)
    assert total == step


def test_limits_survive_a_config_missing_sandbox_timeout():
    iterations, step, total, tasks = _limits_for(
        post_execution_review_max_iterations=99, post_execution_review_max_tasks=99,
        post_execution_review_timeout_seconds=0)
    assert iterations == 8, "iterations are hard-capped"
    assert tasks == 10, "task budget is hard-capped"
    assert step > 0 and total >= step


def test_timeout_interrupts_the_kernel_and_never_raises():
    """Cancelling the wait does not stop a dispatched cell, so the ceiling must
    interrupt the kernel itself or it leaks a running execution."""
    from backend.agents.bioinformatics.supplementary import SupplementaryMixin

    class _Agent(SupplementaryMixin):
        def __init__(self, km):
            self.session_id = "s"
            self.sandbox = SimpleNamespace(km=km)

    calls = []
    _Agent(SimpleNamespace(interrupt_kernel=lambda: calls.append(1)))._interrupt_kernel_after_timeout()
    assert calls == [1]

    def _boom():
        raise RuntimeError("kernel gone")

    # Neither a failing interrupt nor a missing kernel may raise into the round.
    _Agent(SimpleNamespace(interrupt_kernel=_boom))._interrupt_kernel_after_timeout()
    _Agent(None)._interrupt_kernel_after_timeout()


# ---------------------------------------------------------------------------
# Last-direction detection
# ---------------------------------------------------------------------------

def _batch_state(queue, current_direction, **kwargs):
    return _state(directions_queue=queue, current_round_direction=current_direction, **kwargs)


def test_last_direction_ignores_the_entry_this_round_is_running():
    """`directions_queue` is popped one round LATE — planning peeks, the next
    round's _initialize_round pops — so the running direction is still queued
    while it runs. A bare non-empty-queue test never fired on a last direction,
    which is the stale-entry trap 0.30.4 already fixed once for the terminal gate."""
    harness = _Harness()

    # Two-direction batch: not last while D2 is still ahead, last once running D2.
    assert harness._is_last_direction_round_of_batch(
        _batch_state([{"direction": "D2"}], "D1")) is False
    assert harness._is_last_direction_round_of_batch(
        _batch_state([{"direction": "D2"}], "D2")) is True

    # The two queue writers disagree on whether the running entry is present:
    # _pi_planning stores selected_dirs[1:], the user-selection route stores all.
    assert harness._is_last_direction_round_of_batch(
        _batch_state([{"direction": "U1"}], "U1")) is True
    assert harness._is_last_direction_round_of_batch(
        _batch_state([{"direction": "U1"}, {"direction": "U2"}], "U1")) is False


def test_last_direction_does_not_use_the_round_budget():
    """The integration branch budgets one round MORE than the batch it plans
    (the spare slot is the integration round itself), so `current_round >=
    max_rounds` is false on every round of every integration batch — with an
    empty queue included. The predicate must not consult it."""
    harness = _Harness()
    assert harness._is_last_direction_round_of_batch(
        _batch_state([], "D5", current_round=5, max_rounds=6)) is True
    assert harness._is_last_direction_round_of_batch(
        _batch_state([], "D1", current_round=1, max_rounds=8)) is True


def test_last_direction_ignores_blank_queue_entries():
    harness = _Harness()
    assert harness._is_last_direction_round_of_batch(
        _batch_state([{"direction": "  "}, {}], "D1")) is True


# ---------------------------------------------------------------------------
# The executing agent's own suggestions
# ---------------------------------------------------------------------------

def test_agent_suggestions_read_from_both_log_shapes():
    result = {
        "analysis_log": [
            {"step_type": "execute_code", "findings_suggestions": ["ignored"]},
            {"step_type": "report_findings", "findings_suggestions": ["from dict", "dup"]},
            AnalysisStep(step_number=9, step_type="report_findings",
                         findings_suggestions=["from object", "dup"]),
        ]
    }
    assert PostExecutionReviewMixin._collect_agent_suggestions(result) == [
        "from dict", "dup", "from object"
    ]


# ---------------------------------------------------------------------------
# Feature B — the gated pass
# ---------------------------------------------------------------------------

class _PI:
    def __init__(self, plan=None, assessment=None, raise_on_review=False):
        self._plan = plan if plan is not None else {
            "gaps": ["decoupler never ran"],
            "supplementary_tasks": ["Run decoupler run_ulm on the existing object"],
            "out_of_scope": ["Measure protein by western blot"],
            "verdict": "supplement",
        }
        self._assessment = assessment or {"verdict": "resolved", "summary": "Gap closed.",
                                          "resolved_gaps": ["decoupler never ran"],
                                          "remaining_gaps": []}
        self._raise = raise_on_review
        self.review_calls = 0
        self.assessment_inputs = []

    async def review_round_execution(self, **kwargs):
        self.review_calls += 1
        if self._raise:
            raise RuntimeError("planner down")
        return self._plan

    async def assess_supplementary_result(self, **kwargs):
        self.assessment_inputs.append(kwargs)
        return self._assessment


class _Bio:
    def __init__(self, result=None, raise_on_run=False):
        self._result = result
        self._raise = raise_on_run
        self.calls = 0

    async def run_supplementary_tasks(self, **kwargs):
        self.calls += 1
        if self._raise:
            raise RuntimeError("executor down")
        return self._result if self._result is not None else {
            "status": "completed",
            "summary": "Ran the pathway activity task.",
            "findings": [
                "Pathway activity is elevated in the treated group "
                "[execution_id supp-1]."
            ],
            "finding_records": [{
                "text": "Pathway activity is elevated in the treated group "
                        "[execution_id supp-1].",
                "modality": "single_cell",
                "evidence_execution_ids": ["supp-1"],
            }],
            "figures": [],
            "warnings": [],
            "code_history": [_execution(
                "dc.run_ulm(adata)", exec_id="supp-1",
                modalities=["single_cell"],
            )],
            "analysis_log": [],
        }


def _unmet_result():
    """A clean completion whose plan audit will report one unmet step."""
    return {"findings": ["Baseline finding."], "figures": [], "status": "completed",
            "analysis_log": [], "evidence_execution_ids": ["e1"], "code_history": []}


@pytest.mark.asyncio
async def test_disabled_by_config_skips_without_touching_agents():
    pi, bio = _PI(), _Bio()
    cfg = SimpleNamespace(enable_post_execution_review=False, max_rounds=3)
    harness = _Harness(pi=pi, bio=bio, config=cfg)

    meta, merge = await harness._run_post_execution_review(
        _state(_plan([])), _unmet_result(), {}, "full")

    assert meta["review_status"] == "skipped"
    assert merge == {} and pi.review_calls == 0 and bio.calls == 0


@pytest.mark.asyncio
async def test_fully_evidenced_round_does_not_run_the_pass():
    plan = _plan([PlanStep(id=1, title="Communication", method="Run liana rank_aggregate")])
    # Another direction is still queued, so this is not the batch's last one either.
    state = _state(plan, code_history=[_execution("li.mt.rank_aggregate(adata)")],
                   current_round_direction="D1",
                   directions_queue=[{"direction": "D2"}],
                   publication_figures=[_result_figure(
                       plan, 1, name="communication_result"
                   )])
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "skipped"
    assert pi.review_calls == 0 and bio.calls == 0 and merge == {}


@pytest.mark.asyncio
async def test_unmet_step_triggers_pass_and_merges_back():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    result = _unmet_result()
    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert pi.review_calls == 1 and bio.calls == 1
    assert meta["review_status"] == "completed"
    assert meta["assessment"]["verdict"] == "resolved"
    assert merge["findings"] == [
        "Pathway activity is elevated in the treated group [execution_id supp-1]."
    ]

    findings, figures = harness._merge_supplementary_into_round(
        state, result, result["findings"], result["figures"], merge)
    assert findings[-1].startswith("[Supplementary] ")
    assert [c.id for c in state.code_history] == ["supp-1"]
    # The raw result must carry the new execution so a re-audit can see it.
    assert result["findings"] == findings

    # The computation is now evidenced, while its reviewed result figure remains
    # a main-executor responsibility rather than something this restricted pass
    # may fabricate.
    reaudit = harness._audit_round_plan_execution(state, {})
    assert reaudit["n_unmet"] == 1
    assert reaudit["unmet_figure_kinds"] == ["result"]
    assert plan.steps[0].status == "pending"


@pytest.mark.asyncio
async def test_evidence_records_reach_pi_and_meta_without_raw_provenance():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    raw_execution = _execution("dc.run_ulm(adata)", exec_id="supp-evidence")
    evidence_records = [{
        "execution_id": raw_execution.id,
        "status": "completed",
        "stdout_excerpt": "top regulons: A, B",
        "error": "",
    }]
    raw_result = {
        "status": "partial",
        "summary": "The iteration limit was reached.",
        "findings": [],
        "figures": [],
        "warnings": ["max_iterations reached"],
        "evidence_records": evidence_records,
        "code_history": [raw_execution],
        "analysis_log": [],
    }
    pi = _PI(assessment={
        "verdict": "partial",
        "summary": "The execution record is available for review.",
        "resolved_gaps": [],
        "remaining_gaps": ["decoupler never ran"],
    })
    harness = _Harness(pi=pi, bio=_Bio(result=raw_result))

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full"
    )

    assert pi.assessment_inputs
    assessed = pi.assessment_inputs[-1]["supplementary_result"]
    assert assessed["evidence_records"] == evidence_records
    assert meta["review_result"]["evidence_records"] == evidence_records
    assert "code_history" not in assessed and "analysis_log" not in assessed
    assert "code_history" not in meta and "analysis_log" not in meta
    assert merge["code_history"] == [raw_execution]
    assert merge["findings"] == []
    import json
    json.dumps(meta)


def test_merge_discards_unreviewed_automatic_supplementary_figures():
    """Defense in depth: old snapshots and test doubles may still carry the old
    payload shape, but those images have no PublicationFigure review record."""
    state = _state(_plan([]))
    result = {"findings": ["main"], "figures": ["reviewed-main-figure"]}

    findings, figures = _Harness()._merge_supplementary_into_round(
        state,
        result,
        result["findings"],
        result["figures"],
        {
            "findings": ["repair"],
            "finding_records": [{
                "text": "repair", "modality": "single_cell",
                "evidence_execution_ids": ["supp-1"],
            }],
            "figures": ["raw-inline-base64"],
        },
    )

    assert findings == ["main", "[Supplementary] repair"]
    assert figures == ["reviewed-main-figure"]
    assert result["figures"] == ["reviewed-main-figure"]


def test_supplementary_n1_inference_is_discarded_before_merge():
    from backend.agents.bioinformatics.supplementary import (
        _validated_supplementary_findings,
    )
    from backend.models.schemas import (
        AnalysisState, SampleCountRecord, TreatmentGroup,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [
        TreatmentGroup(id="g1", name="Con", sample_count=1),
        TreatmentGroup(id="g2", name="DKO", sample_count=1),
    ]
    state.detected_sample_counts = [
        SampleCountRecord(
            group_id=group_id, group_name=name, section="sequencing",
            data_type="scRNA-seq", detected_count=1, declared_count=1,
            method="obs_column:sample", evidence="one library",
        )
        for group_id, name in (("g1", "Con"), ("g2", "DKO"))
    ]
    execution = _execution(
        "sc.tl.rank_genes_groups(adata, 'condition')",
        exec_id="supp-n1", modalities=["single_cell"],
    )

    records, warnings = _validated_supplementary_findings(
        ["Gene X is significantly increased (p = 0.004) [execution_id supp-n1]."],
        [execution], state,
    )

    assert records == []
    assert any("fewer than two independent" in warning for warning in warnings)


def test_supplementary_batch_correction_uses_the_same_design_guard():
    from backend.agents.bioinformatics.supplementary import SupplementaryMixin
    from backend.models.schemas import AnalysisState, SampleCountRecord, TreatmentGroup

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [
        TreatmentGroup(id="g1", name="Con", sample_count=1),
        TreatmentGroup(id="g2", name="DKO", sample_count=1),
    ]
    state.detected_sample_counts = [
        SampleCountRecord(
            group_id=group_id, group_name=name, section="sequencing",
            data_type="scRNA-seq", detected_count=1, declared_count=1,
            method="obs_column:sample", evidence="one library",
        )
        for group_id, name in (("g1", "Con"), ("g2", "DKO"))
    ]

    class _Agent(SupplementaryMixin):
        _current_analysis_state = state

    error = _Agent()._supplementary_code_is_safe(
        "sc.external.pp.harmony_integrate(adata, 'sample')", "adata",
    )
    assert error and "cannot be separated from condition" in error


@pytest.mark.asyncio
async def test_figure_only_gap_is_deferred_to_main_publication_pipeline():
    plan = _plan([PlanStep(
        id=1,
        title="Show pathway activity",
        method="Summarize the registered score",
        figure_requirements=[{"figure_type": "heatmap"}],
    )])
    state = _state(plan, publication_figures=[])
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full"
    )

    assert meta["review_status"] == "skipped"
    assert "main bioinformatics figure/review pipeline" in meta["review_reason"]
    assert merge == {}
    assert pi.review_calls == 0 and bio.calls == 0


@pytest.mark.asyncio
async def test_review_meta_is_json_serializable():
    """The meta is prompted to the PI, sent over the websocket and written into
    session.json, so a pydantic object anywhere inside it breaks all three. The
    executor's CodeExecution records must reach the caller through the merge
    payload only."""
    import json

    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    harness = _Harness(pi=_PI(), bio=_Bio())

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    json.dumps(meta)
    json.dumps(state.post_execution_reviews)
    assert merge["code_history"], "executions still reach the caller, just not via meta"


@pytest.mark.asyncio
async def test_pass_is_paid_for_once_per_execution():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio)
    audit_summary = harness._audit_round_plan_execution(state, {})
    result = _unmet_result()

    first, _ = await harness._run_post_execution_review(state, result, audit_summary, "full")
    second, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert pi.review_calls == 1 and bio.calls == 1, "re-entry must reuse the stored review"
    assert second["review_attempt_id"] == first["review_attempt_id"]
    assert merge == {}, "a reused review must not merge its findings twice"


@pytest.mark.asyncio
async def test_pi_declining_costs_nothing_further():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    pi = _PI(plan={"gaps": [], "supplementary_tasks": [], "verdict": "no_action",
                   "rationale": "Nothing runnable on the current data."})
    bio = _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "no_action"
    assert bio.calls == 0 and merge == {}


@pytest.mark.asyncio
async def test_zero_task_budget_keeps_the_audit_and_skips_the_pass():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    cfg = SimpleNamespace(enable_post_execution_review=True, post_execution_review_max_tasks=0,
                          post_execution_review_max_iterations=5,
                          post_execution_review_timeout_seconds=900, max_rounds=3)
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio, config=cfg)

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, _ = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "skipped"
    assert meta["plan_audit"]["n_unmet"] == 1, "the audit still runs and is recorded"
    assert pi.review_calls == 0 and bio.calls == 0


@pytest.mark.asyncio
async def test_planner_failure_cannot_fail_the_round():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    harness = _Harness(pi=_PI(raise_on_review=True), bio=_Bio())

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "failed" and merge == {}


@pytest.mark.asyncio
async def test_executor_failure_cannot_fail_the_round():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    harness = _Harness(pi=_PI(), bio=_Bio(raise_on_run=True))

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "failed"
    assert meta["assessment"]["verdict"] in ("resolved", "partial", "unresolved")
    assert merge["findings"] == []


@pytest.mark.asyncio
async def test_no_runtime_means_no_pass():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    harness = _Harness(pi=_PI(), bio=None)

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "skipped" and merge == {}


@pytest.mark.asyncio
async def test_last_direction_triggers_on_an_otherwise_clean_round():
    """The one signal with nothing else behind it: analysis may stop after the
    batch's last direction, so its follow-up work would be discarded. This is the
    real shape it has to fire on — the running direction still queued."""
    plan = _plan([PlanStep(id=1, title="Communication", method="Run liana rank_aggregate")])
    state = _state(plan, current_round=3, max_rounds=3,
                   current_round_direction="D2",
                   directions_queue=[{"direction": "D2"}],
                   code_history=[_execution("li.mt.rank_aggregate(adata)", round_number=3)],
                   publication_figures=[_result_figure(
                       plan, 1, round_number=3, name="communication_result"
                   )])
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    assert audit_summary["n_unmet"] == 0
    meta, _ = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert "Last direction" in meta["review_reason"]
    assert pi.review_calls == 1
