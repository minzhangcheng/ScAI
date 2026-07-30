"""Post-execution review orchestration.

Covers the three contracts the round depends on: the plan audit is the writer of
`PlanStep.status`, the supplementary pass is paid for at most once per execution,
and neither can fail the round it reviews.
"""
import pytest
from types import SimpleNamespace

from backend.models.schemas import AnalysisStep, CodeExecution
from backend.models.schema_parts.session import AnalysisPlan, PlanStep
from backend.workflow.post_execution import PostExecutionReviewMixin


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


def _execution(code, error=None, round_number=2, exec_id=None):
    kwargs = {"code": code, "error": error, "round_number": round_number}
    if exec_id:
        kwargs["id"] = exec_id
    return CodeExecution(**kwargs)


# ---------------------------------------------------------------------------
# Feature A — the audit writes plan step status
# ---------------------------------------------------------------------------

def test_audit_writes_step_status_and_stamps_only_completed():
    plan = _plan([
        PlanStep(id=1, title="Cell communication", method="Run liana rank_aggregate"),
        PlanStep(id=2, title="Pathway activity", method="Run decoupler run_ulm"),
    ])
    state = _state(plan, code_history=[_execution("import liana as li; li.mt.rank_aggregate(adata)")])

    summary = _Harness()._audit_round_plan_execution(state, {})

    assert summary["n_steps"] == 2
    assert summary["n_completed"] == 1
    assert summary["n_unmet"] == 1
    by_id = {s.id: s for s in plan.steps}
    assert by_id[1].status == "completed" and by_id[1].executed_at is not None
    assert by_id[2].status == "pending" and by_id[2].executed_at is None


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


def test_registered_score_vouches_for_gene_scoring():
    plan = _plan([PlanStep(id=1, title="Module scores", method="Use score_genes for the panel")])
    state = _state(plan, score_definitions=[SimpleNamespace(round_number=2)])

    summary = _Harness()._audit_round_plan_execution(state, {})

    assert summary["n_completed"] == 1
    assert plan.steps[0].status == "completed"


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
# Terminal-round detection
# ---------------------------------------------------------------------------

def test_terminal_round_requires_spent_budget_empty_queue_and_closed_integration():
    harness = _Harness()
    assert harness._is_terminal_analysis_round(_state(current_round=2, max_rounds=3)) is False
    assert harness._is_terminal_analysis_round(_state(current_round=3, max_rounds=3)) is True
    assert harness._is_terminal_analysis_round(
        _state(current_round=3, max_rounds=3, directions_queue=[{"direction": "d"}])) is False
    assert harness._is_terminal_analysis_round(
        _state(current_round=3, max_rounds=3, pending_integration=True,
               integration_cycle=0, max_integration_cycles=1)) is False
    # Integration cycle cap reached: nothing can reopen the analysis.
    assert harness._is_terminal_analysis_round(
        _state(current_round=3, max_rounds=3, pending_integration=True,
               integration_cycle=1, max_integration_cycles=1)) is True


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

    async def review_round_execution(self, **kwargs):
        self.review_calls += 1
        if self._raise:
            raise RuntimeError("planner down")
        return self._plan

    async def assess_supplementary_result(self, **kwargs):
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
            "findings": ["Pathway activity is elevated in the treated group."],
            "figures": [],
            "warnings": [],
            "code_history": [_execution("dc.run_ulm(adata)", exec_id="supp-1")],
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
    state = _state(plan, code_history=[_execution("li.mt.rank_aggregate(adata)")])
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
    assert merge["findings"] == ["Pathway activity is elevated in the treated group."]

    findings, figures = harness._merge_supplementary_into_round(
        state, result, result["findings"], result["figures"], merge)
    assert findings[-1].startswith("[Supplementary] ")
    assert [c.id for c in state.code_history] == ["supp-1"]
    # The raw result must carry the new execution so a re-audit can see it.
    assert result["findings"] == findings

    # Re-audit now finds the step evidenced by the supplementary execution.
    assert harness._audit_round_plan_execution(state, {})["n_unmet"] == 0
    assert plan.steps[0].status == "completed"


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
async def test_terminal_round_triggers_on_an_otherwise_clean_round():
    """The one case with no other signal: after this round nothing can run."""
    plan = _plan([PlanStep(id=1, title="Communication", method="Run liana rank_aggregate")])
    state = _state(plan, current_round=3, max_rounds=3,
                   code_history=[_execution("li.mt.rank_aggregate(adata)", round_number=3)])
    pi, bio = _PI(), _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    assert audit_summary["n_unmet"] == 0
    meta, _ = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert "Terminal round" in meta["review_reason"]
    assert pi.review_calls == 1
