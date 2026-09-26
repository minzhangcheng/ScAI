"""Post-execution review orchestration.

Covers the three contracts the round depends on: the audit supplies observations
to the PI without deciding completion, the supplementary pass is paid for at most
once per execution, and neither can fail the round it reviews.
"""
import base64
import hashlib
import pytest
from types import SimpleNamespace

from backend.models.schemas import CodeExecution
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

    def __init__(self, pi=None, bio=None, config=None, state=None, state_callback=None):
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
        self.saves = []
        self._current_analysis_state = state
        async def _save(current, force=False):
            self.saves.append((current, force))
            return True
        self.state_callback = state_callback or _save

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
        restore_branch_id=None,
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


def _execution(
    code,
    error=None,
    round_number=2,
    exec_id=None,
    modalities=None,
    **extra,
):
    kwargs = {"code": code, "error": error, "round_number": round_number}
    if exec_id:
        kwargs["id"] = exec_id
    if modalities is not None:
        kwargs["analysis_modalities"] = modalities
    kwargs.update(extra)
    return CodeExecution(**kwargs)


# ---------------------------------------------------------------------------
# Feature A — the audit records observations without writing plan status
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evidence binding must survive the node's result dict
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Budget derivation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Last-direction detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The executing agent's own suggestions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature B — the gated pass
# ---------------------------------------------------------------------------

class _PI:
    def __init__(self, plan=None, assessment=None, raise_on_review=False):
        self._plan = plan if plan is not None else {
            "gaps": ["decoupler never ran"],
            "step_decisions": [{
                "plan_step_id": 1,
                "status": "pending",
                "reason": "The PI considers this step unfinished.",
            }],
            "supplementary_tasks": ["Run decoupler run_ulm on the existing object"],
            "out_of_scope": ["Measure protein by western blot"],
            "verdict": "supplement",
        }
        self._assessment = assessment or {"verdict": "resolved", "summary": "Gap closed.",
                                          "resolved_gaps": ["decoupler never ran"],
                                          "remaining_gaps": []}
        self._raise = raise_on_review
        self.review_calls = 0
        self.review_inputs = []
        self.assessment_inputs = []

    async def review_round_execution(self, **kwargs):
        self.review_calls += 1
        self.review_inputs.append(kwargs)
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
        self.task_inputs = []

    async def run_supplementary_tasks(self, **kwargs):
        self.calls += 1
        self.task_inputs.append(kwargs)
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
async def test_reservation_save_failure_skips_automatic_execution():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    pi, bio = _PI(), _Bio()
    saves = []

    async def fail_save(_state, force=False):
        saves.append(force)
        return False

    harness = _Harness(
        pi=pi, bio=bio, state_callback=fail_save,
    )
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), {"n_unmet": 1}, "full",
    )

    assert meta["lifecycle"] == "reservation_failed"
    assert meta["review_status"] == "failed"
    assert merge == {}
    assert bio.calls == 0
    assert saves == [True]


@pytest.mark.asyncio
async def test_terminal_save_failure_holds_replay_until_ack_without_rerun():
    """A paid pass is replayable after terminal persistence is acknowledged."""
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    pi, bio = _PI(), _Bio()
    save_outcomes = iter([True, False, True])
    events = []

    async def save(current, force=False):
        events.append(("save", force, bio.calls))
        return next(save_outcomes)

    harness = _Harness(pi=pi, bio=bio, state=state, state_callback=save)
    result = _unmet_result()
    original_result = dict(result)

    first, held_merge = await harness._run_post_execution_review(
        state, result, {"n_unmet": 1}, "full",
    )

    assert bio.calls == 1 and pi.review_calls == 1
    assert held_merge == {}
    assert result == original_result
    assert first["lifecycle"] == "terminal_persistence_pending"
    assert first["persistence_status"] == "pending"
    assert first["merge_payload"]["findings"] == [
        "Pathway activity is elevated in the treated group [execution_id supp-1]."
    ]
    assert first["merge_payload"]["code_history"]
    assert events[:2] == [("save", True, 0), ("save", True, 1)]

    second, replay = await harness._run_post_execution_review(
        state, result, {"n_unmet": 1}, "full",
    )

    assert bio.calls == 1 and pi.review_calls == 1
    assert events == [("save", True, 0), ("save", True, 1), ("save", True, 1)]
    assert second["lifecycle"] == "completed"
    assert second["persistence_status"] == "acknowledged"
    assert replay["findings"] == first["merge_payload"]["findings"]
    findings, _ = harness._merge_supplementary_into_round(
        state, result, result["findings"], result["figures"], replay,
    )
    replay_findings, _ = harness._merge_supplementary_into_round(
        state, result, findings, result["figures"], replay,
    )
    assert replay_findings == findings
    assert [item.id for item in state.code_history].count("supp-1") == 1


def test_supplementary_n1_inference_is_retained_for_pi_review():
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

    assert len(records) == 1
    assert records[0]["text"].startswith("Gene X is significantly increased")
    assert records[0]["evidence_status"] == "unverified"
    assert any("PI review observation" in warning for warning in warnings)
    assert any("fewer than two independent" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_tasks_without_explicit_pi_supplement_verdict_never_execute():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    pi = _PI(plan={
        "gaps": ["Potential gap"],
        "supplementary_tasks": ["Run another analysis"],
        "verdict": "maybe",
    })
    bio = _Bio()
    harness = _Harness(pi=pi, bio=bio)

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full"
    )

    assert meta["review_status"] == "failed"
    assert meta["technical_skip"] is True
    assert bio.calls == 0 and merge == {}


@pytest.mark.asyncio
async def test_planner_failure_cannot_fail_the_round():
    plan = _plan([PlanStep(id=1, title="Pathway activity", method="Run decoupler run_ulm")])
    state = _state(plan)
    harness = _Harness(pi=_PI(raise_on_review=True), bio=_Bio())

    audit_summary = harness._audit_round_plan_execution(state, {})
    meta, merge = await harness._run_post_execution_review(
        state, _unmet_result(), audit_summary, "full")

    assert meta["review_status"] == "failed" and merge == {}
