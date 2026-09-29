"""Tests for Host evidence identity and PI-owned plan review."""
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisPlan, AnalysisState, CodeExecution, PlanStep
from backend.workflow.post_execution_review import audit_plan_execution
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


def test_agent_nominated_linkage_is_not_host_verified_plan_completion():
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

    assert audit.evidence_mode == "agent_nominated_linkage"
    assert [step.status for step in audit.steps] == ["pending", "pending"]
    assert all(step.verifiable is False for step in audit.steps)
    observations = audit.summary()["step_observations"]
    assert observations[0]["technical_linkage"] == "agent_nominated"
    assert observations[1]["technical_linkage"] == "not_nominated"
    assert "receipt-1" in observations[0]["reason"]
    assert "Host verified receipt identity only" in observations[0]["reason"]
    assert "Host verified current successful" not in observations[0]["reason"]


def test_agent_nomination_does_not_hide_missing_planned_method_family():
    plan = AnalysisPlan(
        id="trajectory-plan",
        phase="innovation",
        steps=[PlanStep(
            id=11,
            title="PAGA and DPT ordering",
            method="Run sc.tl.paga, sc.tl.diffmap, and sc.tl.dpt",
        )],
    )

    audit = audit_plan_execution(
        plan,
        [_receipt(
            plan,
            id="unrelated-execution",
            code="summarize_isr_and_er_upr()",
            plan_step_ids=[11],
        )],
        round_number=2,
        target_cohort_digest="a" * 64,
    )

    observation = audit.summary()["step_observations"][0]
    assert observation["technical_linkage"] == "agent_nominated"
    assert audit.summary()["unmet_families"] == ["trajectory"]
    assert "did not contain" in observation["reason"]
    assert "trajectory" in observation["reason"]


def test_semantic_reconciliation_preserves_links_across_cohort_generations():
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

    assert audit.steps[0].status == "completed"
    assert audit.evidence_mode == "supervisor_semantic"


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
