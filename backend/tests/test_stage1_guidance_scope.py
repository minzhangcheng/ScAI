import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import (
    BioinformaticsAgent,
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
)
from backend.models.schemas import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    PlanAnalysisScope,
    PlanStep,
)


def _innovation_state() -> AnalysisState:
    inventory = AcceptedFoundationInventory(
        inventory_id="foundation-v1-r1",
        round_number=1,
        plan_id="round1-plan",
        planning_stage="stage1",
        lineage_id="main",
        adata_checkpoint_id="checkpoint-r1",
        steps=[AcceptedFoundationStep(
            plan_step_id=4,
            title="Accepted clustering baseline",
            method="scanpy.tl.leiden",
            evidence_execution_ids=["receipt-r1-step4"],
        )],
    )
    return AnalysisState(
        user_id="scope-test",
        current_round=2,
        analysis_phase=AnalysisPhase.INNOVATION,
        accepted_foundation_inventories=[inventory],
    )


def _reuse_scope() -> dict:
    return {
        "relationship": "reuse",
        "cohort_scope": "subset",
        "foundation_refs": [{
            "inventory_id": "foundation-v1-r1",
            "round_number": 1,
            "plan_id": "round1-plan",
            "plan_step_ids": [4],
            "execution_ids": ["receipt-r1-step4"],
            "checkpoint_id": "checkpoint-r1",
        }],
        "reason": "Reuse the accepted measurements for a defined subset.",
        "requested_by": None,
    }


def _bare_planner(response: dict) -> BioinformaticsAgent:
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._call_planning_llm = AsyncMock(return_value=json.dumps(response))
    agent._method_decision_guidance_block = MagicMock(return_value="")
    return agent


class TestStage1GuidanceScope:
    """Stage 1 guidance must forbid loading/preprocessing data for Stage 2 analyses."""

    def test_stage1_forbids_downstream_analyses(self):
        text = _STAGE1_GUIDANCE.lower()
        assert "do not include downstream analysis" in text

    def test_stage1_forbids_loading_velocity_layers(self):
        """Stage 1 must explicitly disallow loading spliced/unspliced/velocyto layers."""
        text = _STAGE1_GUIDANCE.lower()
        assert "spliced" in text
        assert "unspliced" in text
        assert "velocity" in text

    def test_stage1_forbids_preparing_stage2_inputs(self):
        """Stage 1 must ban data prep whose only purpose is Stage 2."""
        text = _STAGE1_GUIDANCE.lower()
        assert "do not load" in text or "do not preprocess" in text
        assert "stage 2" in text

    def test_stage1_names_concrete_examples(self):
        """Concrete examples reduce model ambiguity — name the common offenders."""
        text = _STAGE1_GUIDANCE.lower()
        assert "loom" in text
        assert "trajectory" in text or "pseudotime" in text

    def test_stage1_keeps_in_scope_steps(self):
        text = _STAGE1_GUIDANCE.lower()
        for token in ["quality control", "doublet", "normalization", "clustering", "annotation"]:
            assert token in text, f"missing in-scope token: {token}"

    def test_stage_guidance_text_routes_correctly(self):
        assert _stage_guidance_text("stage1") == _STAGE1_GUIDANCE
        assert _stage_guidance_text("stage2") == _STAGE2_GUIDANCE
        assert _stage_guidance_text("full") == ""


@pytest.mark.asyncio
async def test_round_two_full_plan_receives_foundation_contract_and_returns_scope():
    state = _innovation_state()
    scope = _reuse_scope()
    agent = _bare_planner({
        "overview": "Analyze the selected cluster subset.",
        "scope_contract_version": 1,
        "foundation_inventory_id": "foundation-v1-r1",
        "steps": [{
            "id": 1,
            "title": "Cluster subset sensitivity",
            "method": "scanpy.tl.leiden",
            "input": "Accepted clustering measurements for Cluster 18.",
            "expected_output": "Subset sensitivity labels.",
            "figure_requirements": [],
            "allow_text_only": True,
            "text_only_reason": "The plan-only test has no rendered result.",
            "rationale": "Compare the selected subset with accepted clustering.",
            "depends_on": [],
            "analysis_scope": scope,
        }],
    })

    plan = await agent.generate_analysis_plan(
        state=state,
        pi_tasks=["Assess Cluster 18 sensitivity"],
        overview="Targeted subset analysis",
        phase="innovation",
        planning_stage="full",
    )

    prompt = agent._call_planning_llm.await_args.args[0]
    assert "Innovation Scope Contract v1" in prompt
    assert "foundation-v1-r1" in prompt
    assert plan.scope_contract_version == 1
    assert plan.foundation_inventory_id == "foundation-v1-r1"
    assert plan.steps[0].analysis_scope.relationship == "reuse"
    assert plan.steps[0].analysis_scope.foundation_refs[0].plan_step_ids == [4]


@pytest.mark.asyncio
async def test_incremental_revision_keeps_scope_on_completed_and_revised_steps():
    state = _innovation_state()
    completed_scope = PlanAnalysisScope.model_validate(_reuse_scope())
    current_plan = AnalysisPlan(
        id="round2-plan",
        phase="innovation",
        stage="full",
        status="draft",
        scope_contract_version=1,
        foundation_inventory_id="foundation-v1-r1",
        steps=[
            PlanStep(
                id=1,
                title="Completed subset definition",
                method="pandas filtering",
                input="Accepted measurements.",
                expected_output="Selected Cluster 18 cells.",
                rationale="Define the targeted cohort.",
                analysis_scope=completed_scope,
                status="completed",
            ),
            PlanStep(
                id=2,
                title="Current sensitivity analysis",
                method="scanpy.tl.leiden",
                input="Selected Cluster 18 cells.",
                expected_output="Sensitivity labels.",
                rationale="Assess the accepted result.",
                analysis_scope=completed_scope,
            ),
        ],
    )
    state.current_plan = current_plan
    revised_scope = {
        **_reuse_scope(),
        "reason": "Use the accepted baseline for a more focused sensitivity check.",
    }
    agent = _bare_planner({
        "overview": "Revise the remaining sensitivity step.",
        "scope_contract_version": 1,
        "foundation_inventory_id": "foundation-v1-r1",
        "steps": [
            {
                "id": 1,
                "title": "Completed subset definition",
                "method": "pandas filtering",
                "input": "Accepted measurements.",
                "expected_output": "Selected Cluster 18 cells.",
                "rationale": "Define the targeted cohort.",
                "depends_on": [],
                "analysis_scope": None,
            },
            {
                "id": 2,
                "title": "Revised sensitivity analysis",
                "method": "scanpy.tl.leiden",
                "input": "Selected Cluster 18 cells.",
                "expected_output": "Sensitivity labels.",
                "rationale": "Assess the accepted result.",
                "depends_on": [1],
                "analysis_scope": revised_scope,
            },
        ],
    })

    revised = await agent.revise_plan_incrementally(
        state=state,
        current_plan=current_plan,
        pi_feedback="Revise the parameter sensitivity analysis.",
        completed_step_ids=[1],
    )

    prompt = agent._call_planning_llm.await_args.args[0]
    assert "Innovation Scope Contract v1" in prompt
    assert "foundation-v1-r1" in prompt
    assert revised.scope_contract_version == 1
    assert revised.steps[0].analysis_scope == completed_scope
    assert revised.steps[1].analysis_scope.reason == revised_scope["reason"]
    assert revised.steps[1].analysis_scope.foundation_refs[0].execution_ids == [
        "receipt-r1-step4"
    ]
