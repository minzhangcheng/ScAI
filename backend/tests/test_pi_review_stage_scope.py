import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import (
    AcceptedFoundationInventory,
    AcceptedFoundationStep,
    AnalysisPhase,
    AnalysisPlan,
    AnalysisState,
    PlanStep,
)


class TestPIReviewStageScope:
    """review_bioinfo_plan prompt must contain stage-scope review criteria."""

    def _prompt_template(self) -> str:
        return inspect.getsource(PIAgent.review_bioinfo_plan)

    def test_prompt_passes_plan_stage(self):
        src = self._prompt_template()
        assert "plan.stage" in src, "PI review prompt must surface plan.stage"

    def test_prompt_lists_stage_scope_criterion(self):
        src = self._prompt_template().lower()
        assert "stage scope" in src or "scope" in src
        assert "stage 1" in src
        assert "stage 2" in src

    def test_prompt_names_concrete_stage1_exclusions(self):
        src = self._prompt_template().lower()
        for token in ["spliced", "velocity", "trajectory"]:
            assert token in src, f"PI prompt missing scope token: {token}"

    def test_prompt_directs_revise_on_scope_leak(self):
        src = self._prompt_template().lower()
        assert "revise" in src
        assert "downstream" in src or "stage 2" in src

    def test_prompt_reviews_visual_coverage_and_explicit_exceptions(self):
        src = self._prompt_template().lower()
        assert "visual coverage" in src
        assert "figure_requirements" in src
        assert "allow_text_only=true" in src
        assert "text_only_reason" in src
        assert "doublet" in src and "before/after" in src
        assert "missing visual" in src and "revise" in src

    @pytest.mark.asyncio
    async def test_innovation_scope_is_reviewed_for_full_stage_plans(self):
        provider = MagicMock()
        provider.get_model.return_value = "test/model"
        provider.get_system_prompt.return_value = ""
        provider.get_temperature.return_value = 0.3
        provider.get_llm_client.return_value = MagicMock()
        agent = PIAgent(settings_provider=provider)
        agent.chat = AsyncMock(return_value=json.dumps({
            "decision": "approved",
            "feedback": "The declared subset reuse is clear.",
            "modifications": [],
        }))
        inventory = AcceptedFoundationInventory(
            inventory_id="foundation-v1-r1",
            round_number=1,
            plan_id="round1-plan",
            planning_stage="stage1",
            steps=[AcceptedFoundationStep(
                plan_step_id=4,
                title="Accepted clustering baseline",
                method="scanpy.tl.leiden",
                evidence_execution_ids=["receipt-r1-step4"],
            )],
        )
        state = AnalysisState(
            user_id="scope-test",
            current_round=2,
            analysis_phase=AnalysisPhase.INNOVATION,
            accepted_foundation_inventories=[inventory],
        )
        plan = AnalysisPlan(
            phase="innovation",
            stage="full",
            scope_contract_version=1,
            foundation_inventory_id="foundation-v1-r1",
            steps=[PlanStep(
                id=1,
                title="Subset sensitivity",
                method="scanpy.tl.leiden",
                analysis_scope={
                    "relationship": "reuse",
                    "cohort_scope": "subset",
                    "foundation_refs": [{
                        "inventory_id": "foundation-v1-r1",
                        "round_number": 1,
                        "plan_id": "round1-plan",
                        "plan_step_ids": [4],
                        "execution_ids": ["receipt-r1-step4"],
                    }],
                    "reason": "Use the accepted baseline for targeted sensitivity.",
                },
            )],
        )

        await agent.review_bioinfo_plan(state, plan)

        prompt = agent.chat.await_args.args[0]
        assert plan.stage == "full"
        assert "Innovation Scope Contract v1" in prompt
        assert "foundation-v1-r1" in prompt
        assert "method-family wording alone is not a reason to reject" in prompt

    @pytest.mark.asyncio
    async def test_scope_contradiction_is_exposed_to_pi_and_revise_is_preserved(self):
        provider = MagicMock()
        provider.get_model.return_value = "test/model"
        provider.get_system_prompt.return_value = ""
        provider.get_temperature.return_value = 0.3
        provider.get_llm_client.return_value = MagicMock()
        agent = PIAgent(settings_provider=provider)
        agent.chat = AsyncMock(return_value=json.dumps({
            "decision": "revise",
            "feedback": "The stated rebuild contradicts relationship=new.",
            "modifications": [{
                "step_id": 1,
                "suggestion": "Declare an authorized foundation revision.",
            }],
        }))
        inventory = AcceptedFoundationInventory(
            inventory_id="foundation-v1-r1",
            round_number=1,
            plan_id="round1-plan",
            planning_stage="stage1",
            steps=[AcceptedFoundationStep(
                plan_step_id=2,
                title="Accepted full-cohort QC and doublet handling",
                method="accepted method",
                evidence_execution_ids=["receipt-r1-step2"],
            )],
        )
        state = AnalysisState(
            user_id="scope-contradiction",
            current_round=2,
            analysis_phase=AnalysisPhase.INNOVATION,
            accepted_foundation_inventories=[inventory],
        )
        plan = AnalysisPlan(
            phase="innovation",
            stage="full",
            scope_contract_version=1,
            foundation_inventory_id=inventory.inventory_id,
            steps=[PlanStep(
                id=1,
                title="Rebuild the full-cohort foundation",
                method="Repeat full-cohort QC and doublet detection from raw inputs",
                analysis_scope={
                    "relationship": "new",
                    "cohort_scope": "full",
                    "foundation_refs": [],
                    "reason": "",
                },
            )],
        )

        result = await agent.review_bioinfo_plan(state, plan)

        prompt = agent.chat.await_args.args[0]
        assert "Rebuild the full-cohort foundation" in prompt
        assert '"relationship": "new"' in prompt
        assert "return revise and explain the contradiction" in prompt
        assert result["decision"] == "revise"
