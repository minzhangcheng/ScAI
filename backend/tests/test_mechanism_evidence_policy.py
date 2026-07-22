import inspect

from backend.agents.pi_agent import PIAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.services.report.planning import PlanningMixin
from backend.utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY


def test_shared_policy_is_domain_neutral_and_balances_recall_with_caution():
    low = MECHANISM_EVIDENCE_POLICY.lower()
    assert "unbiased analyses" in low
    assert "name it explicitly" in low
    assert "supporting and contradictory evidence" in low
    assert "prespecified hypotheses" in low
    assert "supported, contradicted, or insufficiently supported" in low
    assert "causal mechanism execution" in low
    for hardcoded_example in ("ferroptosis", "apoptosis", "cuproptosis", "nrf2"):
        assert hardcoded_example not in low


def test_pi_discovery_and_synthesis_prompts_use_shared_policy():
    for method in (
        PIAgent.generate_raw_direction_candidates,
        PIAgent.generate_integration_direction_candidates,
        PIAgent.innovation_instruction,
        PIAgent.build_cross_modal_synthesis,
        PIAgent.generate_final_summary,
        PIAgent.generate_integrated_report,
    ):
        assert "MECHANISM_EVIDENCE_POLICY" in inspect.getsource(method)


def test_exploration_plan_preserves_prespecified_hypotheses_without_biasing_baseline():
    src = inspect.getsource(PIAgent.initial_exploration_instruction).lower()
    assert "mandatory broad exploration tasks" in src
    assert "prespecified hypotheses" in src
    assert "support" in src
    assert "contradiction" in src
    assert "insufficient evidence" in src


def test_live_and_fallback_report_prompts_use_shared_policy():
    assert "MECHANISM_EVIDENCE_POLICY" in inspect.getsource(
        SupervisorAgent._build_report_prompt
    )
    assert "MECHANISM_EVIDENCE_POLICY" in inspect.getsource(
        PlanningMixin._generate_section_from_existing_figures
    )
