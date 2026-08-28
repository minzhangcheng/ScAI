import inspect

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.pi_agent import PIAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.services.report.planning import PlanningMixin
from backend.services.report import planning as report_planning
from backend.utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from backend.agents.supervisor import report_prompt


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


def test_policy_leads_with_recall_so_caution_does_not_dominate():
    """Order matters: the clause an agent reads first is the one it applies.

    Leading with the caution produced agents that described a mechanism their
    own results had already identified in generic terms rather than naming it.
    """
    low = MECHANISM_EVIDENCE_POLICY.lower()
    assert low.index("name it explicitly") < low.index("do not assume or privilege")
    assert "never drop one silently" in low


def test_bioinformatics_planning_prompts_use_shared_policy():
    """Planning is where a mechanism is scheduled to be measured at all.

    The policy previously reached only the PI's prompts, so plan drafting and
    revision — the steps that decide what gets computed — ran without it.
    """
    for method in (
        BioinformaticsAgent.generate_analysis_plan,
        BioinformaticsAgent.revise_plan_incrementally,
    ):
        assert "MECHANISM_EVIDENCE_POLICY" in inspect.getsource(method)


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
        report_prompt._build_report_prompt
    )
    assert "MECHANISM_EVIDENCE_POLICY" in inspect.getsource(
        report_planning._build_existing_figure_section_prompt
    )
