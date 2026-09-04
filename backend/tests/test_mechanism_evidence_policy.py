import inspect

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.pi_agent import PIAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.services.report.planning import PlanningMixin
from backend.services.report import planning as report_planning
from backend.utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from backend.agents.supervisor import report_prompt
from backend.agents.bioinformatics.tool_schemas import TOOLS
from backend.workflow.agents import (
    _build_stage2_instruction, _build_exploration_instruction,
)


def _mechanism_program_field(name: str) -> dict:
    tool = next(
        t for t in TOOLS if t["function"]["name"] == "register_mechanism_programs"
    )
    programs = tool["function"]["parameters"]["properties"]["programs"]
    return programs["items"]["properties"][name]


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


def test_policy_separates_undemonstrated_from_undemonstrable():
    """A mechanism no instrument in the study can reach is an assay limit, not a result.

    Agents were reporting "susceptibility, not execution" as a finding when it
    was a restatement of what the modality can never show, and then declining to
    rank the candidates they had already graded.
    """
    low = MECHANISM_EVIDENCE_POLICY.lower()
    assert "strongest class of evidence" in low
    assert "no measurement in this study could demonstrate" in low
    assert "limit of the assay, not a finding" in low
    assert "rank the remaining candidates" in low
    assert "explicit inference" in low


def test_policy_weighs_counter_regulation_in_both_directions():
    """Negative feedback makes a one-directional reading wrong either way.

    An induced compensatory program is not a refutation of the pressure that
    provoked it, and it is not proof that the pressure prevailed. The escape
    hatch is closed deliberately: "that observation is only compensation" has to
    carry its own support, or it explains away every inconvenient result.
    """
    low = MECHANISM_EVIDENCE_POLICY.lower()
    assert "counter-regulation rather than its absence" in low
    assert "rather than proof that the pressure prevailed" in low
    assert "weigh both directions together" in low
    assert "not available whenever an observation disagrees" in low


def test_policy_treats_a_missing_population_as_an_observation_without_presuming_fate():
    """Depletion is data, and it is not evidence of any particular fate.

    This is a general-purpose platform: a population absent from the capture may
    have changed identity, been lost in handling, or left the sampled
    compartment, so the policy may not reach for survival vocabulary, which
    presumes the question it is supposed to leave open.
    """
    low = MECHANISM_EVIDENCE_POLICY.lower()
    assert "depleted or absent from the captured data" in low
    assert "an observation to be explained" in low
    assert "change of state or identity" in low
    assert "dissociation or capture" in low
    for fate_presuming in ("survivor", "surviving", "survive", "survival"):
        assert fate_presuming not in low


def test_cross_modal_synthesis_reports_gaps_alongside_conclusions():
    """Every candidate still declares its gaps; the gap no longer voids the verdict.

    The blanket instruction to present mechanisms as "supported but incomplete"
    rather than conclusive applied to high-confidence candidates too, so a
    candidate the analysis had graded "high" was still written up as unsettled.
    """
    src = inspect.getsource(PIAgent.build_cross_modal_synthesis)
    assert "rather than conclusive" not in src
    assert "missing_evidence for every candidate" in src


def test_final_summary_conclusions_commit_to_a_ranked_mechanism():
    src = inspect.getsource(PIAgent.generate_final_summary)
    assert "most probable" in src.lower()
    assert "labelled as an inference" in src.lower() or "label it as an inference" in src.lower()


def test_mechanism_interpretation_field_does_not_contradict_the_ranking_rule():
    """The field description sits closer to the output than the tool description.

    The policy requires naming the most probable mechanism once the modality
    ceiling is reached, but the field that actually holds the verdict carried a
    blanket "Cautious interpretation; do not claim confirmation" a few tokens
    from the point of generation, which is the instruction a model follows.
    Overclaiming must still be forbidden -- the field has to bound the claim
    without voiding it.
    """
    low = _mechanism_program_field("interpretation")["description"].lower()
    assert "cautious interpretation" not in low
    assert "most probable" in low
    assert "do not claim" in low


def test_mechanism_capture_instruction_does_not_ask_for_a_cautious_interpretation():
    """The sentence after the injected policy must not take back what it says.

    _MECHANISM_CAPTURE_INSTRUCTION interpolates the policy and then closed with
    "and a cautious interpretation", so every phase instruction carried the
    ranking rule and its retraction in the same paragraph.
    """
    for instr in (
        _build_stage2_instruction(has_bulk=True, has_scrna=True),
        _build_exploration_instruction(has_bulk=True, has_scrna=True),
    ):
        low = instr.lower()
        assert "cautious interpretation" not in low
        assert "interpretation stating what the evidence supports" in low


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
