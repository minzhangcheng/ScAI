"""Focused report follow-up projection and static HTML contracts."""

import html
import json
from types import SimpleNamespace

from backend.models.schemas import AnalysisState, RoundResult
from backend.services.report.generator_parts import html as html_parts
from backend.services.report.generator_parts.support import build_report_followups
from backend.services.report.lineage import build_lineage_report_state
from backend.services.report import regenerator
from backend.utils.report_html import markdown_to_html


def _state_with_followups() -> AnalysisState:
    return AnalysisState(
        id="report-followups",
        user_id="owner",
        round_results=[
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Early selected direction",
                lineage_id="main",
                pi_summary="Main direction result",
            ),
            RoundResult(
                round_number=2,
                phase="innovation",
                direction="Branch direction",
                lineage_id="branch-a",
                branch_id="branch-a",
                pi_summary="Branch-only result",
            ),
        ],
        post_execution_reviews=[
            {
                "direction_pass_id": "main:innovation:2:early selected direction",
                "direction_key": "early selected direction",
                "direction": "Early selected direction",
                "lineage_id": "main",
                "round_number": 2,
                "followups": [{
                    "followup_id": "fu-main",
                    "question": "Which evidence would distinguish the alternatives?",
                    "why_it_matters": "It would refine the interpretation.",
                    "disposition": "offer_in_chat",
                    "reason": "The user can prioritize this broader question.",
                    "suggested_instruction": "Compare the alternatives using the retained results.",
                    "source_round": 2,
                    "direction": "Early selected direction",
                }],
            },
            {
                "direction_key": "branch direction",
                "direction": "Branch direction",
                "lineage_id": "branch-a",
                "round_number": 2,
                "followups": [{
                    "followup_id": "fu-branch",
                    "question": "Branch-only question",
                    "disposition": "needs_new_data",
                    "required_input": "A new measurement",
                }],
            },
        ],
    )


def test_report_followups_are_lineage_scoped_and_json_ready():
    state = _state_with_followups()

    main_state = build_lineage_report_state(state, "main")
    branch_state = build_lineage_report_state(state, "branch-a")
    main_followups = build_report_followups(main_state, lineage_id="main")
    branch_followups = build_report_followups(branch_state, lineage_id="branch-a")

    assert [item["followup_id"] for item in main_followups] == ["fu-main"]
    assert [item["followup_id"] for item in branch_followups] == ["fu-branch"]
    assert main_followups[0]["question"].startswith("Which evidence")
    assert main_followups[0]["suggested_instruction"]
    assert json.dumps({"followups": main_followups})
    assert [item["followup_id"] for item in state.post_execution_reviews[0]["followups"]] == ["fu-main"]


def test_generated_and_regenerated_html_render_the_same_readable_followup():
    state = _state_with_followups()
    state.post_execution_reviews[0]["followups"][0]["why_it_matters"] = (
        "It tests **A & B < C**."
    )
    state.post_execution_reviews[0]["followups"][0]["suggested_instruction"] = (
        "Compare the **retained** results."
    )
    report = {"followups": build_report_followups(state, lineage_id="main")}
    bindings = SimpleNamespace(
        html_module=html,
        markdown_to_html=markdown_to_html,
    )

    generated_toc, generated_sections = [], []
    html_parts._append_followups_section(
        report,
        toc_items=generated_toc,
        sections_html=generated_sections,
        bindings=bindings,
    )
    regenerated_toc, regenerated_sections = [], []
    regenerator._append_followups_section(
        report,
        regenerated_toc,
        regenerated_sections,
    )

    generated = "\n".join(generated_toc + generated_sections)
    regenerated = "\n".join(regenerated_toc + regenerated_sections)
    for rendered in (generated, regenerated):
        assert "Which evidence would distinguish the alternatives?" in rendered
        assert "Why it matters" in rendered
        assert "Continue in Chat" in rendered
        assert "<strong>A &amp; B &lt; C</strong>" in rendered
        assert "<strong>retained</strong>" in rendered
        assert rendered.count("&amp;") == 1
        assert rendered.count("&lt;") == 1
        assert "&amp;amp;" not in rendered
        assert "&amp;lt;" not in rendered
        assert "fu-main" not in rendered
    assert "Which evidence would distinguish the alternatives?" in generated
    assert "Which evidence would distinguish the alternatives?" in regenerated


def test_empty_followup_list_omits_static_section():
    generated_toc, generated_sections = [], []
    html_parts._append_followups_section(
        {"followups": []},
        toc_items=generated_toc,
        sections_html=generated_sections,
        bindings=SimpleNamespace(html_module=html, markdown_to_html=lambda value: value),
    )
    regenerated_toc, regenerated_sections = [], []
    regenerator._append_followups_section({}, regenerated_toc, regenerated_sections)

    assert generated_toc == generated_sections == []
    assert regenerated_toc == regenerated_sections == []
