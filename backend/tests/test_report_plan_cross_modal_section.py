import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.report.planning import insert_cross_modal_section


def test_section_inserted_when_displayable_candidate_exists():
    plan = {"sections": [{"title": "Intro"}, {"title": "Conclusions"}]}
    round_summaries = [{
        "round_number": 2,
        "cross_modal_candidates": [
            {"theme": "iron injury", "mechanism_statement": "m", "confidence": "medium",
             "relation_to_current_direction": "uncovered", "report_recommendation": "secondary"},
            {"theme": "noise", "mechanism_statement": "n", "confidence": "low",
             "relation_to_current_direction": "uncovered", "report_recommendation": "do_not_emphasize"},
        ],
    }]
    out = insert_cross_modal_section(plan, round_summaries)
    titles = [s["title"] for s in out["sections"]]
    assert any("Cross-Modal Mechanism Candidates" in t for t in titles)
    sec = next(s for s in out["sections"] if "Cross-Modal" in s["title"])
    assert len(sec["cross_modal_candidates"]) == 1
    assert sec["cross_modal_candidates"][0]["theme"] == "iron injury"


def test_no_section_when_only_do_not_emphasize():
    plan = {"sections": [{"title": "Intro"}]}
    round_summaries = [{
        "round_number": 1,
        "cross_modal_candidates": [
            {"theme": "x", "mechanism_statement": "m", "confidence": "low",
             "relation_to_current_direction": "uncovered", "report_recommendation": "do_not_emphasize"},
        ],
    }]
    out = insert_cross_modal_section(plan, round_summaries)
    assert all("Cross-Modal" not in s["title"] for s in out["sections"])


def test_no_section_when_no_candidates():
    plan = {"sections": [{"title": "Intro"}]}
    out = insert_cross_modal_section(plan, [{"round_number": 1, "cross_modal_candidates": []}])
    assert all("Cross-Modal" not in s["title"] for s in out["sections"])


def test_ordering_headline_first_future_last():
    plan = {"sections": [{"title": "Intro"}]}
    rs = [{"round_number": 1, "cross_modal_candidates": [
        {"theme": "fut", "mechanism_statement": "m", "confidence": "low",
         "relation_to_current_direction": "uncovered", "report_recommendation": "future_direction"},
        {"theme": "head", "mechanism_statement": "m", "confidence": "high",
         "relation_to_current_direction": "central", "report_recommendation": "headline"},
        {"theme": "sec", "mechanism_statement": "m", "confidence": "medium",
         "relation_to_current_direction": "complementary", "report_recommendation": "secondary"},
    ]}]
    out = insert_cross_modal_section(plan, rs)
    sec = next(s for s in out["sections"] if "Cross-Modal" in s["title"])
    order = [c["theme"] for c in sec["cross_modal_candidates"]]
    assert order == ["head", "sec", "fut"]


def test_inserted_before_chinese_conclusion_section():
    plan = {"sections": [{"title": "结果"}, {"title": "结论"}]}
    rs = [{"round_number": 1, "cross_modal_candidates": [
        {"theme": "t", "mechanism_statement": "m", "confidence": "medium",
         "relation_to_current_direction": "uncovered", "report_recommendation": "secondary"},
    ]}]
    out = insert_cross_modal_section(plan, rs)
    titles = [s["title"] for s in out["sections"]]
    assert titles.index("Cross-Modal Mechanism Candidates") < titles.index("结论")


def test_does_not_mutate_input_plan():
    plan = {"sections": [{"title": "Intro"}]}
    rs = [{"round_number": 1, "cross_modal_candidates": [
        {"theme": "t", "mechanism_statement": "m", "confidence": "low",
         "relation_to_current_direction": "uncovered", "report_recommendation": "secondary"},
    ]}]
    insert_cross_modal_section(plan, rs)
    assert len(plan["sections"]) == 1


import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_cross_modal_section_prompt_contains_candidates():
    from backend.services.report.planning import PlanningMixin

    class _Gen(PlanningMixin):
        def __init__(self):
            self.figure_images_cache = {}
            self.generated_figures = []
            self.generated_tables = []
            self.figures_dir = Path("/tmp")
            self.tables_dir = Path("/tmp")

    gen = _Gen()
    gen._pi_chat = AsyncMock(return_value='{"narrative": "n", "figures": []}')

    state = MagicMock()
    state.name = "proj"
    state.initial_instruction = "obj"
    section_plan = {
        "title": "Cross-Modal Mechanism Candidates",
        "analysis_type": "cross_modal",
        "description": "desc",
        "existing_figures": [],
        "existing_tables": [],
        "cross_modal_candidates": [
            {"theme": "THEME_MARKER", "mechanism_statement": "MECH_MARKER",
             "confidence": "medium", "relation_to_current_direction": "uncovered",
             "report_recommendation": "secondary"},
        ],
    }
    await gen._generate_section_from_existing_figures(
        state, section_plan, findings={"bioinformatics": [], "summary": ""}, language="English",
    )
    prompt = gen._pi_chat.call_args[0][0]
    assert "THEME_MARKER" in prompt
    assert "MECH_MARKER" in prompt
