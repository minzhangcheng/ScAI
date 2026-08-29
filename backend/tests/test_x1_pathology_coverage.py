"""X1 contract tests for retained pathology coverage observations."""

import importlib
import json
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, RoundResult
from backend.routes.serialization import serialize_round_results
from backend.services.report.analysis_log import build_round_summaries
from backend.services.report.planning import PlanningMixin
from backend.services.report.quality import (
    build_quality_disclosures,
    render_report_limitations_html,
)
from backend.workflow.synthesis import build_round_summary_payload


def _pathology_node_result() -> dict:
    invalid_batches = [
        {
            "batch_index": index,
            "tile_start": index * 5,
            "tile_end": index * 5 + 4,
            "attempts": 2,
            "errors": ["invalid response"],
        }
        for index in range(25)
    ]
    records = [{
        "image_id": "slide-1",
        "image_type": "H&E",
        "status": "partial",
        "findings": ["Finding from the validated subset."],
        "tile_validation": {
            "status": "partial",
            "expected_tiles": 45,
            "valid_tiles": 25,
            "coverage_ratio": 25 / 45,
            "invalid_batches": invalid_batches,
        },
    }]
    records.extend({
        "image_id": f"slide-{index}",
        "image_type": "IHC",
        "tile_validation": {
            "status": "completed",
            "expected_tiles": 1,
            "valid_tiles": 1,
            "coverage_ratio": 1.0,
            "invalid_batches": [],
        },
    } for index in range(2, 106))
    return {"status": "partial", "findings": [], "detailed_results": records}


def _extract_coverage(pathology_result: dict):
    synthesis = importlib.import_module("backend.workflow.synthesis")
    extract = getattr(synthesis, "extract_pathology_coverage", None)
    assert callable(extract), "pathology coverage extraction is not wired into synthesis"
    return extract(pathology_result)


def _record_dict(record):
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    assert isinstance(record, dict)
    return record


def test_pathology_tile_coverage_is_bounded_and_survives_round_api_and_websocket_serialization():
    coverage = _extract_coverage(_pathology_node_result())

    assert len(coverage) == 100
    first = _record_dict(coverage[0])
    assert first["image_id"] == "slide-1"
    assert first["image_type"] == "H&E"
    assert first["status"] == "partial"
    assert first["expected_tiles"] == 45
    assert first["valid_tiles"] == 25
    assert first["coverage_ratio"] == pytest.approx(25 / 45)
    assert len(first["invalid_batches"]) == 20
    json.dumps(first, allow_nan=False)

    assert "pathology_coverage" in getattr(RoundResult, "model_fields", {})
    round_result = RoundResult(round_number=3, pathology_coverage=coverage[:1])
    state = AnalysisState(user_id="owner", current_round=3, round_results=[round_result])

    api_round = serialize_round_results([round_result])[0]
    assert api_round["pathology_coverage"][0]["valid_tiles"] == 25
    persisted_round = AnalysisState._serialize_round_results(
        state.model_dump(mode="json")["round_results"]
    )[0]
    assert persisted_round["pathology_coverage"][0]["expected_tiles"] == 45
    assert build_round_summaries(state)[0]["pathology_coverage"][0]["status"] == "partial"
    ws_payload = build_round_summary_payload(
        round_result, state, effective_max_rounds=5,
    )
    assert ws_payload["pathology_coverage"][0]["image_type"] == "H&E"


def test_partial_pathology_coverage_is_exactly_disclosed_in_report_limitations():
    coverage = _extract_coverage(_pathology_node_result())[:1]
    assert "pathology_coverage" in getattr(RoundResult, "model_fields", {})
    state = AnalysisState(
        user_id="owner",
        round_results=[RoundResult(round_number=4, pathology_coverage=coverage)],
    )

    quality = build_quality_disclosures(state)
    assert any("25/45 validated tiles" in item for item in quality["cautions"])

    html = render_report_limitations_html({
        "status": "partial",
        "quality_disclosures": quality,
    })
    assert 'class="report-limitations-details"' in html
    assert "25/45 validated tiles" in html
    assert "55.6%" in html
    assert "omitted tiles were not assessed" in html


def test_missing_pathology_coverage_is_neutral_and_nonblocking():
    state = AnalysisState(
        user_id="owner",
        round_results=[RoundResult(round_number=1)],
    )
    quality = build_quality_disclosures(state)
    assert quality["blocking_issues"] == []
    assert not any("pathology" in item.lower() for item in quality["cautions"])

    payload = build_round_summary_payload(
        state.round_results[0], state, effective_max_rounds=3,
    )
    assert payload["pathology_coverage"] == []


@pytest.mark.asyncio
async def test_pathology_coverage_qualification_reaches_nonempty_report_plan_prompt():
    class _Planner(PlanningMixin):
        def __init__(self):
            self.prompt = ""

        def _build_score_definitions_context(self, _state):
            return ""

        async def _pi_chat(self, prompt, **_kwargs):
            self.prompt = prompt
            return '{"title":"Report","sections":[{"id":"results","title":"Results","description":"d","existing_figures":[],"existing_tables":[]}]}'

    coverage = _record_dict(_extract_coverage(_pathology_node_result())[0])
    planner = _Planner()
    state = SimpleNamespace(
        name="Report",
        initial_instruction="Summarize the analysis",
        exploration_report=None,
        user_chosen_directions=[],
        user_chosen_direction="",
        round_results=[],
    )
    analysis_record = {
        "publication_figures": [],
        "round_summaries": [{
            "round_number": 4,
            "round_status": "partial",
            "evidence_status": "substantive",
            "pi_summary": "partial pathology result",
            "pathology_findings": [],
            "pathology_coverage": [coverage],
            "bioinformatics_findings": [],
            "biochemistry_findings": [],
            "literature_findings": [],
        }],
        "analysis_steps": [],
        "total_rounds": 1,
        "total_code_cells": 0,
    }

    await planner._plan_report_from_analysis(
        state,
        analysis_record,
        findings={"summary": "summary"},
        existing_figures=[],
        existing_tables=[],
        language="English",
        analysis_summary={"methodology_overview": "existing non-empty summary"},
    )

    assert "25/45 validated tiles" in planner.prompt
    assert "omitted tiles were not assessed" in planner.prompt
    assert "Do not infer findings for omitted tiles" in planner.prompt
