"""Fallback report planning must retain registered plan lineage."""

from types import SimpleNamespace

import pytest

from backend.services.report.planning import PlanningMixin
from backend.services.report.generator import ensure_report_plan_figure_coverage


def _existing_figure(tmp_path):
    image_path = tmp_path / "figures" / "planned.png"
    image_path.parent.mkdir(exist_ok=True)
    image_path.write_bytes(b"placeholder")
    return {
        "filename": "planned.png",
        "name": "Planned panel",
        "description": "",
        "png_path": str(image_path),
        "pdf_path": None,
        "inferred_type": "qc",
        "plan_step_id": 7,
        "plan_id": "plan-v3",
        "purpose": "result",
    }


def test_fallback_plan_keeps_figure_plan_provenance(tmp_path):
    planner = PlanningMixin()
    figure = _existing_figure(tmp_path)
    state = SimpleNamespace(name="Report")

    plan = planner._create_fallback_plan_with_existing_figures(state, [figure])
    planned = next(
        item
        for section in plan["sections"]
        for item in section["existing_figures"]
    )

    assert planned["plan_step_id"] == 7
    assert planned["plan_id"] == "plan-v3"
    assert planned["purpose"] == "result"


def test_fallback_plan_keeps_unknown_and_future_figure_types(tmp_path):
    planner = PlanningMixin()
    figures = []
    for name, inferred_type in (
        ("trajectory.png", "trajectory"),
        ("network.png", "custom_network_v2"),
    ):
        figure = _existing_figure(tmp_path)
        figure.update({"filename": name, "inferred_type": inferred_type})
        figures.append(figure)

    plan = planner._create_fallback_plan_with_existing_figures(
        SimpleNamespace(name="Report"), figures
    )
    planned_names = [
        figure["filename"]
        for section in plan["sections"]
        for figure in section["existing_figures"]
    ]

    assert planned_names == ["trajectory.png", "network.png"]
    assert plan["sections"][-1]["id"] == "additional_results"


def test_checkpoint_plan_is_reconciled_with_every_scanned_figure(tmp_path):
    included = _existing_figure(tmp_path)
    included["filename"] = "included.png"
    recovered = _existing_figure(tmp_path)
    recovered["filename"] = "recovered.png"
    old_plan = {
        "title": "Old checkpoint",
        "sections": [{
            "id": "results",
            "title": "Results",
            "existing_figures": [included],
        }],
    }

    plan = ensure_report_plan_figure_coverage(
        old_plan, [included, recovered]
    )

    assert [
        figure["filename"]
        for section in plan["sections"]
        for figure in section.get("existing_figures", [])
    ] == ["included.png", "recovered.png"]
    assert plan["sections"][-1]["id"] == "additional_results"


def test_checkpoint_recovery_does_not_reuse_a_completed_catch_all(tmp_path):
    recovered = _existing_figure(tmp_path)
    recovered["filename"] = "recovered.png"
    old_plan = {
        "sections": [{
            "id": "additional_results",
            "title": "Already completed",
            "existing_figures": [],
        }],
    }

    plan = ensure_report_plan_figure_coverage(old_plan, [recovered])

    assert [section["id"] for section in plan["sections"]] == [
        "additional_results", "additional_results_2",
    ]
    assert plan["sections"][-1]["existing_figures"][0]["filename"] == "recovered.png"


@pytest.mark.asyncio
async def test_fallback_section_output_keeps_figure_plan_provenance(tmp_path):
    class Harness(PlanningMixin):
        async def _pi_chat(self, *args, **kwargs):
            return ""

    planner = Harness()
    planner.figures_dir = tmp_path / "figures"
    planner.tables_dir = tmp_path / "tables"
    planner.figures_dir.mkdir()
    planner.tables_dir.mkdir()
    planner.figure_images_cache = {}
    planner.generated_figures = []
    planner.generated_tables = []
    figure = _existing_figure(tmp_path)
    state = SimpleNamespace(name="Report", initial_instruction="Summarize")
    section_plan = {
        "id": "results",
        "title": "Results",
        "description": "Result figures",
        "analysis_type": "general",
        "existing_figures": [figure],
        "existing_tables": [],
    }

    section = await planner._generate_section_from_existing_figures(
        state, section_plan, {"summary": "", "bioinformatics": []}, "en"
    )

    assert section["figures"][0]["plan_step_id"] == 7
    assert section["figures"][0]["plan_id"] == "plan-v3"
    assert section["figures"][0]["purpose"] == "result"
