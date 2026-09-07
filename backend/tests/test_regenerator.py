import pytest
import json
import re
from pathlib import Path
from backend.services.report.regenerator import _build_analysis_log_from_report


def test_build_analysis_log_from_report_ignores_extra_fields():
    report = {
        "analysis_log": [
            {
                "step_number": 1,
                "step_type": "execute_code",
                "timestamp": "2024-01-01T00:00:00",
                "agent_reasoning": "Test reasoning",
                "round_number": 1,
                "code": "print('hello')",
                "code_output": "hello",
                "unknown_field": "should be ignored",
                "another_extra": 42,
            },
        ],
    }
    html = _build_analysis_log_from_report(report, "en", Path("/tmp/figures"))
    assert "execute_code" in html or "Code" in html
    assert "unknown_field" not in html


def test_build_analysis_log_from_report_empty_log():
    report = {"analysis_log": []}
    html = _build_analysis_log_from_report(report, "en", Path("/tmp/figures"))
    assert html == ""


def test_build_analysis_log_from_report_missing_log():
    report = {}
    html = _build_analysis_log_from_report(report, "en", Path("/tmp/figures"))
    assert html == ""


def test_showing_uses_total_rows():
    from backend.services.report.regenerator import _rows_caption
    caption = _rows_caption(preview_len=10, total_rows=5000)
    assert "5000" in caption and "10" in caption

    caption_small = _rows_caption(preview_len=10, total_rows=5)
    assert caption_small == ""


def test_build_analysis_log_from_report_multiple_step_types():
    report = {
        "analysis_log": [
            {"step_number": 1, "step_type": "execute_code", "timestamp": "2024-01-01T00:00:00", "agent_reasoning": None, "round_number": 1, "code": "x=1", "code_output": None, "code_error": None, "code_description": None, "code_step_type": None, "saved_figure_filenames": []},
            {"step_number": 2, "step_type": "text_response", "timestamp": "2024-01-01T00:01:00", "agent_reasoning": None, "round_number": 1, "text_content": "Summary text"},
            {"step_number": 3, "step_type": "consult_pi", "timestamp": "2024-01-01T00:02:00", "agent_reasoning": None, "round_number": 2, "pi_question": "What next?", "pi_context": "ctx", "pi_options": ["A", "B"]},
        ],
    }
    html = _build_analysis_log_from_report(report, "en", Path("/tmp/figures"))
    assert "execute_code" in html or "Code" in html
    assert "text_response" in html or "Response" in html
    assert "consult_pi" in html or "PI" in html


@pytest.mark.asyncio
async def test_regeneration_and_figureless_generator_html_are_stable(tmp_path):
    from backend.services.report.generator import DetailedReportGenerator
    from backend.services.report.regenerator import generate_html_report

    report = {
        "session_id": "session",
        "title": "Figureless",
        "abstract": "A",
        "conclusions": "C",
        "generated_at": "2026-01-01T00:00:00",
        "status": "partial",
        "current_step": "Technical review availability recorded",
        "partial_reason": "analysis_quality",
        "partial_reasons": ["analysis_quality", "figure_coverage"],
        "sections": [{
            "id": "results",
            "title": "Results",
            "analysis_type": "general",
            "content": "No figures",
            "figures": [],
            "tables": [],
        }],
        "score_definitions": [],
        "graphic_abstract": {
            "review_available": False,
            "selection_available": False,
        },
        "quality_disclosures": {
            "status": "incomplete",
            "cautions": ["Scientific evidence remains limited."],
            "post_execution_reviews": [{
                "review_status": "failed",
                "review_reason": "API <down & retry>",
            }],
        },
    }
    original_report = json.loads(json.dumps(report))
    regenerated_dir = tmp_path / "regenerated"
    (regenerated_dir / "figures").mkdir(parents=True)
    (regenerated_dir / "tables").mkdir()
    (regenerated_dir / "report.json").write_text(
        json.dumps(report), encoding="utf-8",
    )
    regenerated = generate_html_report(regenerated_dir)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    generator = DetailedReportGenerator(
        object(), "pi", "bio", str(tmp_path / "generated"), str(workspace),
    )
    generated = await generator._generate_html_report(report, "en", state=None)

    assert report == original_report

    # What this used to assert: two hardcoded sha256 digests of the whole page.
    # 0.49.3 deliberately replaced the status banner and quality-disclosure
    # sections with one consolidated limitations block, the digests were never
    # re-baselined, and the test sat red for two releases saying only "the bytes
    # changed" -- which is why the content 0.49.3 dropped went unnoticed. Assert
    # the properties a regression would actually violate instead.
    regenerated_html = Path(regenerated).read_text(encoding="utf-8")
    generated_html = Path(generated).read_text(encoding="utf-8")
    persisted_report = json.loads(
        (regenerated_dir / "report.json").read_text(encoding="utf-8")
    )
    assert persisted_report == original_report

    for label, page in (("regenerator", regenerated_html), ("generator", generated_html)):
        # Every template slot is filled: an unfilled one renders as literal
        # "{placeholder}" in a shipped report.
        leftovers = sorted(set(re.findall(r"\{[a-z_]{4,}\}", page)))
        assert not leftovers, f"{label}: unresolved template placeholders {leftovers}"
        # No Python object repr or traceback leaked into the page.
        assert "Traceback" not in page, f"{label}: traceback in report"
        assert "<built-in" not in page and "object at 0x" not in page
        # The report's own content survived the template.
        assert "Figureless" in page, f"{label}: title missing"
        assert "Results" in page, f"{label}: section title missing"
        assert "No figures" in page, f"{label}: section content missing"
        assert "technical-review-notice" in page
        assert "API &lt;down &amp; retry&gt;" in page
        # A report with nothing to qualify carries no limitations block.
        assert "Detailed evidence qualifications" not in page, (
            f"{label}: clean report grew an empty limitations block"
        )

    # The two producers render the same report from the same template and must
    # not drift apart. They differ only in indentation, so compare on collapsed
    # whitespace -- the old test pinned them to two INDEPENDENT digests, which
    # could not have caught a divergence between them at all.
    def _collapsed(page):
        page = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "<TIMESTAMP>", page)
        page = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<UUID>", page)
        return re.sub(r"\s+", " ", page).strip()

    assert _collapsed(regenerated_html) == _collapsed(generated_html), (
        "regenerator and generator HTML have diverged"
    )


@pytest.mark.asyncio
async def test_qualified_report_omits_deterministic_blocks_but_retains_json(tmp_path):
    """Qualified state remains auditable without a duplicate HTML block."""
    from backend.services.report.regenerator import generate_html_report

    report = {
        "session_id": "session",
        "title": "Qualified",
        "abstract": "A",
        "conclusions": "C",
        "generated_at": "2026-01-01T00:00:00",
        "status": "partial",
        "partial_reasons": ["analysis_quality"],
        "sections": [{
            "id": "results",
            "title": "Results",
            "analysis_type": "general",
            "content": "Body",
            "figures": [],
            "tables": [],
        }],
        "score_definitions": [],
        "quality_disclosures": {
            "blocking_issues": ["Round 2 evidence is unverified."],
            "cautions": ["Only one biological replicate per condition."],
            "post_execution_reviews": [{
                "round_number": 2,
                "plan_audit": {"unmet_titles": ["Cell communication"]},
                "assessment": {"remaining_gaps": ["Validate interaction"]},
            }],
        },
    }
    out_dir = tmp_path / "qualified"
    (out_dir / "figures").mkdir(parents=True)
    (out_dir / "tables").mkdir()
    (out_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    page = Path(generate_html_report(out_dir)).read_text(encoding="utf-8")

    assert '<div class="report-status-line' not in page
    assert '<details class="report-limitations-details"' not in page
    assert "Detailed evidence qualifications" not in page
    assert "Round 2 evidence is unverified" not in page
    retained = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert retained["status"] == "partial"
    assert retained["quality_disclosures"] == report["quality_disclosures"]
