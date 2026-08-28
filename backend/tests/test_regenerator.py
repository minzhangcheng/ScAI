import pytest
import hashlib
import json
import re
from pathlib import Path
from backend.services.report.regenerator import _build_analysis_log_from_report


def _normalized_html_digest(path):
    html = Path(path).read_text(encoding="utf-8")
    html = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "<TIMESTAMP>", html)
    html = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<UUID>", html,
    )
    return hashlib.sha256(html.encode()).hexdigest()


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
        "sections": [{
            "id": "results",
            "title": "Results",
            "analysis_type": "general",
            "content": "No figures",
            "figures": [],
            "tables": [],
        }],
        "score_definitions": [],
    }
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

    assert _normalized_html_digest(regenerated) == (
        "637825a2d64c3d400d7e00afacbaf92772c7715d2006972f5c5e66d96101eb98"
    )
    assert _normalized_html_digest(generated) == (
        "62beb06001ae0f21ec7abc96b9d8e8b05d6d35421b68853d1a03d32b97645a1b"
    )
