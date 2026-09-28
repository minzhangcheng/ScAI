"""Chat continuation context and report figure provenance."""

from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisState
from backend.models.schema_parts.analysis import AnalysisStep, CodeExecution
from backend.routes.chat_parts import core
from backend.services.report.analysis_log import AnalysisLogMixin
from backend.services.report.chat_sections import build_chat_sections_async
from backend.services.report.lineage import build_lineage_report_state
from backend.services.report.generator import DetailedReportGenerator


def test_chat_context_exposes_unfinished_review_without_round_credit(tmp_path, monkeypatch):
    state = AnalysisState(id="chat-context", user_id="owner")
    state.chat_analysis_reviews.append({
        "id": "chat_review_prior", "status": "accept",
        "request": "Compare the selected cell population",
        "execution_incomplete": True,
        "cycles": [{"execution_ids": ["exec_prior"]}],
        "unexecuted_revision": "Check the remaining samples",
    })
    monkeypatch.setattr(core, "get_session_workspace", lambda _: tmp_path)
    monkeypatch.setattr(core, "resolve_published_report_dir", lambda *_: tmp_path)
    (tmp_path / "quick.png").write_bytes(b"display-only")

    context, figures = core._build_chat_context(state)

    assert "chat_review_prior" in context
    assert "exec_prior" in context
    assert "Check the remaining samples" in context
    assert figures[0]["figure_role"] == "quick_display"


@pytest.mark.asyncio
async def test_chat_report_includes_only_registered_publication_figures(tmp_path):
    for name in ("quick", "publication"):
        (tmp_path / f"{name}.png").write_bytes(b"figure")
    messages = [
        {"role": "user", "content": "Show the results"},
        {"role": "assistant", "content": "Here are the results", "figures": [
            {"name": "quick", "path": str(tmp_path / "quick.png"),
             "figure_role": "quick_display"},
            {"name": "publication", "path": str(tmp_path / "publication.png"),
             "figure_role": "publication", "artifact_id": "figure_1"},
        ]},
    ]

    sections = await build_chat_sections_async(
        messages, str(tmp_path), str(tmp_path),
        publication_figure_paths={"figure_1": str(tmp_path / "publication.png")},
    )

    assert [figure["name"] for figure in sections[0]["figures"]] == ["publication"]
    assert (await build_chat_sections_async(
        messages, str(tmp_path), str(tmp_path),
        publication_figure_paths={},
    ))[0]["figures"] == []


def test_report_projection_keeps_chat_identity_on_its_branch():
    state = AnalysisState(id="chat-report", user_id="owner")
    state.chat_analysis_reviews = [
        {"id": "main_review", "branch_id": "main"},
        {"id": "branch_review", "branch_id": "branch_a"},
    ]
    state.analysis_log.append(AnalysisStep(
        id="chat_step", step_number=1, step_type="report_findings",
        executor_run_kind="chat", executor_run_scope_id="chat_scope",
        chat_review_id="main_review",
    ))
    state.analysis_log.append(AnalysisStep(
        id="branch_step", step_number=1, step_type="report_findings",
        executor_run_kind="chat", executor_run_scope_id="branch_scope",
        chat_review_id="branch_review",
    ))
    state.code_history.extend([
        CodeExecution(
            id="main_execution", code="print(1)", executor_run_kind="chat",
            lineage_id="main", chat_review_id="main_review",
        ),
        CodeExecution(
            id="branch_execution", code="print(2)", executor_run_kind="chat",
            lineage_id="branch_a", chat_review_id="branch_review",
        ),
    ])

    report_state = build_lineage_report_state(state, "main")
    report_log = AnalysisLogMixin._build_full_analysis_log(None, report_state)

    assert [review["id"] for review in report_state.chat_analysis_reviews] == ["main_review"]
    assert report_log[0]["executor_run_kind"] == "chat"
    assert report_log[0]["chat_review_id"] == "main_review"
    assert [execution.id for execution in report_state.code_history] == ["main_execution"]
    assert [step.id for step in report_state.analysis_log] == ["chat_step"]
    assert report_state.round_results == []


@pytest.mark.asyncio
async def test_report_pi_receives_chat_review_status_and_receipt_ids():
    state = AnalysisState(id="chat-pi-report", user_id="owner")
    state.chat_analysis_reviews.append({
        "id": "chat_review_1", "status": "accept",
        "execution_incomplete": True,
        "cycles": [{"executor_status": "max_steps_reached",
                    "execution_ids": ["chat_execution_1"]}],
    })
    writer = DetailedReportGenerator.__new__(DetailedReportGenerator)
    writer._pi_chat = AsyncMock(return_value='{"analysis_workflow": "Recorded"}')

    await writer._pi_comprehensive_review("Code digest", {}, {}, state, "en")

    prompt = writer._pi_chat.call_args.args[0]
    assert "Separate Chat Analysis Reviews" in prompt
    assert "chat_execution_1" in prompt
    assert '"execution_incomplete": true' in prompt
    assert "separate from approved round plans" in prompt
