"""Round synthesis owns narrative order; final reports reuse it without rewriting."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.pi.synthesis import SynthesisMixin
from backend.agents.supervisor import report_content, report_presentation, report_prompt
from backend.agents.supervisor_agent import SupervisorAgent
from backend.models.schemas import (
    AcceptedFoundationInventory, AcceptedFoundationStep,
    AnalysisPlan, AnalysisState, CodeExecution, FigureRevisionTask, FigureReviewStatus,
    PlanStep, PublicationFigure, PublicationTable, RoundResult,
)
from backend.services.report.round_chapters import (
    _chapter_analysis_context,
    assemble_round_chapter, chapter_markdown, report_artifact_names,
    reusable_sections, round_artifact_catalog,
    saved_figure_captions,
)
from backend.services.report.report_outline import build_report_outline
from backend.services.report.variants_parts.detailed import generate_detailed_report_segment_html
from backend.tests.figure_image_helpers import write_valid_image
from backend.utils.formatting import safe_parse_json


def _state(tmp_path):
    figures = []
    for index, name in enumerate(("late_result", "baseline")):
        path = tmp_path / f"{name}.jpg"
        payload = write_valid_image(path, color=(30 + index * 80, 90, 140), min_bytes=2200)
        figures.append(PublicationFigure(
            id=name, name=name, description=f"Original caption for {name}.",
            figure_type="other", png_path=str(path), pdf_path="",
            render_sha256=hashlib.sha256(payload).hexdigest(), pdf_sha256="2" * 64,
            review_status=FigureReviewStatus.APPROVED, round_number=1,
        ))
    return AnalysisState(user_id="u", name="Study", current_round=1, publication_figures=figures)


def _chapter(state):
    return assemble_round_chapter({
        "title": "Evidence in context",
        "blocks": [
            {"text": "First establish the baseline.", "artifact_ids": ["baseline"]},
            {"text": "Then interpret the result.", "artifact_ids": ["late_result"]},
        ],
        "conclusions": "The evidence is descriptive.",
    }, round_artifact_catalog(state), "Summary.", 1)


@pytest.mark.asyncio
async def test_pi_synthesis_composes_and_persists_chapter_in_one_call(tmp_path):
    state = _state(tmp_path)
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = lambda text, default: json.loads(text)
    pi._extract_next_steps = lambda data: []
    chapter = _chapter(state)
    pi.chat = AsyncMock(return_value=json.dumps({
        "summary": "Summary.", "report_chapter": chapter.model_dump(),
    }))
    result = await pi.synthesize_round(state, bioinformatics_result="Measured observations.")
    pi.chat.assert_awaited_once()
    prompt = pi.chat.call_args.args[0]
    assert "catalog order" in prompt
    assert "Original caption for baseline." in prompt
    state.round_results = [result]
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert loaded.round_results[0].report_chapter == chapter
    assert loaded.round_results[0].report_chapter.blocks[0].artifact_ids == ["baseline"]


@pytest.mark.asyncio
async def test_synthesis_places_late_figure_with_its_analysis_step(tmp_path):
    state = _state(tmp_path)
    state.current_plan = AnalysisPlan(id="plan", phase="exploration", round_number=1, steps=[
        PlanStep(id=20, title="Initial measurement", method="Recorded method", status="completed"),
        PlanStep(id=10, title="Downstream comparison", method="Comparison", depends_on=[20]),
    ])
    supplement, baseline = state.publication_figures
    for figure in (supplement, baseline):
        figure.plan_id = "plan"
        figure.plan_step_id = 20
        figure.source_execution_id = "measurement"
    supplement.producer_run_kind = "figure_followup"
    supplement.figure_followup_task_id = "followup"
    supplement.revision_of_id = "old-render"
    state.figure_revision_tasks = [FigureRevisionTask(
        id="followup", logical_figure_id="late_result", reason="visual_revision",
        plan_id="plan", plan_step_id=20,
        pi_instruction="Clarify the diagnostic view of the retained measurement.",
    )]
    state.code_history = [CodeExecution(
        id="measurement", code="recorded_measurement()", round_number=1,
        plan_id="plan", plan_step_ids=[20], execution_status="succeeded",
    )]
    baseline.plan_id = "earlier-plan"
    baseline.source_execution_id = "earlier-measurement"
    state.code_history.append(CodeExecution(
        id="earlier-measurement", code="earlier_measurement()", round_number=1,
        plan_id="earlier-plan", plan_step_ids=[20], execution_status="succeeded",
    ))
    earlier = AnalysisPlan(id="earlier-plan", phase="exploration", round_number=1, steps=[
        PlanStep(id=20, title="Earlier measurement", method="Earlier method"),
    ])
    state.plan_dialogue_history = [{"round": 7, "plan": earlier.model_dump(mode="json")}]
    state.publication_figures.append(baseline.model_copy(update={"id": "other-round", "round_number": 2}))

    async def respond(prompt, **kwargs):
        context = json.loads(prompt.split(
            "(plans are intentions; execution links are not scientific acceptance):\n", 1
        )[1])
        assert [step["id"] for step in context["current_plan"]["steps"]] == [20, 10]
        assert context["current_plan"]["steps"][1]["depends_on"] == [20]
        assert context["historical_plans"][0]["id"] == "earlier-plan"
        assert context["historical_plans"][0]["steps"][0]["id"] == 20
        assert context["historical_plans"][0]["steps"][0]["method"] == "Earlier method"
        assert context["current_plan"]["steps"][0]["method"] == "Recorded method"
        assert context["missing_plan_ids"] == []
        artifacts = {item["id"]: item for item in context["artifacts"]}
        assert set(artifacts) == {"late_result", "baseline"}
        assert artifacts["late_result"]["plan_step_id"] == 20
        assert artifacts["baseline"]["plan_id"] == "earlier-plan"
        assert artifacts["late_result"]["producer_run_kind"] == "figure_followup"
        assert artifacts["late_result"]["revision_of_id"] == "old-render"
        assert artifacts["late_result"]["followup_task"]["pi_instruction"].startswith("Clarify")
        assert context["source_executions"][0]["id"] == "measurement"
        assert context["source_executions"][1]["plan_id"] == "earlier-plan"
        return json.dumps({"summary": "Summary.", "report_chapter": {
            "title": "Analysis steps", "blocks": [
                {"text": "### Initial measurement\nThe diagnostic qualifies this result.",
                 "artifact_ids": ["baseline", "late_result"]},
                {"text": "### Downstream comparison\nThe next planned step remains unperformed.",
                 "artifact_ids": []},
            ],
        }})

    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = safe_parse_json
    pi._extract_next_steps = lambda data: []
    pi.chat = AsyncMock(side_effect=respond)
    result = await pi.synthesize_round(state)
    pi.chat.assert_awaited_once()
    state.round_results = [result]
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    sections = reusable_sections(
        loaded, build_report_outline(loaded),
        {Path(f.png_path).name: Path(f.png_path) for f in (supplement, baseline)}, [],
    )
    content = sections[0]["content"]
    assert content.index("[FIGURE: baseline.jpg]") < content.index("[FIGURE: late_result.jpg]")
    assert content.index("[FIGURE: late_result.jpg]") < content.index("### Downstream comparison")
    assert "The diagnostic qualifies this result." in content
    assert "Additional Recorded Evidence" not in content


@pytest.mark.parametrize("source", ["foundation", "missing", "malformed", "other_round"])
def test_historical_plan_context_discloses_partial_or_missing_history(tmp_path, source):
    state = _state(tmp_path)
    state.current_plan = AnalysisPlan(id="current", phase="exploration", round_number=1, steps=[
        PlanStep(id=1, title="Current step", method="Current method"),
    ])
    state.publication_figures[0].plan_id = "earlier"
    state.publication_figures[0].plan_step_id = 1
    state.publication_figures[1].plan_id = "current"
    if source == "foundation":
        state.accepted_foundation_inventories = [AcceptedFoundationInventory(
            inventory_id="foundation", plan_id="earlier", round_number=1,
            planning_stage="stage1", steps=[AcceptedFoundationStep(
                plan_step_id=1, title="Earlier step", method="Earlier method",
                evidence_execution_ids=["original-execution"],
            )],
        )]
    elif source == "malformed":
        state.plan_dialogue_history = [{"plan": {"id": "earlier", "steps": "invalid"}}]
    elif source == "other_round":
        state.plan_dialogue_history = [{"plan": AnalysisPlan(
            id="earlier", phase="innovation", round_number=2,
            steps=[PlanStep(id=1, title="Wrong round", method="Wrong method")],
        ).model_dump(mode="json")}]
    # A stale snapshot of the current plan must not override its live steps.
    state.plan_dialogue_history.append({"plan": state.current_plan.model_copy(update={
        "steps": [PlanStep(id=1, title="Stale step", method="Stale method")],
    }).model_dump(mode="json")})
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    context = _chapter_analysis_context(loaded, round_artifact_catalog(loaded))
    assert context["current_plan"]["steps"][0]["method"] == "Current method"
    if source == "foundation":
        earlier = context["historical_plans"][0]
        assert earlier["id"] == "earlier"
        assert earlier["steps"][0]["method"] == "Earlier method"
        assert earlier["steps"][0]["evidence_execution_ids"] == ["original-execution"]
        assert "depends_on" not in earlier["steps"][0]
        assert "dependency graph are unavailable" in earlier["context_limitations"]
        assert context["missing_plan_ids"] == []
    else:
        assert context["historical_plans"] == []
        assert context["missing_plan_ids"] == ["earlier"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "separate", "embedded", "ambiguous", "malformed", "nested", "bad_references",
    "with_malformed_draft", "with_empty_draft",
])
async def test_pi_synthesis_receives_separate_chapter_without_merging_json(tmp_path, case):
    state = _state(tmp_path)
    chapter = _chapter(state)
    raw = chapter.model_dump()
    summary = {"summary": "Retained summary.", "should_stop": False}
    extras = [raw]
    if case == "embedded":
        summary["report_chapter"] = raw
        extras = [{**raw, "title": "Must not replace the embedded chapter"}]
    elif case == "ambiguous":
        extras.append({**raw, "title": "Another possible chapter"})
    elif case == "malformed":
        extras = [{"title": "Invalid chapter", "blocks": "not a block list"}]
    elif case == "with_malformed_draft":
        extras.insert(0, {"title": "Invalid draft", "blocks": "not a block list"})
    elif case == "with_empty_draft":
        extras.append({"title": "Empty draft", "blocks": []})
    elif case == "nested":
        extras = [{"metadata": raw}]
    elif case == "bad_references":
        raw["blocks"][0]["artifact_ids"] = ["unknown", "baseline", "baseline"]
        raw["blocks"][1]["artifact_ids"] = []
    # An independent chapter must never overwrite synthesis decisions.
    for extra in extras:
        extra["should_stop"] = True
    response = "\n\n".join(
        "```json\n" + json.dumps(value) + "\n```" for value in [summary, *extras]
    )
    pi = SynthesisMixin()
    pi.refresh_runtime_config_if_needed = lambda: None
    pi.reset_history = lambda: None
    pi._safe_parse_json = safe_parse_json
    pi._extract_next_steps = lambda data: []
    pi.chat = AsyncMock(return_value=response)

    result = await pi.synthesize_round(state)

    pi.chat.assert_awaited_once()
    assert result.pi_summary == "Retained summary."
    assert result.should_stop is False
    state.round_results = [result]
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    saved = loaded.round_results[0].report_chapter
    assert {a.artifact_id for a in saved.artifacts} == {"baseline", "late_result"}
    if case in {"ambiguous", "malformed", "nested"}:
        assert saved.source == "fallback"
        assert saved.blocks[0].text == "Retained summary."
    elif case == "bad_references":
        assert saved.source == "pi"
        assert saved.blocks[0].artifact_ids == ["baseline"]
        assert saved.blocks[-1].artifact_ids == ["late_result"]
    else:
        assert saved == chapter
        sections = reusable_sections(
            loaded, build_report_outline(loaded),
            {Path(f.png_path).name: Path(f.png_path) for f in loaded.publication_figures}, [],
        )
        content = sections[0]["content"]
        assert content.index("[FIGURE: baseline.jpg]") < content.index("[FIGURE: late_result.jpg]")
        assert "Additional Recorded Evidence" not in content


def test_bad_references_and_missing_chapter_preserve_evidence(tmp_path):
    state = _state(tmp_path)
    catalog = round_artifact_catalog(state)
    chapter = assemble_round_chapter({
        "title": "Findings", "blocks": [
            {"text": "Limited evidence.", "artifact_ids": ["unknown", "baseline", "baseline"]},
        ],
    }, catalog, "Original summary.", 1)
    assert chapter.blocks[0].artifact_ids == ["baseline"]
    assert chapter.blocks[1].artifact_ids == ["late_result"]
    assert chapter.assembly_notes
    fallback = assemble_round_chapter(None, catalog, "Original summary.", 1)
    assert fallback.source == "fallback"
    assert fallback.blocks[0].text == "Original summary."
    assert {a.artifact_id for a in fallback.artifacts} == {"baseline", "late_result"}
    assert RoundResult.model_validate({"round_number": 1}).report_chapter is None


def test_round_html_uses_saved_order_and_captions(monkeypatch, tmp_path):
    state = _state(tmp_path)
    result = RoundResult(round_number=1, pi_summary="Summary.", report_chapter=_chapter(state))
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    # A changed live description must not rewrite the saved round caption.
    state.publication_figures[1].description = "Changed metadata."
    html = generate_detailed_report_segment_html(
        state, "exploration_complete", "Round 1", round_result=result,
    )
    assert html.index("First establish") < html.index("Original caption for baseline.")
    assert html.index("Original caption for baseline.") < html.index("Then interpret")
    assert html.index("Then interpret") < html.index("Original caption for late_result.")
    assert "Changed metadata." not in html
    assert "Generated Figures" not in html
    assert html.count('<img ') == 2


def test_final_html_preserves_order_captions_and_ignores_global_filename_mentions(tmp_path):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state))]
    figures = {Path(f.png_path).name: Path(f.png_path) for f in state.publication_figures}
    saved = reusable_sections(state, build_report_outline(state), figures, [])
    captions = saved_figure_captions(state, figures, [])
    state.publication_figures[1].description = "Changed live caption."
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    out = tmp_path / "report"
    out.mkdir()
    sections = [{"id": "methods", "title": "Methods", "analysis_type": "general",
                 "content": "The archived filename is baseline.jpg.", "figures": [], "tables": []}] + saved
    path = agent._generate_html_report(
        {"title": "Study", "figures": [], "tables": [], "_analysis_state": state,
         "round_chapter_captions": captions},
        sections, "Abstract.", "Conclusions.",
        [{"filename": n, "path": str(p), "category": "other"} for n, p in figures.items()],
        figures, {"total_steps_monitored": 0, "steps_with_technical_issues": 0, "pi_escalations": 0},
        "en", out, tmp_path,
    )
    html = path.read_text()
    assert html.count('<img ') == 2
    assert html.index("First establish") < html.index('alt="baseline.jpg"')
    assert html.index('alt="baseline.jpg"') < html.index("Then interpret")
    assert html.index("Then interpret") < html.index('alt="late_result.jpg"')
    assert "Changed live caption." not in html
    assert "Original caption for baseline." in html


@pytest.mark.asyncio
async def test_report_figure_metadata_keeps_saved_caption(tmp_path):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state))]
    figures = {Path(f.png_path).name: Path(f.png_path) for f in state.publication_figures}
    captions = saved_figure_captions(state, figures, [])
    state.publication_figures[1].description = "Changed live caption."
    report_presentation.bind_facade(report_content)
    ctx = {"_progress": AsyncMock(), "workspace_path": tmp_path,
           "figure_map": figures, "figure_metadata_by_stem": {}, "saved_figure_captions": captions}
    await report_presentation.build_report_figures(None, ctx, state)
    baseline = next(f for f in ctx["report_figures"] if f["filename"] == "baseline.jpg")
    assert baseline["description"] == "Original caption for baseline."


def test_round_chapter_retains_table_references(monkeypatch, tmp_path):
    state = _state(tmp_path)
    table_path = tmp_path / "observations.csv"
    table_path.write_text("group,value\na,2\n")
    state.publication_tables = [PublicationTable(
        id="table", name="Observations", description="Recorded observations.",
        csv_path=str(table_path), csv_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        round_number=1, review_status=FigureReviewStatus.APPROVED,
        source_execution_id="execution-1",
        columns=["group", "value"], preview_data=[{"group": "a", "value": 2}],
    )]
    chapter = assemble_round_chapter({
        "title": "Findings", "blocks": [{"text": "Measured contrast.",
        "artifact_ids": ["baseline", "table", "late_result"]}],
    }, round_artifact_catalog(state), "Summary.", 1)
    assert [a.kind for a in chapter.artifacts] == ["figure", "figure", "table"]
    names = report_artifact_names(state, {}, [{"filename": "observations.csv", "table_id": "table"}])
    assert "[TABLE: observations.csv]" in chapter_markdown(chapter, names)
    monkeypatch.setattr("backend.config.get_session_workspace", lambda _: tmp_path)
    html = generate_detailed_report_segment_html(state, "exploration_complete", "Round", round_result=RoundResult(
        round_number=1, report_chapter=chapter,
    ))
    assert "Recorded observations." in html
    assert "<td>a</td>" in html


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("marker", ["FIGURE", "figure", "FiGuRe"])
async def test_final_assembly_keeps_saved_chapter_despite_model_rewrite(tmp_path, fallback, marker):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state))]
    outline = build_report_outline(state)
    figures = {Path(f.png_path).name: Path(f.png_path) for f in state.publication_figures}
    saved = reusable_sections(state, outline, figures, [])
    assert [s["id"] for s in saved] == ["exploration_findings"]
    expected = saved[0]["content"]
    generated = [
        {"id": s.id, "title": s.default_title, "content": "Generated global section.",
         "analysis_type": s.analysis_type, "figures": [], "tables": []}
        for s in outline
    ]
    generated[0]["content"] += f"\n[{marker}: baseline.jpg]\n[{marker}_CAPTION: Wrong caption.]"
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    agent._validate_report_content = lambda _: True
    agent._parse_report_sections = lambda *a, **kw: generated
    report_prompt.bind_facade(report_content)
    ctx = {
        "_progress": AsyncMock(), "report_content": "Generated response", "language": "en",
        "outline": outline, "used_fallback": fallback, "attribution": {n: 1 for n in figures},
        "figure_map": figures, "figure_metadata_by_stem": {}, "client": SimpleNamespace(),
        "saved_sections": saved, "reserved_names": set(figures),
    }
    await report_prompt.parse_and_place_report(agent, ctx, state, str(tmp_path), None, None, "model")
    actual = next(s for s in ctx["sections"] if s["id"] == "exploration_findings")
    assert actual["content"] == expected
    assert ctx["abstract"] == "Generated global section.\n"
    assert "Wrong caption" not in str(ctx["sections"])


@pytest.mark.asyncio
async def test_final_prompt_requests_only_global_sections(tmp_path):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, report_chapter=_chapter(state))]
    figures = {Path(f.png_path).name: Path(f.png_path) for f in state.publication_figures}
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = state.id
    agent.refresh_runtime_config_if_needed = lambda: None
    agent.get_supervision_log = lambda: []
    agent.get_summary_stats = lambda: {}
    agent._build_analysis_narrative = lambda *a: "Methods and measurements."
    agent._collect_figure_paths = lambda *a, **kw: figures
    agent._collect_table_paths = lambda *a, **kw: []
    prompts = []
    def build(*args, **kwargs):
        prompts.append((args, kwargs))
        return "Global prompt."
    agent._build_report_prompt = build
    agent._get_report_system_prompt = lambda _: "Write a report."
    agent._set_llm_context = lambda: None
    client = SimpleNamespace(chat=AsyncMock(), set_context=lambda *a: None)
    agent._budgeted_text_call = AsyncMock(return_value=SimpleNamespace(content="Global response."))
    report_prompt.bind_facade(report_content)
    ctx = await report_prompt.prepare_report_generation(
        agent, state, "model", str(tmp_path / "report"), str(tmp_path), None, client, False,
    )
    args, kwargs = prompts[0]
    assert "exploration_findings" not in {s.id for s in kwargs["outline"]}
    assert args[3] == {}
    sent = agent._budgeted_text_call.call_args.kwargs["messages"][1]["content"]
    assert "already written" in sent
    assert "First establish the baseline." in sent
    assert ctx["saved_sections"][0]["title"] == "Evidence in context"


def test_missing_or_inherited_artifact_keeps_prose_and_exact_identity(tmp_path):
    state = _state(tmp_path)
    chapter = _chapter(state)
    missing = chapter_markdown(chapter, {})
    assert "First establish the baseline." in missing
    assert "Recorded figure unavailable" in missing
    inherited = state.publication_figures[1].model_copy(deep=True)
    inherited.id = "baseline:child"
    inherited.lineage_id = "child"
    inherited.inherited_from_lineage_id = "main"
    state.publication_figures = [inherited]
    state.sandbox_state = {"_report_target_lineage_id": "child"}
    names = report_artifact_names(state, {"baseline.jpg": Path(inherited.png_path)}, [])
    assert names["baseline"] == "baseline.jpg"
    assert "late_result" not in names


def test_saved_reference_cannot_borrow_a_same_named_replacement(tmp_path):
    state = _state(tmp_path)
    original = state.publication_figures[1]
    replacement = tmp_path / "replacement" / "baseline.jpg"
    replacement.parent.mkdir()
    write_valid_image(replacement, color=(220, 10, 10), min_bytes=2200)
    names = report_artifact_names(state, {"baseline.jpg": replacement}, [])
    assert original.id not in names


def test_legacy_outline_and_generation_remain_available(tmp_path):
    state = _state(tmp_path)
    state.round_results = [RoundResult(round_number=1, pi_summary="Legacy summary.")]
    outline = build_report_outline(state)
    assert "data_qc" in {s.id for s in outline}
    assert reusable_sections(state, outline, {}, []) == []
