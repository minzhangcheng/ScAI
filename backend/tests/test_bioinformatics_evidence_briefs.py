"""Bioinformatics briefs retain method/result sources through PI decisions."""
import hashlib
import json
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents import bioinformatics_agent as facade
from backend.agents.bioinformatics import executor_report, executor_report_gates
from backend.agents.bioinformatics.executor_dispatch import ToolAction
from backend.agents.pi_agent import PIAgent
from backend.models.schemas import (
    AnalysisPhase, AnalysisState, CodeExecution, PublicationTable, RoundResult,
    ScoreDefinition, ScoringPlan,
)
from backend.utils.bioinformatics_evidence import attach_evidence_sources, render_evidence_briefs
from backend.utils import bioinformatics_evidence as briefs
from backend.utils.direction_payload import build_direction_projection, build_direction_projection_block
from backend.utils.llm_client import ToolCall
from backend.workflow.synthesis import _build_directions_with_results_for_correlation
from backend.workflow.helpers import WorkflowSupersededError, WorkflowTerminatedError


def _state(tmp_path):
    path = tmp_path / "comparison.csv"
    path.write_text("feature,effect,qvalue\n" + "G0,0,1\n" * 12 + "G1,2.25,0.003\n")
    return AnalysisState(
        user_id="u", current_round=1, analysis_phase=AnalysisPhase.INNOVATION,
        score_definitions=[ScoreDefinition(
            id="score-old", score_name="mixed_response", method="score_genes",
            description="Recorded response panel", gene_list=["G1", "D1", "D2"],
            n_genes_matched=2, round_number=1, plan_id="panel-plan",
            parameters={"matched_genes": ["G1", "D1"], "missing_genes": ["D2"]},
        )],
        scoring_plans=[ScoringPlan(id="panel-plan", score_name="mixed_response",
                                  method="score_genes", panel_genes=["D1", "D2"])],
        publication_tables=[PublicationTable(
            id="table-1", name="Comparison", description="Measured comparison",
            csv_path=str(path), csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_execution_id="exec-1", round_number=1,
        )],
    )


def _finding():
    return {
        "text": "A mixed response is observed; the competing mechanism remains untested.",
        "evidence_execution_ids": ["exec-1"],
        "evidence_summary": {
            "comparison": "Sample-level group comparison and descriptive score_genes panel",
            "supporting_results": ["G1 effect is 2.25; qvalue is 0.003"],
            "opposing_results": ["The counter-regulatory response is also elevated"],
            "alternative_explanations": ["A general stress response"],
            "open_questions": ["The competing mechanism has not been compared"],
            "score_ids": ["score-old"],
            "table_rows": [{"table_id": "table-1", "row_match": {"feature": "G1"},
                            "columns": ["feature", "effect", "qvalue"]}],
        },
    }


def _report_context(state):
    executor_report.bind_facade(facade)
    executor_report_gates.bind_facade(facade)
    agent = facade.BioinformaticsAgent.__new__(facade.BioinformaticsAgent)
    agent._notify = AsyncMock()
    agent.code_history = [CodeExecution(id="exec-1", code="print('measured result')", round_number=1)]
    return SimpleNamespace(
        agent=agent, state=state, messages=[], analysis_log=[], step_count=1,
        agent_reasoning="Report evidence", annotation_completed=True, planning_stage="full",
        completed=False, evidence_gate_degraded=False, evidence_gate_findings=[], all_findings=[],
        accepted_finding_records=[], reported_evidence_ids=[], mechanism_programs_registered=True,
    )


@pytest.mark.asyncio
async def test_report_freezes_exact_panel_and_rows_before_same_name_score_update(tmp_path):
    state = _state(tmp_path)
    context = _report_context(state)
    action = await executor_report.handle_report_findings(context, ToolCall(
        id="report", name="report_findings", arguments={"summary": "Mixed evidence", "findings": [_finding()]},
    ))
    assert action == ToolAction.BREAK and context.completed
    record = context.accepted_finding_records[0]
    assert record["evidence_summary"]["open_questions"] == _finding()["evidence_summary"]["open_questions"]
    sources = record["evidence_sources"]
    assert sources[0]["gene_list"] == ["G1", "D1", "D2"]
    assert sources[0]["frozen_plan"]["panel_genes"] == ["D1", "D2"]
    assert sources[1]["rows"] == [{"row_index": 12, "values": {"feature": "G1", "effect": "2.25", "qvalue": "0.003"}}]
    state.score_definitions[0] = ScoreDefinition(
        id="score-new", score_name="mixed_response", method="another_method",
        description="Later definition", gene_list=["OTHER"], round_number=2,
    )
    state.round_results = [RoundResult(round_number=1, bioinformatics_finding_records=[record])]
    loaded = AnalysisState.model_validate_json(state.model_dump_json())
    assert loaded.round_results[0].bioinformatics_finding_records[0]["evidence_sources"] == sources
    assert loaded.score_definitions[0].gene_list == ["OTHER"]


@pytest.mark.parametrize("missing", [False, True])
def test_unreadable_or_changed_table_does_not_invent_or_remove_findings(tmp_path, missing):
    state = _state(tmp_path)
    path = tmp_path / "comparison.csv"
    path.unlink() if missing else path.write_text("feature,effect,qvalue\nG1,99,0.001\n")
    record = _finding()
    attach_evidence_sources(state, [record])
    source = record["evidence_sources"][1]
    assert source["rows"] == []
    assert "unavailable" in source["note"] if missing else "differs" in source["note"]
    assert record["text"] == _finding()["text"]
    assert record["evidence_summary"] == _finding()["evidence_summary"]


def test_compact_context_keeps_early_and_late_questions_with_disclosed_omissions():
    records = [{"id": f"finding-{index}", "round_number": index,
                "text": f"Interpretation {index}", "evidence_summary": {
                    "supporting_results": ["Support"], "opposing_results": [f"Counter {index}"],
                    "open_questions": [f"Untested {index}"],
                }} for index in range(8)]
    text = render_evidence_briefs(records, max_chars=4000)
    assert len(text) <= 4000
    for index in (0, 1, 6, 7):
        assert f"Counter {index}" in text and f"Untested {index}" in text
    assert "4 brief(s) omitted" in text
    assert render_evidence_briefs([{"text": "Legacy finding"}]) == ""


@pytest.mark.asyncio
async def test_briefs_reach_direction_integration_final_and_chat_context(tmp_path):
    state = _state(tmp_path)
    record = _finding()
    record.update(id="finding-1", round_number=1, lineage_id="main")
    attach_evidence_sources(state, [record])
    state.active_direction_batch = [{"direction": "Compare competing responses"}]
    state.round_results = [RoundResult(
        round_number=1, phase="innovation", direction="Compare competing responses",
        bioinformatics_findings=[record["text"]], bioinformatics_finding_records=[record],
        pi_summary="Earlier interpretation", round_status="completed", evidence_status="substantive",
    )]
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    pi = PIAgent(settings_provider=provider)
    pi.chat = AsyncMock(return_value=json.dumps({"candidates": [], "no_candidate_reason": "Open question retained."}))
    await pi.generate_raw_direction_candidates(state, state.round_results[0])
    prompts = [pi.chat.call_args.args[0]]
    projection = build_direction_projection(state, "Compare competing responses")
    prompts.append(build_direction_projection_block(projection))
    capped = build_direction_projection_block(projection, max_chars=500)
    assert len(capped) <= 500 and "score-old" in capped and "table-1" in capped
    inputs = _build_directions_with_results_for_correlation(state)
    assert inputs[0]["finding_records"] == [record]
    reason = "Qualified the earlier claim because the panel measures a mixed response; table-1 result retained."
    pi.chat.return_value = reason
    assert await pi.analyze_direction_correlations(state, inputs) == reason
    prompts.append(pi.chat.call_args.args[0])
    assert await pi.generate_final_summary(state) == reason
    assert pi.chat.call_args.args[0].count("## Bioinformatics evidence briefs") == 1
    prompts.append(pi.chat.call_args.args[0])
    from backend.routes.chat_parts.delegation import _build_execution_context
    prompts.append(_build_execution_context({"finding_records": [record]}))
    for prompt in prompts:
        for value in ("score-old", "G1", "D1", "D2", "table-1", "2.25", "0.003", "exec-1",
                      "opposing_results", "competing mechanism has not been compared"):
            assert value in prompt
    assert "qualified, superseded" in pi.chat.call_args.args[0]


@pytest.mark.asyncio
async def test_malformed_restored_briefs_keep_valid_fields_and_reach_pi(tmp_path):
    state = _state(tmp_path)
    good = _finding()
    good.update(id="good", round_number=1)
    attach_evidence_sources(state, [good])
    mixed = deepcopy(good)
    mixed.update(id="mixed", evidence_sources=["malformed", {"kind": "score", "id": "bad-source", "parameters": 42}, *good["evidence_sources"]])
    mixed["evidence_summary"]["supporting_results"].append({"effect": 99})
    bad = {"id": "bad", "text": "Original finding", "evidence_execution_ids": ["exec-1", {}], "evidence_summary": 42}
    records = [good, mixed, bad]
    before = deepcopy(records)
    text = render_evidence_briefs(records)
    assert len(text) <= 6000 and records == before
    for value in ("2.25", "opposing_results", "Original finding", "exec-1", "Malformed", "source rendering failed"):
        assert value in text
    assert "exec-1" in briefs.evidence_brief_references(records)
    state.round_results = [RoundResult(round_number=1, bioinformatics_finding_records=records)]
    state = AnalysisState.model_validate_json(state.model_dump_json())
    pi = PIAgent(settings_provider=MagicMock())
    pi.chat = AsyncMock(return_value="Evidence retained.")
    assert await pi.generate_final_summary(state) == "Evidence retained."
    assert "Original finding" in pi.chat.call_args.args[0]


@pytest.mark.parametrize("failure", ["score", "frozen_plan", "table"])
def test_source_failure_preserves_other_snapshots(tmp_path, monkeypatch, failure):
    state = _state(tmp_path)
    state.score_definitions.append(ScoreDefinition(id="good-score", score_name="other", method="score_genes", description="Other panel", gene_list=["G3"]))
    if failure == "table":
        monkeypatch.setattr(briefs, "_table_excerpt", MagicMock(side_effect=RuntimeError("excerpt failed")))
    else:
        model = ScoreDefinition if failure == "score" else ScoringPlan
        real_dump = model.model_dump
        def dump(self, *args, **kwargs):
            if self.id in {"score-old", "panel-plan"}:
                raise RuntimeError("snapshot failed")
            return real_dump(self, *args, **kwargs)
        monkeypatch.setattr(model, "model_dump", dump)
    record = _finding()
    record["evidence_summary"]["score_ids"].append("good-score")
    record["evidence_summary"]["table_rows"].append({"table_id": "table-1", "row_match": {"feature": "G1", "qvalue": {"unparsed": 0.003}}})
    attach_evidence_sources(state, [record])
    sources = record["evidence_sources"]
    assert len(sources) == 3  # A malformed selector must not broaden the CSV query.
    assert next(s for s in sources if s["id"] == "good-score")["gene_list"] == ["G3"]
    failed = next(s for s in sources if s["id"] == ("table-1" if failure == "table" else "score-old"))
    assert "failed (RuntimeError)" in failed["note"]
    if failure != "table":
        assert sources[-1]["rows"][0]["values"]["effect"] == "2.25"
    if failure == "frozen_plan":
        assert failed["gene_list"] == ["G1", "D1", "D2"]
    assert record["text"] == _finding()["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, WorkflowTerminatedError, WorkflowSupersededError, asyncio.CancelledError])
async def test_attachment_boundary_retains_report_and_propagates_control(tmp_path, monkeypatch, error):
    state = _state(tmp_path)
    context = _report_context(state)
    def fail(state, records):
        records[0]["evidence_sources"] = [{"kind": "score", "id": "retained-source", "gene_list": ["G1"]}]
        raise error("injected failure")
    monkeypatch.setattr(executor_report, "attach_evidence_sources", fail)
    finding = _finding()
    finding["evidence_summary"]["supporting_results"].append({"unparsed": 99})
    call = ToolCall(id="report", name="report_findings", arguments={"summary": "Measured observations", "findings": [finding]})
    if error is not RuntimeError:
        with pytest.raises(error):
            await executor_report.handle_report_findings(context, call)
        assert not context.completed
        return
    assert await executor_report.handle_report_findings(context, call) == ToolAction.BREAK
    assert context.completed and not context.evidence_gate_degraded
    record = context.analysis_log[-1].finding_records[0]
    assert record["text"] == _finding()["text"]
    assert record["candidate_evidence_execution_ids"] == ["exec-1"]
    assert record["evidence_sources"][0]["id"] == "retained-source"
    state.round_results = [RoundResult(round_number=1, bioinformatics_finding_records=[record])]
    restored = AnalysisState.model_validate_json(state.model_dump_json()).round_results[0].bioinformatics_finding_records
    assert "source attachment failed" in render_evidence_briefs(restored)
    assert restored[0]["evidence_summary"]["format_notes"]
    assert restored[0]["evidence_summary"]["supporting_results"] == _finding()["evidence_summary"]["supporting_results"]


@pytest.mark.parametrize("target,error", [
    ("_render_card", RuntimeError), ("_render_evidence_briefs", RuntimeError),
    ("_render_card", WorkflowTerminatedError), ("_render_evidence_briefs", WorkflowSupersededError),
    ("_render_evidence_briefs", asyncio.CancelledError),
])
def test_render_fallback_is_bounded_and_does_not_swallow_control(monkeypatch, target, error):
    monkeypatch.setattr(briefs, target, MagicMock(side_effect=error("injected failure")))
    record = {"id": "finding", **_finding()}
    if error is not RuntimeError:
        with pytest.raises(error):
            render_evidence_briefs([record])
        return
    text = render_evidence_briefs([record], max_chars=1000)
    assert len(text) <= 1000
    assert record["text"] in text and "exec-1" in text
    assert "brief rendering failed" in text
