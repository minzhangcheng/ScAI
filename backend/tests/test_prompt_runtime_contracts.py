"""The prompt says X; prove the runtime does not force not-X.

Every case here existed as a defect: an instruction told an agent something was
permitted or required, and code downstream quietly enforced the opposite. The
prompt alone is not evidence, so these assert the runtime behaviour.

Nothing here is a gate. Where a disagreement is detected the expectation is that
it is RECORDED and passed to the PI, never that the run is blocked.
"""
import json
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from backend.agents.pi_agent import PIAgent
from backend.models.schemas import (
    AnalysisState, ExplorationReport, TreatmentGroup,
)
from backend.workflow.agents import build_phase_instruction


def _pi_agent():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = PIAgent(settings_provider=provider)
    agent.set_session("test-session")
    return agent


def _state_with(candidates):
    return AnalysisState(
        user_id="u",
        exploration_report=ExplorationReport(research_direction_candidates=candidates),
    )


CANDIDATES = [
    {"direction_id": "d1", "direction": "LOW one", "overall_priority": "low"},
    {"direction_id": "d2", "direction": "HIGH one", "overall_priority": "high"},
]


def _replies(agent, payload):
    async def fake_chat(*_a, **_kw):
        return payload
    agent.chat = fake_chat


# --------------------------------------------------------------- direction paths
@pytest.mark.asyncio
async def test_pi_selecting_no_direction_is_honoured():
    """The prompt says selecting none is valid; recovery used to overwrite it."""
    agent = _pi_agent()
    _replies(agent, json.dumps({
        "selected_directions": [],
        "rejected_directions": [],
        "selection_rationale": "nothing clears the reliability bar",
    }))
    decision = await agent.select_best_directions(_state_with(CANDIDATES))
    assert decision["selected_directions"] == []
    assert "reliability" in decision["selection_rationale"]


@pytest.mark.asyncio
async def test_unreadable_selection_does_not_let_the_host_pick_the_science():
    """A parse failure is a tooling failure, not a scientific verdict.

    The host must not substitute its own ranking for the PI's judgement; it must
    say the selection was unavailable and let the run finalize.
    """
    agent = _pi_agent()
    _replies(agent, "definitely not json")
    decision = await agent.select_best_directions(_state_with(CANDIDATES))
    assert decision["selected_directions"] == []
    assert "UNAVAILABLE" in decision["selection_rationale"]
    assert "tooling failure" in decision["selection_rationale"]


@pytest.mark.asyncio
async def test_no_candidates_does_not_fabricate_one():
    agent = _pi_agent()
    decision = await agent.select_best_directions(_state_with([]))
    assert decision["selected_directions"] == []
    assert "Continue analysis" not in json.dumps(decision)


@pytest.mark.asyncio
async def test_empty_candidate_proposal_is_not_replaced_by_a_placeholder():
    """Round 1 may legitimately yield no direction worth investigating."""
    from backend.models.schemas import RoundResult

    agent = _pi_agent()
    _replies(agent, json.dumps({"candidates": [], "no_candidate_reason": "nothing stood out"}))
    out = await agent.generate_raw_direction_candidates(
        AnalysisState(user_id="u"), RoundResult(round_number=1)
    )
    assert out == []


@pytest.mark.asyncio
async def test_unreadable_candidate_proposal_yields_no_candidate_at_all():
    """A tooling failure must not travel inside the candidate list.

    It used to be wrapped as a candidate carrying unreadable_response=True and a
    "do not score this" note in its rationale. Nothing downstream read either:
    the workflow sent it to literature evaluation and PI scoring like any other
    proposal. The failure belongs in its own channel.
    """
    from backend.models.schemas import RoundResult

    agent = _pi_agent()
    _replies(agent, "<<garbage>>")
    out = await agent.generate_raw_direction_candidates(
        AnalysisState(user_id="u"), RoundResult(round_number=1)
    )
    assert out == []
    assert "could not be read" in (agent.last_candidate_generation_error or "")


@pytest.mark.asyncio
async def test_a_readable_empty_proposal_reports_no_failure():
    from backend.models.schemas import RoundResult

    agent = _pi_agent()
    _replies(agent, json.dumps({"candidates": [], "no_candidate_reason": "nothing stood out"}))
    out = await agent.generate_raw_direction_candidates(
        AnalysisState(user_id="u"), RoundResult(round_number=1)
    )
    assert out == []
    assert not getattr(agent, "last_candidate_generation_error", None), (
        "an honest empty proposal must not be recorded as a malfunction"
    )


@pytest.mark.asyncio
async def test_a_mixed_selection_list_keeps_the_usable_entries():
    agent = _pi_agent()
    _replies(agent, json.dumps({
        "selected_directions": ["bad", {"direction": "HIGH one"}],
        "rejected_directions": [],
        "selection_rationale": "r",
    }))
    decision = await agent.select_best_directions(_state_with(CANDIDATES))
    assert all(isinstance(d, dict) for d in decision["selected_directions"])
    assert any(d.get("direction") == "HIGH one" for d in decision["selected_directions"])
    assert decision["decision_outcome"] == "selected"


@pytest.mark.asyncio
async def test_a_selection_list_with_no_objects_is_a_tooling_failure():
    """["bad"] must not filter down to a scientific "nothing was worth a round"."""
    agent = _pi_agent()
    _replies(agent, json.dumps({
        "selected_directions": ["bad"],
        "rejected_directions": [],
        "selection_rationale": "r",
    }))
    decision = await agent.select_best_directions(_state_with(CANDIDATES))
    assert decision["selected_directions"] == []
    assert decision["decision_outcome"] == "tooling_failure"
    assert "UNAVAILABLE" in decision["selection_rationale"]


@pytest.mark.asyncio
async def test_candidates_list_with_no_objects_is_retried_then_failed():
    from backend.models.schemas import RoundResult

    agent = _pi_agent()
    calls = []

    async def fake_chat(prompt, temperature=0.3):
        calls.append(prompt)
        return json.dumps({"candidates": ["bad"]})

    agent.chat = fake_chat
    out = await agent.generate_raw_direction_candidates(
        AnalysisState(user_id="u"), RoundResult(round_number=1)
    )
    assert out == []
    assert len(calls) == 2, "a structurally unusable reply must be retried"
    assert "could not be read" in (agent.last_candidate_generation_error or "")


def test_outcome_comes_from_the_selector_not_from_english_prose():
    from backend.workflow.planning_parts.pi import _decision_outcome

    assert _decision_outcome({"decision_outcome": "tooling_failure"}, [], "") == "tooling_failure"
    assert _decision_outcome({"decision_outcome": "none_selected"}, [], "UNAVAILABLE") == "none_selected"


def test_scorer_cannot_invent_a_candidate():
    from backend.agents.pi.shared import _normalize_integration_scored_candidates

    out = _normalize_integration_scored_candidates(
        [{"direction_id": "nope", "direction": "invented", "overall_priority": "high"}],
        [{"direction_id": "d1", "direction": "real"}],
    )
    assert out == []


def test_biochemistry_host_restores_identity_the_model_retyped():
    """A correct id next to another measurement's name/value was accepted verbatim."""
    from backend.agents.biochemistry_parts.records import restore_measurement_identity
    from backend.models.schemas import BiochemistryIndicator, TreatmentGroup

    record = BiochemistryIndicator(
        indicator="ALT", value=45.2, unit="U/L", group_id="g1", sample_number=2,
        reference_range="10-40 U/L",
    )
    state = AnalysisState(user_id="u", treatment_groups=[TreatmentGroup(id="g1", name="Control")])
    restored = restore_measurement_identity(
        [{"measurement_id": record.id, "indicator": "AST", "value": 999,
          "group": "Treated", "sample_number": 7, "status": "high", "basis": "reference_range"}],
        [record], state,
    )[0]
    assert restored["indicator"] == "ALT" and restored["value"] == 45.2
    assert restored["group_id"] == "g1" and restored["group"] == "Control"
    assert restored["sample_number"] == 2
    assert restored["status"] == "high", "the model still owns the judgement"


def test_ambiguous_legacy_match_is_left_unlinked():
    """setdefault() collapsed duplicates, so the uniqueness guard never fired."""
    from backend.agents.biochemistry_parts.records import merge_marker_records

    step1 = [
        {"measurement_id": "m1", "indicator": "ALT", "group": "Control", "sample_number": 1,
         "status": "normal"},
        {"measurement_id": "m2", "indicator": "ALT", "group": "Control", "sample_number": 1,
         "status": "normal"},
    ]
    merged, notes = merge_marker_records(
        step1, [{"indicator": "ALT", "group": "Control", "sample_number": 1, "status": "high"}]
    )
    assert len(merged) == 2
    assert all(m["status"] == "normal" for m in merged)
    assert any("share that indicator/group/sample" in n for n in notes)


def test_step2_marker_without_named_evidence_stays_a_note():
    from backend.agents.biochemistry_parts.records import merge_marker_records

    merged, notes = merge_marker_records(
        [{"measurement_id": "m1", "indicator": "A", "status": "normal"}],
        [{"indicator": "Claimed", "status": "high", "source_step": "step2"}],
    )
    assert {m["indicator"] for m in merged} == {"A"}
    assert any("named no source" in n for n in notes)


@pytest.mark.asyncio
async def test_unreadable_integration_selection_is_not_reported_as_no_novelty():
    agent = _pi_agent()
    _replies(agent, "not json")
    decision = await agent.select_integration_directions(
        AnalysisState(user_id="u"),
        [{"direction_id": "i1", "direction": "X", "overall_priority": "high"}],
        "integration report",
    )
    assert decision["selected_directions"] == []
    assert "UNAVAILABLE" in decision["selection_rationale"]
    assert "tooling failure" in decision["selection_rationale"]
    assert "no novel direction remains" in decision["selection_rationale"], (
        "the rationale must say the malfunction did not establish anything about novelty"
    )


# --------------------------------------------------------------- modality
def test_single_cell_run_still_gets_the_single_cell_pipeline():
    txt = build_phase_instruction("stage1", "exploration", has_bulk=False, has_scrna=True, direction="")
    assert "scrublet" in txt and "annotate_clusters" in txt


async def _captured_pi_prompt(coro_factory, agent):
    """Run a PI instruction builder and return the prompt it actually sent."""
    captured = {}

    async def fake_chat(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return "{}"

    agent.chat = fake_chat
    await coro_factory()
    return captured.get("prompt", "")


def _modality_state(*, scrna: bool, bulk: bool) -> AnalysisState:
    from backend.models.schemas import DataFile

    files = []
    if scrna:
        files.append(DataFile(filename="a.h5ad", file_path="/a.h5ad",
                              section="sequencing", data_type="scRNA-seq"))
    if bulk:
        files.append(DataFile(filename="b.csv", file_path="/b.csv",
                              section="sequencing", data_type="bulk RNA-seq"))
    return AnalysisState(user_id="u", name="P", initial_instruction="obj", data_files=files)


@pytest.mark.asyncio
async def test_single_cell_round1_prompt_still_asks_for_annotation():
    agent = _pi_agent()
    state = _modality_state(scrna=True, bulk=False)
    prompt = await _captured_pi_prompt(
        lambda: agent.initial_exploration_instruction(state, "obj"), agent
    )
    assert "annotate_clusters" in prompt and "liana" in prompt.lower()


@pytest.mark.asyncio
async def test_single_cell_innovation_toolkit_keeps_cell_level_methods():
    agent = _pi_agent()
    state = _modality_state(scrna=True, bulk=False)
    prompt = await _captured_pi_prompt(
        lambda: agent.innovation_instruction(state, "a direction"), agent
    )
    assert "RNA Velocity" in prompt and "Cell-cell communication" in prompt


def test_graphic_context_labels_single_cell_without_implying_bulk():
    from backend.agents.painter_parts.evidence_context import (
        build_graphic_evidence_context,
    )

    context = build_graphic_evidence_context(
        _modality_state(scrna=True, bulk=False)
    )
    assert "Modalities analysed: single-cell sequencing" in context
    assert "bulk sequencing" not in context


def test_graphic_context_lists_pathology_and_biochemistry_inputs():
    from backend.agents.painter_parts.evidence_context import (
        build_graphic_evidence_context,
    )
    from backend.models.schemas import DataFile

    state = _modality_state(scrna=True, bulk=False)
    state.data_files.extend([
        DataFile(filename="slide.png", file_path="/slide.png",
                 section="pathology", data_type="histology"),
        DataFile(filename="assay.csv", file_path="/assay.csv",
                 section="biochemistry", data_type="assay"),
    ])
    context = build_graphic_evidence_context(state)
    assert "pathology" in context
    assert "biochemistry" in context


# --------------------------------------------------------------- biochemistry
def test_upload_metadata_wins_and_conflicts_are_recorded():
    from backend.agents.biochemistry_parts.records import reconcile_provenance

    state = AnalysisState(user_id="u", treatment_groups=[TreatmentGroup(id="g1", name="Control")])
    gid, snum, label, notes = reconcile_provenance(
        file_group_id="g1", file_sample_number=2,
        row_group_label="Control", row_sample_value="sample 5", state=state,
    )
    assert (gid, snum) == ("g1", 2)
    assert any("sample-number conflict" in n for n in notes)


def test_undeclared_group_label_never_becomes_a_group_id():
    from backend.agents.biochemistry_parts.records import reconcile_provenance

    state = AnalysisState(user_id="u", treatment_groups=[TreatmentGroup(id="g1", name="Control")])
    gid, _snum, label, notes = reconcile_provenance(
        file_group_id=None, file_sample_number=None,
        row_group_label="Treated", row_sample_value=None, state=state,
    )
    assert gid is None, "group_id must be a TreatmentGroup.id or nothing"
    assert label == "Treated", "the raw label stays visible instead of being dropped"
    assert any("no declared treatment group" in n for n in notes)


def test_synthesis_cannot_silently_promote_not_classifiable():
    from backend.agents.biochemistry_parts.records import merge_marker_records

    merged, _notes = merge_marker_records(
        [{"measurement_id": "m1", "indicator": "A", "status": "not_classifiable", "basis": "none"}],
        [{"measurement_id": "m1", "indicator": "A", "status": "abnormal", "basis": "published:recalled"}],
    )
    entry = merged[0]
    assert entry["status"] == "not_classifiable"
    assert any("synthesis proposed" in n for n in entry["provenance_notes"]), (
        "the disagreement must be recorded for the PI, not discarded"
    )


def test_synthesis_may_add_an_observation_it_alone_saw_from_an_image():
    from backend.agents.biochemistry_parts.records import merge_marker_records

    merged, _notes = merge_marker_records(
        [{"measurement_id": "m1", "indicator": "A", "status": "normal", "basis": "reference_range"}],
        [{"indicator": "FromImage", "status": "high", "basis": "control_group:c",
          "source_step": "step2", "source_evidence": "blot_1.png"}],
    )
    assert {m["indicator"] for m in merged} == {"A", "FromImage"}


def test_an_unlinked_synthesis_marker_is_not_promoted_to_a_located_measurement():
    """A step-3 row matching nothing and citing no image is an unlinked note."""
    from backend.agents.biochemistry_parts.records import merge_marker_records

    merged, notes = merge_marker_records(
        [{"measurement_id": "m1", "indicator": "A", "status": "normal", "basis": "reference_range"}],
        [{"indicator": "Mystery", "status": "high", "basis": "none"}],
    )
    assert {m["indicator"] for m in merged} == {"A"}
    assert any("unlinked observation" in n for n in notes)


def test_two_groups_and_two_replicates_stay_four_distinct_records():
    """The whole point of measurement identity: nothing collapses or crosses over."""
    from backend.agents.biochemistry_parts.records import merge_marker_records

    step1 = [
        {"measurement_id": f"m{i}", "indicator": "ALT", "group": g, "sample_number": n,
         "status": st, "basis": "control_group:Control"}
        for i, (g, n, st) in enumerate(
            [("Control", 1, "normal"), ("Control", 2, "normal"),
             ("Treated", 1, "high"), ("Treated", 2, "high")]
        )
    ]
    step3 = [{"measurement_id": "m2", "indicator": "ALT", "status": "high",
              "basis": "control_group:Control"}]
    merged, _notes = merge_marker_records(step1, step3)

    assert len(merged) == 4
    assert [(m["group"], m["sample_number"]) for m in merged] == [
        ("Control", 1), ("Control", 2), ("Treated", 1), ("Treated", 2)
    ]
    assert [m["status"] for m in merged] == ["normal", "normal", "high", "high"]


# --------------------------------------------------------------- pathology
def test_severity_inconsistency_is_reported_not_rejected():
    from backend.agents.pathology_parts.image_analysis import _severity_observations

    notes = _severity_observations(
        {"severity_rubric_name": "Ishak", "severity_score": 7, "severity_max": 4,
         "severity_basis": "Ishak"},
        {"severity_score": 3, "severity_max": None, "severity_basis": ""},
    )
    joined = " | ".join(notes)
    assert "outside its own stated range" in joined
    assert "without a maximum" in joined
    assert "without naming the rubric" in joined


def test_consistent_severity_produces_no_noise():
    from backend.agents.pathology_parts.image_analysis import _severity_observations

    assert _severity_observations(
        {"severity_rubric_name": "Ishak", "severity_score": 2, "severity_max": 4,
         "severity_basis": "Ishak"},
        {"severity_rubric_name": "Ishak", "severity_score": 2, "severity_max": 4,
         "severity_basis": "Ishak"},
    ) == []


def test_two_layers_grading_on_different_rubrics_is_flagged():
    from backend.agents.pathology_parts.image_analysis import _severity_observations

    notes = _severity_observations(
        {"severity_rubric_name": "Ishak", "severity_score": 1, "severity_max": 4,
         "severity_basis": "b"},
        {"severity_rubric_name": "NAS", "severity_score": 1, "severity_max": 4,
         "severity_basis": "b"},
    )
    assert any("different rubrics" in n for n in notes)


def test_same_rubric_name_over_different_ranges_is_flagged():
    from backend.agents.pathology_parts.image_analysis import _severity_observations

    notes = _severity_observations(
        {"severity_rubric_name": "Ishak", "severity_score": 3, "severity_max": 4,
         "severity_basis": "b"},
        {"severity_rubric_name": "Ishak", "severity_score": 3, "severity_max": 5,
         "severity_basis": "b"},
    )
    assert any("different ranges" in n for n in notes)


def test_same_rubric_worded_differently_is_not_flagged():
    """Genuinely different prose for one rubric; identity comes from name + range."""
    from backend.agents.pathology_parts.image_analysis import _severity_observations

    assert _severity_observations(
        {"severity_rubric_name": "Ishak fibrosis", "severity_score": 2, "severity_max": 4,
         "severity_basis": "Ishak staging of fibrosis; 0 none, 4 bridging with nodules"},
        {"severity_rubric_name": "Ishak fibrosis", "severity_score": 2, "severity_max": 4,
         "severity_basis": "Fibrosis graded per Ishak, where 4 indicates bridging fibrosis"},
    ) == []


def test_a_score_without_a_rubric_name_is_reported_as_unidentifiable():
    from backend.agents.pathology_parts.image_analysis import _severity_observations

    notes = _severity_observations(
        {"severity_score": 2, "severity_max": 4, "severity_basis": "some scale"}, {},
    )
    assert any("rubric identity unavailable" in n for n in notes)


# --------------------------------------------------------------- literature
@pytest.mark.asyncio
async def test_integration_literature_framing_reaches_the_provider_payload():
    """Both Deep Research clients drop `topic` when a brief is supplied.

    Exercised through the real call, capturing what _do_research is actually
    handed, rather than reading the source and hoping.
    """
    from backend.agents.literature_agent import LiteratureAgent

    provider = MagicMock()
    provider.get_model.return_value = "m"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.3
    provider.get_llm_client.return_value = MagicMock()
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("s")

    captured = {}

    async def fake_research(topic, state, research_brief=None, **_kw):
        captured["topic"] = topic
        captured["brief"] = research_brief
        return {"summary": "", "findings": []}

    agent._do_research = fake_research
    await agent.integration_literature_search(
        AnalysisState(user_id="u", name="P", initial_instruction="obj"),
        "the integration report body",
    )

    brief = captured["brief"] or ""
    assert brief, "a brief is sent, so the framing must be inside it"
    for token in ("contradicts", "null", "negative", "One-sided", "one-sided"):
        if token in brief:
            break
    else:
        raise AssertionError(f"balanced-evidence framing missing from the brief: {brief[:300]}")
    assert "the integration report body" in brief


# --------------------------------------------------------------- report status
def test_report_status_banner_render_hook_is_retired():
    from backend.services.report.quality import render_report_status_line_html

    assert render_report_status_line_html({
        "status": "partial",
        "partial_reasons": ["analysis_quality"],
        "quality_disclosures": {"blocking_issues": ["unverified"], "cautions": []},
    }) == ""
    assert render_report_status_line_html({"status": "completed"}) == ""


def test_regenerated_report_omits_deterministic_qualification_blocks(tmp_path):
    """Partial and quality state remain in JSON without prominent HTML blocks."""
    from backend.services.report.regenerator import generate_html_report

    report = {
        "session_id": "s", "title": "T", "abstract": "A", "conclusions": "C",
        "generated_at": "2026-01-01T00:00:00",
        "status": "partial", "partial_reasons": ["analysis_quality"],
        "sections": [{"id": "results", "title": "Results", "analysis_type": "general",
                      "content": "body", "figures": [], "tables": []}],
        "score_definitions": [],
        "quality_disclosures": {"blocking_issues": ["unverified round"], "cautions": []},
    }
    out = tmp_path / "r"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir()
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")

    page = Path(generate_html_report(out)).read_text(encoding="utf-8")
    assert '<div class="report-status-line' not in page
    assert '<details class="report-limitations-details"' not in page
    assert "Detailed evidence qualifications" not in page
    assert "Conclusions" in page
    retained = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert retained["status"] == "partial"
    assert retained["partial_reasons"] == ["analysis_quality"]
    assert retained["quality_disclosures"]["blocking_issues"] == ["unverified round"]


def test_pathology_projection_reports_what_it_omitted():
    """A bounded projection that does not say it is bounded hides observations."""
    from backend.workflow.synthesis_parts.phases import _pathology_image_projection

    detailed = [
        {"image_id": f"img{i}", "status": "completed",
         "severity_rubric_name": "Ishak", "severity_score": 1, "severity_max": 4,
         "severity_observations": [f"note {i}"],
         "tile_validation": {"valid_tiles": 4, "expected_tiles": 4, "status": "completed"}}
        for i in range(25)
    ]
    projected, counts = _pathology_image_projection(detailed)
    assert len(projected) == 20
    assert counts["total_images"] == 25
    assert counts["omitted_images"] == 5
    assert "5 further image" in counts["note"]
    assert projected[0]["severity_rubric_name"] == "Ishak"
    assert "tile prose" not in json.dumps(projected)


def test_small_pathology_set_reports_no_omission():
    from backend.workflow.synthesis_parts.phases import _pathology_image_projection

    projected, counts = _pathology_image_projection(
        [{"image_id": "a", "status": "completed", "tile_validation": {}}]
    )
    assert counts["omitted_images"] == 0 and "note" not in counts
