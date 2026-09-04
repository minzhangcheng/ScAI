"""Measurements printed in a biochemistry figure become comparable indicators.

A study whose only biochemistry was a multi-panel figure produced zero
``BiochemistryIndicator`` records: the vision pass read every number and every
group label off the panels, wrote them into prose, and the prose never reached
the structured sink that group comparison, mechanism adjudication and evidence
grading all read from. The numbers were then quoted in narrative while the gates
downstream correctly reported that no biochemistry could adjudicate anything.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.biochemistry_agent import BiochemistryAgent
from backend.agents.biochemistry_parts.image_measurements import _MARKER
from backend.models.schemas import (
    AgentType, AnalysisState, DataFile, TreatmentGroup,
)
from backend.utils.llm_client import UpstreamExhaustedError


VISION_PROSE = """EXTRACTION_STATUS: usable

## 4. Specific Values
*   **Panel G (Rel. metal level, normalized to Con = 1):**
    *   Fe: Con = 1.0, DKO ~2.2
*   **Panel H (Rel. ALT activity):** Con ~0.5, DKO ~4.1

## 5. Statistical Significance
*   **Panel G (Metals):** Fe (***)
*   **Panel H:** ALT (***)
"""

EXTRACTED_ROWS = {
    "results": [
        {"indicator": "Relative Fe level", "value": 1.0, "unit": "", "group": "Con",
         "sample": None, "panel": "G", "readout": "chart_estimate",
         "statistical_annotation": None, "reference_range": None},
        {"indicator": "Relative Fe level", "value": 2.2, "unit": "", "group": "DKO",
         "sample": None, "panel": "G", "readout": "chart_estimate",
         "statistical_annotation": "*** vs Con", "reference_range": None},
    ],
    "notes": "",
}


def _agent(llm_client=None):
    agent = BiochemistryAgent.__new__(BiochemistryAgent)
    agent.agent_type = AgentType.BIOCHEMISTRY
    agent.session_id = "s"
    provider = MagicMock()
    provider.get_llm_client.return_value = llm_client
    provider.get_model.return_value = "m"
    provider.get_system_prompt.return_value = "sys"
    agent._provider = provider
    return agent


def _state():
    state = AnalysisState(user_id="u", current_round=1)
    state.treatment_groups = [
        TreatmentGroup(id="g-con", name="Con", order=0),
        TreatmentGroup(id="g-dko", name="DKO", order=1),
    ]
    return state


def _image(**kwargs):
    """The upload that broke this: an image with no group assigned to the FILE.

    Both groups live inside the picture, which is the only place they could
    live -- one figure cannot be uploaded twice under two group labels.
    """
    fields = dict(
        filename="biochem.png",
        file_path="/nope/biochem.png",
        section="biochemistry",
        data_type="image",
        group_id=None,
    )
    fields.update(kwargs)
    return DataFile(**fields)


def _stub_vision(monkeypatch, text=VISION_PROSE):
    async def _analyze(self, bio_file, instruction):
        bio_file.extracted_data = {
            "analysis": text,
            "type": "visual_analysis_v2",
            "input_sha256": "sha-of-biochem-png",
            "extraction_status": "usable",
            "terminal": True,
        }
        return text

    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", _analyze)


@pytest.mark.asyncio
async def test_group_labels_inside_the_figure_produce_grouped_indicators(monkeypatch):
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    state = _state()

    await agent._step0_extract_image_measurements(state, [_image()], "instruction")

    by_group = {i.group_id: i for i in state.biochemistry_indicators}
    assert set(by_group) == {"g-con", "g-dko"}, (
        "a group label read out of the picture must resolve to the declared "
        "TreatmentGroup, or nothing downstream can compare the two arms"
    )
    assert by_group["g-dko"].value == 2.2
    assert by_group["g-dko"].indicator == "Relative Fe level"
    assert by_group["g-con"].value == 1.0


@pytest.mark.asyncio
async def test_chart_reads_declare_their_provenance_and_the_figures_own_statistics(monkeypatch):
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    state = _state()

    await agent._step0_extract_image_measurements(state, [_image()], "instruction")

    dko = next(i for i in state.biochemistry_indicators if i.group_id == "g-dko")
    notes = " | ".join(dko.provenance_notes)
    assert "biochem.png" in notes
    assert "panel G" in notes
    # A value estimated against an axis is not a printed number, and a reader who
    # cannot tell the two apart will over-trust it.
    assert "chart" in notes.lower() and "estimate" in notes.lower()
    # The figure's own significance marker is the statistic that adjudicates this
    # comparison; recomputing one from two group means would invent it.
    assert "*** vs Con" in notes


@pytest.mark.asyncio
async def test_measurements_are_minted_once_per_image_version(monkeypatch):
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    state = _state()
    image = _image()

    await agent._step0_extract_image_measurements(state, [image], "instruction")
    await agent._step0_extract_image_measurements(state, [image], "instruction")

    assert len(state.biochemistry_indicators) == 2, (
        "process() runs every round; re-minting would multiply one figure's "
        "measurements into a fake replicate count"
    )
    assert agent.chat.await_count == 1


@pytest.mark.asyncio
async def test_step2_reuses_the_analysis_instead_of_calling_vision_twice(monkeypatch):
    agent = _agent()
    calls = []

    async def _analyze(self, bio_file, instruction):
        calls.append(bio_file.filename)
        return VISION_PROSE

    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", _analyze)
    agent.chat = AsyncMock(return_value=json.dumps({
        "image_findings": ["f"], "observed_values": [],
        "image_mechanisms": [], "image_suggested_experiments": [],
    }))

    result = await agent._step2_process_images(
        [_image()], "instruction", analyses={"/nope/biochem.png": VISION_PROSE},
    )

    assert result["status"] == "completed"
    assert calls == [], "step 0 already paid for this vision call"


@pytest.mark.asyncio
async def test_image_measurements_reach_step1_classification(monkeypatch):
    """Ordering, end to end: step 1 is what grades a measurement.

    Minting the records after step 1 would leave them ungraded -- present in the
    state, absent from every status, basis and abnormal-marker judgement.
    """
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.refresh_runtime_config_if_needed = MagicMock()
    agent.chat = AsyncMock(side_effect=[
        json.dumps(EXTRACTED_ROWS),                       # step 0 extraction
        json.dumps({"abnormal_markers": [], "findings": [],
                    "mechanisms": [], "suggested_experiments": []}),   # step 1
        json.dumps({"image_findings": [], "observed_values": [],
                    "image_mechanisms": [], "image_suggested_experiments": []}),
        json.dumps({"findings": [], "abnormal_markers": [], "interpretation": "i",
                    "suggested_experiments": [], "conclusions": []}),
    ])
    state = _state()
    state.data_files = [_image()]

    await agent.process(state, "instruction")

    step1_prompt = agent.chat.await_args_list[1].args[0]
    assert "Relative Fe level" in step1_prompt
    assert "group=DKO" in step1_prompt
    assert "Groups present in these measurements" in step1_prompt


# ---------------------------------------------------------------------------
# Idempotence. `process` runs every round, `add_result` is a bare append with no
# dedupe, and the once-only marker is the only thing standing between one figure
# and a fake replicate count -- so every path that can lose or wrongly set it is
# pinned here.
# ---------------------------------------------------------------------------


BARE_ARRAY = json.dumps(EXTRACTED_ROWS["results"])


@pytest.mark.asyncio
async def test_an_unreadable_extraction_response_leaves_the_image_unmarked(monkeypatch):
    """A response that could not be read is not "this figure has no numbers".

    `_safe_parse_json` returns the FIRST balanced object it finds, so a model
    answering with a bare JSON array yields one measurement row and no `results`
    key. Marking the image done there is permanent: it would never be re-read,
    including after the operator switches to a compliant model.
    """
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=BARE_ARRAY)
    state = _state()
    image = _image()

    await agent._step0_extract_image_measurements(state, [image], "instruction")

    assert state.biochemistry_indicators == []
    assert not (image.extracted_data or {}).get(_MARKER), (
        "an unreadable answer must not silence this image for the session"
    )

    # ... and the next round tries again rather than skipping in silence.
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    await agent._step0_extract_image_measurements(state, [image], "instruction")
    assert len(state.biochemistry_indicators) == 2


@pytest.mark.asyncio
async def test_a_readable_answer_with_no_measurements_still_finishes_the_image(monkeypatch):
    """The counterpart: an empty `results` list is a real answer about a
    qualitative panel, and re-paying for it every round would be waste."""
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps({"results": [], "notes": ""}))
    state = _state()
    image = _image()

    await agent._step0_extract_image_measurements(state, [image], "instruction")
    await agent._step0_extract_image_measurements(state, [image], "instruction")

    assert state.biochemistry_indicators == []
    assert agent.chat.await_count == 1


@pytest.mark.asyncio
async def test_a_raising_vision_call_cannot_destroy_the_once_only_marker(monkeypatch):
    """The vision step rewrites `extracted_data` on the path where it RAISES.

    An exhausted upstream writes {"error": ...} and re-raises so the workflow's
    LLM-failure gate can offer Retry -- and that retry re-enters this node. If
    the marker went with it, the retry mints the same figure a second time and
    the duplicates read downstream as replicates the experiment never had.
    """
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    state = _state()
    image = _image()

    await agent._step0_extract_image_measurements(state, [image], "instruction")
    assert len(state.biochemistry_indicators) == 2

    async def _exhausted(self, bio_file, instruction):
        bio_file.extracted_data = {"error": "upstream exhausted"}
        raise UpstreamExhaustedError("gateway gave up")

    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", _exhausted)
    with pytest.raises(UpstreamExhaustedError):
        await agent._step0_extract_image_measurements(state, [image], "instruction")
    assert (image.extracted_data or {}).get(_MARKER), "the marker was lost"

    # The operator retries; vision recovers; nothing is minted twice.
    _stub_vision(monkeypatch)
    await agent._step0_extract_image_measurements(state, [image], "instruction")
    assert len(state.biochemistry_indicators) == 2


@pytest.mark.asyncio
async def test_a_failure_part_way_through_registration_leaves_nothing_behind(monkeypatch):
    """`add_result` commits as it goes and the marker is written only at the end.

    Half a figure plus no marker is the worst of both: the next round mints all
    of it again on top of the rows that survived.
    """
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    state = _state()
    image = _image()

    real_add = BiochemistryAgent.add_result
    calls = {"n": 0}

    async def _flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("row 2 is malformed")
        return await real_add(self, *args, **kwargs)

    monkeypatch.setattr(BiochemistryAgent, "add_result", _flaky)
    await agent._step0_extract_image_measurements(state, [image], "instruction")

    assert state.biochemistry_indicators == [], (
        "a partially registered figure must be rolled back, not left as orphans"
    )
    assert not (image.extracted_data or {}).get(_MARKER)


# ---------------------------------------------------------------------------
# What the picture says about itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_groups_inside_the_figure_outrank_the_group_it_was_uploaded_under(monkeypatch):
    """A figure plots every arm it compares, so it can be uploaded under only one.

    The ordinary rule -- upload-declared group wins, parsed group fills a gap --
    is right for a data file, where the upload is a per-sample statement. For an
    image it collapses a two-arm chart into a single arm, and the group
    comparison then runs that arm against itself.
    """
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    state = _state()

    await agent._step0_extract_image_measurements(
        state, [_image(group_id="g-dko")], "instruction"
    )

    assert {i.group_id for i in state.biochemistry_indicators} == {"g-con", "g-dko"}
    con = next(i for i in state.biochemistry_indicators if i.group_id == "g-con")
    assert any("g-dko" in note for note in con.provenance_notes), (
        "the disagreement with the upload must still be recorded"
    )


@pytest.mark.asyncio
async def test_a_replicate_label_read_off_a_panel_is_not_the_studys_n(monkeypatch):
    """`biochemistry_sample_counts` derives a group's detected n from the distinct
    `sample_number` values on its indicators and reports it as authoritative over
    the declared design. One label read off a chart of group means would restate
    a six-replicate arm as n=1, so the figure's wording is kept as provenance."""
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps({"results": [
        {"indicator": "ALT", "value": 4.1, "unit": "U/L", "group": "DKO",
         "sample": "replicate 1", "panel": "H", "readout": "chart_estimate",
         "statistical_annotation": None, "reference_range": None},
    ]}))
    state = _state()

    await agent._step0_extract_image_measurements(state, [_image()], "instruction")

    record = state.biochemistry_indicators[0]
    assert record.sample_number is None
    assert any("replicate 1" in note for note in record.provenance_notes)


@pytest.mark.asyncio
async def test_one_figure_cannot_flood_the_indicator_list(monkeypatch):
    """Step 1 renders every indicator ever minted into its classification prompt
    with no truncation, so the number one figure may contribute is bounded."""
    _stub_vision(monkeypatch)
    agent = _agent()
    agent.chat = AsyncMock(return_value=json.dumps({"results": [
        {"indicator": f"marker {i}", "value": float(i), "unit": "",
         "group": "Con", "sample": None, "panel": "A",
         "readout": "printed_number", "statistical_annotation": None,
         "reference_range": None}
        for i in range(500)
    ]}))
    state = _state()

    await agent._step0_extract_image_measurements(state, [_image()], "instruction")

    minted = len(state.biochemistry_indicators)
    assert 0 < minted < 500, (
        f"one figure contributed {minted} indicators, unbounded"
    )
