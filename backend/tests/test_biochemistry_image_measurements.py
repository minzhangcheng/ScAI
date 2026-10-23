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
from backend.agents.biochemistry_parts.image_measurements import IMAGE_PREPROCESSING_VERSION, _MARKER
from backend.models.schemas import (
    AgentType, AnalysisState, DataFile, TreatmentGroup,
)


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
            "preprocessing_version": IMAGE_PREPROCESSING_VERSION,
            "extraction_status": "usable",
            "terminal": True,
        }
        return text

    monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", _analyze)


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
async def test_image_uncertainty_survives_persistence_and_indicator_analysis(monkeypatch):
    _stub_vision(monkeypatch)
    agent = _agent()
    payload = {
        "results": [{**EXTRACTED_ROWS["results"][1],
                     "value_description": "DKO approximately 2.2; axis reading is uncertain"}],
        "notes": "Raw numerical data and error-bar definitions are unavailable",
    }
    agent.chat = AsyncMock(return_value=json.dumps(payload))
    state = _state()
    await agent._step0_extract_image_measurements(state, [_image()], "review")
    state = AnalysisState.model_validate_json(state.model_dump_json())
    record = state.biochemistry_indicators[0]
    expected_notes = [
        "value read from image biochem.png, panel G",
        "chart estimate read against an axis, not a printed number",
        "source value description: " + payload["results"][0]["value_description"],
        "image extraction limitations: " + payload["notes"],
    ]
    assert all(note in record.provenance_notes for note in expected_notes)
    assert record.sample_number is None
    api_record = state.to_api_dict()["biochemistry_indicators"][0]
    assert api_record["origin"] == "extracted"
    assert api_record["provenanceNotes"] == record.provenance_notes
    assert api_record["sourceGroupLabel"] == record.source_group_label
    agent.chat = AsyncMock(return_value=json.dumps({"abnormal_markers": [{
        "measurement_id": record.id, "provenance_notes": ["Additional model caveat"],
    }]}))
    result = await agent._step1_process_indicators(state, [], "review")
    prompt = agent.chat.await_args.args[0]
    assert all(note in prompt for note in record.provenance_notes)
    marker = result["abnormal_markers"][0]
    assert marker["provenance_notes"] == record.provenance_notes + ["Additional model caveat"]
    assert marker["value"] == 2.2


# Idempotence. `process` runs every round, `add_result` is a bare append with no
# dedupe, and the once-only marker is the only thing standing between one figure
# and a fake replicate count -- so every path that can lose or wrongly set it is
# pinned here.


BARE_ARRAY = json.dumps(EXTRACTED_ROWS["results"])


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


# What the picture says about itself


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "vision_failure", "json_failure", "registration_failure"])
async def test_preprocessing_upgrade_replaces_only_linked_rows_after_success(monkeypatch, outcome):
    _stub_vision(monkeypatch)
    agent = _agent()
    state = _state()
    manual = await agent.add_result("Manual assay", 10, "", state)
    other_image = await agent.add_result("Other image assay", 20, "", state)
    legacy = await agent.add_result("Legacy image assay", 30, "", state)
    prior_marker = {
        "input_sha256": "sha-of-biochem-png", "indicator_ids": [legacy.id], "count": 1,
    }
    image = _image(extracted_data={_MARKER: prior_marker})
    agent.chat = AsyncMock(return_value=json.dumps(EXTRACTED_ROWS))
    if outcome == "vision_failure":
        monkeypatch.setattr(BiochemistryAgent, "_analyze_single_image", AsyncMock(return_value=None))
    elif outcome == "json_failure":
        agent.chat.return_value = "unparseable response"
    elif outcome == "registration_failure":
        real_add = BiochemistryAgent.add_result
        calls = 0

        async def fail_second_row(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("registration failed")
            return await real_add(self, *args, **kwargs)

        monkeypatch.setattr(BiochemistryAgent, "add_result", fail_second_row)

    await agent._step0_extract_image_measurements(state, [image], "review")
    ids = {row.id for row in state.biochemistry_indicators}
    assert {manual.id, other_image.id} <= ids
    if outcome == "success":
        assert legacy.id not in ids
        assert len(state.biochemistry_indicators) == 4
        assert image.extracted_data[_MARKER]["preprocessing_version"] == IMAGE_PREPROCESSING_VERSION
        await agent._step0_extract_image_measurements(state, [image], "next round")
        assert agent.chat.await_count == 1
        assert {row.id for row in state.biochemistry_indicators} == ids
    else:
        assert ids == {manual.id, other_image.id, legacy.id}
        assert image.extracted_data[_MARKER] == prior_marker
