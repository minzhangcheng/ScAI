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
