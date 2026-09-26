"""Detected replicate counts are resolved independently per modality."""
from backend.models.schemas import AnalysisState, SampleCountRecord


import backend.agents.bioinformatics_agent as bio_mod
from backend.models.schemas import DataFile, TreatmentGroup
from backend.services.sample_design import (
    SCRNA_MODALITY,
    has_replicated_condition_design,
    sequencing_design_instruction,
)


def _state_with(records, groups, files=()):
    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = list(groups)
    state.data_files = list(files)
    state.detected_sample_counts = list(records)
    return state


def _rec(group_id, group_name, section, data_type, detected, declared,
         method="obs_column:sample"):
    return SampleCountRecord(
        group_id=group_id, group_name=group_name, section=section,
        data_type=data_type, detected_count=detected, declared_count=declared,
        method=method, evidence="test", conflicts=detected != declared,
    )


def test_explicit_undetermined_record_does_not_fall_back_to_upload_numbers():
    """A failed measurement must outrank UI file numbering."""
    state = _state_with(
        records=[
            _rec(group_id, name, "sequencing", "scRNA-seq", 0, 3,
                 "undetermined")
            for group_id, name in (("g1", "Con"), ("g2", "DKO"))
        ],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
        files=[
            DataFile(
                filename=f"{group_id}_{number}.h5ad",
                file_path=f"/{group_id}_{number}.h5ad",
                section="sequencing", data_type="scRNA-seq",
                group_id=group_id, sample_number=number,
            )
            for group_id in ("g1", "g2") for number in (1, 2, 3)
        ],
    )

    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "scRNA-seq": {},
    }
    assert has_replicated_condition_design(state, SCRNA_MODALITY) is False


def test_unreplicated_single_cell_design_is_a_pi_fact_not_a_host_decision():
    state = _state_with(
        records=[
            _rec(group_id, name, "sequencing", "scRNA-seq", 1, 3)
            for group_id, name in (("g1", "Con"), ("g2", "DKO"))
        ],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
    )

    rendered = sequencing_design_instruction(state)
    assert "Single-cell independent-unit counts are" in rendered
    assert "Con=1" in rendered and "DKO=1" in rendered
    assert "PI must decide" in rendered
    assert "do not plan" not in rendered.lower()
    assert "blocked" not in rendered.lower()


import json
import pytest

from backend.utils.llm_client import ToolCall


class _FakeResult:
    def __init__(self, output="", error=None):
        self.output = output
        self.error = error


class _FakeSandbox:
    def __init__(self):
        self.calls = []

    async def execute(self, code, **kwargs):
        self.calls.append((code, kwargs))
        return _FakeResult(output="AnnData object with n_obs 8927")


class _FakeResponse:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content
        # None here so profile_data exercises the hand-built assistant-message
        # fallback (the branch that needs json.dumps on call.arguments), the
        # same branch a plain OpenAI-compatible response takes in production.
        self.raw_assistant_message = None


@pytest.mark.asyncio
async def test_profile_data_registers_what_the_model_reports():
    """The loop's only job: run reads, then convert one tool call to records."""
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.sent_messages = []
            self.stream_preferences = []
            self._responses = [
                _FakeResponse([ToolCall(id="c1", name="execute_code", arguments={
                    "code": "import scanpy as sc\nadata = sc.read_loom('Con.loom')",
                    "description": "load", "step_type": "technical",
                    "persistent_outputs": ["adata"],
                })]),
                _FakeResponse([ToolCall(id="c2", name="register_sample_counts",
                                        arguments={"records": [{
                                            "group_name": "Con",
                                            "data_type": "scRNA-seq",
                                            "detected_count": 1,
                                            "method": "obs_column:sample",
                                            "evidence": "1 unique value",
                                        }]})]),
            ]

        async def chat_with_tools(self, **kwargs):
            # Captured so the assistant-message fallback (Correction 2) can be
            # verified, not just exercised: a wrong key or a missing json.dumps
            # would otherwise fail nothing here.
            self.sent_messages.append(list(kwargs["messages"]))
            self.stream_preferences.append(kwargs.get("prefer_stream"))
            return self._responses.pop(0)

        def _validate_code(self, code):
            return True, ""

        async def _notify(self, kind, payload):
            return None

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]

    agent = Agent()
    result = await agent.profile_data(state)

    assert result["status"] == "completed"
    assert result["objects"] == ["adata"]
    assert len(result["records"]) == 1
    record = result["records"][0]
    assert (record.group_id, record.detected_count, record.declared_count) == ("g1", 1, 3)
    assert record.conflicts is True
    assert agent.stream_preferences == [True, True]

    # Correction 2's fallback (raw_assistant_message is None on the fake): the
    # second call must have been handed the hand-built assistant entry from the
    # first response, with arguments re-serialized to a JSON string -- that
    # last detail is the whole point of the json.dumps on that path.
    replayed = agent.sent_messages[1]
    assistant_entries = [m for m in replayed if m.get("role") == "assistant"]
    assert len(assistant_entries) == 1
    assistant_entry = assistant_entries[0]
    assert assistant_entry["content"] == ""
    tool_calls_out = assistant_entry["tool_calls"]
    assert len(tool_calls_out) == 1
    call_out = tool_calls_out[0]
    assert call_out["id"] == "c1"
    assert call_out["type"] == "function"
    assert call_out["function"]["name"] == "execute_code"
    assert isinstance(call_out["function"]["arguments"], str)
    assert json.loads(call_out["function"]["arguments"]) == {
        "code": "import scanpy as sc\nadata = sc.read_loom('Con.loom')",
        "description": "load", "step_type": "technical",
        "persistent_outputs": ["adata"],
    }


@pytest.mark.asyncio
async def test_profile_data_retries_registration_when_a_group_name_is_unrecognized():
    """A group_name that matches no group must not silently succeed as "0
    records registered" -- the model is told which names it sent and which
    names exist, and gets another turn to correct it.
    """
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.messages_seen = []
            self._responses = [
                _FakeResponse([ToolCall(id="c1", name="register_sample_counts",
                                        arguments={"records": [{
                                            "group_name": "Control",
                                            "data_type": "scRNA-seq",
                                            "detected_count": 1,
                                            "method": "obs_column:sample",
                                            "evidence": "1 unique value",
                                        }]})]),
            ]

        async def chat_with_tools(self, **kwargs):
            self.messages_seen.append(list(kwargs["messages"]))
            if self._responses:
                return self._responses.pop(0)
            return _FakeResponse()

        def _validate_code(self, code):
            return True, ""

        async def _notify(self, kind, payload):
            return None

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]

    agent = Agent()
    result = await agent.profile_data(state)

    assert result["status"] == "failed"
    assert result["records"] == []

    reply = next(m for m in agent.messages_seen[1] if m.get("tool_call_id") == "c1")
    assert "Control" in reply["content"]
    assert "Con" in reply["content"]




_MINIMAL_PLAN_JSON = (
    '{"overview":"ok","steps":[{"id":1,"title":"Verify","method":'
    '"scanpy.read_h5ad","depends_on":[]}]}'
)
