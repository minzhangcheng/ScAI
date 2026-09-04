"""Detected replicate counts are resolved independently per modality."""
from backend.models.schemas import AnalysisState, SampleCountRecord


def test_record_flags_a_conflict_between_detected_and_declared():
    record = SampleCountRecord(
        group_id="g1",
        group_name="Con",
        section="sequencing",
        data_type="scRNA-seq",
        detected_count=1,
        declared_count=3,
        method="obs_column:sample",
        evidence="adata.obs['sample'] holds a single value: Con_1",
        conflicts=True,
    )
    assert record.detected_count == 1
    assert record.declared_count == 3
    assert record.conflicts is True


def test_state_starts_with_no_detected_counts():
    state = AnalysisState(name="s", user_id="u")
    assert state.detected_sample_counts == []


def test_detected_counts_survive_a_state_roundtrip():
    state = AnalysisState(name="s", user_id="u")
    state.detected_sample_counts = [
        SampleCountRecord(
            group_id="g1", group_name="Con", section="sequencing",
            data_type="scRNA-seq", detected_count=1, declared_count=3,
            method="obs_column:sample", evidence="one value", conflicts=True,
        )
    ]
    restored = AnalysisState.model_validate(state.model_dump(mode="json"))
    assert restored.detected_sample_counts[0].detected_count == 1
    assert restored.detected_sample_counts[0].conflicts is True


def test_to_api_dict_serializes_detected_sample_counts_to_camelcase():
    """to_api_dict's per-field dispatch falls back to `result[api_key] = val`
    for any field name it doesn't special-case, so a deleted dispatch branch
    and serializer would still leave every other test in this area green --
    coverage only checks _API_KEY_MAP membership, and a raw snake_case dict is
    just as JSON-serializable as a camelCase one. This is the one test that
    looks at the emitted shape itself."""
    state = AnalysisState(name="s", user_id="u")
    state.detected_sample_counts = [
        SampleCountRecord(
            group_id="grp-42",
            group_name="Treatment A",
            section="pathology",
            data_type="IHC",
            detected_count=5,
            declared_count=7,
            method="file_records",
            evidence="5 distinct slide IDs under group Treatment A",
            conflicts=True,
        )
    ]

    entry = state.to_api_dict()["detected_sample_counts"][0]

    assert entry == {
        "groupId": "grp-42",
        "groupName": "Treatment A",
        "section": "pathology",
        "dataType": "IHC",
        "detectedCount": 5,
        "declaredCount": 7,
        "method": "file_records",
        "evidence": "5 distinct slide IDs under group Treatment A",
        "conflicts": True,
    }


import backend.agents.bioinformatics_agent as bio_mod
from backend.models.schemas import DataFile, TreatmentGroup
from backend.services import sample_design
from backend.services.sample_design import (
    BULK_RNASEQ_MODALITY,
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


def test_detected_count_supersedes_the_declaration():
    """The measured failure: declared 3, uploaded 1, resolver returned 3."""
    state = _state_with(
        records=[_rec("g1", "Con", "sequencing", "scRNA-seq", 1, 3),
                 _rec("g2", "DKO", "sequencing", "scRNA-seq", 1, 3)],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
    )
    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "scRNA-seq": {"Con": 1, "DKO": 1},
    }


def test_sample_counts_do_not_collapse_across_modalities():
    """Non-sequencing records cannot change the scRNA replicate count."""
    state = _state_with(
        records=[_rec("g1", "Con", "sequencing", "scRNA-seq", 1, 3),
                 _rec("g1", "Con", "pathology", "H&E", 1, 3, "file_records"),
                 _rec("g1", "Con", "biochemistry", "image", 6, 3,
                      "extracted_data")],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3)],
    )
    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "scRNA-seq": {"Con": 1},
    }


def test_an_undetermined_record_is_skipped_not_counted_as_zero():
    """detected_count=0 means unreadable, not "no samples".

    Counting it would let one unreadable file manufacture an n=0 group.
    """
    state = _state_with(
        records=[_rec("g1", "Con", "sequencing", "Loom", 0, 3, "undetermined"),
                 _rec("g1", "Con", "sequencing", "scRNA-seq", 2, 3)],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3)],
    )
    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "scRNA-seq": {"Con": 2},
    }


def test_a_group_with_nothing_determined_remains_unresolved():
    """A declaration cannot replace a failed profiling result."""
    state = _state_with(
        records=[_rec("g1", "Con", "sequencing", "Loom", 0, 4, "undetermined")],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=4)],
    )
    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "scRNA-seq": {},
    }


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


def test_ungrouped_bulk_total_does_not_invent_group_membership():
    """A total of six columns does not prove the declared three-plus-three split."""
    state = _state_with(
        records=[_rec(None, "", "sequencing", "bulk RNA-seq", 6, None),
                 _rec("g1", "Con", "sequencing", "scRNA-seq", 1, 3),
                 _rec("g2", "DKO", "sequencing", "scRNA-seq", 1, 3)],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
    )
    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "bulk RNA-seq": {},
        "scRNA-seq": {"Con": 1, "DKO": 1},
    }
    assert has_replicated_condition_design(state, BULK_RNASEQ_MODALITY) is False


def test_an_ungrouped_file_is_not_read_as_an_unidentified_extra_sample():
    """The measured failure: one six-column bulk matrix read as a seventh sample.

    `merged_counts.csv` carries no group label of its own, so it lands in the
    ungrouped list, which is rendered in FILES. Its six columns were detected
    and attributed three-and-three to the two groups, which is rendered in
    SAMPLES. Printed side by side those read as "3 + 3 grouped, plus 1
    unidentified", and the PI planned round after round to resolve the identity
    and provenance of a sample that does not exist.
    """
    state = _state_with(
        records=[
            _rec("g1", "Con", "sequencing", "bulk RNA-seq", 3, 3, "matrix_columns"),
            _rec("g2", "DKO", "sequencing", "bulk RNA-seq", 3, 3, "matrix_columns"),
        ],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
        files=[DataFile(
            filename="merged_counts.csv", file_path="/tmp/merged_counts.csv",
            section="sequencing", data_type="bulk RNA-seq", group_id=None,
        )],
    )

    summary = state.build_experimental_design_summary()

    assert "file counts, not sample counts" in summary
    assert "bulk RNA-seq" in summary
    assert "already counted in the groups above" in summary


def test_the_reassurance_is_absent_where_samples_really_are_unattributed():
    """The note states a fact, so it must not appear when the fact is false."""
    state = _state_with(
        records=[_rec("g1", "Con", "sequencing", "scRNA-seq", 1, 3)],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3)],
        files=[DataFile(
            filename="biochem.png", file_path="/tmp/biochem.png",
            section="biochemistry", data_type="image", group_id=None,
        )],
    )

    summary = state.build_experimental_design_summary()

    assert "file counts, not sample counts" in summary
    assert "already counted in the groups above" not in summary


def test_ungrouped_total_must_match_declared_group_sum_before_distribution():
    state = _state_with(
        records=[_rec(None, "", "sequencing", "bulk RNA-seq", 2, None)],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
    )

    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "bulk RNA-seq": {},
    }


def test_mixed_design_replication_is_decided_per_modality():
    state = _state_with(
        records=[
            _rec(group_id, name, "sequencing", "scRNA-seq", 1, 3)
            for group_id, name in (("g1", "Con"), ("g2", "DKO"))
        ] + [
            _rec(group_id, name, "sequencing", "bulk RNA-seq", 3, 3)
            for group_id, name in (("g1", "Con"), ("g2", "DKO"))
        ],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
    )

    assert has_replicated_condition_design(state, SCRNA_MODALITY) is False
    assert has_replicated_condition_design(state, BULK_RNASEQ_MODALITY) is True


def test_partial_group_counts_fail_the_whole_modality_closed():
    """One resolved arm cannot make an omitted comparison group disappear."""
    state = _state_with(
        records=[_rec("g1", "Con", "sequencing", "bulk RNA-seq", 3, 3)],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
    )

    assert bio_mod._gate_samples_per_group_by_modality(state) == {
        "bulk RNA-seq": {},
    }
    assert has_replicated_condition_design(state, BULK_RNASEQ_MODALITY) is False


def test_profiling_rejects_a_nonempty_but_partial_registration():
    from backend.agents.bioinformatics.profiling import (
        _registration_coverage_error,
    )

    state = _state_with(
        records=[],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
        files=[DataFile(
            filename="bulk.csv", file_path="/bulk.csv", section="sequencing",
            data_type="bulk RNA-seq",
        )],
    )
    records = [
        _rec("g1", "Con", "sequencing", "bulk RNA-seq", 3, 3),
    ]

    assert "incomplete" in _registration_coverage_error(state, records)


def test_declarations_alone_do_not_enable_condition_inference():
    state = _state_with(
        records=[],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
        files=[
            DataFile(filename=f"{group}.h5ad", file_path=f"/{group}.h5ad",
                     section="sequencing", data_type="scRNA-seq",
                     group_id=group)
            for group in ("g1", "g2")
        ],
    )

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


def test_sample_design_service_exposes_facts_without_plan_or_code_vetoes():
    assert not hasattr(sample_design, "plan_sample_design_violations")
    assert not hasattr(sample_design, "code_sample_design_violation")


def test_the_profiling_tool_set_is_two_tools_and_excludes_the_analysis_tools():
    """The pass produces no findings, no figures, and no plan.

    Offering report_findings or generate_publication_figure here would let a
    read-only measurement pass emit analysis artifacts that no plan asked for.
    """
    from backend.agents.bioinformatics.tool_schemas import (
        KNOWN_TOOL_NAMES, PROFILING_TOOL_NAMES, PROFILING_TOOLS, TOOLS,
    )
    assert PROFILING_TOOL_NAMES == {"execute_code", "register_sample_counts"}
    assert {t["function"]["name"] for t in PROFILING_TOOLS} == PROFILING_TOOL_NAMES
    # Identity, not name equality: PROFILING_TOOLS must reuse the executor's own
    # execute_code schema object, or persistent_outputs could silently drift
    # between the two passes without any test here noticing.
    assert PROFILING_TOOLS[0] is {t["function"]["name"]: t for t in TOOLS}["execute_code"]
    for forbidden in ("report_findings", "generate_publication_figure",
                      "consult_pi", "search_literature"):
        assert forbidden not in PROFILING_TOOL_NAMES


def test_register_sample_counts_is_not_offered_to_the_main_executor():
    """A count measured before planning must not be rewritable during it."""
    from backend.agents.bioinformatics.tool_schemas import KNOWN_TOOL_NAMES
    assert "register_sample_counts" not in KNOWN_TOOL_NAMES


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
async def test_profile_data_stops_at_its_step_budget_without_raising():
    """Exhausting the budget ends the pass; it is not an error.

    A verification pass that becomes a new single point of failure repeats the
    defect 0.39.40 removed, where an unsatisfiable check trapped the round.
    """
    from backend.agents.bioinformatics.profiling import (
        MAX_PROFILING_STEPS, ProfilingMixin,
    )

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.turns = 0

        async def chat_with_tools(self, **kwargs):
            self.turns += 1
            return _FakeResponse([ToolCall(id=f"c{self.turns}", name="execute_code",
                                           arguments={
                                               "code": "adata",
                                               "description": "peek",
                                               "step_type": "technical",
                                               "persistent_outputs": [],
                                           })])

        def _validate_code(self, code):
            return True, ""

        async def _notify(self, kind, payload):
            return None

    agent = Agent()
    state = AnalysisState(name="s", user_id="u")
    result = await agent.profile_data(state)

    assert result["status"] == "completed"
    assert result["records"] == []
    assert agent.turns <= MAX_PROFILING_STEPS



@pytest.mark.asyncio
async def test_profile_data_survives_a_present_but_null_description():
    """description present as JSON null, not missing.

    args.get(key, default) only falls back to default when the key is absent;
    a present-but-null value is returned as None, and None[:200] raises.

    Only the malformed call is queued -- chat_with_tools returns an empty,
    tool-call-free response once the canned queue runs dry, ending the pass
    naturally -- so a bare swallow-and-continue around the dispatch cannot
    lean on a second, independently-valid call to reach "completed". The
    assertion on the actual tool-reply content is what a missing coercion
    would fail; a second queued call would let it pass regardless.
    """
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.messages_seen = None
            self._responses = [
                _FakeResponse([ToolCall(id="c1", name="execute_code", arguments={
                    "code": "adata", "description": None,
                    "step_type": "technical", "persistent_outputs": [],
                })]),
            ]

        async def chat_with_tools(self, **kwargs):
            self.messages_seen = kwargs["messages"]
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

    assert result["status"] == "completed"
    assert result["records"] == []
    replies = [m for m in agent.messages_seen if m.get("tool_call_id") == "c1"]
    assert len(replies) == 1
    assert replies[0]["content"] == "AnnData object with n_obs 8927"


@pytest.mark.asyncio
async def test_profile_data_survives_a_null_code():
    """code present as JSON null must be reported back, not executed.

    The fake _validate_code raises on non-string input here, the same way the
    real regex-based validator would on a null code -- so if the empty-code
    short-circuit were ever dropped while _validate_code kept being called,
    this would still catch it.

    Only the malformed call is queued, for the same reason as the description
    vector above: a second, independently-valid call would let a bare
    swallow-and-continue reach "completed" without the coercion doing
    anything, and the specific reply content is what actually pins the fix.
    """
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.messages_seen = None
            self._responses = [
                _FakeResponse([ToolCall(id="c1", name="execute_code", arguments={
                    "code": None, "description": "load",
                    "step_type": "technical", "persistent_outputs": [],
                })]),
            ]

        async def chat_with_tools(self, **kwargs):
            self.messages_seen = kwargs["messages"]
            if self._responses:
                return self._responses.pop(0)
            return _FakeResponse()

        def _validate_code(self, code):
            if not isinstance(code, str):
                raise TypeError("expected string or bytes-like object")
            return True, ""

        async def _notify(self, kind, payload):
            return None

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]

    agent = Agent()
    result = await agent.profile_data(state)

    assert result["status"] == "completed"
    assert result["records"] == []
    replies = [m for m in agent.messages_seen if m.get("tool_call_id") == "c1"]
    assert len(replies) == 1
    assert replies[0]["content"] == "No code provided; nothing to execute."


@pytest.mark.asyncio
async def test_profile_data_survives_a_boolean_persistent_outputs():
    """persistent_outputs: true -- `True or []` is `True`, and bool isn't iterable.

    The model sent a bool where the schema names an array of strings. Only
    the malformed call is queued, for the same reason as the description
    vector above: a second, independently-valid call would let a bare
    swallow-and-continue reach "completed" without the coercion doing
    anything.
    """
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.messages_seen = None
            self._responses = [
                _FakeResponse([ToolCall(id="c1", name="execute_code", arguments={
                    "code": "adata", "description": "load",
                    "step_type": "technical", "persistent_outputs": True,
                })]),
            ]

        async def chat_with_tools(self, **kwargs):
            self.messages_seen = kwargs["messages"]
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

    assert result["status"] == "completed"
    assert result["records"] == []
    replies = [m for m in agent.messages_seen if m.get("tool_call_id") == "c1"]
    assert len(replies) == 1
    assert replies[0]["content"] == "AnnData object with n_obs 8927"


@pytest.mark.asyncio
async def test_profile_data_survives_a_boolean_records():
    """records: true on register_sample_counts -- the call meant to register
    counts must not be able to crash the pass. It also must not be able to
    silently succeed as "0 records registered": the model is told why its
    call could not be used, and only after that chance is spent does the pass
    end -- degraded, not completed, since a registration was attempted and
    nothing usable came of it.
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
                                        arguments={"records": True})]),
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

    # Told concretely what was wrong, not left with a bare "Registered 0
    # record(s)." that reads as an accepted, empty call.
    first_reply = next(m for m in agent.messages_seen[1]
                        if m.get("tool_call_id") == "c1")
    assert "Registered 0 record(s)." != first_reply["content"]
    assert "bool" in first_reply["content"]


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


@pytest.mark.asyncio
async def test_profile_data_gives_up_after_a_second_unusable_registration():
    """Told once and still sends nothing usable: the pass ends rather than
    spending its whole step budget on repeated unusable calls.
    """
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()
            self.model = "m"
            self.llm_client = self
            self.config = type("C", (), {"temperature": 0.2})()
            self.calls = 0
            self._responses = [
                _FakeResponse([ToolCall(id="c1", name="register_sample_counts",
                                        arguments={"records": []})]),
                _FakeResponse([ToolCall(id="c2", name="register_sample_counts",
                                        arguments={"records": []})]),
            ]

        async def chat_with_tools(self, **kwargs):
            self.calls += 1
            if self._responses:
                return self._responses.pop(0)
            # A third call would prove the give-up branch did not fire.
            return _FakeResponse([ToolCall(id="c3", name="register_sample_counts",
                                           arguments={"records": []})])

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
    assert agent.calls == 2  # gave up after the second unusable attempt


@pytest.mark.asyncio
async def test_profiling_execute_refuses_a_normalizing_cell():
    """This pass shares execute_code's schema -- and its normalization-
    inviting description -- with the executor, but must stay read-only: a
    mutation here would bump analysis_generation and invalidate same-round
    artifacts before any plan step has run.
    """
    from backend.agents.bioinformatics.profiling import ProfilingMixin

    class Agent(ProfilingMixin):
        def __init__(self):
            self.sandbox = _FakeSandbox()

        def _validate_code(self, code):
            return True, ""

        async def _notify(self, kind, payload):
            return None

    agent = Agent()
    objects = []

    reply = await agent._profiling_execute(
        {"code": "sc.pp.normalize_total(adata)", "description": "normalize",
         "persistent_outputs": []},
        objects,
    )

    assert "read-only" in reply
    assert agent.sandbox.calls == []
    assert objects == []


def test_the_profiling_node_skips_after_round_one():
    from backend.workflow.agent_nodes.profiling_node import should_profile

    state = AnalysisState(name="s", user_id="u")
    state.current_round = 2
    assert should_profile(state) is False


def test_the_profiling_node_skips_a_session_that_already_has_records():
    """Idempotent, so a restore does not re-profile."""
    from backend.workflow.agent_nodes.profiling_node import should_profile

    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="sequencing",
                          data_type="scRNA-seq", detected_count=1)
    ]
    assert should_profile(state) is False


def test_the_profiling_node_runs_in_round_one_with_no_records():
    from backend.workflow.agent_nodes.profiling_node import should_profile

    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    assert should_profile(state) is True


def test_the_profiling_node_still_runs_when_only_other_modalities_have_registered():
    """Pathology and biochemistry write to the same list earlier in round 1.
    Their records must not look like "sequencing already profiled" and skip
    it permanently -- especially across a resume between their pass and this
    one.
    """
    from backend.workflow.agent_nodes.profiling_node import should_profile

    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="pathology",
                          data_type="H&E", detected_count=1),
        SampleCountRecord(group_id="g1", group_name="Con", section="biochemistry",
                          data_type="indicator", detected_count=3),
    ]
    assert should_profile(state) is True


@pytest.mark.asyncio
async def test_the_profiling_node_surfaces_a_failed_status_without_losing_records(caplog):
    """profile_data's own contract: it never raises, it returns status="failed"
    for a modelling or execution problem instead. That return shape must not
    be silently indistinguishable from "nothing to profile" -- it must be
    logged and surfaced to the user, and any records the pass registered
    before failing must survive onto detected_sample_counts rather than being
    discarded.
    """
    from backend.workflow.agents import AgentNodesMixin

    class Host(AgentNodesMixin):
        def __init__(self):
            self.session_id = "s1"
            self.sent = []
            self.bioinformatics_agent = self

        async def _send_message(self, kind, content, agent=None):
            self.sent.append((kind, content))

        async def profile_data(self, state):
            return {
                "status": "failed",
                "records": [
                    SampleCountRecord(group_id="g1", group_name="Con",
                                      section="sequencing", data_type="scRNA-seq",
                                      detected_count=1)
                ],
                "objects": ["adata_con"],
                "summary": "LLM call failed: timeout",
            }

    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    host = Host()

    await host._data_profiling({"analysis_state": state})

    # The record the pass DID register before failing must survive, not be
    # discarded because the pass overall did not complete.
    assert len(state.detected_sample_counts) == 1
    assert state.detected_sample_counts[0].group_name == "Con"

    # A degraded-status message reached the user -- not silence.
    status_messages = [c for k, c in host.sent if k == "status"]
    assert any("declared" in m and "sample" in m for m in status_messages), status_messages

    # And it was logged, naming the node and the summary the pass returned.
    assert "data_profiling" in caplog.text
    assert "timeout" in caplog.text


@pytest.mark.asyncio
async def test_the_profiling_node_preserves_other_sections_when_it_registers_sequencing():
    """A restore can land at data_profiling with pathology and biochemistry
    records already on the state (see should_profile's docstring) -- both
    sections early-return on a cache hit before ever re-appending their own
    records, so a plain overwrite here would silently drop them for the
    round, undoing the exact thing this feature exists to preserve.
    """
    from backend.workflow.agents import AgentNodesMixin

    class Host(AgentNodesMixin):
        def __init__(self):
            self.session_id = "s1"
            self.bioinformatics_agent = self

        async def _send_message(self, kind, content, agent=None):
            return None

        async def profile_data(self, state):
            return {
                "status": "completed",
                "records": [
                    SampleCountRecord(group_id="g1", group_name="Con",
                                      section="sequencing", data_type="scRNA-seq",
                                      detected_count=1)
                ],
                "objects": [],
                "summary": "",
            }

    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="pathology",
                          data_type="H&E", detected_count=1),
        SampleCountRecord(group_id="g1", group_name="Con", section="biochemistry",
                          data_type="indicator", detected_count=3),
    ]
    host = Host()

    await host._data_profiling({"analysis_state": state})

    sections = {r.section for r in state.detected_sample_counts}
    assert sections == {"pathology", "biochemistry", "sequencing"}


def test_pathology_counts_split_stains_into_separate_records():
    """Three stains of one animal are not three animals.

    In the measured session the pathology group held Con-H&E.tif,
    Con-Sirius Red.tif and Con-SMA.tif. Counting files gives 3 -- which happens
    to equal the wrong declaration, so a file-count check would have "verified"
    the bug.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        pathology_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.data_files = [
        DataFile(filename="Con-H&E.tif", file_path="/x/a.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
        DataFile(filename="Con-Sirius Red.tif", file_path="/x/b.tif",
                 section="pathology", data_type="Sirius Red", group_id="g1"),
        DataFile(filename="Con-SMA.tif", file_path="/x/c.tif",
                 section="pathology", data_type="IHC, SMA", group_id="g1"),
    ]

    records = pathology_sample_counts(state)

    assert {r.data_type for r in records} == {"H&E", "Sirius Red", "IHC, SMA"}
    assert [r.detected_count for r in records] == [1, 1, 1]
    assert all(r.method == "file_records" for r in records)
    assert all(r.conflicts for r in records)  # declared 3, detected 1


def test_pathology_counts_multiple_unnumbered_files_of_one_stain_are_undetermined():
    """Three H&E images of ONE specimen, not three stains of one animal --
    the exact case fix #1 exists to stop being silently trusted as n=3.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        pathology_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=1)]
    state.data_files = [
        DataFile(filename="Con-HE-1.tif", file_path="/x/a.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
        DataFile(filename="Con-HE-2.tif", file_path="/x/b.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
        DataFile(filename="Con-HE-3.tif", file_path="/x/c.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
    ]

    records = pathology_sample_counts(state)

    assert len(records) == 1
    record = records[0]
    assert record.detected_count == 0
    assert record.method == "undetermined"
    assert record.conflicts is False  # undetermined is not a disagreement
    assert "3" in record.evidence


def test_pathology_counts_a_single_unnumbered_file_is_one_specimen():
    """One image of one stain for one group is unambiguous -- the motivating
    case: it must not be swept into the same undetermined bucket as multiple
    unnumbered files.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        pathology_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=1)]
    state.data_files = [
        DataFile(filename="Con-HE.tif", file_path="/x/a.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
    ]

    records = pathology_sample_counts(state)

    assert len(records) == 1
    assert records[0].detected_count == 1
    assert records[0].method == "file_records"
    assert records[0].conflicts is False  # declared 1, detected 1


def test_pathology_counts_do_not_disturb_sequencing_records():
    from backend.workflow.agent_nodes.modality_nodes import (
        append_modality_sample_counts, pathology_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.data_files = [
        DataFile(filename="Con-H&E.tif", file_path="/x/a.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
    ]
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="sequencing",
                          data_type="scRNA-seq", detected_count=1,
                          declared_count=3, method="obs_column:sample",
                          conflicts=True)
    ]

    append_modality_sample_counts(state, "pathology", pathology_sample_counts(state))

    sections = [r.section for r in state.detected_sample_counts]
    assert sections == ["sequencing", "pathology"]


def test_re_registering_a_modality_replaces_its_own_records_only():
    """A node can re-run; its second pass must not double the list."""
    from backend.workflow.agent_nodes.modality_nodes import (
        append_modality_sample_counts, pathology_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=1)]
    state.data_files = [
        DataFile(filename="Con-H&E.tif", file_path="/x/a.tif",
                 section="pathology", data_type="H&E", group_id="g1"),
    ]
    for _ in range(2):
        append_modality_sample_counts(state, "pathology",
                                      pathology_sample_counts(state))
    assert len(state.detected_sample_counts) == 1


from backend.models.schemas import BiochemistryIndicator


def test_biochemistry_counts_from_indicator_sample_numbers():
    """Distinct sample numbers on a group's indicators are the replicate
    count -- extracted_data (checked and ruled out; see the report) never
    carries one, but BiochemistryIndicator.sample_number does.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        biochemistry_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.biochemistry_indicators = [
        BiochemistryIndicator(indicator="ALT", value=45.0, unit="U/L",
                              group_id="g1", sample_number=1),
        BiochemistryIndicator(indicator="ALT", value=50.0, unit="U/L",
                              group_id="g1", sample_number=2),
        BiochemistryIndicator(indicator="AST", value=30.0, unit="U/L",
                              group_id="g1", sample_number=3),
    ]

    records = biochemistry_sample_counts(state)
    indicator_records = [r for r in records if r.data_type == "indicator"]

    assert len(indicator_records) == 1
    assert indicator_records[0].detected_count == 3
    assert indicator_records[0].method == "indicator_sample_numbers"
    assert indicator_records[0].conflicts is False  # declared 3, detected 3


def test_biochemistry_indicators_without_sample_numbers_are_undetermined():
    """Three unnumbered indicator entries could be three indicators on one
    specimen, or one indicator on three -- the record cannot tell them
    apart. detected_count must come back 0/undetermined here, not 3: that is
    exactly the number a len(indicators) implementation would confidently
    report instead, which is why this vector exists.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        biochemistry_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.biochemistry_indicators = [
        BiochemistryIndicator(indicator="ALT", value=45.0, unit="U/L", group_id="g1"),
        BiochemistryIndicator(indicator="AST", value=30.0, unit="U/L", group_id="g1"),
        BiochemistryIndicator(indicator="TG", value=1.2, unit="mmol/L", group_id="g1"),
    ]

    records = biochemistry_sample_counts(state)
    indicator_records = [r for r in records if r.data_type == "indicator"]

    assert len(indicator_records) == 1
    assert indicator_records[0].detected_count == 0
    assert indicator_records[0].method == "undetermined"
    # An unreadable count is not a disagreement with the declaration --
    # conflicts stays False, the same way it would for any other
    # undetermined record.
    assert indicator_records[0].conflicts is False


def test_ungrouped_biochemistry_indicators_produce_no_record():
    """Contract item 6: records with no group_id never enter the mapping --
    for indicators that means no record at all, since there is no group to
    attach a count to.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        biochemistry_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.biochemistry_indicators = [
        BiochemistryIndicator(indicator="ALT", value=45.0, unit="U/L",
                              group_id=None, sample_number=1),
    ]

    records = biochemistry_sample_counts(state)

    assert [r for r in records if r.data_type == "indicator"] == []


def test_biochemistry_data_file_is_recorded_undetermined_with_evidence():
    """extracted_data holds no per-sample structure for either extraction
    path, so an uploaded file still gets a record -- an honest undetermined
    one, naming the shape, rather than being silently dropped.
    """
    from backend.workflow.agent_nodes.modality_nodes import (
        biochemistry_sample_counts,
    )

    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.data_files = [
        DataFile(filename="chart.png", file_path="/x/chart.png",
                 section="biochemistry", data_type="image", group_id="g1",
                 extracted_data={"analysis": "bar chart of ALT levels",
                                 "type": "visual_analysis_v2"}),
        DataFile(filename="levels.csv", file_path="/x/levels.csv",
                 section="biochemistry", data_type="file", group_id="g1",
                 extracted_data={"results": [{"indicator": "ALT", "value": 45.0,
                                              "unit": "U/L"}]}),
    ]

    records = biochemistry_sample_counts(state)
    file_records = {r.data_type: r for r in records if r.data_type in ("image", "file")}

    assert set(file_records) == {"image", "file"}
    for record in file_records.values():
        assert record.detected_count == 0
        assert record.method == "undetermined"
        assert record.conflicts is False
    assert "analysis" in file_records["image"].evidence
    assert "results" in file_records["file"].evidence


def test_the_design_summary_reports_detected_counts_and_marks_conflicts():
    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.data_files = [
        DataFile(filename="Con.loom", file_path="/x/Con.loom",
                 section="sequencing", data_type="Loom", group_id="g1"),
    ]
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="sequencing",
                          data_type="Loom", detected_count=1, declared_count=3,
                          method="obs_column:sample",
                          evidence="1 unique value", conflicts=True)
    ]

    summary = state.build_experimental_design_summary()

    assert "detected" in summary.lower()
    assert "Loom: 1" in summary
    assert "3" in summary and "declared" in summary.lower()


def test_the_design_summary_is_unchanged_without_detection():
    """Every session already on disk takes this path."""
    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]

    summary = state.build_experimental_design_summary()

    assert "Con (3 sample(s))" in summary
    assert "detected" not in summary.lower()


from unittest.mock import AsyncMock, MagicMock

from backend.constants import PROFILED_OBJECTS_KEY
from backend.models.schemas import AnalysisPlan, PlanStep

_MINIMAL_PLAN_JSON = (
    '{"overview":"ok","steps":[{"id":1,"title":"Verify","method":'
    '"scanpy.read_h5ad","depends_on":[]}]}'
)


def _plan_agent():
    """A BioinformaticsAgent wired to generate a plan with no real LLM call."""
    provider = MagicMock()
    provider.get_model.return_value = "test-model"
    provider.get_system_prompt.return_value = "system"
    provider.get_temperature.return_value = 0.3
    provider.get_vision_setting.return_value = False
    provider.get_llm_client.return_value = MagicMock()
    agent = bio_mod.BioinformaticsAgent(settings_provider=provider, sandbox=None)
    agent.session_id = "test-session"
    agent.chat_without_history = AsyncMock(return_value=_MINIMAL_PLAN_JSON)
    return agent


@pytest.mark.asyncio
async def test_the_plan_prompt_names_the_preloaded_objects_on_a_round_one_first_plan():
    """Contract 8: round 1's first plan is told the data is already loaded
    and to verify it rather than write a load step or an unsatisfiable
    sample-count acceptance condition.
    """
    agent = _plan_agent()
    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    state.sandbox_state = {PROFILED_OBJECTS_KEY: ["adata_con"]}

    await agent.generate_analysis_plan(
        state=state, pi_tasks=["QC"], overview="overview", phase="exploration",
    )

    prompt = agent.chat_without_history.call_args_list[0].args[0]
    assert "ALREADY LOADED" in prompt
    assert "adata_con" in prompt
    assert "first step should verify the loaded object" in prompt


@pytest.mark.asyncio
async def test_the_plan_prompt_omits_the_preload_note_on_a_revision():
    """A revision must not receive "the first step should verify the loaded
    object" beside revision_context's "Completed steps MUST NOT be changed"
    -- Contract 8 is about the initial plan only. current_round is left at 1
    so this isolates the existing_plan guard from the round guard.
    """
    agent = _plan_agent()
    state = AnalysisState(name="s", user_id="u")
    state.current_round = 1
    state.sandbox_state = {PROFILED_OBJECTS_KEY: ["adata_con"]}
    existing_plan = AnalysisPlan(
        phase="exploration", stage="full", overview="prior",
        steps=[PlanStep(id=1, title="Load", method="scanpy.read_h5ad",
                        depends_on=[])],
    )

    await agent.generate_analysis_plan(
        state=state, pi_tasks=["QC"], overview="overview", phase="exploration",
        existing_plan=existing_plan,
    )

    prompt = agent.chat_without_history.call_args_list[0].args[0]
    assert "ALREADY LOADED" not in prompt
    assert "first step should verify the loaded object" not in prompt


@pytest.mark.asyncio
async def test_the_plan_prompt_omits_the_preload_note_on_round_two():
    """The profiling pass runs once, in round 1, but its sandbox_state entry
    is never cleared -- current_round, not the entry's presence, must gate
    the note.
    """
    agent = _plan_agent()
    state = AnalysisState(name="s", user_id="u")
    state.current_round = 2
    state.sandbox_state = {PROFILED_OBJECTS_KEY: ["adata_con"]}

    await agent.generate_analysis_plan(
        state=state, pi_tasks=["QC"], overview="overview", phase="exploration",
    )

    prompt = agent.chat_without_history.call_args_list[0].args[0]
    assert "ALREADY LOADED" not in prompt


def test_the_undetermined_subline_survives_alongside_a_usable_record():
    """A group can hold one usable record and one undetermined one at once --
    Task 6 registers biochemistry uploads exactly that way when no per-sample
    structure exists, and this sub-line is the only route by which that
    reaches the PI. If it silently dropped, that reasoning would fail with
    nothing anywhere to notice.
    """
    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=1)]
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="sequencing",
                          data_type="Loom", detected_count=1, declared_count=1,
                          method="obs_column:sample", evidence="1 unique value",
                          conflicts=False),
        SampleCountRecord(group_id="g1", group_name="Con", section="biochemistry",
                          data_type="file", detected_count=0, declared_count=1,
                          method="undetermined", evidence="no per-sample structure",
                          conflicts=False),
    ]

    summary = state.build_experimental_design_summary()
    lines = summary.splitlines()

    detected_line = next(l for l in lines if l.startswith("  - Con"))
    undetermined_line = next(l for l in lines if l.strip().startswith("undetermined:"))

    # The per-type breakdown is only for what was actually determined.
    assert "Loom: 1" in detected_line
    assert "file" not in detected_line
    # The undetermined data type has to surface somewhere, or an upload that
    # could not be counted looks indistinguishable from one never profiled.
    assert "file" in undetermined_line


def test_all_undetermined_group_still_shows_declared_count_and_the_gap():
    """A group whose only records are all undetermined must not look
    unprofiled: the declared count is shown as the operative number, and the
    undetermined line names what could not be measured.
    """
    state = AnalysisState(name="s", user_id="u")
    state.treatment_groups = [TreatmentGroup(id="g1", name="Con", sample_count=3)]
    state.detected_sample_counts = [
        SampleCountRecord(group_id="g1", group_name="Con", section="biochemistry",
                          data_type="image", detected_count=0, declared_count=3,
                          method="undetermined", evidence="no per-sample structure",
                          conflicts=False),
        SampleCountRecord(group_id="g1", group_name="Con", section="biochemistry",
                          data_type="file", detected_count=0, declared_count=3,
                          method="undetermined", evidence="no per-sample structure",
                          conflicts=False),
    ]

    summary = state.build_experimental_design_summary()
    lines = summary.splitlines()

    detected_line = next(l for l in lines if l.startswith("  - Con"))
    undetermined_line = next(l for l in lines if l.strip().startswith("undetermined:"))

    assert "3 sample(s)" in detected_line
    assert "image" in undetermined_line and "file" in undetermined_line


def test_the_reassurance_is_withheld_when_the_same_modality_is_also_unresolved():
    """The note asserts a fact; the data must actually establish it.

    Membership was tested against a global set of data types with no link to the
    files being described, so two groups with detected bulk samples PLUS a
    genuinely unassigned bulk file printed "it is not an additional unidentified
    sample" directly above "detected counts not safely assignable to groups:
    bulk RNA-seq: 2" -- and the assertive half was the wrong one. 0.52.4 was
    written to stop a phantom sample being invented; this is the same error
    mirrored, hiding a real one.
    """
    state = _state_with(
        records=[
            _rec("g1", "Con", "sequencing", "bulk RNA-seq", 3, 3, "matrix_columns"),
            _rec("g2", "DKO", "sequencing", "bulk RNA-seq", 3, 3, "matrix_columns"),
            _rec(None, "", "sequencing", "bulk RNA-seq", 2, None, "matrix_columns"),
        ],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
        files=[DataFile(
            filename="extra_cohort.csv", file_path="/tmp/extra_cohort.csv",
            section="sequencing", data_type="bulk RNA-seq", group_id=None,
        )],
    )

    summary = state.build_experimental_design_summary()

    assert "detected counts not safely assignable to groups" in summary
    assert "already counted in the groups above" not in summary


def test_the_reassurance_survives_a_modality_spelled_two_ways():
    """`SampleCountRecord.data_type` is written from a model's own words while
    `DataFile.data_type` is the stored upload type, so "loom" and "scRNA-seq" are
    one modality spelled twice. Every other comparison of the two goes through
    `canonical_modality`; comparing raw strings made the note vanish silently
    for the very uploads it was written for."""
    state = _state_with(
        records=[
            _rec("g1", "Con", "sequencing", "scRNA-seq", 3, 3, "obs_column:sample"),
            _rec("g2", "DKO", "sequencing", "scRNA-seq", 3, 3, "obs_column:sample"),
        ],
        groups=[TreatmentGroup(id="g1", name="Con", sample_count=3),
                TreatmentGroup(id="g2", name="DKO", sample_count=3)],
        files=[DataFile(
            filename="all_samples.loom", file_path="/tmp/all_samples.loom",
            section="sequencing", data_type="loom", group_id=None,
        )],
    )

    summary = state.build_experimental_design_summary()

    assert "already counted in the groups above" in summary
