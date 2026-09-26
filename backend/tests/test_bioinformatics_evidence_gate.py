
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.agents.bioinformatics_agent import (
    BioinformaticsAgent,
    TOOLS,
    execution_supports_evidence,
    successful_evidence_executions,
)
from backend.models.schemas import AgentType, AnalysisPhase, CodeExecution
from backend.models.schemas import AnalysisState


def test_failed_or_explicitly_failed_validation_cannot_support_findings():
    good = CodeExecution(code="print(metric)", output="metric=0.81", round_number=2)
    empty = CodeExecution(code="", output="", round_number=2)
    exception = CodeExecution(code="run()", error="TypeError: bad argument")
    failed_validation = CodeExecution(
        code="bulk_validation()",
        output="Matched in bulk: []\nCRITICAL: No signature genes found. Validation failed.",
    )

    assert execution_supports_evidence(good) is True
    assert execution_supports_evidence(empty) is False
    assert execution_supports_evidence(exception) is False
    assert execution_supports_evidence(failed_validation) is False
    assert successful_evidence_executions(
        [exception, empty, good, failed_validation]
    ) == [good]
    assert successful_evidence_executions(
        [exception, good, failed_validation], round_number=1
    ) == []
    assert successful_evidence_executions(
        [exception, good, failed_validation], round_number=2
    ) == [good]


def test_report_findings_accepts_optional_execution_evidence_candidates():
    tool = next(
        item for item in TOOLS
        if item["function"]["name"] == "report_findings"
    )
    parameters = tool["function"]["parameters"]
    assert "evidence_execution_ids" not in parameters["required"]
    assert parameters["properties"]["evidence_execution_ids"]["items"] == {
        "type": "string"
    }
    finding_item = parameters["properties"]["findings"]["items"]
    assert finding_item["required"] == ["text"]
    assert finding_item["properties"]["modality"]["enum"] == [
        "single_cell", "bulk_rna_seq", "cross_modal",
    ]


def test_finding_cannot_borrow_a_different_executions_modality():
    from backend.services.sample_design import finding_evidence_errors

    bulk = CodeExecution(
        id="bulk-exec", code="DeseqDataSet(counts=bulk_counts)",
        analysis_modalities=["bulk_rna_seq"],
    )
    records = [{
        "text": "single-cell condition effect is significant",
        "modality": "single_cell",
        "evidence_execution_ids": [bulk.id],
    }]

    errors = finding_evidence_errors(records, {bulk.id}, {bulk.id: bulk})
    assert errors == [
        "finding 1 declares 'single_cell', but its executed code does not include "
        "dedicated matching evidence; observed modalities are ['bulk_rna_seq']"
    ]


@pytest.mark.asyncio
async def test_process_preserves_execution_evidence_metadata():
    agent = object.__new__(BioinformaticsAgent)
    agent.refresh_runtime_config_if_needed = lambda: None
    agent.run_autonomous_analysis = AsyncMock(return_value={
        "status": "completed",
        "findings": ["supported finding"],
        "figures": [],
        "analysis_log": [],
        "success": True,
        "steps_executed": 1,
        "mechanism_programs": [],
        "evidence_verified": True,
        "evidence_gate_degraded": True,
        "evidence_execution_ids": ["exec-good"],
        "evidence_gate_findings": [{"gate": "panel-redundancy", "severity": "advise"}],
    })
    state = AnalysisState(
        user_id="owner",
        current_round=1,
        data_files=[{
            "id": "file-1",
            "filename": "matrix.h5ad",
            "file_path": "/tmp/matrix.h5ad",
            "data_type": "scRNA-seq",
            "section": "sequencing",
        }],
    )

    result = await agent.process(state, "Analyze")

    assert result["evidence_verified"] is True
    assert result["evidence_gate_degraded"] is True
    assert result["evidence_execution_ids"] == ["exec-good"]
    assert result["evidence_gate_findings"] == [{
        "gate": "panel-redundancy",
        "severity": "advise",
    }]


@pytest.mark.asyncio
async def test_register_score_limitations_remain_advisory_inside_real_tool_loop(
    monkeypatch, tmp_path,
):
    """Repeated invalid metadata is recorded without degrading acceptance."""
    from backend.tests.split_facade import setattr_facade
    from backend.utils.llm_client import LLMToolResponse, ToolCall

    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.session_id = "score-gate-ceiling"
    agent.code_history = []
    agent.analysis_context = {}
    agent.last_result = None
    agent._recent_sup_feedback = set()
    agent._pause_event = None
    agent.literature_callback = None
    agent.pi_callback = None
    agent.annotate_clusters_callback = None
    agent.message_callback = None
    agent.figure_review_callback = None
    agent.supervisor_callback = None
    agent.species_validation_callback = None
    agent.marker_db_intervention_callback = None
    agent.technique_agent = None
    agent._provider = MagicMock()
    agent._provider.get_config_revision.return_value = 0
    agent._last_config_revision = 0
    agent.agent_type = AgentType.BIOINFORMATICS

    monkeypatch.setattr(
        BioinformaticsAgent, "model", property(lambda _self: "test-model"),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "system_prompt",
        property(lambda _self: "Test bioinformatics agent."),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "config",
        property(lambda _self: MagicMock(temperature=0.1)),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_static_system_prompt",
        property(lambda _self: "Static bioinformatics prompt."),
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_build_dynamic_context",
        lambda _self, _state: "Dynamic test context.",
    )
    monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda _self: None)
    monkeypatch.setattr(BioinformaticsAgent, "_notify", AsyncMock())

    invalid_score = {
        "score_name": "unattributed_score",
        "method": "mean expression",
        "description": "Test score",
        "parameters": {},
        "gene_list": ["GENE1", "GENE2"],
        "gene_list_source": "",
        "plan_id": "missing-plan",
    }
    responses = []
    for index in range(4):
        responses.append(LLMToolResponse(
            content="",
            tool_calls=[ToolCall(
                id=f"score-{index}",
                name="register_score",
                arguments=dict(invalid_score),
            )],
            model="test-model",
            usage={},
            raw_response=None,
            raw_assistant_message=None,
        ))
    evidence = CodeExecution(
        id="exec-score-evidence",
        code="print('descriptive result')",
        output="descriptive result",
        round_number=2,
    )
    responses.append(LLMToolResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="score-ceiling",
                name="register_score",
                arguments=dict(invalid_score),
            ),
            ToolCall(
                id="report-after-ceiling",
                name="report_findings",
                arguments={
                    "findings": ["A descriptive signal was retained."],
                    "summary": "Completed with a provenance qualification.",
                    "suggestions": [],
                    "evidence_execution_ids": [evidence.id],
                },
            ),
        ],
        model="test-model",
        usage={},
        raw_response=None,
        raw_assistant_message=None,
    ))
    fake_client = MagicMock(base_url="https://fake.example/v1")
    fake_client.chat_with_tools = AsyncMock(side_effect=responses)
    monkeypatch.setattr(
        BioinformaticsAgent, "llm_client", property(lambda _self: fake_client),
    )

    agent.sandbox = MagicMock()
    agent.sandbox.state_dir = tmp_path / "states"
    agent.sandbox.state_dir.mkdir()
    agent.sandbox.working_dir = tmp_path
    agent.sandbox.km = MagicMock()
    agent.sandbox.start = AsyncMock()
    agent.sandbox.save_session = AsyncMock()
    agent.sandbox.get_code_history.return_value = []
    agent.sandbox.sync_runtime_persistence_state.return_value = {}

    class _NullCheckpoint:
        def __init__(self, *_args, **_kwargs):
            pass

        async def load(self):
            return None

        async def save(self, _data):
            return None

        def clear(self):
            return None

        def should_checkpoint(self, _step):
            return False

    setattr_facade(
        monkeypatch,
        "backend.agents.bioinformatics_agent",
        "CheckpointManager",
        _NullCheckpoint,
    )
    setattr_facade(
        monkeypatch,
        "backend.agents.bioinformatics_agent",
        "successful_evidence_executions",
        lambda *_args, **_kwargs: [evidence],
    )

    state = AnalysisState(
        id=agent.session_id,
        user_id="owner",
        current_round=2,
        analysis_phase=AnalysisPhase.INNOVATION,
    )
    result = await agent.run_autonomous_analysis(
        state,
        initial_instruction="Exercise score registration.",
        max_steps=5,
        resume_from_checkpoint=False,
    )

    assert fake_client.chat_with_tools.await_count == 5
    assert result["status"] == "completed"
    assert result["evidence_gate_degraded"] is False
    assert result["evidence_verified"] is False
    assert state.evidence_gate_degraded is False
    assert len(state.score_definitions) == 1
    assert "pi_review_observations" in state.score_definitions[0].plan_drift
    gates = {item["gate"] for item in result["evidence_gate_findings"]}
    assert gates == {"register-score-provenance", "register-score-plan"}
    for item in result["evidence_gate_findings"]:
        assert item["severity"] == "advise"
        assert item["forced_release"] is False
        assert item["rejection_count"] == 0
        assert item["rejection_limit"] == 0
        assert item["round_number"] == 2
