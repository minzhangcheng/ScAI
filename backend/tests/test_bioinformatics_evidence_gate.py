import traceback

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.agents.bioinformatics_agent import (
    BioinformaticsAgent,
    TOOLS,
    _build_exec_autosave_wrapper,
    _evidence_gate_record,
    execution_supports_evidence,
    successful_evidence_executions,
)
from backend.models.schemas import AgentType, AnalysisPhase, CodeExecution
from backend.models.schemas import AnalysisState, RoundResult
from backend.services.report.analysis_log import build_round_summaries


def test_failed_or_explicitly_failed_validation_cannot_support_findings():
    good = CodeExecution(code="print(metric)", output="metric=0.81", round_number=2)
    exception = CodeExecution(code="run()", error="TypeError: bad argument")
    failed_validation = CodeExecution(
        code="bulk_validation()",
        output="Matched in bulk: []\nCRITICAL: No signature genes found. Validation failed.",
    )

    assert execution_supports_evidence(good) is True
    assert execution_supports_evidence(exception) is False
    assert execution_supports_evidence(failed_validation) is False
    assert successful_evidence_executions([exception, good, failed_validation]) == [good]
    assert successful_evidence_executions(
        [exception, good, failed_validation], round_number=1
    ) == []
    assert successful_evidence_executions(
        [exception, good, failed_validation], round_number=2
    ) == [good]


def test_report_findings_requires_execution_evidence_ids():
    tool = next(
        item for item in TOOLS
        if item["function"]["name"] == "report_findings"
    )
    parameters = tool["function"]["parameters"]
    assert "evidence_execution_ids" in parameters["required"]
    assert parameters["properties"]["evidence_execution_ids"]["items"] == {
        "type": "string"
    }
    finding_item = parameters["properties"]["findings"]["items"]
    assert finding_item["required"] == [
        "text", "modality", "evidence_execution_ids",
    ]
    assert finding_item["properties"]["modality"]["enum"] == [
        "single_cell", "bulk_rna_seq", "cross_modal",
    ]


def test_execution_modality_is_derived_from_code_not_finding_text():
    from backend.services.sample_design import infer_execution_modalities

    assert infer_execution_modalities(
        "dds = DeseqDataSet(counts=bulk_counts, metadata=meta)"
    ) == ["bulk_rna_seq"]
    assert infer_execution_modalities(
        "sc.tl.rank_genes_groups(adata, 'condition')"
    ) == ["single_cell"]
    assert infer_execution_modalities(
        "bulk_degs = set(bulk_results.index); sc.pl.dotplot(adata, bulk_degs)"
    ) == ["single_cell", "bulk_rna_seq"]
    assert infer_execution_modalities(
        "# import pydeseq2\nprint('bulk_results')"
    ) == []
    assert infer_execution_modalities(
        "from pydeseq2.dds import DeseqDataSet as DDS\nDDS(counts=counts)"
    ) == ["bulk_rna_seq"]


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


def test_mixed_execution_can_support_a_modality_specific_finding():
    from backend.services.sample_design import finding_evidence_errors

    mixed = CodeExecution(
        id="mixed-exec", code="compare(bulk_counts, adata)",
        analysis_modalities=["single_cell", "bulk_rna_seq"],
    )
    records = [{
        "text": "single-cell localization is descriptive",
        "modality": "single_cell",
        "evidence_execution_ids": [mixed.id],
    }]

    assert finding_evidence_errors(
        records, {mixed.id}, {mixed.id: mixed},
    ) == []


def test_force_released_gate_has_structured_audit_identity_and_count():
    record = _evidence_gate_record(
        gate="analysis-unit",
        rule_id="replicate-aware-inference",
        detail="Cell-level inference is not identifiable.",
        severity="block",
        rejection_count=1,
        rejection_limit=1,
        forced_release=True,
    )

    assert record == {
        "gate": "analysis-unit",
        "rule_id": "replicate-aware-inference",
        "detail": "Cell-level inference is not identifiable.",
        "subject": "",
        "severity": "block",
        "rejection_count": 1,
        "rejection_limit": 1,
        "forced_release": True,
    }


def test_user_exception_traceback_points_to_user_source_not_cleanup(tmp_path):
    wrapper = _build_exec_autosave_wrapper(
        "raise TypeError('user failure')",
        str(tmp_path),
        "traceback-source",
    )

    import matplotlib.pyplot as pyplot

    real_close = pyplot.close
    while getattr(real_close, "_scai_noop", False):
        real_close = getattr(real_close, "_scai_prev", real_close)
    real_show = pyplot.show
    while getattr(real_show, "_scai_noop", False):
        real_show = getattr(real_show, "_scai_prev", real_show)
    pyplot.close = real_close
    pyplot.show = real_show
    try:
        with pytest.raises(TypeError) as caught:
            exec(wrapper, {}, {})
    finally:
        pyplot.close = real_close
        pyplot.show = real_show
        real_close("all")
    rendered = "".join(traceback.format_exception(caught.value))

    assert '<scai-user-code>' in rendered
    assert "_scai_cleanup_plt.close('all')" not in rendered


def test_report_round_summary_preserves_evidence_qualification():
    round_result = RoundResult(
        round_number=7,
        phase="innovation",
        bioinformatics_findings=["candidate signal"],
        round_status="partial",
        evidence_status="substantive",
        bioinformatics_evidence_verified=False,
        evidence_execution_ids=["exec-good"],
        evidence_notes=["Validation did not complete."],
        evidence_gate_degraded=True,
        evidence_gate_findings=[{
            "gate": "analysis-unit",
            "rule_id": "replicate-aware-inference",
            "severity": "block",
            "forced_release": True,
        }],
    )
    state = AnalysisState(user_id="owner", round_results=[round_result])

    summary = build_round_summaries(state)[0]
    assert summary["round_status"] == "partial"
    assert summary["bioinformatics_evidence_verified"] is False
    assert summary["evidence_gate_degraded"] is True
    assert summary["evidence_execution_ids"] == ["exec-good"]
    assert summary["evidence_notes"] == ["Validation did not complete."]
    assert summary["evidence_gate_findings"][0]["gate"] == "analysis-unit"


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
async def test_register_score_ceiling_force_releases_inside_real_tool_loop(
    monkeypatch, tmp_path,
):
    """Repeated invalid provenance/plan calls terminate with a degraded audit.

    This drives the real autonomous dispatcher.  A source-only assertion cannot
    prove that the counter survives turns, that the fifth call is registered,
    or that the result and durable session both carry the forced-release record.
    """
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
    assert result["status"] == "partial"
    assert result["evidence_gate_degraded"] is True
    assert result["evidence_verified"] is False
    assert state.evidence_gate_degraded is True
    assert len(state.score_definitions) == 1
    assert "gate_escape" in state.score_definitions[0].plan_drift
    gates = {item["gate"] for item in result["evidence_gate_findings"]}
    assert gates == {"register-score-provenance", "register-score-plan"}
    for item in result["evidence_gate_findings"]:
        assert item["forced_release"] is True
        assert item["rejection_count"] == 4
        assert item["rejection_limit"] == 4
        assert item["round_number"] == 2
