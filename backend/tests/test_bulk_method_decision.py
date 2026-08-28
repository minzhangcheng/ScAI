"""Method-neutral bulk input profiling and method-decision contracts."""

from types import SimpleNamespace

import pytest

from backend.agents import bioinformatics_agent as bio_facade
from backend.agents.bioinformatics import executor_registration
from backend.agents.bioinformatics.tool_schemas import KNOWN_TOOL_NAMES
from backend.models.schemas import (
    AnalysisPlan,
    AnalysisState,
    CodeExecution,
    DataFile,
    PlanStep,
)
from backend.services.bulk_input_profile import ensure_bulk_input_profiles


def test_host_bulk_profile_reports_facts_without_selecting_a_method(tmp_path):
    path = tmp_path / "bulk.csv"
    path.write_text(
        "feature,control_1,treated_1,note\nA,1,2,x\nB,3.5,-1,y\n",
        encoding="utf-8",
    )
    state = AnalysisState(
        id="bulk-profile",
        user_id="user",
        data_files=[DataFile(
            id="bulk-file",
            filename="bulk.csv",
            file_path=str(path),
            data_type="bulk RNA-seq",
            section="sequencing",
        )],
    )

    profile = ensure_bulk_input_profiles(state)[0]

    assert profile.shape == [2, 4]
    assert profile.annotation_columns == ["feature", "note"]
    assert profile.candidate_sample_columns == ["control_1", "treated_1"]
    assert profile.integer_like_fraction == 0.75
    assert profile.negative_value_count == 1
    assert profile.declared_scale == "unknown"
    assert profile.profile_status == "complete"
    assert profile.complete is True
    assert not hasattr(profile, "method")


@pytest.mark.asyncio
async def test_method_decision_must_cite_current_bulk_execution(tmp_path):
    executor_registration.bind_facade(bio_facade)
    path = tmp_path / "bulk.csv"
    path.write_text(
        "feature,control_1,treated_1\nA,1,2\nB,3,4\n",
        encoding="utf-8",
    )
    state = AnalysisState(
        id="bulk-decision",
        user_id="user",
        current_round=1,
        current_plan=AnalysisPlan(
            id="bulk-plan",
            phase="exploration",
            status="approved",
            steps=[PlanStep(id=1, title="Bulk summary", method="Agent-selected")],
        ),
    )
    state.bulk_input_profiles = ensure_bulk_input_profiles(AnalysisState(
        id="profile-source",
        user_id="user",
        data_files=[DataFile(
            id="bulk-file",
            filename="bulk.csv",
            file_path=str(path),
            data_type="bulk RNA-seq",
            section="sequencing",
        )],
    ))
    execution = CodeExecution(
        id="bulk-execution",
        code="analyze_bulk_table()",
        round_number=1,
        analysis_modalities=["bulk_rna_seq"],
        evidence_schema_version=2,
        plan_id=state.current_plan.id,
        execution_status="succeeded",
        input_profile_ids=[state.bulk_input_profiles[0].id],
    )
    context = SimpleNamespace(
        state=state,
        agent=SimpleNamespace(code_history=[execution]),
        messages=[],
        analysis_log=[],
        step_count=1,
        agent_reasoning="choose method",
    )
    call = SimpleNamespace(
        id="call-1",
        arguments={
            "input_profile_id": state.bulk_input_profiles[0].id,
            "claim_level": "descriptive",
            "method": "agent-selected robust summary",
            "rationale": "The observed input facts and unresolved mapping support a descriptive summary.",
            "execution_id": execution.id,
        },
    )

    await executor_registration.handle_register_method_decision(context, call)

    assert len(state.method_decisions) == 1
    assert state.method_decisions[0].execution_id == execution.id
    assert state.method_decisions[0].method == "agent-selected robust summary"
    assert (
        state.method_decisions[0].decision_source
        == "approved_plan_and_bioinformatics"
    )

    stale_execution = execution.model_copy(update={
        "id": "bulk-execution-old-plan",
        "plan_id": "old-plan",
    })
    context.agent.code_history = [stale_execution]
    stale_call = SimpleNamespace(
        id="call-2",
        arguments={**call.arguments, "execution_id": stale_execution.id},
    )

    await executor_registration.handle_register_method_decision(
        context, stale_call
    )

    assert len(state.method_decisions) == 1
    assert "another approved plan" in context.messages[-1]["content"]


def test_method_decision_tool_is_part_of_main_executor_contract():
    assert "register_method_decision" in KNOWN_TOOL_NAMES
