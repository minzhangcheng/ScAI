from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.models.schemas import AnalysisPhase, AnalysisState, RoundResult
from backend.workflow.direction_coverage import clone_round_results_for_branch
from backend.workflow.direction_coverage import normalize_legacy_round_result


def test_current_evidence_schema_is_never_reinterpreted_as_legacy():
    current = RoundResult(
        round_number=4,
        phase="integration",
        integration_report_summary="Retained text does not override a live verdict.",
        evidence_schema_version=1,
        round_status="failed",
        evidence_status="failed",
        direction_coverage_eligible=False,
    )

    normalized = normalize_legacy_round_result(current)

    assert normalized.round_status == "failed"
    assert normalized.evidence_status == "failed"
    assert normalized.direction_coverage_eligible is False


def test_nested_branch_clone_preserves_the_original_evidence_lineage():
    inherited = RoundResult(
        round_number=3,
        phase="innovation",
        direction="Direction A",
        lineage_id="branch-a",
        branch_id="branch-a",
        inherited_from_lineage_id="main",
        restored_from_snapshot_id="source/snapshot-a",
        bioinformatics_finding_records=[{
            "text": "Inherited finding",
            "lineage_id": "main",
            "evidence_execution_ids": ["execution-main"],
        }],
    )

    cloned = clone_round_results_for_branch(
        [inherited],
        branch_id="branch-b",
        branch_label="branch-b",
        source_snapshot_id="source/snapshot-b",
        source_lineage_id="branch-a",
    )

    branch_b = next(result for result in cloned if result.lineage_id == "branch-b")
    assert branch_b.inherited_from_lineage_id == "main"
    assert branch_b.bioinformatics_finding_records[0]["lineage_id"] == "main"


def test_branch_clone_does_not_inherit_unrelated_branch_results():
    main = RoundResult(
        round_number=2,
        phase="innovation",
        direction="Direction A",
        lineage_id="main",
        pi_summary="main completed",
        bioinformatics_findings=["main finding"],
    )
    other_branch = RoundResult(
        round_number=3,
        phase="innovation",
        direction="Direction B",
        lineage_id="branch-x",
        branch_id="branch-x",
        pi_summary="other branch completed",
        bioinformatics_findings=["other branch finding"],
    )

    cloned = clone_round_results_for_branch(
        [main, other_branch],
        branch_id="branch-y",
        branch_label="branch-y",
        source_snapshot_id="snap-y",
        source_lineage_id="main",
    )

    branch_y_directions = {
        r.direction for r in cloned if r.lineage_id == "branch-y"
    }
    assert branch_y_directions == {"Direction A"}
    assert any(r.lineage_id == "branch-x" for r in cloned)


@pytest.mark.asyncio
async def test_synthesis_persist_preserves_restore_origin_on_same_identity_retry():
    """The production synthesis writer keeps branch provenance immutable."""
    from backend.workflow import synthesis
    from backend.workflow.synthesis_parts import phases

    origin = "source/r001"
    state = AnalysisState(
        id="restored-child",
        user_id="owner",
        current_round=3,
        analysis_phase=AnalysisPhase.INNOVATION,
        restore_branch_id="restore-source-r001",
        restore_branch_label="Restored direction",
        last_restored_snapshot_id=origin,
        restore_provenance_snapshot_id=origin,
    )
    graph_state = {
        "analysis_state": state,
        "pathology_result": {"status": "completed", "findings": ["p"]},
        "bioinformatics_result": {"status": "completed", "findings": ["b"]},
        "biochemistry_result": {"status": "completed", "findings": ["c"]},
        "literature_result": {"status": "completed", "findings": ["l"]},
    }
    writer = SimpleNamespace(
        session_id=state.id,
        state_callback=None,
        _build_and_save_detailed_segment=AsyncMock(),
    )
    phases.bind_facade(synthesis)

    async def persist(round_number, direction, summary):
        result = RoundResult(
            round_number=round_number,
            phase="innovation",
            direction=direction,
            pi_summary=summary,
            bioinformatics_findings=[summary],
        )
        await phases.persist_round_result(writer, {
            "state": graph_state,
            "analysis_state": state,
            "round_result": result,
            "bioinformatics_result": graph_state["bioinformatics_result"],
        })
        return result

    first = await persist(3, "Direction A", "first")
    retry = await persist(3, "Direction A", "retry")
    later = await persist(4, "Direction B", "later")

    assert first.restored_from_snapshot_id == origin
    assert retry.restored_from_snapshot_id == origin
    assert later.restored_from_snapshot_id is None
    assert state.restore_provenance_snapshot_id is None
    assert len(state.round_results) == 2
    assert state.round_results[0].pi_summary == "retry"
    assert state.round_results[0].lineage_id == state.restore_branch_id
    assert state.round_results[0].branch_id == state.restore_branch_id
    assert sum(
        result.restored_from_snapshot_id == origin
        for result in state.round_results
    ) == 1
