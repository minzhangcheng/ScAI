from backend.models.schemas import RoundResult
from backend.workflow.direction_coverage import clone_round_results_for_branch
from backend.workflow.direction_coverage import normalize_legacy_round_result
from backend.workflow.synthesis import _upsert_round_result


def test_legacy_round_result_backfill_derives_identity_and_substantive_evidence():
    result = RoundResult(
        round_number=3,
        phase="innovation",
        direction="  Direction A  ",
        pi_summary="completed",
        bioinformatics_findings=["finding"],
    )

    normalized = normalize_legacy_round_result(result)

    assert normalized.lineage_id == "main"
    assert normalized.direction_key == "direction a"
    assert normalized.result_identity_key == "main:innovation:3:direction a"
    assert normalized.evidence_status == "substantive"


def test_branch_clone_inherits_main_ancestor_results_with_branch_identity():
    main = RoundResult(
        round_number=3,
        phase="innovation",
        direction="Direction A",
        pi_summary="completed",
        bioinformatics_findings=["finding"],
    )

    cloned = clone_round_results_for_branch(
        [main],
        branch_id="restore-snap-1",
        branch_label="branch-snap-1",
        source_snapshot_id="snap-1",
    )

    assert [r.lineage_id for r in cloned] == ["main", "restore-snap-1"]
    assert cloned[1].result_identity_key == "restore-snap-1:innovation:3:direction a"
    assert cloned[1].source_snapshot_id == "snap-1"


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


def test_upsert_replaces_same_lineage_identity():
    first = RoundResult(
        round_number=3,
        phase="innovation",
        direction="Direction A",
        pi_summary="old",
        bioinformatics_findings=["old finding"],
    )
    second = RoundResult(
        round_number=3,
        phase="innovation",
        direction="Direction A",
        pi_summary="new",
        bioinformatics_findings=["new finding"],
    )
    results = []

    _upsert_round_result(results, first, lineage_id="main", branch_id=None, branch_label=None, restored_from_snapshot_id=None)
    _upsert_round_result(results, second, lineage_id="main", branch_id=None, branch_label=None, restored_from_snapshot_id=None)

    assert len(results) == 1
    assert results[0].pi_summary == "new"
    assert results[0].result_identity_key == "main:innovation:3:direction a"


def test_upsert_keeps_branch_and_main_separate():
    main = RoundResult(round_number=3, phase="innovation", direction="Direction A", pi_summary="main", bioinformatics_findings=["main"])
    branch = RoundResult(round_number=3, phase="innovation", direction="Direction A", pi_summary="branch", bioinformatics_findings=["branch"])
    results = []

    _upsert_round_result(results, main, lineage_id="main", branch_id=None, branch_label=None, restored_from_snapshot_id=None)
    _upsert_round_result(results, branch, lineage_id="branch-snap-1", branch_id="branch-snap-1", branch_label="branch-A", restored_from_snapshot_id="snap-1")

    assert [r.lineage_id for r in results] == ["main", "branch-snap-1"]
