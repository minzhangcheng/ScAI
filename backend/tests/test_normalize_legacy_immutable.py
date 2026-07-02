from backend.workflow.direction_coverage import normalize_legacy_round_result
from backend.models.schemas import RoundResult


def test_normalize_does_not_mutate_input():
    r = RoundResult(round_number=1, phase="exploration", direction="", lineage_id="", findings=[])
    original_lineage = r.lineage_id
    out = normalize_legacy_round_result(r)
    assert r.lineage_id == original_lineage
    assert out.lineage_id == "main"
