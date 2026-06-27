from backend.utils.direction_evidence import build_fallback_direction_evidence

RAW = [{"direction_id": "dir_001", "direction": "A"}, {"direction": "B"}]
KEYS = {"direction_id", "direction", "status", "search_tier",
        "literature_novelty", "confidence", "summary", "limitations",
        "novelty_status", "support_level", "contradiction_level",
        "prior_art_level", "fallback_tier", "knowledge_gap_summary"}


def test_fallback_has_full_field_set():
    out = build_fallback_direction_evidence(RAW, status="skipped", reason="r")
    assert len(out) == 2
    for row in out:
        assert set(row) == KEYS
        assert row["status"] == "skipped"
        assert row["literature_novelty"] == "unknown"
        assert row["confidence"] == "failed"
        assert row["search_tier"] == "failed"
        assert row["limitations"] == ["r"]


def test_missing_direction_id_is_synthesized():
    out = build_fallback_direction_evidence(RAW, status="failed", reason="boom")
    assert out[0]["direction_id"] == "dir_001"
    assert out[1]["direction_id"] == "dir_002"


def test_custom_limitations_override():
    out = build_fallback_direction_evidence(
        RAW, status="failed", reason="boom", limitations=["conservative"])
    assert out[0]["limitations"] == ["conservative"]


def test_fallback_direction_evidence_includes_enhanced_failure_metadata():
    out = build_fallback_direction_evidence(RAW, status="failed", reason="literature timeout")

    row = out[0]
    assert row["novelty_status"] == "unknown_due_to_failure"
    assert row["support_level"] == "unknown"
    assert row["contradiction_level"] == "unknown"
    assert row["prior_art_level"] == "unknown"
    assert row["fallback_tier"] == "failed"
    assert row["knowledge_gap_summary"] == "literature timeout"
