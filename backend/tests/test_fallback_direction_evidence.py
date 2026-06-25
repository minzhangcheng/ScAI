from backend.utils.direction_evidence import build_fallback_direction_evidence

RAW = [{"direction_id": "dir_001", "direction": "A"}, {"direction": "B"}]
KEYS = {"direction_id", "direction", "status", "search_tier",
        "literature_novelty", "confidence", "summary", "limitations"}


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
