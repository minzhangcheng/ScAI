from backend.services.report.variants import _render_direction_candidates_rows


def test_main_table_uses_scored_fields():
    rows = _render_direction_candidates_rows([
        {
            "direction": "IFN network",
            "rationale": "strong signal",
            "overall_priority": "high",
            "literature_novelty": "medium",
            "data_reliability": "high",
        }
    ])
    assert "IFN network" in rows
    # Priority cell should render High (from overall_priority), not N/A.
    assert "badge-high" in rows
    assert "badge-medium" in rows
    assert "N/A" not in rows


def test_main_table_falls_back_to_legacy_fields():
    rows = _render_direction_candidates_rows([
        {
            "direction": "Legacy direction",
            "rationale": "r",
            "expected_innovation": "high",
            "expected_novelty": "low",
            "data_reliability": "medium",
        }
    ])
    assert "Legacy direction" in rows
    assert "badge-high" in rows   # from expected_innovation fallback
    assert "badge-low" in rows    # from expected_novelty fallback
    assert "N/A" not in rows
