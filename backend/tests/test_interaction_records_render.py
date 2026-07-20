import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    AnalysisState, SelectionEntry, DirectionSelectionRecord,
    AnnotationClusterEntry, AnnotationRecord,
)
from backend.services.report.interaction_records import (
    build_direction_selection_history,
    render_direction_record_html,
    render_direction_selection_history_html,
    render_annotation_record_html,
)


def test_direction_select_marks_selected_and_unselected():
    rec = DirectionSelectionRecord(
        round_number=2, phase="exploration", event_type="select",
        candidates=[
            {"direction": "Ferroptosis", "rationale": "GPX4 down", "recommendation": "prioritize"},
            {"direction": "Angiogenesis", "rationale": "low", "recommendation": "avoid"},
        ],
        selections=[SelectionEntry(direction="Ferroptosis", notes="focus GPX4",
                                   is_custom=False, candidate_index=0)],
    )
    html = render_direction_record_html(rec)
    assert "Ferroptosis" in html
    assert "Angiogenesis" in html
    assert "&#10003;" in html and "&#10007;" in html
    assert "focus GPX4" in html


def test_direction_custom_tag_and_supplementary():
    rec = DirectionSelectionRecord(
        round_number=2, phase="exploration", event_type="select",
        candidates=[{"direction": "A"}],
        selections=[SelectionEntry(direction="My idea", is_custom=True, candidate_index=None)],
        supplementary_requests=["search SPP1 literature"],
    )
    html = render_direction_record_html(rec)
    assert "My idea" in html
    assert "search SPP1 literature" in html


def test_direction_renders_literature_evidence():
    rec = DirectionSelectionRecord(
        round_number=2, phase="exploration", event_type="select",
        candidates=[{"direction": "Ferroptosis", "evidence_ref": "d1"}],
        literature_evidence=[{
            "direction_id": "d1",
            "direct_evidence": [{"pmid": "12345", "title": "GPX4 in fibrosis",
                                 "why_it_matters": "supports"}],
        }],
        selections=[SelectionEntry(direction="Ferroptosis", candidate_index=0)],
    )
    html = render_direction_record_html(rec)
    assert "GPX4 in fibrosis" in html
    assert "12345" in html
    assert "Literature evidence" in html


def test_direction_pi_decide_banner():
    rec = DirectionSelectionRecord(
        round_number=2, phase="exploration", event_type="pi_decide",
        let_pi_decide=True,
        candidates=[{"direction": "A"}, {"direction": "B"}],
        pi_selected=[{"direction": "A", "notes": ""}],
    )
    html = render_direction_record_html(rec)
    assert "PI" in html
    a_pos = html.index("font-weight:600'>A")
    b_pos = html.index("font-weight:600'>B")
    assert html.rfind("&#10003;", 0, a_pos) > html.rfind("&#10007;", 0, a_pos)
    assert html.rfind("&#10007;", 0, b_pos) > html.rfind("&#10003;", 0, b_pos)


def test_direction_renders_supporting_data():
    rec = DirectionSelectionRecord(
        round_number=2,
        phase="exploration",
        event_type="select",
        candidates=[{
            "direction": "Ferroptosis",
            "supporting_data": [
                {"source": "DEG", "evidence": "Gpx4 down", "strength": "high"},
            ],
        }],
        selections=[SelectionEntry(direction="Ferroptosis", candidate_index=0)],
    )
    html = render_direction_record_html(rec)
    assert "Supporting data" in html
    assert "DEG" in html
    assert "Gpx4 down" in html
    assert "high" in html


def test_direction_renders_scalar_list_fields_as_single_items():
    rec = DirectionSelectionRecord(
        round_number=5,
        phase="integration",
        integration_cycle=1,
        event_type="select",
        candidates=[{
            "direction": "Cross-direction stress program",
            "supporting_data": "Round 2 and 3 share an NRF2 signal.",
            "artifact_risks": "Single-library composition uncertainty.",
            "alternative_hypotheses": "A technical stress response.",
            "contradictory_or_missing_evidence": "No independent cohort.",
            "validation_plan": {
                "next_analysis": "Residualize the shared score.",
                "wet_lab_validation": "Perturb the regulator.",
                "decision_criteria": "The shared score should decrease.",
            },
        }],
        selections=[SelectionEntry(
            direction="Cross-direction stress program", candidate_index=0
        )],
    )

    html = render_direction_record_html(rec)

    assert "<li>Round 2 and 3 share an NRF2 signal.</li>" in html
    assert "<li>Single-library composition uncertainty.</li>" in html
    assert "<li>Residualize the shared score.</li>" in html
    assert "<li>A technical stress response.</li>" in html
    assert "<li>No independent cohort.</li>" in html
    assert "<li>Perturb the regulator.</li>" in html
    assert "<li>The shared score should decrease.</li>" in html
    assert "<li>R</li>" not in html
    assert "<li>S</li>" not in html


def test_direction_finalize_note():
    rec = DirectionSelectionRecord(
        round_number=9, phase="integration", integration_cycle=2, event_type="finalize",
    )
    html = render_direction_record_html(rec)
    assert "without adding" in html.lower() or "ended" in html.lower()


def test_direction_selection_history_uses_all_completed_records():
    state = AnalysisState(
        id="history", name="history", initial_instruction="test", user_id="user",
        user_chosen_directions=[{"direction": "Newest only", "notes": "legacy view"}],
        direction_selection_records=[
            DirectionSelectionRecord(
                round_number=1,
                phase="exploration",
                event_type="pi_decide",
                pi_selected=[
                    {"direction": "Direction A", "notes": "PI rationale A"},
                    {"direction": "Direction B", "notes": "PI rationale B"},
                ],
            ),
            DirectionSelectionRecord(
                round_number=4,
                integration_cycle=1,
                phase="integration",
                event_type="select",
                selections=[SelectionEntry(
                    direction="Direction C", notes="User note C", candidate_index=0,
                )],
            ),
            DirectionSelectionRecord(
                round_number=6,
                integration_cycle=2,
                phase="integration",
                event_type="finalize",
            ),
        ],
    )

    rows = build_direction_selection_history(state)

    assert [row["direction"] for row in rows] == [
        "Direction A", "Direction B", "Direction C",
    ]
    assert rows[0]["selection_event"] == "Round 1 · exploration"
    assert rows[0]["decision_mode"] == "PI-selected"
    assert rows[2]["selection_event"] == "Integration Cycle 1 · integration"
    assert rows[2]["decision_mode"] == "User-selected"
    assert "Newest only" not in {row["direction"] for row in rows}


def test_direction_selection_history_legacy_fallback_and_table_escaping():
    state = AnalysisState(
        id="legacy", name="legacy", initial_instruction="test", user_id="user",
        user_chosen_directions=[{
            "direction": "Legacy <direction>", "notes": "A & B",
        }],
    )

    rows = build_direction_selection_history(state)
    html = render_direction_selection_history_html(rows)

    assert rows[0]["selection_event"] == "Current selection"
    assert "Selection Event" in html
    assert "Decision Mode" in html
    assert "Rationale / Notes" in html
    assert "Legacy &lt;direction&gt;" in html
    assert "A &amp; B" in html


def test_annotation_edit_shows_original_struck_and_tag():
    rec = AnnotationRecord(
        organism="mouse", tissue="heart",
        proportion_dimension="sample", proportion_group_labels=["A", "B"],
        clusters=[
            AnnotationClusterEntry(cluster_id="0", original_cell_type="Fibroblast",
                                   final_cell_type="Myofibroblast", user_override=True,
                                   confidence="medium", key_markers=["Col1a1"],
                                   group_proportions={"A": 10.0, "B": 5.0}),
        ],
    )
    html = render_annotation_record_html(rec)
    assert "Myofibroblast" in html
    assert "<s" in html or "line-through" in html
    assert "Fibroblast" in html
    assert "Status" not in html and "状态" not in html


def test_annotation_grouped_proportions():
    rec = AnnotationRecord(
        proportion_group_labels=["A", "B"],
        clusters=[AnnotationClusterEntry(cluster_id="0", original_cell_type="X",
                                         final_cell_type="X",
                                         group_proportions={"A": 12.5, "B": 3.0})],
    )
    html = render_annotation_record_html(rec)
    assert "12.5" in html and "3.0" in html
