import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    AnalysisState, AnnotationConfirmationState, ClusterAnnotation,
    DirectionSelectionRecord,
)
from backend.services.interaction_record_capture import (
    get_or_open_direction_record, find_open_pi_decide_record,
    resolve_integration_gate, find_open_integration_record,
    map_selections, build_annotation_record,
)


def _state():
    return AnalysisState(user_id="test", id="s", name="n", initial_instruction="i")


def test_get_or_open_creates_then_reuses():
    st = _state()
    r1 = get_or_open_direction_record(st, round_number=2, phase="exploration")
    r2 = get_or_open_direction_record(st, round_number=2, phase="exploration")
    assert r1.record_id == r2.record_id
    assert len(st.direction_selection_records) == 1


def test_get_or_open_separate_by_integration_cycle():
    st = _state()
    a = get_or_open_direction_record(st, round_number=5, phase="integration", integration_cycle=1)
    b = get_or_open_direction_record(st, round_number=5, phase="integration", integration_cycle=2)
    assert a.record_id != b.record_id
    assert len(st.direction_selection_records) == 2


def test_closed_record_not_reused():
    st = _state()
    r1 = get_or_open_direction_record(st, round_number=2, phase="exploration")
    r1.event_type = "select"
    r2 = get_or_open_direction_record(st, round_number=2, phase="exploration")
    assert r1.record_id != r2.record_id


def test_find_open_pi_decide_record_ignores_round_number():
    st = _state()
    r = get_or_open_direction_record(st, round_number=1, phase="exploration")
    r.let_pi_decide = True
    assert find_open_pi_decide_record(st).record_id == r.record_id
    r.event_type = "pi_decide"
    assert find_open_pi_decide_record(st) is None


def test_map_selections_matches_candidate_index():
    candidates = [{"direction": "A"}, {"direction": "B"}]
    dirs = [
        {"direction": "B", "notes": "n"},
        {"direction": "Custom", "notes": "", "is_custom": True},
    ]
    entries = map_selections(dirs, candidates)
    assert entries[0].candidate_index == 1
    assert entries[0].is_custom is False
    assert entries[1].candidate_index is None
    assert entries[1].is_custom is True


def test_map_selections_infers_custom_when_no_match():
    entries = map_selections([{"direction": "X"}], [{"direction": "A"}])
    assert entries[0].is_custom is True
    assert entries[0].candidate_index is None


def test_build_annotation_record_original_vs_final():
    st = _state()
    st.annotation_state = AnnotationConfirmationState(
        organism="mouse", tissue="heart", proportion_dimension="sample",
        proportion_group_labels=["A", "B"],
        cluster_annotations=[
            ClusterAnnotation(cluster_id="0", cell_type="Fibroblast",
                              user_override="Myofibroblast", confidence="medium",
                              key_markers=["Col1a1"]),
            ClusterAnnotation(cluster_id="1", cell_type="Cardiomyocyte",
                              confidence="high"),
        ],
    )
    st.annotation_supplementary_requests = ["look at cluster 3"]
    rec = build_annotation_record(st, auto_confirmed=False)
    assert rec.clusters[0].original_cell_type == "Fibroblast"
    assert rec.clusters[0].final_cell_type == "Myofibroblast"
    assert rec.clusters[0].user_override is True
    assert rec.clusters[1].final_cell_type == "Cardiomyocyte"
    assert rec.clusters[1].user_override is False
    assert rec.supplementary_requests == ["look at cluster 3"]
    assert rec.proportion_dimension == "sample"
    assert rec.auto_confirmed is False


def test_resolve_integration_gate_preserves_candidates_when_clear_gate_false():
    state = AnalysisState(id="test", name="Test", user_id="test", current_round=5, integration_cycle=1)
    state.integration_gate_candidates = [{"direction": "Dir A"}]
    state.integration_gate_literature_evidence = [{"direction_id": "d1"}]
    rec = resolve_integration_gate(state, "open", clear_gate=False)
    assert state.integration_gate_candidates == [{"direction": "Dir A"}]
    assert state.integration_gate_literature_evidence == [{"direction_id": "d1"}]
    assert rec.event_type == "open"


def test_resolve_integration_gate_clears_candidates_by_default():
    state = AnalysisState(id="test", name="Test", user_id="test", current_round=5, integration_cycle=1)
    state.integration_gate_candidates = [{"direction": "Dir A"}]
    state.integration_gate_literature_evidence = [{"direction_id": "d1"}]
    resolve_integration_gate(state, "select")
    assert state.integration_gate_candidates == []
    assert state.integration_gate_literature_evidence == []


def test_find_open_integration_record_finds_current_cycle():
    state = AnalysisState(id="test", name="Test", user_id="test", integration_cycle=2)
    rec1 = DirectionSelectionRecord(round_number=5, phase="integration", integration_cycle=1, event_type="open", let_pi_decide=True)
    rec2 = DirectionSelectionRecord(round_number=6, phase="integration", integration_cycle=2, event_type="open", let_pi_decide=True)
    state.direction_selection_records = [rec1, rec2]
    result = find_open_integration_record(state)
    assert result is rec2


def test_find_open_integration_record_returns_none_for_wrong_cycle():
    state = AnalysisState(id="test", name="Test", user_id="test", integration_cycle=2)
    rec1 = DirectionSelectionRecord(round_number=5, phase="integration", integration_cycle=1, event_type="open", let_pi_decide=True)
    state.direction_selection_records = [rec1]
    result = find_open_integration_record(state)
    assert result is None


def test_find_open_integration_record_returns_none_when_empty():
    state = AnalysisState(id="test", name="Test", user_id="test", integration_cycle=1)
    result = find_open_integration_record(state)
    assert result is None
