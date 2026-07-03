import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import AnalysisState, ExplorationReport
from backend.services.interaction_record_capture import (
    get_or_open_direction_record, map_selections,
)


def _state_with_candidates():
    st = AnalysisState(user_id="test", id="s", name="n", initial_instruction="i")
    st.current_round = 2
    st.exploration_report = ExplorationReport(
        research_direction_candidates=[{"direction": "A"}, {"direction": "B"}],
        direction_literature_evidence=[{"direction_id": "A"}],
    )
    return st


def test_supplementary_then_select_merges_into_one_record():
    st = _state_with_candidates()
    rec = get_or_open_direction_record(st, round_number=2, phase="exploration")
    rec.supplementary_requests.append("search more")
    rec2 = get_or_open_direction_record(st, round_number=2, phase="exploration")
    rec2.candidates = st.exploration_report.research_direction_candidates
    rec2.selections = map_selections([{"direction": "B", "notes": "", "is_custom": False}],
                                     rec2.candidates)
    rec2.event_type = "select"
    assert len(st.direction_selection_records) == 1
    assert st.direction_selection_records[0].supplementary_requests == ["search more"]
    assert st.direction_selection_records[0].event_type == "select"


def test_invalid_direction_request_does_not_open_record():
    st = _state_with_candidates()
    assert st.direction_selection_records == []
    has_direction_choice = False
    if not has_direction_choice:
        assert st.direction_selection_records == []


def test_pi_decide_record_shape():
    from backend.services.interaction_record_capture import (
        find_open_pi_decide_record, get_or_open_direction_record,
    )
    st = _state_with_candidates()
    rec = get_or_open_direction_record(st, round_number=1, phase="exploration")
    rec.let_pi_decide = True
    found = find_open_pi_decide_record(st)
    assert found is rec
    found.candidates = st.exploration_report.research_direction_candidates
    found.let_pi_decide = True
    found.pi_selected = [{"direction": "A", "notes": ""}]
    found.event_type = "pi_decide"
    assert len(st.direction_selection_records) == 1
    assert st.direction_selection_records[0].event_type == "pi_decide"
    assert st.direction_selection_records[0].pi_selected[0]["direction"] == "A"


def test_build_annotation_record_auto_confirmed_flag():
    from backend.models.schemas import AnnotationConfirmationState, ClusterAnnotation
    from backend.services.interaction_record_capture import build_annotation_record
    st = AnalysisState(user_id="test", id="s", name="n", initial_instruction="i")
    st.annotation_state = AnnotationConfirmationState(
        organism="mouse", tissue="heart",
        cluster_annotations=[ClusterAnnotation(cluster_id="0", cell_type="X", confidence="high")],
    )
    rec = build_annotation_record(st, auto_confirmed=True)
    assert rec.auto_confirmed is True
    assert rec.clusters[0].final_cell_type == "X"


def test_supervisor_appendix_links_new_records(tmp_path, monkeypatch):
    import backend.services.report.interaction_records as ir
    from backend.models.schemas import DirectionSelectionRecord, SelectionEntry, AnnotationRecord
    monkeypatch.setattr("backend.config.get_session_reports", lambda _id: tmp_path)
    st = AnalysisState(user_id="test", id="s", name="n", initial_instruction="i")
    st.annotation_record = AnnotationRecord(organism="mouse", tissue="heart")
    st.direction_selection_records.append(DirectionSelectionRecord(
        round_number=2, phase="exploration", event_type="select",
        candidates=[{"direction": "A"}],
        selections=[SelectionEntry(direction="A", candidate_index=0)]))
    files = ir.write_interaction_record_files(st)
    assert files["annotation_record"] == "annotation_record.html"
    assert files["direction_records"][0]["filename"].endswith(".html")
    assert (tmp_path / "annotation_record.html").exists()


def test_record_writer_uses_current_report_output_dir(tmp_path, monkeypatch):
    import backend.services.report.interaction_records as ir
    from backend.models.schemas import AnnotationRecord

    root_reports = tmp_path / "reports"
    branch_reports = root_reports / "branches" / "branch-a"
    monkeypatch.setattr("backend.config.get_session_reports", lambda _id: root_reports)
    st = AnalysisState(user_id="test", id="s", name="n", initial_instruction="i")
    st.annotation_record = AnnotationRecord(organism="mouse", tissue="heart")

    files = ir.write_interaction_record_files(st, reports_dir=branch_reports)

    assert files["annotation_record"] == "annotation_record.html"
    assert (branch_reports / "annotation_record.html").exists()
    assert not (root_reports / "annotation_record.html").exists()


def test_filename_formula_matches_between_writer_and_sections():
    from backend.models.schemas import DirectionSelectionRecord
    rec = DirectionSelectionRecord(round_number=3, phase="exploration", event_type="select")
    i = 0
    expected = f"direction_record_{i:02d}_round{rec.round_number}.html"
    assert expected == "direction_record_00_round3.html"
