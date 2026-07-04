from backend.models.schemas import AnalysisState, DirectionSelectionRecord
from backend.services.report import interaction_records as ir


def test_no_mkdir_when_nothing_to_write(tmp_path, monkeypatch):
    calls = {"mkdir": 0}
    real_mkdir = ir.Path.mkdir

    def spy_mkdir(self, *a, **k):
        calls["mkdir"] += 1
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(ir.Path, "mkdir", spy_mkdir)
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.annotation_record = None
    s.direction_selection_records = []
    result = ir.write_interaction_record_files(s, reports_dir=tmp_path / "reports")
    assert result == {"annotation_record": None, "direction_records": []}
    assert calls["mkdir"] == 0


def test_direction_record_entries_matches_writer(tmp_path):
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    rec = DirectionSelectionRecord(round_number=3, phase="innovation", event_type="select")
    s.direction_selection_records = [rec]
    entries = ir.direction_record_entries(s)
    written = ir.write_interaction_record_files(s, reports_dir=tmp_path)["direction_records"]
    assert entries == written


def test_integration_entry_label_matches_rendered_record_header():
    from backend.models.schemas import DirectionSelectionRecord

    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    rec = DirectionSelectionRecord(
        round_number=5,
        phase="integration",
        integration_cycle=2,
        event_type="finalize",
    )
    s.direction_selection_records = [rec]
    entry = ir.direction_record_entries(s)[0]
    html = ir.render_direction_record_html(rec)
    assert entry["label"] == "Integration Cycle 2 · integration"
    assert "Integration Cycle 2 · integration" in html


def test_write_interaction_record_files_does_not_use_next_sync():
    import inspect

    src = inspect.getsource(ir.write_interaction_record_files)
    assert "next(" not in src
    assert "iter(entries)" not in src
