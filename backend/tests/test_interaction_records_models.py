import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.schemas import (
    SelectionEntry, DirectionSelectionRecord,
    AnnotationClusterEntry, AnnotationRecord,
)


def test_direction_record_defaults():
    rec = DirectionSelectionRecord(round_number=2, phase="exploration")
    assert rec.event_type == "open"
    assert rec.record_id
    assert rec.integration_cycle is None
    assert rec.selections == []
    assert rec.candidates == []
    assert rec.literature_evidence == []


def test_selection_entry_roundtrip():
    e = SelectionEntry(direction="D", notes="n", is_custom=True, candidate_index=None)
    assert e.model_dump()["is_custom"] is True


def test_annotation_record_defaults():
    rec = AnnotationRecord(organism="mouse", tissue="heart")
    assert rec.auto_confirmed is False
    assert rec.proportion_dimension == "group"
    assert rec.clusters == []
    assert rec.supplementary_requests == []


def test_annotation_cluster_entry():
    c = AnnotationClusterEntry(
        cluster_id="0", original_cell_type="Fibroblast",
        final_cell_type="Myofibroblast", user_override=True, confidence="medium",
    )
    assert c.user_override is True
