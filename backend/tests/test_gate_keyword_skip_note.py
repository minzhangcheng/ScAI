"""gate reaching max rejections must emit a visible skip note (not silence)."""
from backend.agents.bioinformatics_agent import build_gate_skip_note


def test_skip_note_mentions_gate_label():
    note = build_gate_skip_note("bulk/scRNA-seq cross-validation")
    assert "bulk/scRNA-seq cross-validation" in note
    assert "not" in note.lower()
