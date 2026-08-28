"""Report outline: rounds carry the report, and integration rounds are placed
by what follows them rather than by what they are."""
from datetime import datetime, timedelta
from types import SimpleNamespace

from backend.services.report.report_outline import (
    GATE_SECTION_WORD_LIMIT,
    align_sections_to_outline,
    build_report_outline,
    section_for_round,
)

_T0 = datetime(2026, 1, 1, 0, 0, 0)


def _round(n, phase, direction="", minutes=None):
    return SimpleNamespace(
        round_number=n,
        phase=phase,
        direction=direction,
        timestamp=_T0 + timedelta(minutes=minutes if minutes is not None else n * 10),
    )


def _state(rounds, publication_figures=None):
    return SimpleNamespace(
        round_results=rounds,
        publication_figures=publication_figures or [],
    )


def _ids(outline):
    return [sec.id for sec in outline]


def test_mid_run_integration_is_a_gate_final_one_closes():
    """1 explore → 2,3 directions → 4 integrate → 5 direction → 6 integrate."""
    state = _state([
        _round(1, "exploration"),
        _round(2, "innovation", "direction A"),
        _round(3, "innovation", "direction B"),
        _round(4, "integration"),
        _round(5, "innovation", "direction C"),
        _round(6, "integration"),
    ])
    outline = build_report_outline(state)

    assert _ids(outline) == [
        "abstract", "methods", "exploration_findings",
        "direction_1", "direction_2", "integration_gate_1",
        "direction_3", "integrated_findings",
        "mechanism_candidates", "conclusions",
    ]
    gate = next(s for s in outline if s.id == "integration_gate_1")
    final = next(s for s in outline if s.id == "integrated_findings")
    # Round 4 has innovation work after it, so it is a hinge; round 6 does not.
    assert gate.kind == "integration_gate" and gate.is_brief
    assert final.kind == "integration_final" and not final.is_brief
    assert gate.rounds == (4,) and final.rounds == (6,)
    assert gate.is_text_only and not gate.expects_figures
    assert final.is_text_only and not final.expects_figures
    assert next(s for s in outline if s.id == "direction_1").expects_figures
    conclusions = next(s for s in outline if s.id == "conclusions")
    assert "exactly one **Limitations**" in conclusions.guidance


def test_every_round_owns_exactly_one_section():
    state = _state([
        _round(1, "exploration"),
        _round(2, "innovation"),
        _round(3, "integration"),
        _round(4, "innovation"),
        _round(5, "integration"),
    ])
    outline = build_report_outline(state)
    for rn in (1, 2, 3, 4, 5):
        owner = section_for_round(outline, rn)
        assert owner is not None, f"round {rn} has no section"
        assert sum(1 for s in outline if rn in s.rounds) == 1


def test_methods_and_framing_sections_own_no_figures():
    """Methods describes procedure; the chapters that used to hold result
    figures under type headings are gone, so nothing can drift into them."""
    state = _state([
        _round(1, "exploration"),
        _round(2, "innovation"),
        _round(3, "integration"),
    ])
    outline = build_report_outline(state)
    by_id = {s.id: s for s in outline}
    for sid in ("abstract", "methods", "mechanism_candidates", "conclusions"):
        assert by_id[sid].rounds == ()
        assert not by_id[sid].owns_figures
    # The type-organised chapters no longer exist in a multi-round report.
    for gone in ("data_qc", "clustering_annotation", "differential_expression",
                 "analysis_overview"):
        assert gone not in by_id


def test_multiple_gates_are_numbered_in_round_order():
    state = _state([
        _round(1, "exploration"),
        _round(2, "innovation"),
        _round(3, "integration"),
        _round(4, "innovation"),
        _round(5, "integration"),
        _round(6, "innovation"),
        _round(7, "integration"),
    ])
    outline = build_report_outline(state)
    gates = [s for s in outline if s.kind == "integration_gate"]
    assert [g.id for g in gates] == ["integration_gate_1", "integration_gate_2"]
    assert [g.rounds for g in gates] == [(3,), (5,)]
    assert next(s for s in outline if s.kind == "integration_final").rounds == (7,)


def test_run_ending_on_a_direction_still_gets_a_closing_section():
    state = _state([
        _round(1, "exploration"),
        _round(2, "innovation"),
        _round(3, "integration"),
        _round(4, "innovation"),
    ])
    outline = build_report_outline(state)
    final = next(s for s in outline if s.id == "integrated_findings")
    # No integration round closes the run, so the section exists but owns no
    # figures of its own — round 4's figures stay with direction_2.
    assert final.rounds == ()
    assert section_for_round(outline, 4).id == "direction_2"


def test_single_pass_analysis_keeps_the_type_organised_outline():
    """Without innovation rounds there is one QC step and one DE step, so
    organising by analysis type is unambiguous."""
    outline = build_report_outline(_state([_round(1, "exploration")]))
    assert _ids(outline) == [
        "abstract", "analysis_overview", "data_qc", "clustering_annotation",
        "differential_expression", "mechanism_candidates", "key_findings",
        "conclusions",
    ]


def test_rounds_out_of_order_are_sorted_before_planning():
    state = _state([
        _round(3, "innovation", "B"),
        _round(1, "exploration"),
        _round(2, "innovation", "A"),
    ])
    outline = build_report_outline(state)
    assert _ids(outline).index("exploration_findings") < _ids(outline).index("direction_1")
    assert section_for_round(outline, 2).id == "direction_1"
    assert section_for_round(outline, 3).id == "direction_2"


def test_align_restores_outline_order_and_keeps_extras_last():
    state = _state([_round(1, "exploration"), _round(2, "innovation"),
                    _round(3, "integration")])
    outline = build_report_outline(state)
    emitted = [
        {"id": "conclusions", "content": "c"},
        {"id": "direction_1", "content": "d"},
        {"id": "abstract", "content": "a"},
        {"id": "an_invented_section", "content": "x"},
    ]
    aligned = align_sections_to_outline(emitted, outline)
    ids = [s["id"] for s in aligned]
    assert ids == ["abstract", "direction_1", "conclusions", "an_invented_section"]


def test_align_materialises_required_sections_the_model_skipped():
    state = _state([_round(1, "exploration"), _round(2, "innovation"),
                    _round(3, "integration")])
    outline = build_report_outline(state)
    aligned = align_sections_to_outline(
        [{"id": "abstract", "content": "a"}], outline,
        require_ids={"direction_1"},
    )
    ids = [s["id"] for s in aligned]
    assert ids == ["abstract", "direction_1"]
    created = aligned[1]
    assert created["title"] == "Innovation Round 2"
    assert created["content"] == ""


def test_gate_word_limit_is_stated_in_the_guidance():
    state = _state([
        _round(1, "exploration"), _round(2, "innovation"),
        _round(3, "integration"), _round(4, "innovation"),
        _round(5, "integration"),
    ])
    gate = next(
        s for s in build_report_outline(state) if s.kind == "integration_gate"
    )
    assert str(GATE_SECTION_WORD_LIMIT) in gate.guidance
    assert "provisional" in gate.guidance.lower()
