"""register_score must reject unattributable panels and record real panel size.

The schema change alone is advisory -- a model can omit or invent a field. These
tests pin the handler-side enforcement and the report rendering, which are the
parts a model cannot talk its way around.
"""
import inspect

import backend.agents.bioinformatics_agent as bio_mod
from backend.models.schemas import ScoreDefinition
from backend.services.report.score_section import (
    build_score_definitions_section_html,
    panel_size_cell,
    provenance_label,
    write_score_definitions_csv,
)


def _register_score_schema():
    for tool in bio_mod.TOOLS:
        if tool["function"]["name"] == "register_score":
            return tool["function"]["parameters"]
    raise AssertionError("register_score tool schema not found")


def test_gene_list_source_is_required_by_schema():
    schema = _register_score_schema()
    assert "gene_list_source" in schema["required"], (
        "an optional provenance field is one the model will skip"
    )


def test_schema_exposes_matched_gene_count():
    schema = _register_score_schema()
    assert "n_genes_matched" in schema["properties"]


def test_handler_rejects_invalid_provenance_and_returns_a_tool_message():
    """A rejection must still emit a role=tool message for the tool_call_id.

    Otherwise the call is orphaned and the next request fails -- the same
    invariant test_unknown_tool_dispatch pins for unknown tools.
    """
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "register_score":')
    end = src.index('elif tool_name == "register_mechanism_programs":', start)
    block = src[start:end]

    assert "parse_provenance" in block, "handler does not validate provenance"
    assert "prov_error" in block
    # The rejection path emits a tool message then skips registration.
    assert '"role": "tool"' in block
    assert "continue" in block
    # has_register_score must not be set before validation succeeds, or a
    # rejected call would satisfy the "did you register a score" gate.
    reject_idx = block.index("prov_error")
    flag_idx = block.index("has_register_score = True")
    assert flag_idx > reject_idx, (
        "has_register_score is set before provenance validation, so a rejected "
        "score would still satisfy the registration gate"
    )


def test_handler_records_power_assessment():
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "register_score":')
    end = src.index('elif tool_name == "register_mechanism_programs":', start)
    block = src[start:end]
    assert "assess_panel_power" in block
    assert "power=power" in block
    assert "n_genes=len(panel_members)" in block


def test_power_is_assessed_on_the_frozen_panel_not_the_reported_excerpt():
    """gene_list is documented as an excerpt; the frozen plan is the panel.

    Sizing power from gene_list understates every panel longer than the excerpt
    cap, which is exactly the case where the underpowered flag matters. The
    handler must take its counts and its hash from plan.panel_genes when a plan
    is cited, and use gene_list only for the drift comparison.
    """
    src = inspect.getsource(bio_mod)
    start = src.index('elif tool_name == "register_score":')
    end = src.index('elif tool_name == "register_mechanism_programs":', start)
    block = src[start:end]
    assert "panel_members = list(plan.panel_genes)" in block
    assert "assess_panel_power(len(panel_members)" in block
    assert "panel_hash(panel_members)" in block
    assert "diff_panels(plan.panel_genes, gene_list)" in block, (
        "drift must still compare the frozen panel against what was reported"
    )


def test_score_definition_carries_provenance_fields():
    sd = ScoreDefinition(
        score_name="s", method="m", description="d",
        gene_list=["A", "B"], gene_list_source="database:res:v1:SET",
        provenance={"kind": "database", "resource": "res",
                    "version": "v1", "set_id": "SET"},
        n_genes=2, n_genes_matched=2,
        power={"underpowered": True, "effective_n": 2},
    )
    assert sd.provenance["kind"] == "database"
    assert sd.n_genes_matched == 2
    assert sd.power["underpowered"]


def test_provenance_label_prefers_parsed_form():
    sd = {"provenance": {"kind": "database", "resource": "res",
                         "version": "v1", "set_id": "SET"},
          "gene_list_source": "database:res:v1:SET"}
    assert provenance_label(sd) == "res v1: SET"


def test_provenance_label_falls_back_for_legacy_scores():
    """Scores registered before enforcement must still render, not vanish."""
    sd = {"provenance": {}, "gene_list_source": "curated gene set"}
    assert provenance_label(sd) == "curated gene set"


def test_panel_size_cell_marks_underpowered():
    sd = {"n_genes": 5, "n_genes_matched": 5, "gene_list": ["a"] * 5,
          "power": {"underpowered": True}}
    assert panel_size_cell(sd) == "5/5 (underpowered)"


def test_panel_size_cell_shows_matched_over_nominal():
    sd = {"n_genes": 200, "n_genes_matched": 180, "gene_list": [],
          "power": {"underpowered": False}}
    assert panel_size_cell(sd) == "180/200"


def test_report_section_renders_panel_size_column():
    sd = ScoreDefinition(
        score_name="s", method="sc.tl.score_genes", description="d",
        gene_list=["A"] * 5, gene_list_source="bespoke:" + "x" * 30,
        provenance={"kind": "bespoke", "justification": "x" * 30},
        n_genes=5, n_genes_matched=5,
        power={"underpowered": True, "effective_n": 5},
    )
    html = build_score_definitions_section_html([sd])
    assert "Panel Size" in html
    assert "underpowered" in html
    assert "hand-assembled" in html


def test_csv_export_includes_provenance_columns(tmp_path):
    sd = ScoreDefinition(
        score_name="s", method="m", description="d",
        gene_list=["A", "B"], gene_list_source="database:res:v1:SET",
        provenance={"kind": "database", "resource": "res",
                    "version": "v1", "set_id": "SET"},
        n_genes=2, n_genes_matched=1,
        power={"underpowered": True},
    )
    path = write_score_definitions_csv([sd], tmp_path)
    text = path.read_text()
    assert "Provenance Kind" in text
    assert "Panel Genes Matched" in text
    assert "database" in text
