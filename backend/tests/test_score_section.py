from backend.services.report.score_section import friendly_method, _normalize, build_score_definitions_section_html, write_score_definitions_csv


def test_friendly_method_known():
    assert "Mean expression" in friendly_method("sc.tl.score_genes")
    assert "ssGSEA" in friendly_method("gseapy.ssgsea") or "GSEA" in friendly_method("gseapy.ssgsea")
    assert "decoupler" in friendly_method("dc.run_ulm").lower()


def test_friendly_method_unknown_returns_empty():
    assert friendly_method("some.unknown.fn") == ""
    assert friendly_method("") == ""


def test_normalize_from_dict():
    sd = {"score_name": "S", "method": "m", "gene_list": ["A", "B"]}
    out = _normalize(sd)
    assert out["score_name"] == "S"
    assert out["gene_list"] == ["A", "B"]
    assert out["parameters"] == {}
    assert out["gene_list_source"] == ""
    assert out["round_number"] == 0


def test_normalize_from_object():
    class Fake:
        score_name = "S"
        method = "m"
        description = "d"
        parameters = {"ctrl_size": 50}
        gene_list = ["A"]
        gene_list_source = "MSigDB"
        round_number = 2
    out = _normalize(Fake())
    assert out["parameters"] == {"ctrl_size": 50}
    assert out["gene_list_source"] == "MSigDB"
    assert out["round_number"] == 2


def _sample_defs():
    return [{
        "score_name": "Fibrosis_Score",
        "method": "sc.tl.score_genes",
        "description": "Fibrosis level",
        "parameters": {"ctrl_size": 50, "n_bins": 25},
        "gene_list": [f"GENE{i}" for i in range(15)],
        "gene_list_source": "MSigDB Hallmark",
        "round_number": 1,
    }]


def test_build_html_empty_returns_empty():
    assert build_score_definitions_section_html([]) == ""


def test_build_html_contains_download_link():
    html = build_score_definitions_section_html(_sample_defs())
    assert 'href="tables/score_definitions.csv"' in html
    assert "download-btn csv" in html


def test_build_html_shows_all_genes_in_details():
    html = build_score_definitions_section_html(_sample_defs())
    # All 15 genes present (not truncated to 10), inside a collapsible <details>
    assert "<details" in html
    assert "GENE0" in html and "GENE14" in html
    assert "15 genes" in html


def test_build_html_shows_friendly_computation():
    html = build_score_definitions_section_html(_sample_defs())
    assert "Mean expression of the gene set" in html
    assert "<code>sc.tl.score_genes</code>" in html


def test_build_html_escapes_content():
    defs = _sample_defs()
    defs[0]["description"] = "<script>x</script>"
    html = build_score_definitions_section_html(defs)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_csv_empty_returns_none(tmp_path):
    assert write_score_definitions_csv([], tmp_path / "tables") is None


def test_write_csv_creates_file_in_tables(tmp_path):
    path = write_score_definitions_csv(_sample_defs(), tmp_path / "tables")
    assert path is not None
    assert path.name == "score_definitions.csv"
    assert path.parent.name == "tables"
    content = path.read_text(encoding="utf-8")
    assert "Score Name" in content and "Computation" in content
    assert "Fibrosis_Score" in content
    assert "GENE0" in content and "GENE14" in content  # full gene list, not truncated
    assert "Mean expression of the gene set" in content
