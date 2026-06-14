"""Shared rendering + CSV export for the report 'Score Definitions' section.

Used by both the live report generator (generator.py) and the report
regenerator (regenerator.py) so the two paths stay in sync (DRY).
"""
import csv
import html as html_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.utils.report_html import render_table_html

# Map known computation methods (code function names) to a biologist-friendly
# explanation of how the score is aggregated. Unknown methods fall back to "".
METHOD_AGGREGATION: Dict[str, str] = {
    "sc.tl.score_genes": "Mean expression of the gene set, relative to a random control gene set",
    "gseapy.ssgsea": "Single-sample GSEA enrichment (rank-weighted)",
    "gseapy.prerank": "Pre-ranked GSEA enrichment (rank-weighted)",
    "gseapy.enrichr": "Over-representation enrichment (Enrichr)",
    "dc.run_ulm": "Weighted linear model activity (decoupler ULM)",
    "dc.run_ora": "Over-representation analysis (decoupler ORA)",
    "dc.run_consensus": "Consensus of multiple enrichment methods (decoupler)",
}


def friendly_method(method: str) -> str:
    """Return a plain-language aggregation description for a known method, else ''."""
    if not method:
        return ""
    key = method.strip()
    # `scanpy.tl.*` and `sc.tl.*` are the same function under scanpy's two import
    # names; normalize to the canonical `sc.` prefix so the map needs one entry.
    if key.startswith("scanpy."):
        key = "sc." + key[len("scanpy."):]
    return METHOD_AGGREGATION.get(key, "")


def _normalize(sd: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Normalize a ScoreDefinition (pydantic object) or a report.json dict to a plain dict."""
    if isinstance(sd, dict):
        get = lambda k, d=None: sd.get(k, d)
    else:
        get = lambda k, d=None: getattr(sd, k, d)
    return {
        "score_name": get("score_name", "") or "",
        "method": get("method", "") or "",
        "description": get("description", "") or "",
        "parameters": get("parameters", {}) or {},
        "gene_list": get("gene_list", []) or [],
        "gene_list_source": get("gene_list_source", "") or "",
        "round_number": get("round_number", 0) or 0,
    }


SCORE_DEFINITIONS_FILENAME = "score_definitions.csv"
DEFAULT_CSV_HREF = f"tables/{SCORE_DEFINITIONS_FILENAME}"


def build_score_definitions_section_html(
    score_defs: List[Union[Dict[str, Any], Any]],
    csv_href: str = DEFAULT_CSV_HREF,
) -> str:
    """Build the enhanced 'Score Definitions' HTML section. Returns '' if no scores.

    Enhanced for biologists: full gene list in a collapsible <details>, a
    plain-language computation column derived from the method, and a CSV
    download link above the table.
    """
    defs = [_normalize(sd) for sd in score_defs]
    if not defs:
        return ""
    esc = html_module.escape
    rows: List[List[str]] = []
    for sd in defs:
        method = sd["method"]
        friendly = friendly_method(method)
        code = f"<code>{esc(method)}</code>" if method else ""
        if friendly:
            comp_cell = f"{esc(friendly)}<br>{code}"
        else:
            comp_cell = code or "N/A"

        gene_list = sd["gene_list"]
        n = len(gene_list)
        if n:
            full = esc(", ".join(gene_list))
            genes_cell = (
                f'<details class="score-genes"><summary>{n} genes</summary>'
                f'<div style="margin-top:6px">{full}</div></details>'
            )
        else:
            genes_cell = "N/A"

        params = sd["parameters"]
        params_str = esc(", ".join(f"{k}={v}" for k, v in params.items()) if params else "default")

        rows.append([
            esc(sd["score_name"]),
            comp_cell,
            esc(sd["description"]),
            params_str,
            esc(sd["gene_list_source"] or "N/A"),
            genes_cell,
        ])

    table_html = render_table_html(
        headers=["Score Name", "Computation Method", "Description", "Parameters", "Source", "Gene List"],
        rows=rows,
        style="font-size:0.9em",
    )
    download_link = (
        f'<a href="{esc(csv_href)}" download class="download-btn csv">Download CSV</a>'
    )
    return f"""
        <div class="section score-definitions" id="score-definitions">
            <div class="section-header">
                <h2>Score Definitions</h2>
            </div>
            <div class="section-content">
                <p>The following scores were computed during the analysis. Each score's gene set and computation method are documented for transparency and reproducibility. Expand "Gene List" to view all genes.</p>
                <div style="margin-bottom:12px">{download_link}</div>
                {table_html}
            </div>
        </div>"""


def write_score_definitions_csv(
    score_defs: List[Union[Dict[str, Any], Any]],
    tables_dir: Union[str, Path],
) -> Optional[Path]:
    """Write score_definitions.csv into tables_dir. Returns the path, or None if empty."""
    defs = [_normalize(sd) for sd in score_defs]
    if not defs:
        return None
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / SCORE_DEFINITIONS_FILENAME
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Score Name", "Method", "Computation", "Description",
            "Parameters", "Gene List", "Gene List Source", "Round Number",
        ])
        for sd in defs:
            params = sd["parameters"]
            params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            genes_str = ", ".join(sd["gene_list"]) if sd["gene_list"] else ""
            writer.writerow([
                sd["score_name"], sd["method"], friendly_method(sd["method"]),
                sd["description"], params_str, genes_str,
                sd["gene_list_source"] or "", sd["round_number"],
            ])
    return csv_path
