"""Shared rendering + CSV export for the report 'Score Definitions' section.

Used by both the live report generator (generator.py) and the report
regenerator (regenerator.py) so the two paths stay in sync (DRY).
"""
import csv
import html as html_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote as url_quote

from backend.services.gene_set_provenance import GeneSetProvenance
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
    artifacts = []
    for item in get("definition_artifacts", []) or []:
        if isinstance(item, dict):
            role = item.get("role", "")
            table_id = item.get("table_id", "")
        else:
            role = getattr(item, "role", "")
            table_id = getattr(item, "table_id", "")
        if role and table_id:
            artifacts.append({"role": str(role), "table_id": str(table_id)})
    return {
        "score_name": get("score_name", "") or "",
        "method": get("method", "") or "",
        "description": get("description", "") or "",
        "parameters": get("parameters", {}) or {},
        "definition_kind": get("definition_kind", "unspecified") or "unspecified",
        "definition_status": get("definition_status", "legacy") or "legacy",
        "definition_artifacts": artifacts,
        "gene_list": get("gene_list", []) or [],
        "gene_list_source": get("gene_list_source", "") or "",
        "provenance": get("provenance", {}) or {},
        "n_genes": get("n_genes", 0) or 0,
        "n_genes_matched": get("n_genes_matched", None),
        "power": get("power", {}) or {},
        "round_number": get("round_number", 0) or 0,
    }


def _effective_definition_kind(sd: Dict[str, Any]) -> str:
    """Infer pre-contract flat panels without changing their stored identity."""
    kind = str(sd.get("definition_kind") or "unspecified")
    if kind == "unspecified" and sd.get("gene_list"):
        return "gene_panel"
    return kind


def _normalize_table(record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if isinstance(record, dict):
        get = lambda k, d=None: record.get(k, d)
    else:
        get = lambda k, d=None: getattr(record, k, d)
    filename = str(
        get("filename", "") or get("path", "") or get("csv_path", "") or ""
    )
    return {
        "table_id": str(get("table_id", "") or get("id", "") or ""),
        "filename": Path(filename).name if filename else "",
        "name": str(get("name", "") or ""),
        "total_rows": int(get("total_rows", 0) or 0),
        "csv_sha256": str(get("csv_sha256", "") or ""),
    }


ARTIFACT_ROLE_LABELS = {
    "gene_set_membership": "Gene-set membership",
    "regulatory_network": "Regulatory network",
    "ranked_input": "Ranked input",
    "leading_edge": "Leading edge",
}


def provenance_label(sd: Dict[str, Any]) -> str:
    """Human-readable source, preferring the parsed provenance over the raw string.

    Scores registered before provenance enforcement have no parsed form; their
    raw ``gene_list_source`` is shown as-is rather than being dropped.
    """
    prov = GeneSetProvenance.from_dict(sd.get("provenance") or {})
    if prov is not None:
        return prov.citation()
    return sd.get("gene_list_source", "") or ""


def panel_size_cell(sd: Dict[str, Any]) -> str:
    """Nominal / matched panel size plus a small-matched-panel marker.

    The matched count is what governs what the panel can detect; this is a
    size heuristic, not a power calculation. A panel that looks large but
    whose genes are largely absent from the analysed matrix cannot support a
    negative conclusion, and that has to be visible in the report rather than
    buried.
    """
    if _effective_definition_kind(sd) != "gene_panel":
        return "Structured definition"
    n = sd.get("n_genes") or len(sd.get("gene_list") or [])
    matched = sd.get("n_genes_matched")
    if not n:
        return "Not recorded"
    text = str(n) if matched is None else f"{matched}/{n}"
    if (sd.get("power") or {}).get("underpowered"):
        text += " (small matched panel)"
    return text


SCORE_DEFINITIONS_FILENAME = "score_definitions.csv"
DEFAULT_CSV_HREF = f"tables/{SCORE_DEFINITIONS_FILENAME}"


def build_score_definitions_section_html(
    score_defs: List[Union[Dict[str, Any], Any]],
    csv_href: str = DEFAULT_CSV_HREF,
    table_records: Optional[List[Union[Dict[str, Any], Any]]] = None,
) -> str:
    """Build the enhanced 'Score Definitions' HTML section. Returns '' if empty.

    Flat panels show their full genes. Structured definitions link to the exact
    registered tables packaged with the report.
    """
    defs = [_normalize(sd) for sd in score_defs]
    if not defs:
        return ""
    esc = html_module.escape
    table_by_id = {
        table["table_id"]: table
        for table in (_normalize_table(item) for item in (table_records or []))
        if table["table_id"]
    }
    rows: List[List[str]] = []
    for sd in defs:
        method = sd["method"]
        friendly = friendly_method(method)
        code = f"<code>{esc(method)}</code>" if method else ""
        if friendly:
            comp_cell = f"{esc(friendly)}<br>{code}"
        else:
            comp_cell = code or "N/A"

        definition_kind = _effective_definition_kind(sd)
        gene_list = sd["gene_list"]
        if definition_kind == "gene_panel" and gene_list:
            full = esc(", ".join(gene_list))
            definition_cell = (
                f'<details class="score-genes"><summary>{len(gene_list)} genes</summary>'
                f'<div style="margin-top:6px">{full}</div></details>'
            )
        elif definition_kind in {"gene_set_collection", "regulatory_network"}:
            artifact_lines = []
            for reference in sd["definition_artifacts"]:
                role = reference["role"]
                table_id = reference["table_id"]
                label = ARTIFACT_ROLE_LABELS.get(role, role.replace("_", " ").title())
                table = table_by_id.get(table_id)
                if table and table["filename"]:
                    href = "tables/" + url_quote(table["filename"])
                    table_name = table["name"] or table["filename"]
                    details = []
                    if table["total_rows"]:
                        details.append(f"{table['total_rows']:,} rows")
                    if table["csv_sha256"]:
                        details.append(f"SHA-256 {table['csv_sha256'][:12]}...")
                    suffix = f" ({'; '.join(details)})" if details else ""
                    artifact_lines.append(
                        f'<a href="{esc(href)}" download>{esc(label)}: '
                        f'{esc(table_name)}</a>{esc(suffix)}'
                    )
                else:
                    artifact_lines.append(
                        f"{esc(label)}: Registered table unavailable: {esc(table_id)}"
                    )
            definition_cell = "<br>".join(artifact_lines) if artifact_lines else (
                '<span class="score-definition-missing">'
                "Definition artifact not registered</span>"
            )
        else:
            definition_cell = (
                '<span class="score-definition-missing">'
                "Definition not recorded for this legacy score</span>"
            )

        params = sd["parameters"]
        params_str = esc(", ".join(f"{k}={v}" for k, v in params.items()) if params else "default")

        rows.append([
            esc(sd["score_name"]),
            comp_cell,
            esc(sd["description"]),
            params_str,
            esc(provenance_label(sd) or "Not recorded"),
            esc(panel_size_cell(sd)),
            definition_cell,
        ])

    table_html = render_table_html(
        headers=["Score Name", "Computation Method", "Description", "Parameters",
                 "Source", "Panel Size", "Gene List / Definition"],
        rows=rows,
        style="font-size:0.9em",
    )
    download_link = (
        f'<a href="{esc(csv_href)}" download class="download-btn csv">Download CSV</a>'
    )
    return f"""
        <details class="section score-definitions report-collapsible" id="score-definitions">
            <summary class="section-header report-collapsible-summary">
                <h2>Score Definitions</h2>
            </summary>
            <div class="section-content report-collapsible-content">
                <p>The following scores were computed during the analysis. Flat gene panels are shown in full; structured gene-set collections and regulatory networks link to the registered, hash-bound tables that define their membership or edges.</p>
                <p style="font-size:0.9em;color:#555"><strong>Reading "Source" and "Panel Size":</strong> a source of the form <code>resource version: set_id</code> identifies a named set from a versioned external resource and can be re-fetched exactly; <em>derived</em> panels are reproducible from this notebook; <em>hand-assembled</em> panels were assembled for this analysis and carry correspondingly less weight. Panel Size shows genes matched in the analysed matrix over the nominal panel size. A panel marked <em>(small matched panel)</em> matched fewer than 10 genes -- a size heuristic, not a power analysis; a positive result may still be reported as an observation, but a non-significant result from such a panel does not support a negative conclusion.</p>
                <div style="margin-bottom:12px">{download_link}</div>
                {table_html}
            </div>
        </details>"""


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
            "Parameters", "Definition Kind", "Definition Status",
            "Definition Artifacts", "Gene List", "Gene List Source", "Provenance Kind",
            "Provenance Resource", "Provenance Version", "Provenance Set ID",
            "Panel Genes", "Panel Genes Matched", "Small Matched Panel", "Round Number",
        ])
        for sd in defs:
            params = sd["parameters"]
            params_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            genes_str = ", ".join(sd["gene_list"]) if sd["gene_list"] else ""
            artifacts_str = "; ".join(
                f"{item['role']}:{item['table_id']}"
                for item in sd["definition_artifacts"]
            )
            prov = sd.get("provenance") or {}
            power = sd.get("power") or {}
            writer.writerow([
                sd["score_name"], sd["method"], friendly_method(sd["method"]),
                sd["description"], params_str, sd["definition_kind"],
                sd["definition_status"], artifacts_str, genes_str,
                sd["gene_list_source"] or "",
                prov.get("kind", ""), prov.get("resource", ""),
                prov.get("version", ""), prov.get("set_id", ""),
                sd.get("n_genes", "") or "",
                "" if sd.get("n_genes_matched") is None else sd["n_genes_matched"],
                "yes" if power.get("underpowered") else "",
                sd["round_number"],
            ])
    return csv_path


def ensure_score_definition_tables(
    score_defs: List[Union[Dict[str, Any], Any]],
    available_tables: List[Dict[str, Any]],
    generated_tables: List[Dict[str, Any]],
) -> List[str]:
    """Include tables referenced by scores even when no narrative section chose them.

    Returns referenced table IDs that were not available for packaging. The order
    follows score registration and artifact registration, making report output
    deterministic.
    """
    available_by_id = {}
    for item in available_tables:
        table_id = _normalize_table(item)["table_id"]
        if table_id:
            available_by_id[table_id] = item
    generated_ids = set()
    for item in generated_tables:
        table_id = _normalize_table(item)["table_id"]
        if table_id:
            generated_ids.add(table_id)
    missing = []
    seen_missing = set()
    for score in score_defs:
        for reference in _normalize(score)["definition_artifacts"]:
            table_id = reference["table_id"]
            if table_id in generated_ids:
                continue
            table = available_by_id.get(table_id)
            if table is not None:
                generated_tables.append(table)
                generated_ids.add(table_id)
            elif table_id not in seen_missing:
                missing.append(table_id)
                seen_missing.add(table_id)
    return missing
