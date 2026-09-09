"""Pure semantic-section renderers for report variants."""
from __future__ import annotations

from backend.models.schemas import AnalysisState


def generate_integration_report_html(
    state,
    candidates,
    literature_evidence,
    *,
    bindings,
):
    project_name = bindings.html_module.escape(state.name or "Untitled Analysis")
    timestamp = bindings.datetime.now().strftime("%Y-%m-%d %H:%M")

    sections: List[tuple[str, str]] = []

    correlation_text = bindings._inline_session_text(state.multi_direction_correlation, state)
    if correlation_text:
        sections.append(("Cross-Direction Correlation", bindings.markdown_to_html(correlation_text)))

    innovation_rounds = [r for r in (state.round_results or []) if r.phase == "innovation"]
    if innovation_rounds:
        round_cards = ""
        for r in innovation_rounds:
            direction = bindings.html_module.escape(r.direction or "N/A")
            pi_summary = bindings.markdown_to_html(bindings._inline_session_text(r.pi_summary or "N/A", state))
            bio_items = "".join(f"<li>{bindings.html_module.escape(str(f))}</li>" for f in r.bioinformatics_findings[:5])
            path_items = "".join(f"<li>{bindings.html_module.escape(str(f))}</li>" for f in r.pathology_findings[:5])
            biochem_items = "".join(f"<li>{bindings.html_module.escape(str(f))}</li>" for f in r.biochemistry_findings[:5])
            lit_items = "".join(f"<li>{bindings.html_module.escape(str(f))}</li>" for f in r.literature_findings[:5])
            round_cards += f"""
            <div class="findings-card">
                <div class="findings-card-header" style="background:var(--primary-light);color:var(--primary-color);">
                    Round {r.round_number}: {direction}
                </div>
                <div class="findings-card-content">
                    <div class="pi-assessment" style="margin-bottom:16px;">
                        <h3 style="margin-top:0;">PI Summary</h3>
                        {pi_summary}
                    </div>
                    <div class="findings-grid">
                        <div class="findings-card">
                            <div class="findings-card-header bio">Bioinformatics</div>
                            <div class="findings-card-content"><ul>{bio_items or "<li>No findings</li>"}</ul></div>
                        </div>
                        <div class="findings-card">
                            <div class="findings-card-header path">Pathology</div>
                            <div class="findings-card-content"><ul>{path_items or "<li>No findings</li>"}</ul></div>
                        </div>
                        <div class="findings-card">
                            <div class="findings-card-header biochem">Biochemistry</div>
                            <div class="findings-card-content"><ul>{biochem_items or "<li>No findings</li>"}</ul></div>
                        </div>
                        <div class="findings-card">
                            <div class="findings-card-header lit">Literature</div>
                            <div class="findings-card-content"><ul>{lit_items or "<li>No findings</li>"}</ul></div>
                        </div>
                    </div>
                </div>
            </div>"""
        sections.append(("Innovation Round Summaries", round_cards))

    candidates_rows = bindings._render_integration_candidates_rows(candidates)
    if candidates_rows:
        candidate_table = f"""
            <table class="direction-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Direction &amp; Rationale</th>
                        <th>Rank</th>
                        <th>Novelty</th>
                        <th>Evidence</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>{candidates_rows}</tbody>
            </table>"""
        sections.append(("Research Direction Candidates", candidate_table))

    evidence_html = bindings._render_integration_evidence_section(candidates, literature_evidence)
    if evidence_html:
        sections.append(("Literature-Gated Direction Evaluation", evidence_html))

    numbered_sections_html = "\n".join(
        bindings._render_integration_section(index, title, content_html)
        for index, (title, content_html) in enumerate(sections, 1)
    )

    score_section = bindings.build_score_definitions_section_html(state.score_definitions)

    style_block = bindings._INTEGRATION_STYLE

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Integration Report - {project_name}</title>
<style>
{style_block}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Integration Report</h1>
        <div class="meta">{project_name} | {timestamp}</div>
    </div>
    <div class="content">
        {numbered_sections_html}
        {score_section}
    </div>
    <div class="footer">
        Generated by ScAI Multi-Agent Analysis System | {timestamp}
    </div>
</div>
</body>
</html>"""

    reports_dir = bindings.get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "integration_report.html"
    tmp_path = html_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    bindings.os.replace(str(tmp_path), str(html_path))

    bindings.logger.info(f"Integration report HTML saved to: {html_path}")
    return str(html_path)


def generate_annotation_context_report_html(
    state: AnalysisState,
    graph_state,
    cluster_annotations: list,
    *,
    bindings,
) -> str:
    """
    Generate a self-contained HTML report providing full analysis context
    for the cell annotation confirmation step. No LLM calls.

    Args:
        state: AnalysisState with session info
        graph_state: Current workflow graph state dict
        cluster_annotations: List of ClusterAnnotation objects

    Returns:
        Path to the saved HTML file
    """
    from backend.config import get_session_reports, get_session_workspace

    project_name = bindings.html_module.escape(state.name or "Untitled Analysis")
    exp_design = bindings.html_module.escape(state.build_experimental_design_summary())
    timestamp = bindings.datetime.now().strftime("%Y-%m-%d %H:%M")

    (
        biochem_html,
        patho_html,
        bio_steps_html,
        figures_html,
        lit_html,
        ann_html,
    ) = build_annotation_sections(
        state,
        graph_state=graph_state,
        cluster_annotations=cluster_annotations,
        bindings=bindings,
    )
    # --- Full HTML ---
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Annotation Context Report - {project_name}</title>
<style>
{bindings._ANNOTATION_STYLE}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Annotation Context Report</h1>
        <div class="meta">{project_name} | {timestamp}</div>
    </div>
    <div class="content">
        <div class="section">
            <div class="section-header"><h2>1. Experimental Design</h2></div>
            <div class="section-content">
                <pre style="background:#f1f5f9;padding:12px;border-radius:8px;font-size:0.85rem;white-space:pre-wrap;">{exp_design}</pre>
            </div>
        </div>
        {biochem_html}
        {patho_html}
        {bio_steps_html}
        {figures_html}
        {lit_html}
        {ann_html}
        {bindings.build_score_definitions_section_html(state.score_definitions)}
    </div>
    <div class="footer">
        Generated by ScAI Multi-Agent Analysis System | {timestamp}
    </div>
</div>
</body>
</html>"""

    # Save HTML file
    reports_dir = get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "annotation_context_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    bindings.logger.info(f"Annotation context report HTML saved to: {html_path}")
    return str(html_path)


def build_annotation_sections(
    state,
    *,
    graph_state,
    cluster_annotations,
    bindings,
):
    from backend.config import get_session_workspace

    # --- Biochemistry & Pathology findings ---
    biochem_html = ""
    patho_html = ""
    if graph_state:
        biochem = graph_state.get("biochemistry_result")
        if biochem and biochem.get("findings"):
            items = "\n".join(f"<li>{bindings.html_module.escape(str(f))}</li>" for f in biochem["findings"])
            biochem_html = f"""
        <div class="section">
            <div class="section-header"><h2>2. Biochemistry Findings</h2></div>
            <div class="section-content"><ul>{items}</ul></div>
        </div>"""

        patho = graph_state.get("pathology_result")
        if patho and patho.get("findings"):
            items = "\n".join(f"<li>{bindings.html_module.escape(str(f))}</li>" for f in patho["findings"])
            patho_html = f"""
        <div class="section">
            <div class="section-header"><h2>3. Pathology Findings</h2></div>
            <div class="section-content"><ul>{items}</ul></div>
        </div>"""

    # --- Build figure description map ---
    figure_desc_map = bindings._build_figure_description_map(state.analysis_log)
    eligible_figures = bindings.eligible_artifacts(state.publication_figures)
    excluded_figure_stems = bindings.excluded_artifact_stems(
        state.publication_figures,
        eligible_figures,
        invalidated_stems=state.invalidated_artifact_stems,
    )
    workspace_dir = get_session_workspace(state.id) if state.id else None
    discovered_figures = []
    if workspace_dir and workspace_dir.exists():
        bindings.ensure_legacy_artifact_inventory(state, workspace_dir)
        discovered_figures = bindings.prepare_discovered_figures(
            bindings.discover_figures_in_workspace(workspace_dir),
            state.publication_figures,
            invalidated_stems=state.invalidated_artifact_stems,
            report_only=True,
            # This report is the review context for annotation validation, not
            # an archival publication surface. Its own unregistered diagnostic
            # figures are useful here; registered ineligible figures were already
            # removed by bindings.prepare_discovered_figures.
            allow_unregistered_diagnostics=True,
            registry_version=state.artifact_registry_version,
            legacy_artifacts=state.legacy_artifacts,
        )
    discovered_by_name = {
        bindings.Path(candidate["path"]).name: candidate for candidate in discovered_figures
    }
    discovered_by_stem = {
        candidate["name"]: candidate for candidate in discovered_figures
    }
    eligible_records_by_stem = {}
    for figure in eligible_figures:
        for stem in bindings.artifact_stems([figure]):
            eligible_records_by_stem.setdefault(stem, figure)

    def _resolve_fig_path(fname: str):
        """Resolve through the shared recursive discovery index."""
        if not workspace_dir:
            return None
        record = eligible_records_by_stem.get(bindings.Path(fname).stem)
        if record is not None:
            # A step log records only a filename. If that stem belongs to a
            # reviewed artifact, resolve the record-owned render/preview so an
            # overwritten same-name workspace file cannot inherit its approval.
            return bindings.resolve_archival_render(record, workspace_dir)
        candidate = discovered_by_name.get(bindings.Path(fname).name)
        if candidate is None:
            candidate = discovered_by_stem.get(bindings.Path(fname).stem)
        if candidate is not None:
            return bindings.Path(candidate["path"])
        return None

    # --- Integrated bioinformatics steps + figures ---
    bio_steps_html = ""
    bio_steps = [s for s in state.analysis_log if s.step_type == "execute_code"]
    linked_stems: set = set()

    if bio_steps:
        step_items = []
        for s in bio_steps:
            desc = bindings.html_module.escape(s.code_description or f"Step {s.step_number}")

            output_html = ""
            if s.code_output:
                output_preview = bindings.html_module.escape(s.code_output[:400])
                output_html = (
                    f'<details class="step-output-details">'
                    f'<summary>代码输出</summary>'
                    f'<pre class="step-output">{output_preview}</pre>'
                    f'</details>'
                )

            explanation_html = ""
            if s.agent_reasoning:
                explanation_html = (
                    f'<div class="step-explanation">'
                    f'{bindings.markdown_to_html(s.agent_reasoning[:800])}'
                    f'</div>'
                )

            step_fig_blocks = []
            for fname in (s.saved_figure_filenames or []):
                stem = bindings.Path(fname).stem
                if stem in excluded_figure_stems:
                    continue
                linked_stems.add(stem)
                fpath = _resolve_fig_path(fname)
                if fpath:
                    try:
                        b64 = bindings.compress_image_to_base64(str(fpath), max_width=1100, quality=85)
                        caption = bindings.html_module.escape(
                            figure_desc_map.get(stem, s.code_description or fname)
                        )
                        step_fig_blocks.append(bindings._make_figure_block(b64, caption))
                    except Exception:
                        bindings.logger.warning("Failed to embed step figure: %s", fpath, exc_info=True)

            figures_block = "".join(step_fig_blocks)
            step_items.append(
                f'<div class="analysis-step">'
                f'<div class="step-header"><span class="step-marker">&#9656;</span>'
                f'<strong class="step-desc">{desc}</strong></div>'
                f'{explanation_html}'
                f'{output_html}'
                f'{figures_block}'
                f'</div>'
            )

        bio_steps_html = f"""
        <div class="section">
            <div class="section-header"><h2>4. Bioinformatics Analysis Steps</h2></div>
            <div class="section-content">{"".join(step_items)}</div>
        </div>"""

    # --- Orphan figures (workspace figures not linked to any step) ---
    figures_html = ""
    orphan_blocks = []
    seen_stems: set = set(linked_stems)

    for candidate in discovered_figures:
        img_file = bindings.Path(candidate["path"])
        if img_file.stem in excluded_figure_stems or img_file.stem in seen_stems:
            continue
        seen_stems.add(img_file.stem)
        try:
            b64 = bindings.compress_image_to_base64(str(img_file), max_width=1100, quality=85)
            caption = bindings.html_module.escape(
                candidate.get("description")
                or figure_desc_map.get(
                    img_file.stem, img_file.stem.replace("_", " ").title()
                )
            )
            orphan_blocks.append(bindings._make_figure_block(b64, caption))
        except Exception:
            bindings.logger.warning("Failed to embed figure in HTML: %s", img_file, exc_info=True)

    # From publication_figures
    for pf in eligible_figures:
        pf_name = getattr(pf, 'name', '') or 'Figure'
        pf_stems = bindings.artifact_stems([pf])
        if not (pf_stems & seen_stems) and workspace_dir:
            fig_path = bindings.resolve_archival_render(pf, workspace_dir)
        else:
            fig_path = None
        if fig_path is not None:
            seen_stems.update(pf_stems)
            title = bindings.html_module.escape(pf_name)
            try:
                b64_data = bindings.compress_image_to_base64(
                    str(fig_path), max_width=1100, quality=85
                )
            except Exception:
                bindings.logger.warning(
                    "Failed to embed publication figure: %s", fig_path,
                    exc_info=True,
                )
            else:
                orphan_blocks.append(bindings._make_figure_block(b64_data, title))

    if orphan_blocks:
        section_num = "5" if bio_steps_html else "4"
        figures_html = f"""
        <div class="section">
            <div class="section-header"><h2>{section_num}. Additional Figures</h2></div>
            <div class="section-content">{"".join(orphan_blocks)}</div>
        </div>"""

    # --- Literature research ---
    lit_html = ""
    lit_steps = [s for s in state.analysis_log if s.step_type == "search_literature"]
    if lit_steps:
        lit_items = []
        for s in lit_steps:
            topic = bindings.html_module.escape(s.literature_topic or "Unknown topic")
            result = bindings.html_module.escape((s.literature_result or "No findings")[:500])
            lit_items.append(f"<li><strong>{topic}</strong><br/><span class=\"lit-result\">{result}</span></li>")
        lit_html = f"""
        <div class="section">
            <div class="section-header"><h2>6. Literature Research</h2></div>
            <div class="section-content"><ul>{"".join(lit_items)}</ul></div>
        </div>"""

    # --- Cluster annotation table ---
    ann_rows = ""
    for ca in cluster_annotations:
        cid = bindings.html_module.escape(str(ca.cluster_id))
        ct = bindings.html_module.escape(ca.cell_type)
        subtypes = bindings.html_module.escape(", ".join(ca.subtypes)) if ca.subtypes else ""
        markers = bindings.html_module.escape(", ".join(ca.key_markers[:10]))
        conf = ca.confidence
        evidence = bindings.html_module.escape(ca.evidence[:300]) if ca.evidence else ""

        conf_class = "badge-high" if conf == "high" else ("badge-medium" if conf == "medium" else "badge-low")
        conf_badge = f'<span class="{conf_class}">{bindings.html_module.escape(conf)}</span>'

        alt_types = ""
        if ca.alternative_cell_types:
            alt_list = ", ".join(bindings.html_module.escape(t) for t in ca.alternative_cell_types)
            alt_types = f'<div class="alt-types">Alternatives: {alt_list}</div>'

        reason = ""
        if ca.low_confidence_reason:
            reason = f'<div class="low-conf-reason">{bindings.html_module.escape(ca.low_confidence_reason)}</div>'

        # Citations. An empty cell would read as "no literature supports this",
        # which is only one of the four things an empty list can mean, so the
        # status is spelled out instead.
        citations = list(getattr(ca, "citations", None) or [])
        status = getattr(ca, "citation_status", "unavailable")
        if citations:
            used = [c for c in citations if getattr(c, "used_by_model", False)]
            listed = used or citations
            items = []
            for c in listed[:5]:
                pmid = bindings.html_module.escape(str(getattr(c, "pmid", "") or ""))
                title = bindings.html_module.escape(str(getattr(c, "title", "") or "")[:110])
                mark = " (read, not cited)" if not getattr(c, "used_by_model", False) else ""
                stale = " (predates a label change)" if getattr(c, "stale", False) else ""
                items.append(
                    f'<li><a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
                    f'target="_blank" rel="noopener">{pmid}</a> {title}{mark}{stale}</li>'
                )
            cites_html = f'<ul class="citations">{"".join(items)}</ul>'
        elif status == "no_evidence":
            cites_html = '<span class="citations-none">Searched, nothing found</span>'
        else:
            cites_html = '<span class="citations-none">Not recorded</span>'

        ann_rows += f"""
            <tr class="{'ann-low' if conf == 'low' else ('ann-medium' if conf == 'medium' else '')}">
                <td class="cluster-id">{cid}</td>
                <td>{ct}{f'<div class="subtypes">{subtypes}</div>' if subtypes else ''}{alt_types}{reason}</td>
                <td class="text-center">{conf_badge}</td>
                <td class="markers">{markers}</td>
                <td class="evidence">{evidence}</td>
                <td class="citations">{cites_html}</td>
            </tr>"""

    ann_html = f"""
        <div class="section">
            <div class="section-header"><h2>7. Cluster Annotation Details</h2></div>
            <div class="section-content">
                <table class="ann-table">
                    <thead>
                        <tr>
                            <th>Cluster</th>
                            <th>Cell Type</th>
                            <th>Confidence</th>
                            <th>Key Markers</th>
                            <th>Evidence</th>
                            <th>Citations</th>
                        </tr>
                    </thead>
                    <tbody>{ann_rows}</tbody>
                </table>
            </div>
        </div>"""

    return (
        biochem_html,
        patho_html,
        bio_steps_html,
        figures_html,
        lit_html,
        ann_html,
    )
from backend.models.schemas import AnalysisState


def generate_exploration_report_html(
    state: AnalysisState,
    exploration_report,
    round_result,
    *,
    bindings,
) -> str:
    """
    Generate a static HTML report for the exploration phase (Report A).
    No LLM calls — data is already structured.

    Args:
        state: AnalysisState with session info
        exploration_report: ExplorationReport model
        round_result: RoundResult from Round 1

    Returns:
        Path to the saved HTML file
    """
    from backend.config import get_session_reports, get_session_workspace

    # Prepare data
    project_name = bindings.html_module.escape(state.name or "Untitled Analysis")
    exp_design = bindings.html_module.escape(state.build_experimental_design_summary())
    cluster_overview = bindings.html_module.escape(exploration_report.cluster_overview or "N/A")
    pi_assessment = bindings.markdown_to_html(exploration_report.pi_direction_assessment or "N/A")

    # Data quality warnings
    dq_items = ""
    for w in (exploration_report.data_quality_warnings or []):
        dq_items += f"<li>{bindings.html_module.escape(w)}</li>\n"
    if not dq_items:
        dq_items = "<li>No data quality warnings</li>"

    # Key findings sections
    def build_list(items, empty_msg="None identified"):
        if not items:
            return f"<li>{empty_msg}</li>"
        return "\n".join(f"<li>{bindings.html_module.escape(str(i))}</li>" for i in items)

    heterogeneity_items = build_list(exploration_report.strongest_heterogeneity)
    pathway_items = build_list(exploration_report.key_pathway_changes)
    abnormal_items = build_list(exploration_report.abnormal_markers)
    low_conf_items = build_list(exploration_report.low_confidence_annotations)

    # Round 1 findings
    bio_findings = build_list(round_result.bioinformatics_findings[:10], "No bioinformatics findings")
    path_findings = build_list(round_result.pathology_findings[:5], "No pathology findings")
    biochem_findings = build_list(round_result.biochemistry_findings[:5], "No biochemistry findings")
    lit_findings = build_list(round_result.literature_findings[:5], "No literature findings")

    # Key analysis figures from Round 1
    key_figures_html = ""
    try:
        workspace_dir = get_session_workspace(state.id) if state.id else None
        if workspace_dir and workspace_dir.exists():
            bindings.ensure_legacy_artifact_inventory(state, workspace_dir)
            fig_desc_map = bindings._build_figure_description_map(state.analysis_log)
            figure_blocks = []
            seen_stems: set = set()
            # Resolve registered artifacts first. Disk discovery alone cannot
            # recover a reviewed figure after its original png_path moved or was
            # overwritten, while the record-owned preview still can.
            discovered = bindings.resolved_record_figure_candidates(
                state.publication_figures,
                workspace_dir,
                report_only=True,
            )
            discovered.extend(bindings.prepare_discovered_figures(
                bindings.discover_figures_in_workspace(workspace_dir),
                state.publication_figures,
                invalidated_stems=state.invalidated_artifact_stems,
                report_only=True,
                allow_unregistered_diagnostics=False,
                registry_version=state.artifact_registry_version,
                legacy_artifacts=state.legacy_artifacts,
            ))
            for candidate in discovered:
                img_file = bindings.Path(candidate["path"])
                if img_file.stem in seen_stems:
                    continue
                seen_stems.add(img_file.stem)
                try:
                    b64 = bindings.compress_image_to_base64(str(img_file), max_width=1100, quality=85)
                    caption = bindings.html_module.escape(
                        candidate.get("description")
                        or fig_desc_map.get(
                            img_file.stem, img_file.stem.replace("_", " ").title()
                        )
                    )
                    figure_blocks.append(bindings._make_figure_block(b64, caption))
                except Exception:
                    bindings.logger.warning("Failed to embed figure in exploration report: %s", img_file, exc_info=True)
            if figure_blocks:
                key_figures_html = f"""
        <div class="section">
            <div class="section-header">
                <h2>4b. Key Analysis Figures</h2>
            </div>
            <div class="section-content">
                {"".join(figure_blocks)}
            </div>
        </div>"""
    except Exception:
        bindings.logger.warning("Could not build figures section for exploration report", exc_info=True)

    # Direction candidates table
    candidates = exploration_report.research_direction_candidates or []
    direction_rows = bindings._render_direction_candidates_rows(candidates)

    evidence_section_html = bindings._render_direction_evidence_section(exploration_report)

    timestamp = bindings.datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exploration Report - {project_name}</title>
<style>
{bindings._EXPLORATION_STYLE}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Exploration Report</h1>
        <div class="meta">{project_name} | {timestamp}</div>
    </div>
    <div class="content">
        <!-- Experimental Design -->
        <div class="section">
            <div class="section-header">
                <h2>1. Experimental Design</h2>
            </div>
            <div class="section-content">
                <pre style="background:#f1f5f9;padding:16px;border-radius:8px;font-size:0.9rem;white-space:pre-wrap;">{exp_design}</pre>
            </div>
        </div>

        <!-- Cluster Overview -->
        <div class="section">
            <div class="section-header">
                <h2>2. Cluster Overview</h2>
            </div>
            <div class="section-content">
                <p>{cluster_overview}</p>
            </div>
        </div>

        <!-- Data Quality Warnings -->
        <div class="section">
            <div class="section-header">
                <h2>3. Data Quality Assessment</h2>
            </div>
            <div class="section-content">
                <ul class="warning-list">{dq_items}</ul>
            </div>
        </div>

        <!-- Round 1 Findings -->
        <div class="section">
            <div class="section-header">
                <h2>4. Round 1 Findings</h2>
            </div>
            <div class="section-content">
                <div class="findings-grid">
                    <div class="findings-card">
                        <div class="findings-card-header bio">🧬 Bioinformatics</div>
                        <div class="findings-card-content"><ul>{bio_findings}</ul></div>
                    </div>
                    <div class="findings-card">
                        <div class="findings-card-header path">🔬 Pathology</div>
                        <div class="findings-card-content"><ul>{path_findings}</ul></div>
                    </div>
                    <div class="findings-card">
                        <div class="findings-card-header biochem">🧪 Biochemistry</div>
                        <div class="findings-card-content"><ul>{biochem_findings}</ul></div>
                    </div>
                    <div class="findings-card">
                        <div class="findings-card-header lit">📚 Literature</div>
                        <div class="findings-card-content"><ul>{lit_findings}</ul></div>
                    </div>
                </div>
            </div>
        </div>

        {key_figures_html}

        <!-- Key Findings -->
        <div class="section">
            <div class="section-header">
                <h2>5. Key Findings</h2>
            </div>
            <div class="section-content">
                <h3 style="margin-bottom: 8px; color: var(--primary-color);">Strongest Heterogeneity</h3>
                <ul>{heterogeneity_items}</ul>

                <h3 style="margin: 16px 0 8px; color: var(--primary-color);">Key Pathway Changes</h3>
                <ul>{pathway_items}</ul>

                <h3 style="margin: 16px 0 8px; color: var(--primary-color);">Abnormal Markers</h3>
                <ul>{abnormal_items}</ul>

                <h3 style="margin: 16px 0 8px; color: var(--primary-color);">Low Confidence Annotations</h3>
                <ul>{low_conf_items}</ul>
            </div>
        </div>

        <!-- Direction Candidates -->
        <div class="section">
            <div class="section-header">
                <h2>6. Research Direction Candidates</h2>
            </div>
            <div class="section-content">
                <table class="direction-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Direction & Rationale</th>
                            <th>Priority</th>
                            <th>Literature Novelty</th>
                            <th>Data Reliability</th>
                        </tr>
                    </thead>
                    <tbody>
                        {direction_rows}
                    </tbody>
                </table>
            </div>
        </div>

        {evidence_section_html}

        <!-- Score Definitions -->
        {bindings.build_score_definitions_section_html(state.score_definitions)}

        <!-- PI Assessment -->
        <div class="section">
            <div class="section-header">
                <h2>{"9" if state.score_definitions else "8"}. PI Overall Assessment</h2>
            </div>
            <div class="section-content">
                <div class="pi-assessment">
                    <h3>🎯 PI Direction Assessment</h3>
                    {pi_assessment}
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        Generated by ScAI Multi-Agent Analysis System | {timestamp}
    </div>
</div>
</body>
</html>"""

    # Save HTML file
    reports_dir = get_session_reports(state.id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "exploration_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    bindings.logger.info(f"Exploration report HTML saved to: {html_path}")
    return str(html_path)
