"""Detailed-report figure, persistence, and HTML presentation phases."""
from __future__ import annotations


def bind_facade(facade) -> None:
    globals().update(
        {name: value for name, value in vars(facade).items() if not name.startswith("__")}
    )

async def build_report_figures(self, ctx, analysis_state):
    _progress = ctx["_progress"]
    workspace_path = ctx["workspace_path"]
    figure_map = ctx["figure_map"]
    figure_metadata_by_stem = ctx["figure_metadata_by_stem"]
    await _progress(70, "Embedding figures...")

    # Build figures list for report
    report_figures = []
    eligible_records = eligible_artifacts(analysis_state.publication_figures)
    records_by_stem: Dict[str, Any] = {}
    for record in eligible_records:
        stems = set(artifact_stems([record]))
        resolved_record_path = (
            resolve_archival_render(record, workspace_path)
            if workspace_path is not None else None
        )
        if resolved_record_path is not None:
            stems.add(resolved_record_path.stem)
        for stem in stems:
            records_by_stem.setdefault(stem, record)
    legacy_by_stem = {
        Path(str(getattr(item, "relative_path", "") or "")).stem: item
        for item in (getattr(analysis_state, "legacy_artifacts", None) or [])
        if str(getattr(item, "relative_path", "") or "").strip()
    }
    for fig_name, fig_path in figure_map.items():
        stem = Path(fig_name).stem
        record = records_by_stem.get(stem) or records_by_stem.get(fig_path.stem)
        metadata = figure_metadata_by_stem.get(stem, {})
        legacy_record = legacy_by_stem.get(stem) or legacy_by_stem.get(fig_path.stem)
        if record is not None:
            metadata = {
                **metadata,
                "publication_name": getattr(record, "name", "") or "",
                "type": getattr(record, "figure_type", "") or "other",
                "description": getattr(record, "description", "") or "",
                "interpretation": getattr(record, "interpretation", "") or "",
                "plan_step_id": getattr(record, "plan_step_id", None),
                "plan_id": getattr(record, "plan_id", None),
                "purpose": getattr(record, "purpose", "result") or "result",
                "logical_figure_id": getattr(record, "logical_figure_id", None),
                "render_attempt_id": getattr(record, "render_attempt_id", None),
                "revision_of_id": getattr(record, "revision_of_id", None),
                "supersedes_id": getattr(record, "supersedes_id", None),
                "is_published_head": bool(
                    getattr(record, "is_published_head", False)
                ),
                "reservation_notice": figure_reservation_notice(record),
            }
        fig_entry = {
            "filename": fig_name,
            "path": str(fig_path),
            "category": metadata.get("type") or self._categorize_figure(fig_name),
            "name": metadata.get("publication_name") or Path(fig_name).stem,
            "description": metadata.get("description") or "",
            "interpretation": metadata.get("interpretation") or "",
            "plan_step_id": metadata.get("plan_step_id"),
            "plan_id": metadata.get("plan_id"),
            "purpose": metadata.get("purpose") or "result",
            "logical_figure_id": metadata.get("logical_figure_id"),
            "render_attempt_id": metadata.get("render_attempt_id"),
            "revision_of_id": metadata.get("revision_of_id"),
            "supersedes_id": metadata.get("supersedes_id"),
            "is_published_head": bool(metadata.get("is_published_head")),
            "reservation_notice": metadata.get("reservation_notice") or {},
            "provenance_kind": (
                "registered_publication"
                if record is not None
                else (
                    getattr(legacy_record, "provenance_kind", None)
                    or "unregistered"
                )
            ),
            "is_legacy": legacy_record is not None and record is None,
            "legacy_artifact_id": (
                getattr(legacy_record, "id", None)
                if legacy_record is not None and record is None else None
            ),
        }
        # A same-name PDF can be overwritten independently of the raster.
        # Registered figures resolve through their review-time digest;
        # unregistered legacy figures at least need a structurally valid PDF.
        pdf_path = (
            resolve_archival_pdf(record, workspace_path)
            if record is not None else fig_path.with_suffix(".pdf")
        )
        if pdf_path is not None and publication_pdf_is_valid(pdf_path):
            fig_entry["pdf_path"] = str(pdf_path)
        report_figures.append(fig_entry)

    reported_figure_names = {
        str(figure.get("filename") or "") for figure in report_figures
    }
    unreported_figures = sorted(
        name for name in figure_map if name not in reported_figure_names
    )

    ctx.update(
        report_figures=report_figures, unreported_figures=unreported_figures,
    )

async def assemble_and_persist_report(self, ctx, analysis_state):
    _progress = ctx["_progress"]
    output_path = ctx["output_path"]
    workspace_path = ctx["workspace_path"]
    stats = ctx["stats"]
    used_fallback = ctx["used_fallback"]
    unreported_figures = ctx["unreported_figures"]
    figureless_sections = ctx["figureless_sections"]
    missing_sections = ctx.get("missing_sections") or []
    abstract = ctx["abstract"]
    conclusions = ctx["conclusions"]
    filtered_sections = ctx["filtered_sections"]
    report_figures = ctx["report_figures"]
    figure_map = ctx["figure_map"]
    language = ctx["language"]
    table_list = ctx["table_list"]
    all_records = ctx["all_records"]
    await _progress(80, "Building report...")

    report_id = str(uuid.uuid4())
    record_files = write_interaction_record_files(analysis_state, reports_dir=output_path)
    appendix_data = self._build_appendix(analysis_state, stats, record_files, reports_dir=output_path)

    # Persist graphic abstract for downloadable zip. Wrapped so a
    # malformed/unopenable image degrades gracefully (ga_path stays None;
    # the HTML embed still uses the base64 from analysis_state) instead of
    # crashing report generation — matches services/report/generator.py.
    ga_obj = getattr(analysis_state, "graphic_abstract", None)
    ga_path = None
    try:
        ga_path = persist_graphic_abstract(ga_obj, output_path / "figures")
    except Exception:
        logger.exception("Failed to save graphic abstract")

    # A report can be structurally complete while still missing a visual for
    # an ordinary findings section. Preserve the report, but do not label it
    # fully complete: callers and the UI need a machine-readable signal that
    # figure coverage is incomplete. Explicit text-only sections (for
    # example integration gates/syntheses) never enter this list.
    quality_disclosures = build_quality_disclosures(analysis_state)
    (
        report_status,
        report_current_step,
        partial_reason,
        partial_reasons,
    ) = build_report_partial_fields(
        degraded_fallback=used_fallback,
        unreported_figures=unreported_figures,
        figureless_sections=figureless_sections,
        missing_sections=missing_sections,
        approved_partial=bool(
            getattr(analysis_state, "allow_partial_report", False)
        ),
        quality_disclosures=quality_disclosures,
    )

    appendix_data["report_kind"] = report_status
    appendix_data["partial_reason"] = partial_reason
    appendix_data["partial_reasons"] = partial_reasons

    report = {
        "id": report_id,
        "session_id": analysis_state.id,
        "branch_id": getattr(analysis_state, "restore_branch_id", None),
        "title": analysis_state.name,
        "abstract": abstract,
        "sections": filtered_sections,
        "conclusions": conclusions,
        "figures": report_figures,
        "tables": table_list,
        "graphic_abstract_path": ga_path,  # Path for zip download
        "generated_at": datetime.now().isoformat(),
        "status": report_status,
        "progress": 100,
        "current_step": report_current_step,
        "appendix": appendix_data,
        # Recorded so an incomplete report is a reportable condition rather
        # than something an operator has to notice by eye.
        "figureless_sections": figureless_sections,
        "missing_sections": missing_sections,
        "unreported_figures": unreported_figures,
        "partial_reason": partial_reason,
        "partial_reasons": partial_reasons,
        "quality_disclosures": quality_disclosures,
        "analysis_log": all_records,
        "score_definitions": [sd.model_dump(mode="json") for sd in analysis_state.score_definitions] if analysis_state.score_definitions else [],
    }

    # Sanitize NaN/Infinity
    report = self._sanitize(report)

    await _progress(90, "Generating HTML report...")

    # Temporarily attach analysis_state for HTML generation. It is NOT
    # serialized into report.json; it is popped back off after HTML rendering
    # so it never leaks into the on-disk report or downstream consumers.
    report["_analysis_state"] = analysis_state

    # Generate HTML
    html_path = self._generate_html_report(
        report, filtered_sections, abstract, conclusions,
        report_figures, figure_map, stats, language, output_path, workspace_path,
    )
    report.pop("_analysis_state", None)
    report["html_path"] = str(html_path)

    # Persist only paths to the copied generation-owned bytes.  Workspace
    # files remain mutable after publication and therefore cannot be the
    # provenance carried by report.json.
    serialized_figures = list(report.get("figures") or [])
    serialized_tables = list(report.get("tables") or [])
    for section in report.get("sections") or []:
        if isinstance(section, dict):
            serialized_figures.extend(section.get("figures") or [])
            serialized_tables.extend(section.get("tables") or [])

    for figure in serialized_figures:
        if not isinstance(figure, dict):
            continue
        filename = Path(
            str(figure.get("filename") or figure.get("path") or "")
        ).name
        staged_png = output_path / "figures" / filename
        if not filename or not staged_png.is_file():
            raise ReportGenerationError(
                f"report figure was not copied into staging: {filename}"
            )
        figure["path"] = str(staged_png)
        figure["png_path"] = str(staged_png)
        raw_pdf = figure.get("pdf_path")
        if raw_pdf:
            staged_pdf = output_path / "figures" / Path(str(raw_pdf)).name
            if not staged_pdf.is_file():
                raise ReportGenerationError(
                    f"report figure PDF was not copied into staging: {raw_pdf}"
                )
            figure["pdf_path"] = str(staged_pdf)

    for table in serialized_tables:
        if not isinstance(table, dict):
            continue
        filename = Path(
            str(
                table.get("filename")
                or table.get("csv_path")
                or table.get("path")
                or ""
            )
        ).name
        staged_csv = output_path / "tables" / filename
        if not filename or not staged_csv.is_file():
            raise ReportGenerationError(
                f"report table was not copied into staging: {filename}"
            )
        table["path"] = str(staged_csv)
        table["csv_path"] = str(staged_csv)

    await _progress(100, report_current_step, report_status)
    return report

def prepare_html_render(
    self, report, sections, abstract, conclusions, report_figures, figure_map,
    stats, language, output_path, workspace_path,
):
    import html as html_module

    def _maybe_escape_html(text: str) -> str:
        """Return *text* as-is if it already contains HTML tags, otherwise escape it.

        Caption text from [FIGURE_CAPTION:]/[TABLE_CAPTION:] markers has been
        through _markdown_to_html() and may contain safe HTML.  Captions from
        pub_fig_descriptions are plain text and need escaping.
        """
        if '<' in text and '>' in text:
            return text
        return html_module.escape(text)

    # Build figure base64 cache for embedding
    figure_b64_cache: Dict[str, str] = {}
    for fig_name, fig_path in figure_map.items():
        try:
            with open(fig_path, "rb") as f:
                raw = f.read()
            figure_b64_cache[fig_name] = base64.b64encode(raw).decode()
        except Exception as e:
            logger.warning(f"Could not read figure {fig_path}: {e}")

    # PDF paths in ``report_figures`` have already passed the artifact
    # integrity checks in ReportContentMixin.  Keep that validated mapping
    # authoritative here: a same-stem PDF beside a raster may have been
    # overwritten after review and must never become downloadable merely
    # because it happens to exist on disk.
    validated_pdf_by_figure: Dict[str, Path] = {}
    legacy_figure_stems = {
        Path(str(figure.get("filename") or figure.get("path") or "")).stem
        for figure in report_figures
        if figure.get("is_legacy")
    }
    for figure in report_figures:
        raw_pdf_path = figure.get("pdf_path")
        if not raw_pdf_path:
            continue
        pdf_path = Path(raw_pdf_path)
        if not pdf_path.is_file():
            continue
        for raw_name in (figure.get("filename"), figure.get("path")):
            if raw_name:
                validated_pdf_by_figure.setdefault(
                    os.path.basename(str(raw_name)), pdf_path
                )

    def _validated_pdf_for(fig_name: str) -> Optional[Path]:
        return (
            validated_pdf_by_figure.get(fig_name)
            or validated_pdf_by_figure.get(_alt_ext_name(fig_name))
        )

    # Build TOC
    toc_items = []
    for sec in sections:
        a_type = sec.get("analysis_type", "general")
        type_class = "general" if a_type == "general" else "specific"
        type_label = "General" if a_type == "general" else "Specific"
        toc_items.append(
            f'<li><a href="#{html_module.escape(sec["id"])}">'
            f'{html_module.escape(sec["title"])}</a>'
            f'<span class="section-type {type_class}">{type_label}</span></li>'
        )

    # Build publication figure description lookup
    pub_fig_descriptions: Dict[str, Dict[str, str]] = {}
    pub_fig_descriptions_by_stem: Dict[str, Dict[str, str]] = {}
    analysis_state = report.get("_analysis_state")
    eligible_figures = eligible_artifacts(
        analysis_state.publication_figures if analysis_state else []
    )
    if eligible_figures:
        for pf in eligible_figures:
            fname = os.path.basename(pf.png_path) if pf.png_path else ""
            metadata = {
                "name": pf.name,
                "figure_type": pf.figure_type or "other",
                "description": pf.description or "",
                "interpretation": pf.interpretation or "",
                "reservation_notice": figure_reservation_notice(pf),
            }
            if fname:
                pub_fig_descriptions[fname] = metadata
            # A restored record may have an empty or obsolete absolute path
            # while ``resolve_archival_render`` materializes its preview under
            # a new extension.  Match archival metadata by every record-owned
            # stem as well as by the old exact basename.
            for stem in artifact_stems([pf]):
                pub_fig_descriptions_by_stem.setdefault(stem, metadata)

    def _alt_ext_name(name: str) -> str:
        """Return filename with alternate image extension (.png↔.jpg/.jpeg)."""
        stem, ext = os.path.splitext(name)
        lower_ext = ext.lower()
        if lower_ext in (".jpg", ".jpeg"):
            return stem + ".png"
        return stem + ".jpg"

    def _build_figure_html(fig_name: str, caption: str = "") -> str:
        """Build inline figure HTML block for a given figure filename."""
        fig_b64 = figure_b64_cache.get(fig_name, "")
        # Fallback: try alternate extension (.png ↔ .jpg/.jpeg) if exact name not found
        if not fig_b64:
            alt_name = _alt_ext_name(fig_name)
            fig_b64 = figure_b64_cache.get(alt_name, "")
            if fig_b64:
                logger.info(f"Figure extension fallback: '{fig_name}' → '{alt_name}'")
                fig_name = alt_name
        if not fig_b64:
            return ""
        # Determine MIME type from extension
        mime = "image/jpeg" if fig_name.lower().endswith((".jpg", ".jpeg")) else "image/png"

        # Registered metadata is authoritative. A model-written caption may
        # add context only for an unregistered/legacy figure; otherwise it
        # could relabel a declared barplot as a UMAP despite the record next
        # to it saying exactly what was rendered.
        fig_desc = (
            pub_fig_descriptions.get(fig_name)
            or pub_fig_descriptions_by_stem.get(Path(fig_name).stem)
            or {}
        )
        display_name = fig_desc.get("name", fig_name.replace("_", " ").rsplit(".", 1)[0].title())
        description = fig_desc.get("description", "")
        interpretation = fig_desc.get("interpretation", "")
        reservation_html = render_figure_reservation_notice_html(
            fig_desc.get("reservation_notice")
        )
        if not description and not interpretation:
            interpretation = caption

        # Description paragraph
        desc_html = ""
        if description:
            desc_html = f'<p><em>{html_module.escape(description)}</em></p>'
        legacy_html = ""
        if Path(fig_name).stem in legacy_figure_stems:
            legacy_html = (
                '<p class="artifact-provenance legacy-unreviewed">'
                "Legacy unregistered artifact; no publication review record is available."
                "</p>"
            )

        # Interpretation block
        interp_html = ""
        if interpretation:
            interp_content = _maybe_escape_html(interpretation)
            interp_html = (
                f'<div class="figure-interpretation">'
                f'<strong>Interpretation</strong>'
                f'<p>{interp_content}</p>'
                f'</div>'
            )

        # Download buttons — label matches actual file extension
        img_ext = os.path.splitext(fig_name)[1].lstrip('.').upper() or "PNG"
        img_btn = (
            f'<a href="figures/{html_module.escape(fig_name)}" '
            f'download class="download-btn png">Download {img_ext}</a>'
        )
        pdf_btn = ""
        pdf_path = _validated_pdf_for(fig_name)
        if pdf_path is not None:
            pdf_btn = (
                f'<a href="figures/{html_module.escape(pdf_path.name)}" '
                f'download class="download-btn pdf">Download PDF</a>'
            )

        return f'''
            <div class="figure-container">
                <div class="figure-image">
                    <img src="data:{mime};base64,{fig_b64}"
                         alt="{html_module.escape(fig_name)}"/>
                </div>
                <div class="figure-caption">
                    <h4>{html_module.escape(display_name)}</h4>
                    {legacy_html}
                    {desc_html}
                    {reservation_html}
                    {interp_html}
                    <div class="figure-actions">
                        {img_btn}
                        {pdf_btn}
                    </div>
                </div>
            </div>
        '''

    # Track which figures have been placed
    placed_figures: set = set()

    # Build table lookup for inline embedding
    table_lookup: Dict[str, Dict] = {}
    for tbl in report.get("tables", []):
        table_lookup[tbl["filename"]] = tbl
    placed_tables: set = set()

    def _build_table_html(tbl_filename: str, caption: str = "") -> str:
        """Build inline table HTML block for a given CSV filename."""
        tbl_info = table_lookup.get(tbl_filename)
        if not tbl_info:
            return ""
        tbl_path = tbl_info.get("path", "")
        if not tbl_path or not Path(tbl_path).exists():
            return ""

        display_name = tbl_info.get("name", tbl_filename.replace("_", " ").rsplit(".", 1)[0].title())
        csv_url = url_quote(tbl_filename)

        # Read preview data (up to 10 rows) from CSV
        columns = tbl_info.get("columns", [])
        total_rows = tbl_info.get("total_rows", 0)
        preview_data = []
        try:
            import pandas as pd
            df_preview = pd.read_csv(tbl_path, nrows=10)
            columns = list(df_preview.columns)
            preview_data = df_preview.to_dict('records')
            if not total_rows:
                total_rows = sum(1 for _ in open(tbl_path)) - 1
        except Exception:
            logger.debug("Failed to read CSV preview for HTML table: %s", tbl_path)

        if not columns:
            return ""

        table_header = "".join(
            f"<th>{html_module.escape(str(col))}</th>" for col in columns
        )
        table_rows = ""
        for row in preview_data[:10]:
            row_cells = "".join(
                f"<td>{html_module.escape(str(row.get(col, '-')))}</td>" for col in columns
            )
            table_rows += f"<tr>{row_cells}</tr>"

        row_info = ""
        if total_rows > 10:
            row_info = f'<div class="table-info">Showing 10 of {total_rows} rows</div>'

        caption_html = ""
        if caption:
            caption_html = (
                f'<div class="table-interpretation">'
                f'<strong>Interpretation</strong>'
                f'<p>{_maybe_escape_html(caption)}</p>'
                f'</div>'
            )

        return f'''
            <div class="table-container">
                <div class="table-header">
                    <h4>{html_module.escape(display_name)}</h4>
                    <a href="tables/{csv_url}" download class="download-btn csv">
                        Download CSV
                    </a>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead><tr>{table_header}</tr></thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
                {row_info}
                {caption_html}
            </div>
        '''

    return {
        "report": report, "sections": sections, "abstract": abstract,
        "conclusions": conclusions, "report_figures": report_figures,
        "figure_map": figure_map, "stats": stats, "language": language,
        "output_path": output_path, "workspace_path": workspace_path,
        "analysis_state": analysis_state, "html_module": html_module,
        "toc_items": toc_items, "figure_b64_cache": figure_b64_cache,
        "validated_pdf_for": _validated_pdf_for, "build_figure_html": _build_figure_html,
        "placed_figures": placed_figures, "table_lookup": table_lookup,
        "placed_tables": placed_tables, "build_table_html": _build_table_html,
    }

def render_html_sections(self, ctx):
    sections = ctx["sections"]
    html_module = ctx["html_module"]
    figure_b64_cache = ctx["figure_b64_cache"]
    placed_figures = ctx["placed_figures"]
    _build_figure_html = ctx["build_figure_html"]
    table_lookup = ctx["table_lookup"]
    placed_tables = ctx["placed_tables"]
    _build_table_html = ctx["build_table_html"]
    # Build sections HTML with inline figure embedding
    sections_html = []
    for sec in sections:
        raw_content = sec.get("content", "")
        content_html = self._markdown_to_html(raw_content)

        # Replace [FIGURE: filename] + optional [FIGURE_CAPTION: ...] markers
        def _replace_figure_marker(m):
            fname = m.group(1).strip()
            caption = (m.group(2) or "").strip() if m.lastindex >= 2 else ""
            # A figure embeds once. Repeat markers were previously honoured,
            # so a filename cited three times was inlined three times — twice
            # of it back to back in the same section.
            if fname in placed_figures:
                logger.info(f"Figure already embedded, dropping repeat reference: {fname}")
                return ""
            fig_html = _build_figure_html(fname, caption=caption)
            if fig_html:
                placed_figures.add(fname)
                return fig_html
            logger.info(f"Figure referenced in report but not found, skipping: {fname}")
            return ""

        # Match [FIGURE: name] optionally followed by [FIGURE_CAPTION: text]
        content_html = re.sub(
            r'\[FIGURE:\s*([^\]]+)\](?:\s*(?:<[^>]*>\s*)*\[FIGURE_CAPTION:\s*([^\]]+)\])?',
            _replace_figure_marker,
            content_html,
        )

        # Replace [TABLE: filename] + optional [TABLE_CAPTION: ...] markers
        def _replace_table_marker(m):
            fname = m.group(1).strip()
            caption = (m.group(2) or "").strip() if m.lastindex >= 2 else ""
            if fname in placed_tables:
                logger.info(f"Table already embedded, dropping repeat reference: {fname}")
                return ""
            tbl_html = _build_table_html(fname, caption=caption)
            if tbl_html:
                placed_tables.add(fname)
                return tbl_html
            logger.info(f"Table referenced in report but not found, skipping: {fname}")
            return ""

        # Match [TABLE: name] optionally followed by [TABLE_CAPTION: text]
        content_html = re.sub(
            r'\[TABLE:\s*([^\]]+)\](?:\s*(?:<[^>]*>\s*)*\[TABLE_CAPTION:\s*([^\]]+)\])?',
            _replace_table_marker,
            content_html,
        )

        # Fallback: embed figures referenced by filename in text but without markers
        remaining_fig_parts = []
        for fig_name in figure_b64_cache:
            if fig_name in placed_figures:
                continue
            if fig_name.lower() in raw_content.lower():
                fig_html = _build_figure_html(fig_name)
                if fig_html:
                    placed_figures.add(fig_name)
                    remaining_fig_parts.append(fig_html)

        # Fallback: embed tables referenced by filename in text but without markers
        remaining_tbl_parts = []
        for tbl_filename in table_lookup:
            if tbl_filename in placed_tables:
                continue
            if tbl_filename.lower() in raw_content.lower():
                tbl_html = _build_table_html(tbl_filename)
                if tbl_html:
                    placed_tables.add(tbl_filename)
                    remaining_tbl_parts.append(tbl_html)

        a_type = sec.get("analysis_type", "general")
        type_class = "general" if a_type == "general" else "specific"
        type_label = "General" if a_type == "general" else "Specific"

        sec_html = f'''
        <div class="section" id="{html_module.escape(sec["id"])}">
            <div class="section-header">
                <h2>
                    {html_module.escape(sec["title"])}
                    <span class="analysis-type-badge {type_class}">{type_label}</span>
                </h2>
            </div>
            <div class="section-content">
                {content_html}
                {"".join(remaining_fig_parts)}
                {"".join(remaining_tbl_parts)}
            </div>
        </div>
        '''
        sections_html.append(sec_html)

    # Append any unplaced figures as a supplementary section
    unplaced = [fn for fn in figure_b64_cache if fn not in placed_figures and not fn.startswith("graphic_abstract.")]
    if unplaced:
        supp_title = "Supplementary Figures"
        supp_figs = "".join(_build_figure_html(fn) for fn in unplaced)
        sections_html.append(f'''
        <div class="section" id="supplementary_figures">
            <div class="section-header">
                <h2>{supp_title}</h2>
            </div>
            <div class="section-content">
                {supp_figs}
            </div>
        </div>
        ''')

    # Append any unplaced tables as a supplementary section
    unplaced_tables = [fn for fn in table_lookup if fn not in placed_tables]
    if unplaced_tables:
        supp_tbl_title = "Supplementary Data Tables"
        supp_tbl_parts = []
        for fn in unplaced_tables:
            tbl_html = _build_table_html(fn)
            if tbl_html:
                supp_tbl_parts.append(tbl_html)
        if supp_tbl_parts:
            sections_html.append(f'''
            <div class="section" id="supplementary_tables">
                <div class="section-header">
                    <h2>{supp_tbl_title}</h2>
                </div>
                <div class="section-content">
                    <p>The following data tables were generated during analysis and are provided for download and reference.</p>
                    {"".join(supp_tbl_parts)}
                </div>
            </div>
            ''')

    ctx["sections_html"] = sections_html

def persist_html_render(self, ctx):
    report = ctx["report"]
    analysis_state = ctx["analysis_state"]
    html_module = ctx["html_module"]
    toc_items = ctx["toc_items"]
    abstract = ctx["abstract"]
    conclusions = ctx["conclusions"]
    sections_html = ctx["sections_html"]
    output_path = ctx["output_path"]
    figure_map = ctx["figure_map"]
    _validated_pdf_for = ctx["validated_pdf_for"]
    title = report.get("title", "Analysis Report")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Graphic abstract HTML (derived from analysis_state, embedded as <img>)
    _ga_obj = getattr(analysis_state, "graphic_abstract", None) if analysis_state else None
    ga_b64 = (getattr(_ga_obj, "image_base64", "") or "") if _ga_obj else ""
    ga_html = ""
    if ga_b64:
        ga_html = (
            '<section id="graphic-abstract"><h2>Graphic Abstract</h2>'
            f'<figure><img src="data:image/jpeg;base64,{ga_b64}" '
            'alt="Graphic Abstract" style="max-width:100%"/></figure></section>'
        )

    html_content = f'''<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html_module.escape(title)}</title>
    <style>
    {_REPORT_CSS}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
    <h1>{html_module.escape(title)}</h1>
    <div class="meta">Generated: {generated_at}</div>
        </div>
        <div class="content">
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>{"".join(toc_items)}</ul>
    </div>

    <div class="abstract">
        <h2>Abstract</h2>
        {self._markdown_to_html(abstract)}
    </div>

    {render_report_status_line_html(report)}

    {ga_html}
    {"".join(sections_html)}

    <div class="conclusions">
        <h2>Conclusions</h2>
        {self._markdown_to_html(conclusions)}
    </div>

    {render_report_limitations_html(report)}

    {self._build_score_definitions_html(report)}

    {self._build_appendix_html_section(report)}
        </div>
    </div>
    </body>
    </html>'''

    html_file = output_path / "report.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Copy figures to output for download links
    figures_out = output_path / "figures"
    figures_out.mkdir(exist_ok=True)
    for fig_name, fig_path in figure_map.items():
        try:
            shutil.copy2(fig_path, figures_out / fig_name)
            # Only package the PDF whose validated path was carried by the
            # report entry. Never infer a companion from the raster stem.
            pdf_path = _validated_pdf_for(fig_name)
            if pdf_path is not None:
                shutil.copy2(pdf_path, figures_out / pdf_path.name)
        except Exception as e:
            logger.warning(f"Could not copy figure {fig_name}: {e}")

    # Copy tables to output for download links
    tables_out = output_path / "tables"
    tables_out.mkdir(exist_ok=True)
    for tbl in report.get("tables", []):
        tbl_path = tbl.get("path", "")
        if tbl_path and Path(tbl_path).exists():
            try:
                shutil.copy2(tbl_path, tables_out / os.path.basename(tbl_path))
            except Exception as e:
                logger.warning(f"Could not copy table {tbl_path}: {e}")

    # Score definitions CSV — backs the "Download CSV" link in the section.
    try:
        from ...services.report.score_section import write_score_definitions_csv
        write_score_definitions_csv(report.get("score_definitions", []), tables_out)
    except Exception as e:
        logger.warning(f"Could not write score_definitions.csv: {e}")

    _analysis_state = report.get("_analysis_state")

    # Combined literature research report (linked from the appendix).
    if _analysis_state and _analysis_state.literature_reports:
        try:
            from ...services.report.literature_files import write_literature_report
            write_literature_report(_analysis_state.literature_reports, output_path)
        except Exception as e:
            logger.warning(f"Could not write literature report: {e}")

    # Copy notebook to output for download
    if _analysis_state:
        from ...config import get_session_states
        from ...utils.notebook_media import export_notebook_with_inline_media
        notebook_src = Path(get_session_states(_analysis_state.id)) / f"{_analysis_state.id}.ipynb"
        if notebook_src.exists():
            try:
                export_notebook_with_inline_media(
                    notebook_src,
                    output_path / f"{_analysis_state.id}.ipynb",
                )
            except Exception as e:
                logger.warning(f"Could not copy notebook: {e}")

    logger.info(f"HTML report generated: {html_file}")
    return html_file
