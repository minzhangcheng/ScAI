"""Staged detailed-report generation orchestration."""
from __future__ import annotations

from typing import Any, Dict

from backend.models.schemas import AnalysisState


async def generate_report_staged(
    generator,
    state: AnalysisState,
    resume_from_checkpoint: bool = True,
    force_fallback: bool = False,
    *,
    bindings,
) -> Dict[str, Any]:
    """
    Generate a detailed research report by compiling existing analysis results.

    Supports checkpoint-based resumption: if a previous attempt was interrupted
    (e.g. API rate limit), the method resumes from the last successfully
    completed step instead of starting over.

    NO code execution or Jupyter sandbox — the report is built entirely from:
    - state.code_history (what analyses were done)
    - state.round_results (what was found)
    - state.publication_figures (PI-reviewed figures)
    - Workspace files (figures and tables generated during analysis)

    Args:
        state: Analysis state with results
        resume_from_checkpoint: Whether to attempt resuming from a saved checkpoint

    Returns:
        Detailed report structure
    """
    # Initialize checkpoint manager for report generation
    checkpoint_mgr = bindings.CheckpointManager(
        session_id=state.id,
        state_dir=str(bindings.get_session_states(state.id)),
        prefix="report"
    )

    # Try loading existing checkpoint
    checkpoint = None
    last_step = 0
    if resume_from_checkpoint:
        checkpoint = await checkpoint_mgr.load()
        if checkpoint and not checkpoint.get('completed', False):
            last_step = checkpoint.get('last_step', 0)
            bindings.logger.info(f"Resuming report generation from step {last_step}")
        else:
            checkpoint = None

    report_id = checkpoint.get('report_id', str(bindings.uuid.uuid4())) if checkpoint else str(bindings.uuid.uuid4())
    # Force English output for all reports
    language = "en"

    # Filter round_results to target lineage for branch isolation
    report_state = bindings.build_lineage_report_state(
        state,
        generator.target_lineage_id,
        workspace_path=generator.workspace_dir,
    )

    await generator._report_progress(2, "Initializing report compilation...")

    # Step 1: Build structured analysis record from state data
    # (Pure data extraction, fast — always re-run to pick up latest state)
    await generator._report_progress(5, "Reading analysis records...")
    analysis_record = generator._build_analysis_record(report_state)
    bindings.logger.info(
        f"Analysis record: {analysis_record['total_code_cells']} code cells, "
        f"{len(analysis_record['analysis_steps'])} steps with figures, "
        f"{analysis_record['total_rounds']} rounds"
    )

    # Step 2: Scan workspace for existing figures and tables, copy to report dir
    await generator._report_progress(10, "Collecting analysis figures and tables...")
    existing_figures = generator._scan_existing_figures(
        publication_figures=getattr(report_state, "publication_figures", None),
        analysis_log=getattr(report_state, "analysis_log", None),
        invalidated_stems=set(
            getattr(report_state, "invalidated_artifact_stems", None) or []
        ),
        artifact_registry_version=getattr(
            report_state, "artifact_registry_version", None
        ),
        legacy_artifacts=getattr(report_state, "legacy_artifacts", None),
    )
    existing_tables = generator._scan_existing_tables(
        publication_tables=getattr(report_state, "publication_tables", None),
        invalidated_stems=bindings.excluded_artifact_stems(
            getattr(report_state, "publication_tables", None),
            bindings.eligible_artifacts(getattr(report_state, "publication_tables", None)),
            invalidated_stems=getattr(
                report_state, "invalidated_artifact_stems", None
            ),
        )
    )
    if checkpoint:
        checkpoint = bindings.rebind_checkpoint_asset_paths(
            checkpoint,
            figures_dir=generator.figures_dir,
            tables_dir=generator.tables_dir,
        )
    bindings.logger.info(f"Collected {len(existing_figures)} figures and {len(existing_tables)} tables for report")

    # Step 3: Extract findings from round results
    await generator._report_progress(15, "Extracting analysis findings...")
    findings = generator._extract_findings(report_state)

    # Extract cluster annotations from state data (no sandbox needed)
    generator._extract_cluster_annotations_from_state(report_state)

    # Step 3.5: PI reviews actual analysis code, figures, and tables (expensive LLM call)
    if force_fallback:
        analysis_summary = generator._empty_analysis_summary()
        analysis_summary["methodology_overview"] = (
            "This report was assembled deterministically from the "
            "retained analysis state after the report-writing model became "
            "unavailable."
        )
    elif last_step >= 4 and checkpoint:
        analysis_summary = checkpoint.get('analysis_summary', {})
        bindings.logger.info("Restored analysis_summary from checkpoint")
    else:
        await generator._report_progress(16, "PI Agent reviewing analysis code and figures...")
        analysis_summary = await generator._pi_review_analysis(
            report_state, analysis_record, findings, existing_figures, existing_tables, language
        )
        bindings.logger.info(f"Analysis review complete: methodology={bool(analysis_summary.get('methodology_overview'))}, "
                     f"workflow={bool(analysis_summary.get('workflow_summary'))}, "
                     f"results={bool(analysis_summary.get('key_results'))}")
        # Checkpoint after expensive LLM call
        await checkpoint_mgr.save({
            'report_id': report_id, 'last_step': 4, 'step_name': 'pi_review_analysis',
            'analysis_summary': analysis_summary, 'language': language,
        })

    # Step 4: PI plans report structure based on analysis record + analysis summary (expensive LLM call)
    if force_fallback:
        report_plan = generator._create_fallback_plan_with_existing_figures(
            report_state, existing_figures
        )
    elif last_step >= 5 and checkpoint:
        report_plan = checkpoint.get('report_plan', {})
        bindings.logger.info("Restored report_plan from checkpoint")
    else:
        await generator._report_progress(30, "PI Agent planning report structure...")
        report_plan = await generator._plan_report_from_analysis(
            report_state, analysis_record, findings, existing_figures, existing_tables, language,
            analysis_summary=analysis_summary
        )
        await checkpoint_mgr.save({
            'report_id': report_id, 'last_step': 5, 'step_name': 'plan_report',
            'analysis_summary': analysis_summary, 'report_plan': report_plan,
            'language': language,
        })

    # Reconcile even a restored plan against the figures scanned for this
    # run. This closes the checkpoint path where an older plan predates a
    # newly recovered archival figure.
    report_plan = bindings.ensure_report_plan_figure_coverage(
        report_plan, existing_figures
    )

    sections, abstract, conclusions = await _generate_sections(
        generator,
        state=state,
        report_state=report_state,
        checkpoint=checkpoint,
        last_step=last_step,
        checkpoint_mgr=checkpoint_mgr,
        report_id=report_id,
        report_plan=report_plan,
        findings=findings,
        language=language,
        analysis_summary=analysis_summary,
        force_fallback=force_fallback,
        bindings=bindings,
    )
    return await _assemble_report(
        generator,
        state=state,
        report_state=report_state,
        report_id=report_id,
        report_plan=report_plan,
        sections=sections,
        abstract=abstract,
        conclusions=conclusions,
        existing_figures=existing_figures,
        language=language,
        force_fallback=force_fallback,
        bindings=bindings,
    )


async def _generate_sections(
    generator,
    *,
    state,
    report_state,
    checkpoint,
    last_step,
    checkpoint_mgr,
    report_id,
    report_plan,
    findings,
    language,
    analysis_summary,
    force_fallback,
    bindings,
):
    # Step 5: Generate each section using existing figures + PI narrative (with vision + analysis context)
    # Restore already-completed sections from checkpoint
    sections = checkpoint.get('sections', []) if checkpoint and last_step >= 6 else []
    completed_section_count = len(sections)
    total_sections = len(report_plan.get("sections", []))

    # Restore generated_figures/tables/cluster_annotations from checkpoint BEFORE
    # the section loop, so that new sections append to the already-recovered list
    # and per-section checkpoint saves contain the full cumulative data.
    if checkpoint and last_step >= 6:
        saved_figs = checkpoint.get('generated_figures', [])
        saved_tbls = checkpoint.get('generated_tables', [])
        if saved_figs:
            generator.generated_figures = saved_figs
        if saved_tbls:
            generator.generated_tables = saved_tbls
        saved_ca = checkpoint.get('cluster_annotations', {})
        if saved_ca:
            generator.cluster_annotations = saved_ca

    # Repopulate figure_images_cache when resuming from step 4+
    # (since _pi_review_analysis is skipped on resume, the cache stays empty)
    if last_step >= 4 and checkpoint and not generator.figure_images_cache:
        bindings.logger.info("Repopulating figure_images_cache after checkpoint restore")
        generator._prepare_figure_images(existing_figures)

    for i, section_plan in enumerate(report_plan.get("sections", [])):
        # Skip sections already completed in checkpoint
        if i < completed_section_count:
            continue

        progress = 35 + (i / max(total_sections, 1)) * 45
        section_title = section_plan.get("title", f"Section {i+1}")
        await generator._report_progress(
            progress,
            f"Writing: {section_title}"
        )

        section = await generator._generate_section_from_existing_figures(
            report_state,
            section_plan,
            findings,
            language,
            analysis_summary=analysis_summary,
            force_fallback=force_fallback,
        )
        sections.append(section)

        # Checkpoint after each section (per-section granularity)
        await checkpoint_mgr.save({
            'report_id': report_id, 'last_step': 6, 'step_name': f'section_{i+1}',
            'analysis_summary': analysis_summary, 'report_plan': report_plan,
            'sections': sections, 'generated_figures': generator.generated_figures,
            'generated_tables': generator.generated_tables, 'language': language,
            'cluster_annotations': generator.cluster_annotations,
        })

    # Free figure image cache — no longer needed after section writing
    generator.figure_images_cache.clear()

    # Step 6: PI writes abstract and conclusions (expensive LLM calls)
    if force_fallback:
        abstract = await generator._generate_abstract(
            report_state,
            sections,
            findings,
            language,
            analysis_summary=analysis_summary,
            force_fallback=True,
        )
    elif last_step >= 7 and checkpoint:
        abstract = checkpoint.get('abstract', '')
        bindings.logger.info("Restored abstract from checkpoint")
    else:
        await generator._report_progress(85, "Writing abstract...")
        abstract = await generator._generate_abstract(
            report_state, sections, findings, language,
            analysis_summary=analysis_summary,
        )
        await checkpoint_mgr.save({
            'report_id': report_id, 'last_step': 7, 'step_name': 'abstract',
            'analysis_summary': analysis_summary, 'report_plan': report_plan,
            'sections': sections, 'generated_figures': generator.generated_figures,
            'generated_tables': generator.generated_tables, 'abstract': abstract,
            'language': language, 'cluster_annotations': generator.cluster_annotations,
        })

    if force_fallback:
        conclusions = await generator._generate_conclusions(
            report_state,
            sections,
            findings,
            language,
            analysis_summary=analysis_summary,
            force_fallback=True,
        )
    elif last_step >= 8 and checkpoint:
        conclusions = checkpoint.get('conclusions', '')
        bindings.logger.info("Restored conclusions from checkpoint")
    else:
        await generator._report_progress(92, "Writing conclusions...")
        conclusions = await generator._generate_conclusions(
            report_state, sections, findings, language,
            analysis_summary=analysis_summary,
        )
        await checkpoint_mgr.save({
            'report_id': report_id, 'last_step': 8, 'step_name': 'conclusions',
            'analysis_summary': analysis_summary, 'report_plan': report_plan,
            'sections': sections, 'generated_figures': generator.generated_figures,
            'generated_tables': generator.generated_tables, 'abstract': abstract,
            'conclusions': conclusions, 'language': language,
            'cluster_annotations': generator.cluster_annotations,
        })

    return sections, abstract, conclusions


async def _assemble_report(
    generator,
    *,
    state,
    report_state,
    report_id,
    report_plan,
    sections,
    abstract,
    conclusions,
    existing_figures,
    language,
    force_fallback,
    bindings,
):
    # Save graphic abstract image to figures dir if available
    graphic_abstract_path = None
    if report_state.graphic_abstract and report_state.graphic_abstract.image_base64:
        try:
            from ...utils.image_utils import save_graphic_abstract_image
            graphic_abstract_path = save_graphic_abstract_image(
                report_state.graphic_abstract.image_base64, generator.figures_dir
            )
            bindings.logger.info(f"Saved graphic abstract to {graphic_abstract_path}")
        except Exception:
            bindings.logger.exception("Failed to save graphic abstract")

    # Build appendix data for report.json (so regenerate can use it)
    record_files = bindings.write_interaction_record_files(report_state, reports_dir=generator.output_dir)
    from backend.services.report.download_section import build_related_reports_metadata
    related_reports = build_related_reports_metadata(
        report_state,
        record_files,
        reports_dir=generator.output_dir,
    )
    appendix_data = {
        "project_name": report_state.name,
        "project_description": report_state.initial_instruction,
        "created_at": report_state.created_at.isoformat() if hasattr(report_state.created_at, 'isoformat') else str(report_state.created_at),
        "scai_version": report_state.scai_version,
        "agent_models": bindings.build_agent_models_appendix(report_state),
        "agent_temperatures": {k: v for k, v in report_state.custom_temperatures.items()} if report_state.custom_temperatures else {},
        "treatment_groups": [
            {"name": g.name, "description": g.description, "sample_count": g.sample_count}
            for g in report_state.treatment_groups
        ],
        "data_files": [
            {
                "filename": f.filename,
                "section": f.section,
                "data_type": f.data_type,
                "group": (report_state.get_group_by_id(f.group_id).name if report_state.get_group_by_id(f.group_id) else None),
            }
            for f in report_state.data_files
        ],
        "biochemistry_indicators_count": len(report_state.biochemistry_indicators),
        "max_rounds": report_state.max_rounds,
        "cross_modal_synthesis": bindings.build_cross_modal_appendix(report_state),
        "round_results": [
            {
                "round_number": r.round_number,
                "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, 'isoformat') else str(r.timestamp),
                "pathology_findings_count": len(r.pathology_findings),
                "bioinformatics_findings_count": len(r.bioinformatics_findings),
                "biochemistry_findings_count": len(r.biochemistry_findings),
                "literature_findings_count": len(r.literature_findings),
            }
            for r in report_state.round_results
        ],
        "human_messages": [
            {
                "message": hm.get("message", ""),
                "files": [f.get("filename", f.get("name", "file")) for f in hm.get("files", [])],
                "timestamp": hm.get("timestamp", ""),
                "round_number": hm.get("round_number", "?"),
            }
            for hm in report_state.human_messages
        ],
        "user_interactions": {
            "cluster_annotations": [
                {
                    "cluster_id": ca.cluster_id,
                    "cell_type": ca.cell_type,
                    "user_override": ca.user_override or "",
                    "confirmed": ca.user_confirmed,
                    "confidence": ca.confidence,
                }
                for ca in report_state.annotation_state.cluster_annotations
            ] if report_state.annotation_state and report_state.annotation_state.cluster_annotations else [],
            "direction_selection": bindings.build_direction_selection_history(report_state),
        },
        "related_reports": related_reports,
        "notebook_filename": f"{report_state.id}.ipynb",
        "literature_report": bindings.LITERATURE_REPORT_FILENAME if report_state.literature_reports else "",
        "branch_id": None if generator.target_lineage_id == "main" else generator.target_lineage_id,
        "branch_label": getattr(state, "restore_branch_label", None),
        "report_kind": "partial" if getattr(state, "allow_partial_report", False) else "complete",
    }

    unreported_figures = bindings.unreported_figure_names(
        existing_figures, generator.generated_figures
    )
    figureless_sections = bindings.figureless_outline_sections(report_state, sections)
    quality_disclosures = bindings.build_quality_disclosures(report_state)
    (
        report_status,
        report_current_step,
        partial_reason,
        partial_reasons,
    ) = bindings.build_report_partial_fields(
        unreported_figures=unreported_figures,
        figureless_sections=figureless_sections,
        approved_partial=bool(getattr(state, "allow_partial_report", False)),
        quality_disclosures=quality_disclosures,
        degraded_fallback=force_fallback,
    )
    appendix_data["report_kind"] = report_status
    appendix_data["partial_reason"] = partial_reason
    appendix_data["partial_reasons"] = partial_reasons
    report = {
        "id": report_id,
        "session_id": report_state.id,
        "title": report_plan.get("title") or report_state.name,
        "abstract": abstract,
        "sections": sections,
        "conclusions": conclusions,
        "figures": generator.generated_figures,
        "tables": generator.generated_tables,
        "graphic_abstract_path": graphic_abstract_path,
        "generated_at": bindings.datetime.now().isoformat(),
        "status": report_status,
        "progress": 100,
        "current_step": report_current_step,
        "partial_reason": partial_reason,
        "partial_reasons": partial_reasons,
        "figureless_sections": figureless_sections,
        "unreported_figures": unreported_figures,
        "quality_disclosures": quality_disclosures,
        "degraded_fallback": force_fallback,
        "appendix": appendix_data,
        "analysis_log": generator._build_full_analysis_log(report_state),
        "score_definitions": [sd.model_dump(mode="json") for sd in report_state.score_definitions] if report_state.score_definitions else [],
        "bulk_input_profiles": [item.model_dump(mode="json") for item in report_state.bulk_input_profiles],
        "method_decisions": [item.model_dump(mode="json") for item in report_state.method_decisions],
    }

    # Sanitize NaN/Infinity values before saving (pandas data can contain these)
    def _sanitize(obj):
        import math
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj
    report = _sanitize(report)

    # Copy notebook to report directory for download
    from backend.utils.live_paths import live_notebook_path
    from backend.utils.notebook_media import export_notebook_with_inline_media

    notebook_src = live_notebook_path(bindings.get_session_states(report_state.id), report_state.id)
    if notebook_src.exists():
        try:
            export_notebook_with_inline_media(
                notebook_src,
                generator.output_dir / f"{report_state.id}.ipynb",
            )
        except Exception as e:
            bindings.logger.warning(f"Could not copy notebook: {e}")

    # Write the combined literature research report for appendix download
    lit_path = bindings.write_literature_report(report_state.literature_reports, generator.output_dir)
    if lit_path is not None:
        bindings.logger.info(f"Literature report saved to: {lit_path}")

    report["reproducibility_path"] = generator._maybe_generate_reproducibility_bundle(
        state=report_state,
        notebook_src=notebook_src,
    )

    # Generate HTML report
    await generator._report_progress(95, "Generating HTML report...")
    html_path = await generator._generate_html_report(report, language, state=report_state)
    report["html_path"] = str(html_path)

    await generator._report_progress(
        100, report_current_step, report_status
    )

    return report
