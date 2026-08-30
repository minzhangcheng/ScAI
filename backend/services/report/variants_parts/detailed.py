"""Detailed-report segments and the combined detailed report.

Moved verbatim out of backend/services/report/variants.py.
"""
import os
import html as html_module
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from backend.utils.report_html import markdown_to_html, _build_figure_description_map
from backend.utils.image_utils import compress_image_to_base64
from backend.services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    excluded_artifact_stems,
)
from backend.services.report.figure_discovery import (
    discover_figures_in_workspace,
    ensure_legacy_artifact_inventory,
    prepare_discovered_figures,
)
from backend.models.schemas import AnalysisState
from .figure_records import resolved_record_figure_candidates

logger = __import__('logging').getLogger(__name__)


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _matching_round_record(records, round_number: int):
    for record in reversed(list(records or [])):
        if int(_record_value(record, "round_number", 0) or 0) == round_number:
            return record
    return None


def _technical_evidence_observations_html(
    state: "AnalysisState",
    round_result: Any,
) -> str:
    """Render non-blocking technical evidence facts for one completed round."""
    round_number = int(_record_value(round_result, "round_number", 0) or 0)
    verified = _record_value(
        round_result, "bioinformatics_evidence_verified", None
    )
    verification_label = {
        True: "verified",
        False: "not verified",
        None: "not recorded",
    }[verified]
    execution_ids = list(
        _record_value(round_result, "evidence_execution_ids", []) or []
    )
    notes = [
        str(value).strip()
        for value in (_record_value(round_result, "evidence_notes", []) or [])
        if str(value).strip()
    ]

    reconciliation = _matching_round_record(
        getattr(state, "plan_evidence_reconciliations", []), round_number
    )
    links = list(_record_value(reconciliation, "links", []) or [])
    link_counts = {
        status: sum(
            1
            for link in links
            if str(_record_value(link, "status", "")).lower() == status
        )
        for status in ("evidenced", "partial", "unverified")
    }

    parts = [
        '<div class="segment-section evidence-observations">',
        '<h3>Technical Evidence Observations</h3>',
        (
            '<p>These are technical execution and linkage observations for PI '
            'review. They do not accept or reject the scientific conclusions '
            'and do not block this report.</p>'
        ),
        '<ul>',
        (
            '<li><strong>Bioinformatics evidence verification:</strong> '
            f'{html_module.escape(verification_label)}</li>'
        ),
        (
            '<li><strong>Confirmed execution receipts:</strong> '
            f'{len(execution_ids)}</li>'
        ),
    ]
    if reconciliation is not None:
        status = str(_record_value(reconciliation, "status", "not recorded"))
        parts.append(
            '<li><strong>Plan/evidence reconciliation:</strong> '
            f'{html_module.escape(status)}; '
            f'{link_counts["evidenced"]} evidenced, '
            f'{link_counts["partial"]} partial, '
            f'{link_counts["unverified"]} unverified plan steps</li>'
        )
        reason = str(_record_value(reconciliation, "overall_reason", "") or "").strip()
        if reason:
            parts.append(
                '<li><strong>Reconciliation note:</strong> '
                f'{html_module.escape(reason)}</li>'
            )
    elif verified is False:
        parts.append(
            '<li><strong>Plan/evidence reconciliation:</strong> no retained '
            'round-level reconciliation record</li>'
        )
    for note in notes[:20]:
        parts.append(
            '<li><strong>Evidence note:</strong> '
            f'{html_module.escape(note)}</li>'
        )
    parts.extend(['</ul>', '</div>'])
    return "\n".join(parts)


def _completed_round_result(state: "AnalysisState", round_number: int):
    return _matching_round_record(getattr(state, "round_results", []), round_number)


def _with_historical_evidence_observations(
    state: "AnalysisState",
    segment: Any,
) -> str:
    """Add current evidence disclosure when combining older stored segments."""
    content = str(_record_value(segment, "content_html", "") or "")
    if not content or 'class="segment-section evidence-observations"' in content:
        return content
    phase = str(_record_value(segment, "phase", "") or "")
    if phase == "pre_annotation":
        return content
    round_number = int(_record_value(segment, "round_number", 0) or 0)
    round_result = _completed_round_result(state, round_number)
    if round_result is None:
        return content
    observation = _technical_evidence_observations_html(state, round_result)
    timestamp_end = content.find("</p>")
    if timestamp_end >= 0:
        insert_at = timestamp_end + len("</p>")
        return content[:insert_at] + "\n" + observation + content[insert_at:]
    return observation + "\n" + content


def generate_detailed_report_segment_html(
    state: "AnalysisState",
    phase: str,
    title: str,
    graph_state: Optional[Dict[str, Any]] = None,
    round_result=None,
) -> str:
    """Generate one segment of the incremental detailed report as HTML.

    Args:
        state: Current AnalysisState
        phase: "pre_annotation" or "exploration_complete" or "supplementary_*"
        title: Human-readable segment title
        graph_state: Current LangGraph WorkflowState (for in-progress results)
        round_result: Completed RoundResult (for post-round segments)

    Returns:
        HTML string for this segment
    """
    parts = []
    parts.append(f'<div class="report-segment" id="{phase}">')
    parts.append(f'<h2 class="segment-title">{html_module.escape(title)}</h2>')
    parts.append(f'<p class="segment-timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')

    try:
        exp_summary = state.build_experimental_design_summary()
        if exp_summary:
            parts.append('<div class="segment-section">')
            parts.append('<h3>Experimental Design</h3>')
            parts.append(f'<div class="content">{markdown_to_html(exp_summary)}</div>')
            parts.append('</div>')
    except Exception as e:
        logger.debug(f"Experimental design section failed: {e}")
        pass

    if round_result is not None:
        parts.append(_technical_evidence_observations_html(state, round_result))

    # --- Section: Agent Findings (from graph_state if available) ---
    if graph_state:
        for agent_key, agent_label in [
            ("biochemistry_result", "Biochemistry Findings"),
            ("pathology_result", "Pathology Findings"),
            ("bioinformatics_result", "Bioinformatics Findings"),
            ("literature_result", "Literature Research"),
        ]:
            result = graph_state.get(agent_key)
            if result and isinstance(result, dict):
                findings = result.get("findings", [])
                detail = result.get("analysis_detail") or result.get("summary", "")
                if findings or detail:
                    parts.append(f'<div class="segment-section"><h3>{agent_label}</h3>')
                    if findings:
                        parts.append('<ul>')
                        for f in findings:
                            parts.append(f'<li>{html_module.escape(str(f))}</li>')
                        parts.append('</ul>')
                    if detail:
                        parts.append(f'<div class="detail">{markdown_to_html(str(detail)[:3000])}</div>')
                    parts.append('</div>')

    segment_round = int(getattr(state, "current_round", 0) or 0)
    segment_steps = [
        step for step in (state.analysis_log or [])
        if int(getattr(step, "round_number", 0) or 0) in (0, segment_round)
    ]

    # --- Section: Analysis Log Summary ---
    if segment_steps:
        recent_steps = segment_steps[-50:]
        parts.append('<div class="segment-section"><h3>Analysis Steps Executed</h3>')
        parts.append('<table class="steps-table"><thead><tr><th>#</th><th>Step</th><th>Status</th><th>Details</th></tr></thead><tbody>')
        for i, step in enumerate(recent_steps, 1):
            # Derive status from code_error field (AnalysisStep has no status field)
            has_error = bool(step.code_error)
            status_label = "error" if has_error else "completed"
            status_class = "error" if has_error else "success"
            detail_text = ""
            if step.step_type == "search_literature" and step.literature_topic:
                detail_text = html_module.escape(step.literature_topic[:120])
            elif step.code:
                detail_text = html_module.escape(step.code[:100])
            parts.append(f'<tr><td>{i}</td><td>{html_module.escape(step.step_type)}</td>'
                         f'<td class="{status_class}">{html_module.escape(status_label)}</td>'
                         f'<td>{detail_text}</td></tr>')
        parts.append('</tbody></table></div>')

    try:
        from backend.config import get_session_workspace
        workspace_dir = get_session_workspace(state.id)
        if workspace_dir and workspace_dir.exists():
            ensure_legacy_artifact_inventory(state, workspace_dir)
            fig_desc_map = _build_figure_description_map(segment_steps)
            eligible_figures = eligible_artifacts(state.publication_figures)
            excluded_stems = excluded_artifact_stems(
                state.publication_figures,
                eligible_figures,
                invalidated_stems=state.invalidated_artifact_stems,
            )
            round_step_stems = {
                Path(fname).stem
                for step in segment_steps
                for fname in (step.saved_figure_filenames or [])
            }
            eligible_round_stems = artifact_stems(
                fig for fig in eligible_figures
                if int(getattr(fig, "round_number", 0) or 0) in (0, segment_round)
            )
            allowed_stems = round_step_stems | eligible_round_stems
            # Start from record-owned bytes so a reviewed figure remains
            # reportable after its original path moved or was overwritten.
            discovered = resolved_record_figure_candidates(
                state.publication_figures,
                workspace_dir,
                report_only=True,
                round_numbers={0, segment_round},
            )
            discovered.extend(prepare_discovered_figures(
                discover_figures_in_workspace(workspace_dir),
                state.publication_figures,
                invalidated_stems=state.invalidated_artifact_stems,
                report_only=True,
                allow_unregistered_diagnostics=False,
                registry_version=state.artifact_registry_version,
                legacy_artifacts=state.legacy_artifacts,
            ))
            if discovered:
                rendered_blocks = []
                rendered_keys = set()
                for fig_info in discovered:
                    try:
                        file_stem = fig_info.get("name", "")
                        figure_key = fig_info.get("record_stem") or file_stem
                        if (
                            figure_key in rendered_keys
                            or figure_key in excluded_stems
                            or figure_key not in allowed_stems
                        ):
                            continue
                        fig_path = fig_info.get("path", "")
                        if not fig_path:
                            continue
                        b64 = compress_image_to_base64(fig_path, max_width=1000, quality=82)
                        caption = (
                            fig_info.get("description")
                            or fig_desc_map.get(
                                figure_key, figure_key.replace("_", " ").title()
                            )
                        )
                        key = html_module.escape(str(figure_key), quote=True)
                        rendered_blocks.append(
                            f'<div class="figure-item" data-figure-key="{key}"><img src="{b64}" />'
                            f'<p class="caption">{html_module.escape(str(caption))}</p></div>'
                        )
                        rendered_keys.add(figure_key)
                    except Exception as e:
                        logger.debug(f"Figure embedding failed for {fig_info.get('name', '?')}: {e}")
                if rendered_blocks:
                    parts.append('<div class="segment-section"><h3>Generated Figures</h3>')
                    parts.append('<div class="figure-gallery">')
                    parts.extend(rendered_blocks)
                    parts.append('</div></div>')
    except Exception as e:
        logger.warning(f"Figure section generation failed: {e}")

    # --- Section: Round Result (if provided) ---
    if round_result:
        parts.append('<div class="segment-section"><h3>PI Synthesis</h3>')
        if hasattr(round_result, 'pi_summary') and round_result.pi_summary:
            parts.append(f'<div class="content">{markdown_to_html(round_result.pi_summary)}</div>')
        if hasattr(round_result, 'conflicts') and round_result.conflicts:
            parts.append('<h4>Identified Conflicts</h4><ul>')
            for c in round_result.conflicts:
                parts.append(f'<li>{html_module.escape(c.get("description", str(c)))}</li>')
            parts.append('</ul>')
        if hasattr(round_result, 'next_steps') and round_result.next_steps:
            parts.append('<h4>Recommended Next Steps</h4><ul>')
            for s in round_result.next_steps:
                parts.append(f'<li>{html_module.escape(str(s))}</li>')
            parts.append('</ul>')
        parts.append('</div>')

    # --- Section: Exploration Report (if available) ---
    if phase == "exploration_complete" and state.exploration_report:
        er = state.exploration_report
        parts.append('<div class="segment-section"><h3>Exploration Report Summary</h3>')
        if er.cluster_overview:
            parts.append(f'<p><strong>Cluster Overview:</strong> {html_module.escape(er.cluster_overview)}</p>')
        if er.data_quality_warnings:
            parts.append('<h4>Data Quality Warnings</h4><ul>')
            for w in er.data_quality_warnings:
                parts.append(f'<li class="warning">{html_module.escape(w)}</li>')
            parts.append('</ul>')
        if er.pi_direction_assessment:
            parts.append(f'<h4>PI Direction Assessment</h4>'
                         f'<div class="content">{markdown_to_html(er.pi_direction_assessment)}</div>')
        parts.append('</div>')

    parts.append('</div>')  # Close report-segment
    return "\n".join(parts)

def _deduplicate_detailed_figure_blocks(html: str) -> str:
    """Keep the first occurrence of each keyed detailed-report figure block."""
    seen_figure_keys: set[str] = set()
    figure_block_re = re.compile(
        r'<div class="figure-item" data-figure-key="([^"]+)">.*?</div>',
        re.DOTALL,
    )

    def _dedupe_figure(match: re.Match) -> str:
        key = html_module.unescape(match.group(1))
        if key in seen_figure_keys:
            return ""
        seen_figure_keys.add(key)
        return match.group(0)

    return figure_block_re.sub(_dedupe_figure, html)

def generate_combined_detailed_report_html(state: "AnalysisState") -> str:
    """Generate combined HTML from all detailed report segments.

    Returns:
        Path to the saved HTML file
    """
    from backend.config import get_session_reports
    reports_dir = Path(get_session_reports(state.id))
    reports_dir.mkdir(parents=True, exist_ok=True)

    segments_html = "\n<hr class='segment-divider'>\n".join(
        content
        for seg in state.detailed_report_segments
        if (content := _with_historical_evidence_observations(state, seg))
    )
    # A figure can be visible at several workflow gates. Keep its first segment
    # occurrence so the combined report does not repeat the same image gallery.
    segments_html = _deduplicate_detailed_figure_blocks(segments_html)

    toc_items = "".join(
        f'<li><a href="#{seg.phase}">{html_module.escape(seg.title)}'
        f' (Round {seg.round_number})</a></li>'
        for seg in state.detailed_report_segments
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Detailed Analysis Report - {html_module.escape(state.name)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f8fafc; color: #1e293b; }}
.report-header {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; }}
.report-header h1 {{ margin: 0 0 0.5rem; font-size: 1.5rem; }}
.toc {{ background: white; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.toc h2 {{ margin-top: 0; font-size: 1.1rem; }}
.toc ul {{ padding-left: 1.5rem; }}
.toc li {{ margin: 0.3rem 0; }}
.toc a {{ color: #6366f1; text-decoration: none; }}
.report-segment {{ background: white; padding: 2rem; border-radius: 0.75rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.segment-title {{ color: #6366f1; margin-top: 0; }}
.segment-timestamp {{ color: #94a3b8; font-size: 0.85rem; }}
.segment-section {{ margin: 1.5rem 0; padding: 1rem; background: #f1f5f9; border-radius: 0.5rem; }}
.segment-section h3 {{ color: #334155; margin-top: 0; font-size: 1rem; }}
.segment-section h4 {{ color: #475569; font-size: 0.9rem; }}
.steps-table {{ width: 100%; max-width: 100%; border-collapse: collapse; font-size: 0.85rem; display: block; overflow-x: auto; }}
.steps-table th, .steps-table td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #e2e8f0; text-align: left; }}
.steps-table th {{ background: #e2e8f0; }}
.success {{ color: #16a34a; }}
.error {{ color: #dc2626; }}
.warning {{ color: #d97706; }}
.figure-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(700px, 1fr)); gap: 28px; }}
.figure-item {{ border: 1px solid #e2e8f0; border-radius: 0.5rem; overflow: hidden; }}
.figure-item img {{ width: 100%; height: auto; display: block; }}
.caption {{ padding: 0.75rem 1rem; font-size: 0.88rem; color: #334155; line-height: 1.5; background: #f8fafc; }}
.segment-divider {{ border: none; border-top: 2px dashed #cbd5e1; margin: 2rem 0; }}
.detail {{ font-size: 0.9rem; color: #475569; }}
.content {{ line-height: 1.6; }}
</style>
</head>
<body>
<div class="report-header">
  <h1>Detailed Analysis Report</h1>
  <p>{html_module.escape(state.name)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
<div class="toc">
  <h2>Table of Contents</h2>
  <ul>{toc_items}</ul>
</div>
{segments_html}
</body>
</html>"""

    output_path = reports_dir / "detailed_report.html"
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(str(tmp_path), str(output_path))

    state.detailed_report_html_path = str(output_path)
    logger.info(f"Combined detailed report saved to: {output_path}")
    return str(output_path)
