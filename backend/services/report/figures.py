"""Figures mixin for DetailedReportGenerator."""
import os
import io
import base64
import hashlib
import re as _re
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image

from backend.services.report.artifacts import (
    artifact_stems,
    copy_verified_artifact,
    eligible_artifacts,
    excluded_artifact_stems,
    resolve_archival_pdf,
    resolve_archival_render,
)
from backend.services.report.figure_discovery import (
    discover_figures_in_workspace,
    prepare_discovered_figures,
    publication_figure_metadata,
)

logger = logging.getLogger(__name__)


class FiguresMixin:
    """Provides figure and table scanning/preparation methods for report generation."""

    # Priority order for figure types
    _FIGURE_TYPE_PRIORITY = {
        "annotation": 0, "umap": 1, "clustering": 2, "heatmap": 3,
        "volcano": 4, "dotplot": 5, "marker_genes": 6, "violin": 7,
        "barplot": 8, "bar_chart": 8, "feature_plot": 9, "pathway": 10,
        "trajectory": 11, "velocity": 12, "scatter": 13,
        "qc": 14, "overview": 15, "other": 16, "unknown": 16
    }

    def _load_image_for_vision(self, image_path: str, max_width: int = 600, quality: int = 70) -> Optional[str]:
        """
        Load and compress an image to raw base64 for chat_with_vision.
        Returns raw base64 string (no data: URI prefix), or None on error.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    channels = img.split()
                    if len(channels) >= 4:
                        background.paste(img, mask=channels[3])
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large
                width, height = img.size
                if width > max_width:
                    ratio = max_width / width
                    new_size = (max_width, int(height * ratio))
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS
                    img = img.resize(new_size, resample_method)

                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return None

    @staticmethod
    def _infer_figure_type(filename: str, declared_type: Optional[str] = None) -> str:
        """Infer figure type for a generated figure.

        P2-10: When the agent declares ``figure_type`` (via the
        ``generate_publication_figure`` tool, stored on
        ``PublicationFigure.figure_type``), that value is authoritative —
        filename heuristics only work for scanpy-named outputs and miss figures
        produced by matplotlib directly or via R/reticulate. The filename
        keyword scan is retained as a fallback for legacy / workspace-scanned
        figures that have no associated PublicationFigure record.
        """
        if declared_type:
            declared = str(declared_type).strip().lower()
            if declared:
                return declared
        name = filename.lower()
        if "celltype" in name or "cell_type" in name or "annotation" in name:
            return "annotation"
        elif "umap" in name:
            return "umap"
        elif "violin" in name:
            return "violin"
        elif "heatmap" in name:
            return "heatmap"
        elif "dotplot" in name or "dot_plot" in name:
            return "dotplot"
        elif "volcano" in name:
            return "volcano"
        elif "bar" in name or "composition" in name:
            return "bar_chart"
        elif "feature" in name or "expression" in name:
            return "feature_plot"
        elif "cluster" in name or "leiden" in name:
            return "clustering"
        elif "fold_change" in name:
            return "fold_change"
        elif "marker" in name:
            return "marker_genes"
        elif "qc" in name or "quality" in name:
            return "qc"
        elif "trajectory" in name or "dpt" in name or "pseudotime" in name:
            return "trajectory"
        elif "velocity" in name:
            return "velocity"
        elif "pathway" in name or "enrichment" in name:
            return "pathway"
        elif "scatter" in name:
            return "scatter"
        elif "overview" in name:
            return "overview"
        elif "zonation" in name:
            return "zonation"
        elif "stress" in name:
            return "stress_markers"
        return "unknown"

    def _scan_existing_figures(
        self,
        publication_figures: Optional[List[Any]] = None,
        analysis_log: Optional[List[Any]] = None,
        invalidated_stems: Optional[set[str]] = None,
        artifact_registry_version: Optional[int] = None,
        legacy_artifacts: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan the WORKSPACE directory for existing images from analysis phase.
        Copies figures to the report output directory for HTML embedding.
        Returns a list of figure info dicts with path, name, and inferred type.

        P2-10: when ``publication_figures`` is provided (typically
        ``analysis_state.publication_figures``), agent-declared ``figure_type``
        values are preferred over filename heuristics.
        """
        existing_figures = []
        seen_filenames = set()

        eligible_publication_figures = eligible_artifacts(publication_figures)
        excluded_publication_stems = excluded_artifact_stems(
            publication_figures,
            eligible_publication_figures,
            invalidated_stems=invalidated_stems,
        )
        metadata_by_stem = publication_figure_metadata(
            eligible_publication_figures, report_only=True,
        )
        records_by_stem: Dict[str, Any] = {}
        for figure in eligible_publication_figures:
            for stem in artifact_stems([figure]):
                records_by_stem.setdefault(stem, figure)

        # Build filename-stem -> declared figure_type map from eligible metadata.
        declared_type_by_stem: Dict[str, str] = {}
        if eligible_publication_figures:
            for pf in eligible_publication_figures:
                declared = getattr(pf, "figure_type", None)
                if not declared:
                    continue
                for stem in artifact_stems([pf]):
                    declared_type_by_stem[stem] = str(declared)

        if not self.workspace_dir or not self.workspace_dir.exists():
            return existing_figures

        # Registered records are the authoritative tier. A restored session may
        # have lost its old absolute path while retaining a valid record preview;
        # resolve/materialize that preview before the generic workspace scan so
        # this fallback generator obeys the same deliverability contract as the
        # Supervisor report path.
        for pf in eligible_publication_figures:
            source = resolve_archival_render(pf, self.workspace_dir)
            if source is None:
                continue
            stem = source.stem
            if stem in excluded_publication_stems or stem in seen_filenames:
                continue
            seen_filenames.add(stem)
            seen_filenames.update(artifact_stems([pf]))
            dest_png = self.figures_dir / source.name
            if not copy_verified_artifact(
                source, dest_png, pf, kind="image"
            ):
                continue

            pdf_file = resolve_archival_pdf(pf, self.workspace_dir)
            dest_pdf = self.figures_dir / pdf_file.name if pdf_file else None
            if pdf_file and not copy_verified_artifact(
                pdf_file, dest_pdf, pf, kind="pdf"
            ):
                dest_pdf = None

            existing_figures.append({
                "filename": source.name,
                "name": getattr(pf, "name", "") or stem.replace("_", " ").title(),
                "description": getattr(pf, "description", "") or "",
                "png_path": str(dest_png),
                "pdf_path": str(dest_pdf) if dest_pdf and dest_pdf.exists() else None,
                "inferred_type": self._infer_figure_type(
                    stem, declared_type=getattr(pf, "figure_type", None),
                ),
                "plan_step_id": getattr(pf, "plan_step_id", None),
                "plan_id": getattr(pf, "plan_id", None),
                "purpose": getattr(pf, "purpose", "result") or "result",
                "logical_figure_id": getattr(pf, "logical_figure_id", None),
                "render_attempt_id": getattr(pf, "render_attempt_id", None),
                "revision_of_id": getattr(pf, "revision_of_id", None),
                "supersedes_id": getattr(pf, "supersedes_id", None),
                "is_published_head": bool(getattr(pf, "is_published_head", False)),
                "review_status": str(
                    getattr(getattr(pf, "review_status", None), "value", getattr(pf, "review_status", ""))
                    or ""
                ),
                "review_unavailable": bool(getattr(pf, "review_unavailable", False)),
                "is_publication": True,
                "provenance_kind": "registered_publication",
                "is_legacy": False,
            })

        discovered = prepare_discovered_figures(
            discover_figures_in_workspace(self.workspace_dir),
            publication_figures,
            invalidated_stems=invalidated_stems,
            report_only=True,
            allow_unregistered_diagnostics=False,
            registry_version=artifact_registry_version,
            legacy_artifacts=legacy_artifacts,
        )
        admitted_by_stem = {
            Path(str(candidate.get("path") or "")).stem: candidate
            for candidate in discovered
            if candidate.get("path")
        }
        for candidate in discovered:
            png_file = Path(candidate["path"])
            if png_file.stem in excluded_publication_stems:
                continue
            if png_file.stem in seen_filenames:
                continue
            seen_filenames.add(png_file.stem)

            dest_png = self.figures_dir / png_file.name
            if not copy_verified_artifact(
                png_file, dest_png, candidate, kind="image"
            ):
                continue

            pdf_file = Path(candidate["pdf_path"]) if candidate.get("pdf_path") else None
            dest_pdf = self.figures_dir / pdf_file.name if pdf_file else None
            if pdf_file and not copy_verified_artifact(
                pdf_file, dest_pdf, candidate, kind="pdf"
            ):
                dest_pdf = None
            elif not pdf_file:
                dest_pdf = None

            existing_figures.append({
                "filename": png_file.name,
                "name": candidate.get("publication_name")
                        or png_file.stem.replace("_", " ").title(),
                "description": candidate.get("description") or "",
                "png_path": str(dest_png),
                "pdf_path": str(dest_pdf) if dest_pdf and dest_pdf.exists() else None,
                "inferred_type": self._infer_figure_type(
                    png_file.stem,
                    declared_type=(candidate.get("type")
                                   or declared_type_by_stem.get(png_file.stem)),
                ),
                "plan_step_id": candidate.get("plan_step_id"),
                "plan_id": candidate.get("plan_id"),
                "purpose": candidate.get("purpose") or "result",
                "logical_figure_id": candidate.get("logical_figure_id"),
                "render_attempt_id": candidate.get("render_attempt_id"),
                "revision_of_id": candidate.get("revision_of_id"),
                "supersedes_id": candidate.get("supersedes_id"),
                "is_published_head": bool(candidate.get("is_published_head")),
                "review_status": candidate.get("review_status") or "",
                "review_unavailable": bool(candidate.get("review_unavailable")),
                "is_publication": bool(candidate.get("is_publication", False)),
                "provenance_kind": candidate.get("provenance_kind") or "unregistered",
                "is_legacy": bool(candidate.get("is_legacy")),
                "legacy_artifact_id": candidate.get("legacy_artifact_id"),
            })

        if analysis_log:
            logged_basenames: List[str] = []
            seen_logged: set = set()
            for step in analysis_log:
                if isinstance(step, dict):
                    fnames = step.get("saved_figure_filenames") or []
                else:
                    fnames = getattr(step, "saved_figure_filenames", None) or []
                for fname in fnames:
                    base = os.path.basename(str(fname))
                    if os.path.splitext(base)[0] in excluded_publication_stems:
                        continue
                    if base and base not in seen_logged:
                        seen_logged.add(base)
                        logged_basenames.append(base)

            for base in logged_basenames:
                logged_stem = Path(base).stem
                if logged_stem in seen_filenames:
                    continue
                registered = records_by_stem.get(logged_stem)
                admitted = admitted_by_stem.get(logged_stem)
                if registered is not None:
                    src = resolve_archival_render(registered, self.workspace_dir)
                else:
                    # A log entry proves only that a filename was emitted. Its
                    # bytes must still be admitted by the same record/legacy
                    # inventory resolver used by the workspace scan above.
                    # In registry-v1 sessions ``admitted`` is therefore absent
                    # for every unregistered file.
                    src = (
                        Path(str(admitted["path"]))
                        if admitted is not None else None
                    )
                if src is None:
                    continue
                # A logged file in annotation_validation is archival only when
                # an eligible publication record vouches for it. The annotation
                # context report has a separate explicit diagnostic exception.
                try:
                    rel_parts = src.relative_to(self.workspace_dir).parts[:-1]
                except ValueError:
                    rel_parts = ()
                metadata = (
                    metadata_by_stem.get(logged_stem)
                    or metadata_by_stem.get(src.stem)
                )
                if "annotation_validation" in rel_parts and metadata is None:
                    continue
                if (
                    registered is None
                    and any("supplement" in part.lower() for part in rel_parts)
                ):
                    continue
                seen_filenames.add(src.stem)
                if registered is not None:
                    seen_filenames.update(artifact_stems([registered]))

                dest_png = self.figures_dir / src.name
                source_record = registered if registered is not None else admitted
                if not copy_verified_artifact(
                    src, dest_png, source_record, kind="image"
                ):
                    continue

                pdf_file = (
                    resolve_archival_pdf(registered, self.workspace_dir)
                    if registered is not None else src.with_suffix(".pdf")
                )
                dest_pdf = self.figures_dir / pdf_file.name if pdf_file else None
                if pdf_file and not copy_verified_artifact(
                    pdf_file, dest_pdf, source_record, kind="pdf"
                ):
                    dest_pdf = None

                existing_figures.append({
                    "filename": src.name,
                    "name": (metadata or {}).get("publication_name")
                            or src.stem.replace("_", " ").title(),
                    "description": (metadata or {}).get("description") or "",
                    "png_path": str(dest_png),
                    "pdf_path": str(dest_pdf) if dest_pdf and dest_pdf.exists() else None,
                    "inferred_type": self._infer_figure_type(
                        src.stem,
                        declared_type=(metadata or {}).get("type")
                                      or declared_type_by_stem.get(src.stem),
                    ),
                    "plan_step_id": (metadata or {}).get("plan_step_id"),
                    "plan_id": (metadata or {}).get("plan_id"),
                    "purpose": (metadata or {}).get("purpose") or "result",
                    "review_status": (metadata or {}).get("review_status") or "",
                    "review_unavailable": bool((metadata or {}).get("review_unavailable")),
                    "is_publication": registered is not None,
                    "provenance_kind": (
                        (admitted or {}).get("provenance_kind")
                        or ("registered_publication" if registered is not None else "")
                    ),
                    "is_legacy": bool((admitted or {}).get("is_legacy")),
                    "legacy_artifact_id": (admitted or {}).get(
                        "legacy_artifact_id"
                    ),
                })

        return existing_figures

    def _scan_existing_tables(
        self,
        publication_tables: Optional[List[Any]] = None,
        invalidated_stems: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Collect registered, semantically confirmed CSV tables for the report.
        Copies admitted tables to the report output directory.
        Returns a list of table info dicts.
        """
        existing_tables = []

        if not self.workspace_dir or not self.workspace_dir.exists():
            return existing_tables

        candidates = []
        for record in list(eligible_artifacts(publication_tables)):
            csv_file = Path(str(getattr(record, "csv_path", "") or ""))
            if not csv_file.exists():
                for candidate in (
                    self.workspace_dir / "tables" / csv_file.name,
                    self.workspace_dir / csv_file.name,
                ):
                    if candidate.exists():
                        csv_file = candidate
                        break
            if not csv_file.exists():
                continue
            if csv_file.stem in (invalidated_stems or set()):
                continue
            candidates.append((record, csv_file))

        basename_counts: Dict[str, int] = {}
        for _record, csv_file in candidates:
            basename_counts[csv_file.name] = basename_counts.get(csv_file.name, 0) + 1

        for record, csv_file in candidates:
            source_basename = csv_file.name
            destination_basename = source_basename
            if basename_counts[source_basename] > 1:
                table_id = str(getattr(record, "id", None) or "").strip()
                identity = table_id or str(
                    getattr(record, "csv_sha256", None) or csv_file.resolve()
                )
                token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
                destination_basename = (
                    f"{csv_file.stem}--{token}{csv_file.suffix}"
                )
            dest_csv = self.tables_dir / destination_basename
            if not copy_verified_artifact(
                csv_file, dest_csv, record, kind="table"
            ):
                continue
            tbl_info = {
                "filename": destination_basename,
                "name": getattr(record, "name", "") or csv_file.stem,
                "description": getattr(record, "description", "") or "",
                "csv_path": str(dest_csv),
                "source_filename": source_basename,
                "csv_sha256": getattr(record, "csv_sha256", None),
                "columns": list(getattr(record, "columns", None) or []),
                "preview_data": list(
                    getattr(record, "preview_data", None) or []
                ),
                "total_rows": int(getattr(record, "total_rows", 0) or 0),
                "interpretation": getattr(record, "interpretation", None),
                "table_id": getattr(record, "id", None),
                "plan_id": getattr(record, "plan_id", None),
                "plan_step_ids": list(
                    getattr(record, "plan_step_ids", None) or []
                ),
                "inferred_type": "registered_publication",
            }
            existing_tables.append(tbl_info)

        return existing_tables

    def _prepare_figure_images(
        self,
        existing_figures: List[Dict[str, Any]],
        max_figures: int = 20
    ) -> Dict[str, str]:
        """
        Load and compress figure images into raw base64 for vision API.
        Prioritizes by figure type importance.
        Stores results in self.figure_images_cache for downstream reuse.
        """
        # Sort by priority
        sorted_figures = sorted(
            existing_figures,
            key=lambda f: self._FIGURE_TYPE_PRIORITY.get(f.get("inferred_type", "unknown"), 99)
        )

        # Load up to max_figures
        cache: Dict[str, str] = {}
        for fig in sorted_figures[:max_figures]:
            png_path = fig.get("png_path", "")
            filename = fig.get("filename", "")
            if not png_path or not filename:
                continue
            if not os.path.exists(png_path):
                continue

            img_b64 = self._load_image_for_vision(png_path)
            if img_b64:
                cache[filename] = img_b64

        self.figure_images_cache = cache
        return cache

    def _prepare_table_summaries(
        self,
        existing_tables: List[Dict[str, Any]],
        max_rows: int = 30,
        max_tables: int = 10,
        analysis_state: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Read CSV tables and produce text summaries for PI context.
        Returns dict mapping filename -> formatted text summary.
        """
        import pandas as pd

        # P2-11: Read DEG thresholds from session-configurable analysis_state.
        deg_thresholds = getattr(analysis_state, "deg_thresholds", None) or {}
        logfc_cutoff = float(deg_thresholds.get("logfc", 1.0))
        padj_cutoff = float(deg_thresholds.get("padj", 0.05))

        summaries: Dict[str, str] = {}

        for tbl in existing_tables[:max_tables]:
            csv_path = tbl.get("csv_path", "")
            filename = tbl.get("filename", "")
            if not csv_path or not os.path.exists(csv_path):
                continue

            try:
                is_de = tbl.get("inferred_type") == "differential_expression" or filename.startswith("de_")
                # Read full file for DE tables (need sig gene counts); limit rows for others
                df = pd.read_csv(csv_path) if is_de else pd.read_csv(csv_path, nrows=max_rows)
                parts = [
                    f"Table: {filename}",
                    f"Shape: {df.shape[0]}{'+'  if not is_de and df.shape[0] == max_rows else ''} rows x {df.shape[1]} columns",
                    f"Columns: {', '.join(df.columns.tolist()[:20])}",
                ]

                # For DE tables, count significant genes
                if is_de:
                    fc_col = next((c for c in df.columns if "log" in c.lower() and "fold" in c.lower()), None)
                    pval_col = next((c for c in df.columns if "padj" in c.lower() or "p_val_adj" in c.lower() or "fdr" in c.lower()), None)
                    if fc_col and pval_col:
                        try:
                            sig = df[(df[fc_col].abs() > logfc_cutoff) & (df[pval_col] < padj_cutoff)]
                            parts.append(f"Significant genes (|logFC|>{logfc_cutoff}, padj<{padj_cutoff}): {len(sig)}")
                            up = sig[sig[fc_col] > 0]
                            down = sig[sig[fc_col] < 0]
                            parts.append(f"  Upregulated: {len(up)}, Downregulated: {len(down)}")
                        except Exception:
                            pass

                # Add data preview
                preview = df.head(max_rows).to_string(max_colwidth=40)
                parts.append(f"\nData preview (first {min(max_rows, len(df))} rows):\n{preview}")

                # Basic stats for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    stats = df[numeric_cols[:8]].describe().to_string()
                    parts.append(f"\nNumeric column statistics:\n{stats}")

                summary = "\n".join(parts)
                # Cap at ~2000 chars
                if len(summary) > 2000:
                    summary = summary[:2000] + "\n... (truncated)"
                summaries[filename] = summary
            except Exception as e:
                pass

        return summaries

    def _extract_cluster_annotations_from_state(self, state: "AnalysisState"):
        """
        Extract cluster annotations from state data, using multiple sources in priority order:
        1. annotation_state (confirmed by user — highest quality)
        2. analysis_log annotate_clusters steps
        3. bioinformatics findings (regex fallback)
        """
        # Source 1: annotation_state (confirmed annotations from human-in-the-loop dialog)
        if state.annotation_state and state.annotation_state.cluster_annotations:
            for ca in state.annotation_state.cluster_annotations:
                cell_type = ca.user_override or ca.cell_type
                self.cluster_annotations[ca.cluster_id] = {
                    "cell_type": cell_type,
                    "n_cells": getattr(ca, "n_cells", 0) or 0,
                    "confidence": ca.confidence,
                    "key_markers": ca.key_markers if ca.key_markers else [],
                    "evidence": ca.evidence or "",
                }
            logger.info(f"Extracted {len(self.cluster_annotations)} cluster annotations from annotation_state")
            return

        # Source 2: analysis_log annotate_clusters steps
        for step in state.analysis_log:
            if step.step_type == "annotate_clusters" and step.annotation_result_summary:
                for m in _re.finditer(
                    r'[Cc]luster\s+(\S+)\s*[:\-–]\s*(.+?)(?:\s*\(confidence:\s*(\w+)\)|\s*$)',
                    step.annotation_result_summary,
                ):
                    cid, cell_type = m.group(1), m.group(2).strip().rstrip('.,;')
                    confidence = m.group(3) or "medium"
                    if cid not in self.cluster_annotations:
                        self.cluster_annotations[cid] = {
                            "cell_type": cell_type,
                            "n_cells": 0,
                            "confidence": confidence,
                        }
        if self.cluster_annotations:
            logger.info(f"Extracted {len(self.cluster_annotations)} cluster annotations from analysis_log")
            return

        # Source 3: bioinformatics findings (regex fallback)
        for r in state.round_results:
            for finding in (r.bioinformatics_findings or []):
                for m in _re.finditer(r'[Cc]luster\s+(\d+)\s*[:\-–]\s*(\S.+?)(?:\s*\(|,|\.|;|$)', finding):
                    cid, cell_type = m.group(1), m.group(2).strip().rstrip('.,;')
                    if cid not in self.cluster_annotations:
                        self.cluster_annotations[cid] = {
                            "cell_type": cell_type,
                            "n_cells": 0,
                        }

        logger.info(f"Cluster annotations extracted from state: {len(self.cluster_annotations)} clusters")
