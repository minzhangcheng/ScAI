"""Figures mixin for DetailedReportGenerator."""
import os
import io
import base64
import re as _re
import shutil
import logging
from typing import Dict, Any, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class FiguresMixin:
    """Provides figure and table scanning/preparation methods for report generation."""

    # Priority order for figure types
    _FIGURE_TYPE_PRIORITY = {
        "annotation": 0, "umap": 1, "clustering": 2, "heatmap": 3,
        "volcano": 4, "dotplot": 5, "marker_genes": 6, "violin": 7,
        "bar_chart": 8, "feature_plot": 9, "pathway": 10,
        "trajectory": 11, "velocity": 12, "scatter": 13,
        "qc": 14, "overview": 15, "unknown": 16
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
    def _infer_figure_type(filename: str) -> str:
        """Infer figure type from filename."""
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

    def _scan_existing_figures(self) -> List[Dict[str, Any]]:
        """
        Scan the WORKSPACE directory for existing images from analysis phase.
        Copies figures to the report output directory for HTML embedding.
        Returns a list of figure info dicts with path, name, and inferred type.
        """
        existing_figures = []
        seen_filenames = set()

        if not self.workspace_dir or not self.workspace_dir.exists():
            return existing_figures

        # Scan both workspace root and figures/ subdirectory
        scan_dirs = [self.workspace_dir]
        figures_subdir = self.workspace_dir / "figures"
        if figures_subdir.exists():
            scan_dirs.append(figures_subdir)

        for scan_dir in scan_dirs:
            # Scan for both JPG and PNG (JPG preferred, PNG for backward compatibility)
            image_files = sorted(scan_dir.glob("*.jpg")) + sorted(scan_dir.glob("*.png"))
            for png_file in image_files:
                if png_file.stem in seen_filenames:
                    continue  # Skip PNG if JPG already found for same stem
                seen_filenames.add(png_file.stem)

                # Copy to report figures directory for HTML embedding
                dest_png = self.figures_dir / png_file.name
                try:
                    shutil.copy2(str(png_file), str(dest_png))
                except Exception as e:
                    continue

                # Also copy PDF if it exists
                pdf_file = png_file.with_suffix('.pdf')
                dest_pdf = self.figures_dir / pdf_file.name
                if pdf_file.exists():
                    try:
                        shutil.copy2(str(pdf_file), str(dest_pdf))
                    except Exception:
                        pass

                fig_info = {
                    "filename": png_file.name,
                    "name": png_file.stem.replace("_", " ").title(),
                    "png_path": str(dest_png),
                    "pdf_path": str(dest_pdf) if pdf_file.exists() else None,
                    "inferred_type": self._infer_figure_type(png_file.stem)
                }
                existing_figures.append(fig_info)

        return existing_figures

    def _scan_existing_tables(self) -> List[Dict[str, Any]]:
        """
        Scan the workspace directory for existing CSV tables from analysis phase.
        Copies tables to the report output directory.
        Returns a list of table info dicts.
        """
        import pandas as pd

        existing_tables = []

        if not self.workspace_dir or not self.workspace_dir.exists():
            return existing_tables

        tables_subdir = self.workspace_dir / "tables"
        csv_search_dir = tables_subdir if tables_subdir.exists() else self.workspace_dir
        for csv_file in sorted(csv_search_dir.glob("*.csv")):
            # Copy to report tables directory
            dest_csv = self.tables_dir / csv_file.name
            try:
                shutil.copy2(str(csv_file), str(dest_csv))
            except Exception as e:
                continue

            # Read preview data
            columns = []
            preview_data = []
            total_rows = 0
            try:
                df_preview = pd.read_csv(csv_file, nrows=10)
                columns = list(df_preview.columns)
                preview_data = df_preview.to_dict('records')
                total_rows = pd.read_csv(csv_file, usecols=[0]).shape[0]
            except Exception:
                pass

            tbl_info = {
                "filename": csv_file.name,
                "name": csv_file.stem.replace("_", " ").title(),
                "csv_path": str(dest_csv),
                "columns": columns,
                "preview_data": preview_data,
                "total_rows": total_rows,
                "inferred_type": "differential_expression" if csv_file.stem.startswith("de_") else "data"
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
        max_tables: int = 10
    ) -> Dict[str, str]:
        """
        Read CSV tables and produce text summaries for PI context.
        Returns dict mapping filename -> formatted text summary.
        """
        import pandas as pd

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
                            sig = df[(df[fc_col].abs() > 1) & (df[pval_col] < 0.05)]
                            parts.append(f"Significant genes (|logFC|>1, padj<0.05): {len(sig)}")
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

