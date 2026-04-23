"""Analysis log mixin for DetailedReportGenerator."""
import os
import re as _re
from typing import Dict, Any, List

from backend.models.schemas import AnalysisState


class AnalysisLogMixin:
    """Provides analysis log building methods for report generation."""

    def _extract_findings(self, state: AnalysisState) -> Dict[str, Any]:
        """Extract key findings from analysis rounds"""
        findings = {
            "pathology": [],
            "bioinformatics": [],
            "biochemistry": [],
            "literature": [],
            "summary": state.final_summary or "",
            "round_count": len(state.round_results)
        }

        for round_result in state.round_results:
            findings["pathology"].extend(round_result.pathology_findings)
            findings["bioinformatics"].extend(round_result.bioinformatics_findings)
            findings["biochemistry"].extend(round_result.biochemistry_findings)
            findings["literature"].extend(round_result.literature_findings)

        # Include cluster annotations so LLM can reference cell types by name
        if self.cluster_annotations:
            annotation_lines = []
            for cid, info in sorted(self.cluster_annotations.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                annotation_lines.append(f"Cluster {cid}: {info.get('cell_type', 'Unknown')} ({info.get('n_cells', 0)} cells)")
            findings["cluster_annotations"] = "\n".join(annotation_lines)
        else:
            findings["cluster_annotations"] = "Not available"

        return findings

    def _read_notebook(self, state: AnalysisState) -> List[Dict[str, Any]]:
        """
        Read the analysis notebook (.ipynb) directly and extract all cells with outputs.

        Returns a chronological list of cell dicts:
        [
            {
                "cell_index": int,
                "code": str,
                "text_output": str,        # Combined stream + execute_result text
                "has_inline_figure": bool,  # True if display_data has image/png
                "saved_figures": [str],     # Filenames from savefig() calls
                "execution_count": int,
            }
        ]

        Falls back to state.code_history if the .ipynb file is not found.
        """
        notebook_path = os.path.join(
            self._get_session_states(state.id), f"{state.id}.ipynb"
        ) if hasattr(self, '_get_session_states') else None

        # Use config helper instead
        from backend.config import get_session_states
        notebook_path = get_session_states(state.id) / f"{state.id}.ipynb"

        cells = []

        if notebook_path.exists():
            try:
                import nbformat
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)

                for i, cell in enumerate(nb.cells):
                    if cell.cell_type != "code":
                        continue

                    code = cell.source or ""

                    # Filter out trivial cells
                    if len(code.strip()) < 20:
                        continue

                    # Filter out pure import/comment cells
                    lines = [l for l in code.strip().split("\n") if l.strip()]
                    import_pattern = _re.compile(r'^(\s*$|#.*|import\s|from\s.*\simport\s)')
                    if lines and all(import_pattern.match(l) for l in lines):
                        continue

                    # Extract text outputs
                    text_parts = []
                    has_inline_figure = False
                    for out in cell.outputs:
                        if out.output_type == "stream":
                            text_parts.append(out.text)
                        elif out.output_type == "execute_result":
                            data = dict(out.data) if hasattr(out.data, '__iter__') else {}
                            if "text/plain" in data:
                                text_parts.append(data["text/plain"])
                        elif out.output_type == "display_data":
                            data = dict(out.data) if hasattr(out.data, '__iter__') else {}
                            if "image/png" in data:
                                has_inline_figure = True
                            if "text/plain" in data:
                                text_parts.append(data["text/plain"])

                    text_output = "\n".join(text_parts).strip()

                    # Extract saved figure filenames from code
                    saved_figures = []
                    for match in _re.findall(r'["\']([^"\']*\.(?:png|jpg|jpeg))["\']', code):
                        fname = os.path.basename(match)
                        if fname:
                            saved_figures.append(fname)

                    cells.append({
                        "cell_index": i,
                        "code": code,
                        "text_output": text_output,
                        "has_inline_figure": has_inline_figure,
                        "saved_figures": saved_figures,
                        "execution_count": cell.execution_count or i,
                    })

                return cells

            except Exception as e:
                pass

        # Fallback: build from state.code_history
        for i, cell in enumerate(state.code_history):
            code = cell.code or ""
            if len(code.strip()) < 20:
                continue

            lines = [l for l in code.strip().split("\n") if l.strip()]
            import_pattern = _re.compile(r'^(\s*$|#.*|import\s|from\s.*\simport\s)')
            if lines and all(import_pattern.match(l) for l in lines):
                continue

            text_output = cell.output or ""

            saved_figures = []
            for match in _re.findall(r'["\']([^"\']*\.(?:png|jpg|jpeg))["\']', code):
                fname = os.path.basename(match)
                if fname:
                    saved_figures.append(fname)

            cells.append({
                "cell_index": i,
                "code": code,
                "text_output": text_output,
                "has_inline_figure": bool(cell.figures),
                "saved_figures": saved_figures,
                "execution_count": i,
            })

        return cells

    def _build_notebook_digest(self, cells: List[Dict[str, Any]], max_chars: int = 25000) -> str:
        """
        Build a compact but informative chronological digest from notebook cells.

        For each cell:
        - Shows the code (intelligently truncated for very long cells)
        - Shows text output highlights (focuses on quantitative results)
        - Notes saved figures and inline figure displays
        - Capped at max_chars total
        """
        digest_parts = []
        total_chars = 0

        for cell in cells:
            idx = cell["cell_index"]
            code = cell["code"]
            text_output = cell["text_output"]
            saved_figures = cell["saved_figures"]
            has_inline_figure = cell["has_inline_figure"]

            # Cell header
            cell_text = f"\n## Cell {idx + 1}\n"

            # Code: truncate very long cells but keep key lines
            if len(code) > 2000:
                code = self._truncate_code_smartly(code, max_len=2000)
            cell_text += f"```python\n{code}\n```\n"

            # Text output: extract highlights
            if text_output:
                output_highlight = self._extract_output_highlights(text_output, max_len=800)
                if output_highlight:
                    cell_text += f"Output:\n{output_highlight}\n"

            # Figure annotations
            if saved_figures:
                cell_text += f"→ Saved figures: {', '.join(saved_figures)}\n"
            if has_inline_figure:
                cell_text += "→ [Figure displayed inline]\n"

            # Check total length
            if total_chars + len(cell_text) > max_chars and digest_parts:
                remaining = max_chars - total_chars
                if remaining > 200:
                    cell_text = cell_text[:remaining] + "\n... (remaining cells omitted)\n"
                    digest_parts.append(cell_text)
                break

            digest_parts.append(cell_text)
            total_chars += len(cell_text)

        return "\n".join(digest_parts)

    def _truncate_code_smartly(self, code: str, max_len: int = 2000) -> str:
        """
        Truncate long code cells while keeping the most informative lines:
        comments, function calls, key assignments, save commands.
        """
        lines = code.split("\n")
        key_lines = []
        other_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            is_key = False
            # Comments describe intent
            if stripped.startswith("#"):
                is_key = True
            # Scanpy / analysis function calls
            elif _re.search(r'sc\.(tl|pp|pl|external)\.\w+', stripped):
                is_key = True
            # Key function calls
            elif any(kw in stripped.lower() for kw in [
                'savefig', 'save_figure', 'to_csv', 'read_h5ad', 'write',
                'rank_genes_groups', 'enrichr', 'gsea', 'harmonypy',
                'scvelo', 'cellrank', 'liana', 'decoupler'
            ]):
                is_key = True
            # Variable assignments with analysis keywords
            elif _re.search(r'\b(adata|cell_type|cluster|marker|deg_|enrichment)\b', stripped) and '=' in stripped:
                is_key = True
            # Print/display statements
            elif stripped.startswith(('print(', 'display(')):
                is_key = True

            if is_key:
                key_lines.append(line)
            else:
                other_lines.append(line)

        # If key lines fit, use them + as many other lines as possible
        result = "\n".join(key_lines)
        if len(result) < max_len and other_lines:
            remaining = max_len - len(result) - 50
            other_text = "\n".join(other_lines)
            if len(other_text) <= remaining:
                result = code  # Original fits with key lines
            else:
                result += f"\n# ... ({len(other_lines)} auxiliary lines omitted)\n"

        return result[:max_len]

    def _extract_output_highlights(self, output: str, max_len: int = 800) -> str:
        """
        Extract key quantitative results and important information from code output.
        Prioritizes lines with numbers, statistics, cell counts, gene counts, etc.
        """
        if not output:
            return ""

        highlight_lines = []
        other_lines = []

        for line in output.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            is_highlight = False
            # Lines with biological/analysis quantities
            if _re.search(r'\d+[\.\d]*\s*(cells?|genes?|clusters?|samples?|observations?|variables?|batches?)', stripped, _re.IGNORECASE):
                is_highlight = True
            # Lines with percentages or p-values
            elif _re.search(r'(\d+\.?\d*\s*%|p[\s_-]?val|fdr|adjusted)', stripped, _re.IGNORECASE):
                is_highlight = True
            # Lines with filtering/processing results
            elif _re.search(r'(filtered|remaining|total|identified|found|significant|removed|retained|normalized)', stripped, _re.IGNORECASE):
                is_highlight = True
            # Matrix dimensions (e.g., "25000 x 3000")
            elif _re.search(r'\d+\s*[x×]\s*\d+', stripped):
                is_highlight = True
            # AnnData repr
            elif _re.search(r'AnnData\s+object', stripped):
                is_highlight = True

            if is_highlight:
                highlight_lines.append(stripped)
            else:
                other_lines.append(stripped)

        if highlight_lines:
            result = "\n".join(highlight_lines[:15])
            return result[:max_len]

        # Fallback: return first few lines of output
        return output[:max_len]

    def _build_analysis_record(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Build a structured analysis record from state data.
        Connects code_history → saved figures → round findings to create
        a comprehensive timeline of the analysis.

        Returns dict with:
        - analysis_steps: list of notable code cells with their saved figures
        - round_summaries: list of round results with per-agent findings
        - publication_figures: list of PI-reviewed figure metadata
        - tables: list of CSV tables found in workspace
        """
        # 1. Parse code_history to find cells that saved figures
        analysis_steps = []
        for i, cell in enumerate(state.code_history):
            code = cell.code or ""
            output = cell.output or ""
            timestamp = cell.timestamp.isoformat() if hasattr(cell.timestamp, 'isoformat') else str(cell.timestamp)

            # Extract saved figure filenames from code
            saved_figures = []
            # Match patterns like: savefig("name.png"), save_figure(fig, path / "name")
            for match in _re.findall(r'["\']([^"\']*\.(?:png|jpg|jpeg))["\']', code):
                # Get just the filename, not full path
                fname = os.path.basename(match)
                if fname:
                    saved_figures.append(fname)

            # Extract a description from the first meaningful comment or code line
            description = ""
            for line in code.strip().split("\n"):
                line = line.strip()
                if line.startswith("#") and not line.startswith("# ==="):
                    description = line.lstrip("# ").strip()
                    break

            # Only include cells that produced figures or had significant output
            has_display_figures = len(cell.figures) > 0
            has_saved_figures = len(saved_figures) > 0
            has_error = bool(cell.error)

            if has_display_figures or has_saved_figures:
                analysis_steps.append({
                    "cell_index": i,
                    "timestamp": timestamp,
                    "description": description or f"Code cell {i}",
                    "saved_figures": saved_figures,
                    "has_display_figures": has_display_figures,
                    "output_summary": output[:300] if output else "",
                    "had_error": has_error,
                })

        # 2. Build round summaries from round_results
        round_summaries = []
        for r in state.round_results:
            round_summaries.append({
                "round_number": r.round_number,
                "pi_summary": r.pi_summary or "",
                "pathology_findings": r.pathology_findings or [],
                "bioinformatics_findings": r.bioinformatics_findings or [],
                "biochemistry_findings": r.biochemistry_findings or [],
                "literature_findings": r.literature_findings or [],
            })

        # 3. Collect publication figure metadata
        pub_figures = []
        for fig in state.publication_figures:
            pub_figures.append({
                "name": fig.name,
                "description": fig.description,
                "figure_type": fig.figure_type,
                "filename": os.path.basename(fig.png_path) if fig.png_path else "",
                "review_status": fig.review_status.value if hasattr(fig.review_status, 'value') else str(fig.review_status),
            })

        # 4. Scan workspace for CSV tables
        tables = []
        if self.workspace_dir and self.workspace_dir.exists():
            tables_subdir = self.workspace_dir / "tables"
            csv_search_dir = tables_subdir if tables_subdir.exists() else self.workspace_dir
            for csv_file in sorted(csv_search_dir.glob("*.csv")):
                tables.append({
                    "filename": csv_file.name,
                    "name": csv_file.stem.replace("_", " ").title(),
                })

        return {
            "analysis_steps": analysis_steps,
            "round_summaries": round_summaries,
            "publication_figures": pub_figures,
            "tables": tables,
            "total_code_cells": len(state.code_history),
            "total_rounds": len(state.round_results),
        }

    def _build_full_analysis_log(self, state: AnalysisState) -> List[Dict[str, Any]]:
        """
        Build full chronological analysis log for report.
        Uses state.analysis_log if available, falls back to code_history.
        No base64 image data — only filenames. HTML renderer embeds from disk.
        """
        if state.analysis_log:
            log_entries = []
            for step in state.analysis_log:
                entry = {
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "timestamp": step.timestamp.isoformat() if hasattr(step.timestamp, 'isoformat') else str(step.timestamp),
                    "agent_reasoning": step.agent_reasoning,
                }
                if step.step_type == "execute_code":
                    entry.update({
                        "code_description": step.code_description,
                        "code_step_type": step.code_step_type,
                        "code": step.code[:3000] if step.code else None,
                        "code_output": step.code_output[:3000] if step.code_output else None,
                        "code_error": step.code_error[:3000] if step.code_error else None,
                        "saved_figure_filenames": step.saved_figure_filenames,
                    })
                elif step.step_type == "consult_pi":
                    entry.update({
                        "pi_question": step.pi_question,
                        "pi_context": step.pi_context,
                        "pi_options": step.pi_options,
                        "pi_response": step.pi_response,
                    })
                elif step.step_type == "search_literature":
                    entry.update({
                        "literature_topic": step.literature_topic,
                        "literature_result": (step.literature_result or "")[:2000],
                    })
                elif step.step_type == "annotate_clusters":
                    entry.update({
                        "cluster_markers_summary": step.cluster_markers_summary,
                        "annotation_tissue": step.annotation_tissue,
                        "annotation_organism": step.annotation_organism,
                        "annotation_result_summary": (step.annotation_result_summary or "")[:2000],
                    })
                elif step.step_type == "report_findings":
                    entry.update({
                        "findings": step.findings,
                        "findings_summary": step.findings_summary,
                        "findings_suggestions": step.findings_suggestions,
                    })
                elif step.step_type == "generate_publication_figure":
                    entry.update({
                        "pub_figure_name": step.pub_figure_name,
                        "pub_figure_description": step.pub_figure_description,
                        "pub_figure_type": step.pub_figure_type,
                        "pub_figure_code": step.pub_figure_code,
                        "pub_figure_result": step.pub_figure_result,
                    })
                elif step.step_type == "text_response":
                    entry["text_content"] = step.text_content
                log_entries.append(entry)
            return log_entries

        # Fallback: build from code_history (backward compatibility)
        log_entries = []
        for i, cell in enumerate(state.code_history):
            _saved = [os.path.basename(p) for p in _re.findall(
                r'\[Auto-saved\] (?:JPG|PNG): (.+\.(?:jpg|png))', cell.output or ""
            )]
            log_entries.append({
                "step_number": i + 1,
                "step_type": "execute_code",
                "timestamp": cell.timestamp.isoformat() if hasattr(cell.timestamp, 'isoformat') else str(cell.timestamp),
                "agent_reasoning": None,
                "code": (cell.code[:3000] if cell.code else None),
                "code_output": (cell.output[:3000] if cell.output else None),
                "code_error": cell.error,
                "code_description": None,
                "saved_figure_filenames": _saved,
            })
        return log_entries

    def _build_analysis_log_html(
        self,
        report: Dict[str, Any],
        language: str,
    ) -> str:
        """Build the full analysis log HTML section. Delegates to shared implementation."""
        from .regenerator import _build_analysis_log_from_report
        return _build_analysis_log_from_report(report, language, self.figures_dir)
