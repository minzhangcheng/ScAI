"""Restricted supplementary execution for the post-execution review round.

Structurally a sibling of ``ValidationMixin.validate_annotations_with_plan``:
bounded iterations, a wall-clock timeout, a static denylist, a main-object
invariant check after every cell, and a body that returns a status instead of
raising. It differs in the two ways the review round requires — it runs on the
LIVE main object rather than a copy, and its executions are saved to history so
the caller can cite their ``CodeExecution`` ids as evidence provenance.
"""
from typing import Any, Dict, List, Optional, Set
import asyncio
import ast
import hashlib
import json
import logging
import time

from . import supplementary_support
from datetime import datetime
from pathlib import Path

from backend.services.report.artifacts import artifact_stems
from backend.services.report.figure_discovery import discover_figures_in_workspace
from backend.services.sample_design import code_sample_design_violation

from .shared import (
    AnalysisStep,
    CodeExecution,
    compact_traceback,
    code_is_statically_unsafe,
    execution_supports_evidence,
    truncate_for_llm,
)

logger = logging.getLogger(__name__)

_VALID_STATUSES = ("completed", "partial", "failed")
_validated_supplementary_findings = (
    supplementary_support._validated_supplementary_findings
)


class SupplementaryMixin:
    """Bounded supplementary task execution driven by the post-execution review."""

    # Operations a supplementary cell may never contain. The annotation validator's
    # denylist also bans analysis (rank_genes_groups, score_genes, ...), which is
    # wrong here: differential expression, module scoring and pathway activity are
    # precisely what a supplement is asked to add. What remains is the set that
    # would corrupt the round rather than extend it:
    #   sc.pp.filter_*        re-running QC changes the cell population every
    #                         already-reported number was computed on
    #   normalize_total /
    #   log1p / scale         rewrites .X in place, silently invalidating the
    #                         round's expression-derived results
    #   leiden / louvain /
    #   umap / neighbors      re-clustering or re-embedding renames the groups all
    #                         prior findings and figures refer to
    #   .write( / .write_h5ad overwrites persisted data that later rounds, reports
    #                         and snapshot restore read back
    _SUPPLEMENTARY_DENYLIST = (
        "sc.pp.filter_", "sc.pp.normalize_total", "sc.pp.log1p", "sc.pp.scale",
        "sc.tl.leiden", "sc.tl.louvain", "sc.tl.umap", "sc.pp.neighbors",
        ".write(", ".write_h5ad",
    )

    # This executor repairs computational evidence only.  Any result figure must
    # return to the main autonomous loop, where it receives a PublicationFigure
    # record, layout lint, PI review and plan attribution.  These calls cover the
    # normal ways Python plotting libraries write images; AST matching also catches
    # arbitrary whitespace and aliases such as ``fig.savefig`` that a substring
    # denylist would miss.
    _SUPPLEMENTARY_FILE_WRITE_CALLS = frozenset({
        "save", "savefig", "imsave", "imwrite", "write_image",
        "print_png", "print_jpg", "print_jpeg", "print_figure",
        "write_bytes", "write_text",
    })
    _SUPPLEMENTARY_EXACT_WRITE_CALLS = frozenset({
        "os.open", "os.write", "os.pwrite", "os.truncate", "os.ftruncate",
        "os.rename", "os.replace", "os.remove", "os.unlink", "os.link",
        "os.symlink", "os.mkdir", "os.makedirs", "os.rmdir",
        "os.removedirs", "os.utime", "shutil.copy", "shutil.copy2",
        "shutil.copyfile", "shutil.copytree", "shutil.move",
    })

    @staticmethod
    def _supplementary_image_versions(
        working_dir: Any,
    ) -> Dict[str, tuple[str, int, int, str]]:
        """Snapshot image content identity, not only attacker-restorable mtime."""
        if not working_dir:
            return {}
        versions: Dict[str, tuple[str, int, int, str]] = {}
        for candidate in discover_figures_in_workspace(Path(working_dir)):
            path = Path(str(candidate.get("path") or ""))
            try:
                metadata = path.stat()
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                versions[str(path)] = (
                    path.stem,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    digest,
                )
            except OSError:
                continue
        return versions

    def _quarantine_supplementary_images(
        self,
        state: Any,
        before: Dict[str, tuple[str, int, int, str]],
    ) -> Set[str]:
        """Withhold images created/overwritten outside the reviewed figure tool."""
        sandbox = getattr(self, "sandbox", None)
        after = self._supplementary_image_versions(getattr(sandbox, "working_dir", None))
        changed = {
            identity[0] for path, identity in after.items()
            if path not in before or before[path][1:] != identity[1:]
        }
        if not changed or state is None:
            return changed
        state.invalidated_artifact_stems = sorted(
            set(getattr(state, "invalidated_artifact_stems", None) or []) | changed
        )
        reason = (
            "Image bytes were written by the restricted computational supplementary "
            "executor without publication-figure review."
        )
        for figure in getattr(state, "publication_figures", None) or []:
            if (
                (getattr(figure, "lineage_id", None) or "main")
                != (getattr(state, "restore_branch_id", None) or "main")
            ):
                continue
            if int(getattr(figure, "round_number", 0) or 0) != int(
                getattr(state, "current_round", 0) or 0
            ):
                continue
            if not bool(getattr(figure, "is_current", True)):
                continue
            if not (artifact_stems([figure]) & changed):
                continue
            figure.is_current = False
            figure.invalidated_at = datetime.now()
            figure.invalidation_reason = reason
        return changed

    # Signature keys whose movement means the round's evidence no longer describes
    # the object it was computed on. Deliberately narrower than
    # MAIN_OBJECT_SIGNATURE_KEYS: `obs_columns`, `layers`, `layers_fingerprint`,
    # `id` and `x_id` all move when a supplement adds a derived column or layer,
    # which is the point of the pass.
    _SUPPLEMENTARY_FATAL_SIGNATURE_KEYS = (
        "n_obs", "n_vars", "x_sum", "x_nnz", "obs_hash", "var_hash", "cluster_counts",
    )

    def _supplementary_code_is_safe(self, code: str, main_var: str) -> Optional[str]:
        """Return a reason string if a supplementary cell is statically unsafe, else None."""
        design_error = code_sample_design_violation(
            code, getattr(self, "_current_analysis_state", None),
        )
        if design_error:
            return design_error
        unsafe = code_is_statically_unsafe(code, main_var, self._SUPPLEMENTARY_DENYLIST)
        if unsafe:
            return unsafe

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # The sandbox will return the syntax error through the normal execution
            # record.  Static safety is not a second Python parser.
            return None

        module_aliases: Dict[str, str] = {}
        symbol_aliases: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    root = item.name.split(".")[0]
                    local = item.asname or root
                    # ``import package.submodule`` binds only ``package``;
                    # an explicit alias binds the full dotted module.
                    module_aliases[local] = item.name if item.asname else root
            elif isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    if item.name == "*":
                        continue
                    local = item.asname or item.name
                    symbol_aliases[local] = f"{node.module}.{item.name}"

        def qualified_call_name(func: ast.expr) -> str:
            if isinstance(func, ast.Name):
                return symbol_aliases.get(func.id, func.id).lower()
            if not isinstance(func, ast.Attribute):
                return ""
            parts: List[str] = [func.attr]
            value = func.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                root = module_aliases.get(value.id, value.id)
                parts.append(root)
            return ".".join(reversed(parts)).lower()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            qualified = qualified_call_name(func)
            if qualified in self._SUPPLEMENTARY_EXACT_WRITE_CALLS:
                return f"disallowed file-writing call '{qualified}'"
            if isinstance(func, ast.Name) and func.id.lower() in self._SUPPLEMENTARY_FILE_WRITE_CALLS:
                return f"disallowed file-writing call '{func.id.lower()}'"
            if isinstance(func, ast.Name):
                canonical_tail = qualified.rsplit(".", 1)[-1]
                if canonical_tail in self._SUPPLEMENTARY_FILE_WRITE_CALLS:
                    return f"disallowed file-writing call '{qualified}'"
            if isinstance(func, ast.Attribute):
                call_name = func.attr.lower()
                if call_name in self._SUPPLEMENTARY_FILE_WRITE_CALLS:
                    return f"disallowed file-writing call '{call_name}'"

                # Scanpy plots write to sc.settings.figdir when ``save`` is any
                # value other than False/None.  A bare substring check is both too
                # weak (multiline calls) and too strong (save=False).
                if qualified.startswith(("sc.pl.", "scanpy.pl.")):
                    for keyword in node.keywords:
                        if keyword.arg != "save":
                            continue
                        if isinstance(keyword.value, ast.Constant) and keyword.value.value in (
                            False, None,
                        ):
                            continue
                        return "disallowed Scanpy figure write via 'save='"

            # A direct ``open(path, 'wb')`` can otherwise manufacture a PNG that
            # recursive report discovery would mistake for an archival artifact.
            is_builtin_open = (
                isinstance(func, ast.Name)
                and (
                    func.id == "open"
                    or qualified in {
                        "builtins.open", "io.open", "codecs.open",
                        "gzip.open", "bz2.open", "lzma.open",
                    }
                )
            )
            is_attribute_open = (
                isinstance(func, ast.Attribute) and func.attr.lower() == "open"
            )
            if is_builtin_open or is_attribute_open:
                mode_node = None
                mode_was_supplied = False
                for keyword in node.keywords:
                    if keyword.arg == "mode":
                        mode_node = keyword.value
                        mode_was_supplied = True
                        break

                if not mode_was_supplied:
                    if is_builtin_open and len(node.args) >= 2:
                        mode_node = node.args[1]
                        mode_was_supplied = True
                    elif is_attribute_open:
                        module_open = qualified in {
                            "builtins.open", "io.open", "codecs.open",
                            "gzip.open", "bz2.open", "lzma.open",
                        }
                        mode_index = 1 if module_open else 0
                        if len(node.args) > mode_index:
                            mode_node = node.args[mode_index]
                            mode_was_supplied = True

                if mode_was_supplied:
                    if not (
                        isinstance(mode_node, ast.Constant)
                        and isinstance(mode_node.value, str)
                    ):
                        return (
                            "disallowed file-writing call 'open': mode is not "
                            "statically proven read-only"
                        )
                    if any(flag in mode_node.value for flag in "wax+"):
                        return "disallowed file-writing call 'open'"
        return None

    def _supplementary_fatal_signature_changes(self, changed: List[str]) -> List[str]:
        """Destructive subset of a signature diff from `signatures_match`.

        `obs_hash` needs one carve-out: it hashes whole obs rows, so ADDING a column
        moves it even when no existing value was touched. When `obs_columns` moved
        too the two causes are indistinguishable, and a new score column is the
        expected outcome of a supplement — an in-place edit of an existing column
        still surfaces as `obs_hash` with `obs_columns` unchanged.
        """
        fatal = [key for key in changed if key in self._SUPPLEMENTARY_FATAL_SIGNATURE_KEYS]
        if "obs_columns" in changed and "obs_hash" in fatal:
            fatal.remove("obs_hash")
        return fatal

    async def run_supplementary_tasks(
        self,
        tasks: List[str],
        context_summary: str,
        direction: str,
        max_iterations: int = 5,
        timeout_seconds: int = 5520,
        total_timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute PI-proposed supplementary tasks on the live analysis object."""
        return await supplementary_support.run_supplementary_tasks(
            self,
            tasks,
            context_summary,
            direction,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )


    def _interrupt_kernel_after_timeout(self) -> None:
        """Best-effort kernel interrupt when the pass ceiling fires mid-execution.

        Mirrors what the sandbox does on its own timeout (sandbox_parts/execution.py).
        Swallows everything: the pass is already returning a partial result, and
        failing to interrupt must not turn that into an exception in the round.
        """
        km = getattr(getattr(self, "sandbox", None), "km", None)
        if km is None:
            return
        try:
            km.interrupt_kernel()
            logger.warning(
                f"[{getattr(self, 'session_id', '?')}] kernel interrupted after "
                f"supplementary pass timeout"
            )
        except Exception as e:
            compact_error = compact_traceback(str(e), max_chars=1000)
            logger.warning(
                f"[{getattr(self, 'session_id', '?')}] could not interrupt kernel "
                f"after supplementary timeout: {compact_error}"
            )

    @staticmethod
    def _render_object_metadata(metadata: Dict[str, Any]) -> str:
        """Render the live object's metadata field by field, longest last.

        Slicing the serialised document instead cuts it mid-structure: the
        per-column value summaries dominate the payload, so a fixed character
        budget is spent inside them and the object's shape, its layers and its
        raw-slot flag — the fields a task needs in order to be addressable at
        all — fall off the end. Truncating the lists rather than the string keeps
        every field present and keeps the block parseable.
        """
        metadata = metadata or {}
        if not metadata:
            return "(no live object detected)"

        def _names(key: str, limit: int) -> str:
            values = [str(v) for v in (metadata.get(key) or [])]
            if not values:
                return ""
            shown = values[:limit]
            suffix = f" (+{len(values) - len(shown)} more)" if len(values) > len(shown) else ""
            return ", ".join(shown) + suffix

        lines: List[str] = []
        for key, label in (
            ("var", "Object variable"),
            ("cluster_key", "Cluster key"),
            ("n_obs", "Observations"),
            ("n_vars", "Features"),
            ("has_raw", "Has .raw"),
            ("output_dir_writable", "Workspace writable"),
        ):
            value = metadata.get(key)
            if value not in (None, ""):
                lines.append(f"- {label}: {value}")
        for key, label, limit in (
            ("layers", "Layers", 20),
            ("candidate_cluster_keys", "Candidate cluster keys", 20),
            ("condition_like", "Condition-like columns", 20),
            ("sample_like", "Sample-like columns", 20),
            ("batch_like", "Batch-like columns", 20),
            ("qc_like", "QC-like columns", 30),
            ("obs_columns", "All obs columns", 120),
        ):
            rendered = _names(key, limit)
            if rendered:
                lines.append(f"- {label}: {rendered}")

        summaries = metadata.get("obs_summaries") or {}
        if isinstance(summaries, dict) and summaries:
            # Last, and capped: useful for choosing a grouping, but it is the
            # field that used to consume the whole budget.
            rows = []
            for column, info in list(summaries.items())[:25]:
                if not isinstance(info, dict):
                    continue
                top = ", ".join(str(t) for t in (info.get("top") or [])[:5])
                rows.append(f"  - {column}: {info.get('n_categories', '?')} categories [{top}]")
            if rows:
                lines.append("- Category columns and their values:")
                lines.extend(rows)
        return "\n".join(lines)

    def _build_supplementary_executor_prompt(
        self,
        tasks: List[str],
        context_summary: str,
        direction: str,
        metadata: Dict[str, Any],
        transcript_tail: str,
        registered_data_files: str = "",
    ) -> str:
        from backend.services.execution_evidence import (
            GENE_INDEX_ALIGNMENT_GUIDANCE,
            LIANA_RUNTIME_GUIDANCE,
        )
        main_var = metadata.get("var")
        task_list = "\n".join(f"{i}. {t}" for i, t in enumerate(tasks, start=1))
        return f"""You are a RESTRICTED SUPPLEMENTARY code generator. This round's analysis has
already finished and been reported; you are closing specific gaps the PI found in its
results. You may ONLY generate Python that runs in the existing Jupyter kernel — no
analysis tools, literature search, PI consultation, or autonomous re-planning.

## Supplementary tasks
{task_list}

## Research direction
{direction or '(not specified)'}

## What this round already established
{(context_summary or '(no summary available)')[:4000]}

## Active object metadata
{self._render_object_metadata(metadata)}

## Registered uploaded data files (authoritative)
{registered_data_files or '(no registered uploaded data files)'}
Use these exact registered paths before inspecting guessed kernel variable names or
workspace globs. A file that is not currently loaded into the kernel is NOT missing.
Do not report an uploaded modality as absent until every matching registered path has
been checked.

## Installed-runtime guidance
{LIANA_RUNTIME_GUIDANCE}
{GENE_INDEX_ALIGNMENT_GUIDANCE}

## Hard rules
- Work on the LIVE object `{main_var}`. NEVER assign to `{main_var}` itself; write
  derived results into new columns/keys or new variables.
- Do NOT re-run QC filtering, normalization, scaling, clustering, the neighbourhood
  graph, or embeddings, and do NOT write data files. Those would invalidate results
  this round has already reported.
- Do NOT render, display, or save plots or images. This pass repairs quantitative
  evidence only. Any visual result belongs in the main publication-figure pipeline,
  which registers, reviews and contract-checks it.
- Statistical analysis is expected and allowed: differential expression, module or
  signature scoring, pathway activity, enrichment, composition testing.
- Address ONE task per reply, in the order listed. Do not batch tasks into one cell.

## Evidence rules
- Every finding you report must come from a cell you ran in THIS pass. The transcript
  labels each cell with its execution_id; cite that id in the finding.
- Do not restate earlier-round results as new findings.
- If a task cannot be completed — package missing, required metadata absent, too few
  replicates or cells, assumptions unmet — say so plainly and move to the next task.
  NEVER silently substitute a weaker proxy: if you use a different method than the
  task names, state the substitution and the reason in the finding itself.

## Output safety
- Never print raw matrices, full .X, full .obs, or gene-by-cell tables.
- Print compact quantitative summaries: effect sizes, test statistics, adjusted
  p-values, group means, detection fractions, capped top-N tables.
- Report compact quantitative values in text; do not substitute an inline plot. Any
  incidental inline render is discarded and cannot enter the archival report or
  close a required publication-figure gap.

## Prior execution transcript (recent)
{transcript_tail or '(none yet)'}

Reply with STRICT JSON only:
- To run the next step: {{"code": "<python>", "done": false}}
- When every task is either addressed or shown to be impossible:
  {{"done": true, "result": {{
      "status": "completed|partial|failed",
      "summary": "<what this pass added to the round, and what it could not>",
      "findings": ["<one finding per string, each naming the execution_id it came from>"],
      "warnings": ["<each task that was not completed, and why>"]}}}}
Use "partial" when some tasks were completed and others could not be, "failed" when
none could be."""
