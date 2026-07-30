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
import json
import logging
import re
import time

from .shared import AnalysisStep, CodeExecution, execution_supports_evidence, truncate_for_llm

logger = logging.getLogger(__name__)

# Matches the caller's sanitize_review_result() figure cap: base64 payloads beyond
# what will survive sanitizing are held in memory for nothing.
_MAX_CARRIED_FIGURES = 12

_VALID_STATUSES = ("completed", "partial", "failed")


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
        low = code.lower()
        for pat in self._SUPPLEMENTARY_DENYLIST:
            if pat in low:
                return f"disallowed operation '{pat}'"
        # Same assignment guard as _validation_code_is_safe: '<main_var> =' but not
        # '<main_var>_score ='. Rebinding the main variable would strand every
        # execution id the round has already cited as evidence.
        if re.search(rf"(^|\n)\s*{re.escape(main_var)}\s*=(?!=)", code):
            return f"assignment to main object '{main_var}'"
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
        """Execute PI-proposed supplementary tasks on the LIVE analysis object.

        Restricted to code execution: no tool loop, no literature, no PI
        consultation, no publication-figure pipeline (a supplement must not consume
        the round's figure budget). Never raises — every failure path returns a
        status.

        Two independent budgets, because one shared ceiling cannot serve both:
        `timeout_seconds` bounds ONE step (its code generation, its execution and
        the invariant check), so a heavy analysis — GRN inference, deconvolution
        training — gets its full allowance instead of inheriting whatever the
        earlier steps left over; `total_timeout_seconds` (default: the sum of the
        per-step budgets) only backstops a pass that never converges. A step that
        overruns interrupts the kernel and is reported back to the executor, which
        then picks a cheaper route or stops — it does not abandon the pass.

        Returns the raw dict for `sanitize_review_result`, plus `code_history` (the
        CodeExecution records produced, for evidence provenance) and `analysis_log`.
        """
        # Imported here, as the annotation validator does: backend.workflow's
        # package __init__ pulls in the workflow core, which imports the agents.
        from ...workflow.annotation_validation import signatures_match

        max_iterations = max(1, min(int(max_iterations), 8))
        step_timeout = max(1, int(timeout_seconds))
        total_timeout = int(total_timeout_seconds or 0)
        if total_timeout <= 0:
            total_timeout = step_timeout * max_iterations
        # A total below one step's budget would re-create the problem this split
        # exists to fix: the pass ceiling cutting off a single legitimate analysis.
        total_timeout = max(total_timeout, step_timeout)
        clean_tasks = [str(t).strip()[:600] for t in (tasks or []) if str(t).strip()]

        # Accumulated in the enclosing scope so the timeout and exception handlers
        # can still hand back what ran. A supplementary execution IS evidence the
        # caller cites, unlike an annotation-validation cell, so dropping it on a
        # timeout would orphan cells that are already in the notebook.
        produced: List[CodeExecution] = []
        analysis_log: List[AnalysisStep] = []
        figures: List[str] = []
        warnings: List[str] = []

        def _envelope(status: str, summary: str, extra_warnings: List[str]) -> Dict[str, Any]:
            return {
                "status": status,
                "summary": summary,
                "findings": [],
                "figures": list(figures),
                "warnings": warnings + list(extra_warnings),
                "code_history": list(produced),
                "analysis_log": list(analysis_log),
            }

        if not clean_tasks:
            return _envelope("completed", "No supplementary tasks were requested.", [])

        metadata = await self.introspect_validation_metadata(None)
        var_name = metadata.get("var")
        cluster_key = metadata.get("cluster_key")
        if not var_name:
            return _envelope(
                "failed",
                "No active analysis object in the kernel; supplementary tasks cannot run.",
                ["no active analysis object"],
            )
        round_number = int(getattr(self._current_analysis_state, "current_round", 0) or 0)

        async def _run() -> Dict[str, Any]:
            baseline = await self.capture_main_object_signature(var_name, cluster_key)
            if not baseline:
                return _envelope(
                    "failed",
                    "Supplementary pass aborted: could not capture the main-object signature.",
                    ["pre-supplement main-object signature unavailable"],
                )

            transcript: List[str] = []
            tolerated: Set[str] = set()
            final_result: Optional[Dict[str, Any]] = None

            abort_envelope: Optional[Dict[str, Any]] = None

            async def _one_step() -> str:
                """One executor iteration: generate, run, verify the invariant.

                Returns a loop signal — "continue", "done" or "abort" — so the
                caller can time-box a whole step without this body knowing that it
                is being timed.
                """
                nonlocal baseline, final_result, abort_envelope
                prompt = self._build_supplementary_executor_prompt(
                    clean_tasks, context_summary, direction, metadata,
                    "\n\n".join(transcript[-3:]))
                parsed = self._safe_parse_json(
                    await self.chat(prompt, temperature=0.2),
                    default={},
                )
                if not isinstance(parsed, dict):
                    transcript.append("Executor returned unparseable response; reply with valid JSON.")
                    return "continue"
                if parsed.get("done"):
                    final_result = parsed.get("result") or {}
                    return "done"

                code = parsed.get("code", "")
                if not code or not isinstance(code, str):
                    transcript.append("No code provided; provide `code` or set done=true.")
                    return "continue"

                unsafe = self._supplementary_code_is_safe(code, var_name)
                if unsafe:
                    transcript.append(
                        f"[rejected] {unsafe}. Add derived results without re-running "
                        f"preprocessing or rebinding the main object.")
                    return "continue"

                started = time.perf_counter()
                exec_result = await self.sandbox.execute(code)
                execution_time = time.perf_counter() - started

                # Cap at source so the record, the LLM transcript and the analysis
                # log all share one truncated string.
                output = truncate_for_llm(getattr(exec_result, "output", "") or "")
                error = getattr(exec_result, "error", None)
                if error:
                    error = truncate_for_llm(error)
                exec_figures = list(getattr(exec_result, "figures", None) or [])

                record = CodeExecution(
                    code=code,
                    output=output,
                    error=error,
                    figures=exec_figures,
                    execution_time=execution_time,
                    round_number=round_number,
                )
                produced.append(record)
                self.code_history.append(record)
                self.last_result = exec_result
                analysis_log.append(AnalysisStep(
                    step_number=len(analysis_log) + 1, step_type="execute_code",
                    code=code, code_output=output[-3000:],
                    code_error=error[:3000] if error else None,
                    code_description=f"Supplementary execution {len(produced)}",
                    code_step_type="technical", round_number=round_number,
                ))
                # Figures are carried through as captured. The publication-figure
                # pipeline is deliberately not invoked: a supplement must not spend
                # the round's figure budget.
                for figure in exec_figures:
                    if len(figures) < _MAX_CARRIED_FIGURES:
                        figures.append(figure)

                # The execution id is shown so the model can cite the exact cell a
                # finding came from, which is what makes the findings auditable.
                transcript.append(
                    f"[execution_id {record.id}]\n[code]\n{code[:1500]}\n"
                    f"[stdout]\n{output[:6000]}\n[error]\n{error or 'none'}")

                post_sig = await self.capture_main_object_signature(var_name, cluster_key)
                ok, changed = signatures_match(baseline, post_sig)
                if not ok:
                    fatal = self._supplementary_fatal_signature_changes(changed)
                    if fatal:
                        logger.error(
                            f"[{getattr(self, 'session_id', '?')}] supplementary pass "
                            f"corrupted main object: {fatal}")
                        abort_envelope = _envelope(
                            "failed",
                            f"Supplementary pass aborted: main object was modified ({fatal}).",
                            [f"main-object invariant changed: {fatal}"],
                        )
                        return "abort"
                    # Re-baseline on the tolerated state so the next cell is compared
                    # against the object it actually starts from; comparing every cell
                    # to the original signature would let a later in-place edit hide
                    # behind an earlier column addition.
                    tolerated.update(changed)
                    baseline = post_sig
                return "continue"

            for _ in range(max_iterations):
                try:
                    signal = await asyncio.wait_for(_one_step(), timeout=step_timeout)
                except asyncio.TimeoutError:
                    # The step, not the pass, ran out of time. Interrupt the cell it
                    # left running, tell the executor what happened, and let it pick
                    # a cheaper route — abandoning the whole pass here would throw
                    # away the steps that already succeeded.
                    logger.warning(
                        f"[{getattr(self, 'session_id', '?')}] supplementary step "
                        f"exceeded its {step_timeout}s budget")
                    self._interrupt_kernel_after_timeout()
                    warnings.append(
                        f"A supplementary step exceeded its {step_timeout}s budget and "
                        f"was interrupted.")
                    transcript.append(
                        "[timeout] The previous step exceeded its time budget and the "
                        "kernel was interrupted. Do not retry that computation as-is: "
                        "use a cheaper approach, reduce its scope, or set done=true and "
                        "report what is already available.")
                    # `baseline` is deliberately NOT re-captured: an interrupted cell
                    # may have left the object half-modified, and the next step's
                    # invariant check is what has to catch that.
                    continue
                if signal == "done":
                    break
                if signal == "abort":
                    return abort_envelope or _envelope(
                        "failed", "Supplementary pass aborted.", ["aborted"])

            if tolerated:
                warnings.append(
                    "Supplementary pass extended the analysis object: "
                    + ", ".join(sorted(tolerated)))

            if final_result is None:
                return _envelope(
                    "partial",
                    "Supplementary pass reached its iteration limit before reporting a result.",
                    ["max_iterations reached"],
                )

            status = str(final_result.get("status", "") or "").strip().lower()
            if status not in _VALID_STATUSES:
                status = "completed"
            result = _envelope(status, str(final_result.get("summary", "") or "").strip()
                               or "Supplementary analysis completed.", [])
            result["findings"] = [
                str(f).strip() for f in (final_result.get("findings") or []) if str(f).strip()
            ]
            result["warnings"] += [
                str(w).strip() for w in (final_result.get("warnings") or []) if str(w).strip()
            ]

            # A finding must trace to a cell this pass actually ran; with nothing
            # citable the report is the model's recollection of earlier rounds.
            if not any(execution_supports_evidence(r) for r in produced):
                if result["findings"]:
                    result["warnings"].append(
                        "Discarded findings with no successful supplementary execution to cite.")
                    result["findings"] = []
                if result["status"] == "completed":
                    result["status"] = "partial"
            return result

        try:
            return await asyncio.wait_for(_run(), timeout=total_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"[{getattr(self, 'session_id', '?')}] supplementary pass exceeded its "
                f"total {total_timeout}s budget")
            # Cancelling the coroutine only stops US waiting: a cell already
            # dispatched keeps running in the kernel and would hold it against the
            # rest of the workflow. The sandbox interrupts on its OWN timeout, not
            # on ours, so this ceiling has to do it itself.
            self._interrupt_kernel_after_timeout()
            return _envelope("partial", "Supplementary executor timed out.", ["timeout"])
        except Exception as e:
            logger.exception(f"[{getattr(self, 'session_id', '?')}] supplementary executor failed")
            return _envelope("failed", f"Supplementary executor error: {e}", [str(e)])

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
            logger.warning(
                f"[{getattr(self, 'session_id', '?')}] could not interrupt kernel "
                f"after supplementary timeout: {e}"
            )

    def _build_supplementary_executor_prompt(
        self,
        tasks: List[str],
        context_summary: str,
        direction: str,
        metadata: Dict[str, Any],
        transcript_tail: str,
    ) -> str:
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
{json.dumps(metadata, ensure_ascii=False)[:2000]}

## Hard rules
- Work on the LIVE object `{main_var}`. NEVER assign to `{main_var}` itself; write
  derived results into new columns/keys or new variables.
- Do NOT re-run QC filtering, normalization, scaling, clustering, the neighbourhood
  graph, or embeddings, and do NOT write data files. Those would invalidate results
  this round has already reported.
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
- Figures are optional. When you make one, ALSO print the quantitative values behind
  it so the finding stands on its text alone.

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
