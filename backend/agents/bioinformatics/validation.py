# Annotation validation cycle, species/gene checks, and swallowed-error lookup.
# Moved verbatim out of backend/agents/bioinformatics_agent.py; composed into BioinformaticsAgent.
"""
Bioinformatics Agent - Handles snRNA-seq analysis with iterative code execution
Supports autonomous technical steps and PI consultation for biological decisions
With step-level checkpointing for recovery from interruptions
"""
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
import asyncio
import functools
import json
import logging
import base64
import io
import uuid
import re
import ast
import textwrap
import time
from pathlib import Path
from datetime import datetime

import aiofiles

from ..base import BaseAgent, AgentConfig
from ...utils.settings_provider import AgentSettingsProvider
from ...utils.sandbox import (
    validate_code_safety, DANGEROUS_PATTERNS, extract_code_from_markdown,
    LocalJupyterSandbox, CodeExecutionResult, _TPL_NOOP_INSTALL,
)
from ...utils.checkpoint import CheckpointManager
import inspect
import json as _json
from backend.services import ora_kernel
from backend.services.marker_priors import MarkerDB, summarize_database_priors

from ...utils.formatting import format_revision_history, truncate_for_llm
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY

# Configure logger
logger = logging.getLogger(__name__)


from .shared import (  # noqa: F401
    Dict,
    Any,
    List,
    Optional,
    Callable,
    Awaitable,
    Tuple,
    asyncio,
    functools,
    json,
    logging,
    base64,
    io,
    uuid,
    re,
    ast,
    textwrap,
    time,
    Path,
    datetime,
    aiofiles,
    BaseAgent,
    AgentConfig,
    AgentSettingsProvider,
    validate_code_safety,
    DANGEROUS_PATTERNS,
    extract_code_from_markdown,
    LocalJupyterSandbox,
    CodeExecutionResult,
    _TPL_NOOP_INSTALL,
    CheckpointManager,
    inspect,
    _json,
    ora_kernel,
    MarkerDB,
    summarize_database_priors,
    _ORA_KERNEL_SRC,
    format_revision_history,
    truncate_for_llm,
    MECHANISM_EVIDENCE_POLICY,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _OVERCLAIM_WORDS,
    _NEGATION_WORDS,
    _MAX_PROGRAMS,
    _MAX_GENES,
    _MAX_OBS_LEN,
    _DEFAULT_MISSING,
    _REQUIRED_FIELDS,
    _ensure_list,
    _overclaim_unnegated,
    validate_mechanism_program_payload,
    TOOLS,
    KNOWN_TOOL_NAMES,
    build_gate_skip_note,
    count_saved_figures,
    AgentType,
    AnalysisPhase,
    AnalysisState,
    MessageRole,
    CodeExecution,
    DataDrivenMechanismProgram,
    PublicationFigure,
    FigureReviewStatus,
    AnalysisStep,
    PlanStage,
    DATA_TYPE_SCRNASEQ,
    DATA_TYPE_BULK_RNASEQ,
    OpenAICompatibleClient,
    ToolCall,
    UpstreamExhaustedError,
    EMPTY_CONTENT_PLACEHOLDER,
    tool_calls_are_paired,
    append_missing_tool_results,
    PLAN_REVISION_NEEDED_TAG,
    strip_ansi,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_invalidated_file_stems,
    record_representation_change,
    unchanged_artifact_stems,
    workspace_artifact_versions,
    SupervisorEvalResult,
    _DYNAMIC_CONTEXT_PREFIX,
    _APPROVED_PLAN_PREFIX,
    _CELL_TYPE_ASSIGN_RE,
    _FAILED_EVIDENCE_OUTPUT_RE,
    build_approved_plan_context,
    inject_revised_plan_context,
    code_is_statically_unsafe,
    execution_supports_evidence,
    successful_evidence_executions,
    _STAGE1_GUIDANCE,
    _STAGE2_GUIDANCE,
    _stage_guidance_text,
    _build_swallowed_error_feedback,
    _MARKER_DB,
    _get_marker_db,
    marker_priors_write,
)


class ValidationMixin:
    """Annotation validation cycle, species/gene checks, and swallowed-error lookup."""

    def _validation_code_is_safe(self, code: str, main_var: str) -> Optional[str]:
        """Return a reason string if code is statically unsafe, else None."""
        return code_is_statically_unsafe(code, main_var, self._VALIDATION_DENYLIST)

    async def validate_annotations_with_plan(
        self,
        validation_plan: Dict[str, Any],
        annotations: Dict[str, Any],
        cluster_markers: Dict[str, Any],
        tissue: str,
        organism: str,
        context: Dict[str, Any],
        max_iterations: int = 5,
        timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """Run annotation-validation code on a COPY (`adata_val`) of the live
        AnnData, re-checking main-object invariants after EVERY cell. Never
        touches CheckpointManager / autonomous chat history; exposes only code
        execution. Bounded by max_iterations (cap 8) and a wall-clock timeout."""
        import asyncio as _asyncio
        from ...workflow.annotation_validation import signatures_match, sanitize_validation_result

        max_iterations = max(1, min(int(max_iterations), 8))
        var_name = context.get("var")
        cluster_key = context.get("cluster_key")
        empty = {"status": "failed", "cluster_results": {}, "suggested_annotation_changes": {},
                 "artifacts": [], "warnings": []}
        if not var_name:
            return {**empty, "summary": "No active AnnData found for validation."}

        # Create an absolute artifact dir under the session workspace and surface it
        # to the executor prompt. The sandbox cwd is the workspace, and matplotlib's
        # savefig redirect only fires for workspace-ROOT paths, so an absolute subdir
        # is left untouched and artifacts land where they are reported.
        ctx = dict(context)
        try:
            wd = getattr(self.sandbox, "working_dir", None)
            if wd:
                art_dir = Path(wd) / "annotation_validation"
                art_dir.mkdir(parents=True, exist_ok=True)
                ctx["artifact_dir"] = str(art_dir)
        except Exception as e:
            logger.warning(f"[{getattr(self, 'session_id', '?')}] could not create artifact dir: {e}")

        async def _run() -> Dict[str, Any]:
            pre_sig = await self.capture_main_object_signature(var_name, cluster_key)
            if not pre_sig:
                logger.error(f"[{getattr(self, 'session_id', '?')}] could not capture pre-validation signature")
                return {**empty,
                        "summary": "Validation aborted: could not capture pre-validation main-object signature.",
                        "warnings": ["pre-validation main-object signature unavailable"]}
            transcript: List[str] = []
            final_result: Optional[Dict[str, Any]] = None

            # Copy the main object once.  Re-copying a large AnnData before every
            # validation cell multiplies both runtime and peak memory and was the
            # main reason the bounded executor exhausted its wall-clock budget.
            init_result = await self.sandbox.execute(
                f"adata_val = {var_name}.copy()",
                save_to_history=False,
                skip_safety_check=True,
            )
            init_error = getattr(init_result, "error", None)
            if init_error:
                return {**empty,
                        "summary": f"Validation aborted: could not create isolated copy ({init_error}).",
                        "warnings": ["validation copy creation failed"]}

            try:
                for _ in range(max_iterations):
                    prompt = self._build_validation_executor_prompt(
                        validation_plan, annotations, cluster_markers, tissue, organism,
                        ctx, "\n\n".join(transcript[-3:]))
                    parsed = self._safe_parse_json(
                        await self.chat(prompt, temperature=0.2),
                        default={},
                    )
                    if not isinstance(parsed, dict):
                        transcript.append("Executor returned unparseable response; reply with valid JSON.")
                        continue
                    if parsed.get("done"):
                        final_result = parsed.get("validation_result") or {}
                        break

                    code = parsed.get("code", "")
                    if not code or not isinstance(code, str):
                        transcript.append("No code provided; provide `code` or set done=true.")
                        continue

                    unsafe = self._validation_code_is_safe(code, var_name)
                    if unsafe:
                        transcript.append(f"[rejected] {unsafe}. Operate only on adata_val with read-only summaries.")
                        continue

                    exec_result = await self.sandbox.execute(
                        code, save_to_history=False, skip_safety_check=True
                    )
                    out = (getattr(exec_result, "output", "") or "")[:6000]
                    err = getattr(exec_result, "error", None)
                    transcript.append(f"[code]\n{code[:1500]}\n[stdout]\n{out}\n[error]\n{err or 'none'}")

                    # Per-iteration invariant check: abort immediately on first mutation.
                    post_sig = await self.capture_main_object_signature(var_name, cluster_key)
                    ok, changed = signatures_match(pre_sig, post_sig)
                    if not ok:
                        logger.error(f"[{getattr(self, 'session_id', '?')}] validation mutated main object: {changed}")
                        return {**empty,
                                "summary": f"Validation aborted: main object invariant changed ({changed}).",
                                "warnings": [f"main-object invariant changed: {changed}"]}
            finally:
                # The validation copy can be large; release it on success, error,
                # cancellation, or timeout instead of retaining it in the kernel.
                # sandbox.execute() returns a CodeExecutionResult rather than
                # raising, so check .error like the init-copy call above.
                release_result = await self.sandbox.execute(
                    "globals().pop('adata_val', None)",
                    save_to_history=False,
                    skip_safety_check=True,
                )
                if getattr(release_result, "error", None):
                    logger.warning(
                        f"[{getattr(self, 'session_id', '?')}] could not release validation copy"
                    )

            if final_result is None:
                return {**empty, "status": "partial",
                        "summary": "Validation reached its iteration limit without a final result.",
                        "warnings": ["max_iterations reached"]}
            return sanitize_validation_result(final_result)

        try:
            return await _asyncio.wait_for(_run(), timeout=timeout_seconds)
        except _asyncio.TimeoutError:
            logger.warning(f"[{getattr(self, 'session_id', '?')}] validation executor timed out")
            return {**empty, "status": "partial", "summary": "Validation executor timed out.",
                    "warnings": ["timeout"]}
        except Exception as e:
            logger.exception(f"[{getattr(self, 'session_id', '?')}] validation executor failed")
            return {**empty, "summary": f"Validation executor error: {e}", "warnings": [str(e)]}

    def _build_validation_executor_prompt(
        self, validation_plan, annotations, cluster_markers, tissue, organism, context, transcript_tail: str,
    ) -> str:
        import json as _j
        main_var = context.get("var")
        artifact_dir = context.get("artifact_dir") or "annotation_validation"
        return f"""You are a restricted ANNOTATION-VALIDATION code generator for {organism} {tissue}.
You may ONLY generate Python that runs in the existing Jupyter kernel. No analysis tools, literature,
PI consultation, or autonomous analysis.

## Hard rules
- One isolated copy `adata_val = {main_var}.copy()` is created before the validation loop. Use ONLY `adata_val`.
  NEVER assign to or mutate `{main_var}` (the main object).
- Do NOT filter cells, recluster, reintegrate, normalize, or change preprocessing.
- Do NOT run DEG between conditions, pathway enrichment, communication, trajectory, or velocity.
- Validation-only QC/doublet computation on `adata_val` is allowed when a needed field is absent.

## Output safety
- Never print raw matrices, full .X, full .obs, or gene-by-cell tables.
- Use grouped summaries (.groupby().mean()/median(), detection fractions, small crosstabs); cap rows/cols.
- Save large tables/figures under the ABSOLUTE directory {artifact_dir}/ (it already exists) using
  absolute paths, e.g. plt.savefig(f'{artifact_dir}/<name>.jpg') or df.to_csv(f'{artifact_dir}/<name>.csv'),
  and print only the path + a compact summary. Do NOT save with bare filenames.
- For EVERY figure, ALSO print a text summary with quantitative values (group means, detection fractions,
  enrichment direction, QC distributions, per-cluster verdicts). PI vision may be disabled; text must stand alone.

## Available metadata
{_j.dumps(context, ensure_ascii=False)[:2000]}

## Validation plan (PI questions)
{_j.dumps(validation_plan, ensure_ascii=False)[:3000]}

## Preliminary annotations
{_j.dumps({k: (v.get('cell_type') if isinstance(v, dict) else v) for k, v in annotations.items()}, ensure_ascii=False)[:2000]}

## Prior execution transcript (recent)
{transcript_tail or '(none yet)'}

Reply with STRICT JSON only:
- To run a check:  {{"code": "<python on adata_val>", "done": false}}
- When finished:   {{"done": true, "validation_result": {{
      "status": "completed", "summary": "<short>",
      "cluster_results": {{"<id>": {{"support_level": "supported|ambiguous|contradicted",
          "evidence_for": [], "evidence_against": [], "qc_flags": [], "suggested_action": "<text>",
          "requires_reclustering": false, "reclustering_reason": "", "suggested_reclustering_scope": null}}}},
      "suggested_annotation_changes": {{"<id>": {{"confidence": "low|medium|high", "low_confidence_reason": "<text>"}}}},
      "artifacts": [{{"type": "figure|table", "name": "<n>", "path": "{artifact_dir}/<f>",
          "summary": "<REQUIRED for every figure: the quantitative values it shows>"}}],
      "warnings": []}}}}
Every figure artifact MUST include a non-empty "summary" with its quantitative values, or the result is marked incomplete.
Set requires_reclustering true ONLY for strong intra-cluster mixed/bimodal lineage evidence not explained by QC."""

    async def _auto_validate_species_genes(self) -> bool:
        """Automatically validate gene names against session species.

        Returns True only if validation actually completed (sandbox sampling + PI
        assessment ran). Returns False on any early exit so callers can decide
        whether to mark the gate as resolved.
        """
        state = self._current_analysis_state
        if not state or not state.species:
            return False

        # Sample gene names from sandbox
        sample_code = (
            "import json as _json\n"
            "import random as _rnd\n"
            "for _name, _obj in list(globals().items()):\n"
            "    if hasattr(_obj, 'var_names') and not _name.startswith('_'):\n"
            "        _genes = list(_obj.var_names)\n"
            "        _sample = _rnd.sample(_genes, min(50, len(_genes)))\n"
            "        print(_json.dumps(_sample))\n"
            "        break\n"
        )
        import json as _json_mod
        try:
            result = await self.sandbox.execute(sample_code, save_to_history=False, skip_safety_check=True)
            if not result.output or result.error:
                return False
            # Scan all lines, return the first one that parses as a JSON list.
            gene_names = None
            for _line in (result.output or "").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _parsed = _json_mod.loads(_line)
                except Exception:
                    continue
                if isinstance(_parsed, list):
                    gene_names = _parsed
                    break
            if gene_names is None:
                logger.warning("[Bioinformatics] Could not locate JSON gene list in sandbox output")
                return False
        except Exception as e:
            logger.warning(f"[Bioinformatics] Failed to sample gene names: {e}")
            return False

        # Call PI LLM to assess match
        assessment = await self._assess_gene_species_match(gene_names, state.species)

        if assessment.get("match", True):
            await self._notify("status",
                f"Gene names validated: consistent with {state.species}")
            return True

        # Mismatch — trigger HITL if callback available
        if self.species_validation_callback:
            action = await self.species_validation_callback(gene_names, state.species, assessment)
            if action.get("action") == "custom":
                self.analysis_context["species_user_directive"] = action.get("user_input", "")
            return True
        else:
            await self._notify("warning",
                f"Gene names may not match {state.species}: {assessment.get('reasoning', '')}")
            return False

    async def _detect_and_lookup_swallowed_error(self, result, code: str):
        """Detect a technical error the agent caught in-code (try/except) and then
        skipped, and consult the Technique Agent for the correct API.

        Such errors never surface as ``result.error``, so the consecutive-error
        counter and the Supervisor technique trigger never see them. Returns
        ``(swallowed_type, doc_hint)``; ``(None, "")`` when there is nothing to do
        (hard error already surfaced, no swallowed error, or no technique agent).
        """
        if result.error:
            return None, ""
        swallowed_type = detect_swallowed_error(result.output or "")
        if not swallowed_type or not self.technique_agent:
            return swallowed_type, ""
        logger.info(
            f"[Bioinformatics] Detected swallowed {swallowed_type} with "
            f"skip decision; consulting Technique Agent"
        )
        try:
            doc_hint = await self.technique_agent.lookup_from_error(
                self.sandbox, result.output or "", code
            ) or ""
        except Exception as e:
            logger.warning(f"[Bioinformatics] Swallowed-error doc lookup failed: {e}")
            doc_hint = ""
        return swallowed_type, doc_hint

    async def _assess_gene_species_match(self, gene_names: list, species: str) -> dict:
        """Ask PI LLM whether gene names match the expected species."""
        gene_sample = ", ".join(gene_names[:50])
        prompt = f"""Examine these gene names sampled from a dataset and determine if they match species "{species}".

Gene names: {gene_sample}

Gene naming conventions:
- Human (Homo sapiens): ALL UPPERCASE (TP53, BRCA1, CD4, GAPDH)
- Mouse (Mus musculus): Capitalize first letter only (Tp53, Brca1, Cd4, Gapdh)
- Rat (Rattus norvegicus): Similar to mouse (Tp53, Brca1, Cd4, Gapdh)

IMPORTANT: Reply with ONLY a JSON object, no other text:
{{"match": true/false, "likely_species": "scientific name if different", "reasoning": "brief explanation", "suggestions": ["alternative species"]}}"""
        try:
            if self.pi_callback:
                response = await self.pi_callback(prompt, "species_validation")
            else:
                logger.warning("[Bioinformatics] pi_callback is None — skipping gene-species assessment")
                return {"match": True}
            if not response:
                logger.warning("[Bioinformatics] Empty pi_callback response for gene-species assessment — defaulting to match=True")
                return {"match": True}
            from backend.utils.formatting import safe_parse_json as _safe_parse_json2
            result = _safe_parse_json2(response)
            if result is not None:
                return result
            logger.warning("[Bioinformatics] Could not parse JSON from gene-species assessment response — defaulting to match=True")
            return {"match": True}
        except Exception as e:
            logger.warning(f"[Bioinformatics] Gene-species assessment failed: {e}")
            return {"match": True}
