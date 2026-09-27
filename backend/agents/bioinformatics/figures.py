# Publication-figure generation, review loop, and revision.
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
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

import aiofiles

from ...constants import NO_OUTPUT_TOKEN_CAP
from ...utils.llm_client_parts.prompt_cache import PromptCachePlan
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

from ...utils.formatting import (
    format_revision_history,
    normalize_text_list,
    truncate_for_llm,
)
from ...utils.error_parsing import compact_traceback
from ...utils.llm_client import (
    is_vision_capability_mismatch,
    is_vision_request_shape_error,
    retry_replay_safe_llm_call,
)
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...services.figure_layout import (
    blocking_violations,
    format_violations_for_prompt,
    lint_ran,
    parse_layout_violations,
    summarize_violations,
)
from ...services.figure_content import empty_render_reason
from ...services.report.artifacts import (
    archival_render_available,
    eligible_artifacts,
    image_file_sha256,
    is_report_eligible,
    logical_figure_key,
    pdf_file_sha256,
    preview_render_sha256,
    publication_pdf_is_valid,
)
from ...services.report.figure_transactions import (
    acquire_figure_publish_locks,
    apply_publish_record_paths,
    begin_figure_publish_transaction,
    commit_figure_publish_transaction,
    mark_figure_record_recovery_durable,
    mark_publish_record_applied,
    refreeze_pending_figure_attempt,
    register_pending_figure_attempt,
    rollback_figure_publish_transaction,
)
from .figure_dependencies import FigureDependencies
from . import figure_preflight as _figure_preflight_owner
from . import figure_generation as _figure_generation_owner
from . import figure_persistence as _figure_persistence_owner
from .figure_generation import execute_initial_render
from .figure_persistence import (
    finalize_render,
    guard_attempt_await,
    journal_pending_figure,
)
from .figure_preflight import FigureRequestContext, prepare_figure_request
from .figure_review import run_review_loop

# Configure logger
logger = logging.getLogger(__name__)

# Reading back a rendered figure so a worse revision can be rolled back is
# cheap for a normal plot and pointless for a 10 MB rasterized UMAP; above this
# the rollback capability is dropped rather than held in memory.
_MAX_ROLLBACK_SNAPSHOT_BYTES = 24 * 1024 * 1024

# Printed by the publication-figure wrapper when the code ran without raising
# but left no figure to save. The previous file is then still on disk, so every
# file-based success test passes on stale bytes.
_NO_FIGURE_MARKER = "No figures found"

# The wrapper catches a failing savefig and only prints, so result.error stays
# None while the file on disk is the previous render — same stale-bytes problem
# from the other direction.
_SAVE_FAILED_MARKER = "could not save JPG"
_PDF_SAVE_FAILED_MARKER = "could not save PDF"
_JPG_SAVED_MARKER = "Saved JPG"
_PDF_SAVED_MARKER = "Saved PDF"


def _new_render_attempt_paths(figures_dir: Path, safe_name: str, attempt_id: str):
    """Return a hidden staging directory and its JPG/PDF paths."""
    return _figure_preflight_owner._new_render_attempt_paths(
        figures_dir, safe_name, attempt_id
    )


def _discard_render_attempt(attempt_dir: Path) -> None:
    return _figure_generation_owner._discard_render_attempt(
        attempt_dir, bindings=sys.modules[__name__]
    )


def _publish_render_attempt(
    attempt_dir: Path,
    staged_jpg: Path,
    staged_pdf: Path,
    final_jpg: Path,
    final_pdf: Path,
    *,
    archived_record: Any = None,
) -> str | None:
    """Expose a validated pair and preserve the previous registered render.

    A same-name redraw must not destroy the bytes referenced by an older
    reportable record.  Copies of that pair are therefore placed in an explicit
    record-owned archive before either public path is replaced.  If either
    replacement fails, the old pair is restored before the attempt is removed.
    """
    return _figure_generation_owner._publish_render_attempt(
        attempt_dir,
        staged_jpg,
        staged_pdf,
        final_jpg,
        final_pdf,
        archived_record=archived_record,
        bindings=sys.modules[__name__],
    )


def _render_landed(output: str | None) -> bool:
    """Whether this execution actually replaced the figure on disk."""
    return _figure_preflight_owner._render_landed(
        output, bindings=sys.modules[__name__]
    )


def _publication_output_error(
    jpg_path: Path, pdf_path: Path, output: str | None
) -> str | None:
    """Explain why a promised JPG+PDF render is not publication-ready."""
    return _figure_preflight_owner._publication_output_error(
        jpg_path, pdf_path, output, bindings=sys.modules[__name__]
    )


def _layout_unverified_marker() -> Dict[str, Any]:
    """The record for a render the linter returned no verdict for.

    ``parse_layout_violations`` returns ``[]`` both for "measured, nothing wrong"
    and for "no verdict at all", and only the second is a gap — yet stored on the
    figure the two are the same value, so every consumer reads an unmeasured
    figure as a measured-clean one. Record the gap instead, reusing the
    ``lint_skipped`` kind the kernel already emits when it drops one of its own
    checks rather than inventing a second representation.

    Advisory by construction: a linter that did not run has measured no defect,
    and a redraw cannot fix one that was never found. Blocking here would spend
    the revision budget on an unfixable objection and drop the figure from every
    report — the false-positive cost this gate is built to avoid.
    """
    return _figure_preflight_owner._layout_unverified_marker()


def _layout_feedback_block(violations) -> str:
    """Render measured layout violations as a feedback section, or ''."""
    return _figure_preflight_owner._layout_feedback_block(
        violations, bindings=sys.modules[__name__]
    )


def _violation_must_fix_items(violations) -> List[str]:
    """One must-fix line per measured violation, for the reviewer's list."""
    return _figure_preflight_owner._violation_must_fix_items(
        violations, bindings=sys.modules[__name__]
    )


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
    compact_traceback,
    extract_error_type,
    TECHNICAL_ERROR_TYPES,
    detect_swallowed_error,
    build_expression_runtime_code,
    build_expression_preflight_code,
    extract_sensitive_expression_inputs,
    has_direct_expression_contract_call,
    instrument_expression_contract_calls,
    mutates_expression_representation,
    parse_expression_contracts,
    prepares_expression_before_sensitive_call,
    record_expression_contract,
    record_executed_representation_change,
    workspace_artifact_versions,
    strip_expression_contract_markers,
    SupervisorEvalResult,
    _DYNAMIC_CONTEXT_PREFIX,
    _APPROVED_PLAN_PREFIX,
    _CELL_TYPE_ASSIGN_RE,
    _FAILED_EVIDENCE_OUTPUT_RE,
    build_approved_plan_context,
    inject_revised_plan_context,
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


class FiguresMixin:
    """Publication-figure generation, review loop, and revision."""

    async def _call_figure_revision_llm(self, call):
        """Run one side-effect-free revision call with bounded outcome retries."""
        client = self.llm_client

        async def notify_retry(attempt, max_retries, delay, error):
            notify = getattr(self, "_notify", None)
            if callable(notify):
                await notify(
                    "status",
                    "Figure revision LLM response was incomplete; retrying the "
                    f"same request ({attempt}/{max_retries}) in {delay:.0f}s.",
                )

        return await retry_replay_safe_llm_call(
            call,
            agent_name="bioinformatics",
            model_name=str(self.model or ""),
            llm_server=str(getattr(client, "base_url", "") or ""),
            on_retry=notify_retry,
        )

    async def _environment_api_hint(
        self,
        state,
        error: str | None,
        failed_code: str = "",
    ) -> str:
        """Return one bounded, session-scoped note for a recognizable API mismatch."""
        return await _figure_preflight_owner.environment_api_hint(
            self,
            state,
            error,
            failed_code,
            bindings=sys.modules[__name__],
        )

    async def _handle_generate_publication_figure(
        self,
        code: str,
        figure_name: str,
        description: str,
        figure_type: str,
        state: AnalysisState,
        review_callback: Optional[
            Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]
        ] = None,
        diagnostic: bool = False,
        plan_step_id: Optional[int] = None,
        purpose: Optional[str] = None,
        logical_figure_id: Optional[str] = None,
        source_execution_id: Optional[str] = None,
        submitted_plan_step_id: Optional[int] = None,
        plan_requirement_key: Optional[str] = None,
        supervisor_binding_decision: Optional[str] = None,
        supervisor_binding_reason: Optional[str] = None,
        producer_run_kind: str = "ordinary",
        producer_run_scope_id: Optional[str] = None,
        figure_followup_task_id: Optional[str] = None,
        figure_followup_scope: Optional[Dict[str, Any]] = None,
        analysis_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Handle generate_publication_figure tool call.
        Generates the figure, saves in dual format, and optionally gets PI review.

        Args:
            code: Python plotting code
            figure_name: Name for the figure file
            description: Figure description
            figure_type: Type of figure
            state: Analysis state to store the figure
            review_callback: Optional callback for PI review
            diagnostic: Platform-rendered diagnostic rather than a result figure.
                Recorded on the figure so report eligibility does not depend on a
                review that, for this kind of figure, never happens.
            plan_step_id: Approved plan step this figure fulfils, when known.
            purpose: Semantic role of the figure (result, baseline, diagnostic).
            logical_figure_id: Stable semantic identity reused by later redraws.

        Returns:
            Tool result message
        """
        dependencies = FigureDependencies.from_facade()
        request = FigureRequestContext(
            agent=self,
            code=code,
            figure_name=figure_name,
            description=description,
            figure_type=figure_type,
            state=state,
            review_callback=review_callback,
            diagnostic=diagnostic,
            plan_step_id=plan_step_id,
            purpose=purpose,
            logical_figure_id=logical_figure_id,
            source_execution_id=source_execution_id,
            submitted_plan_step_id=submitted_plan_step_id,
            plan_requirement_key=plan_requirement_key,
            supervisor_binding_decision=supervisor_binding_decision,
            supervisor_binding_reason=supervisor_binding_reason,
            dependencies=dependencies,
            producer_run_kind=producer_run_kind,
            producer_run_scope_id=producer_run_scope_id,
            figure_followup_task_id=figure_followup_task_id,
            figure_followup_scope=figure_followup_scope,
            analysis_metadata=analysis_metadata,
        )
        attempt, error = await prepare_figure_request(request)
        if error:
            return error
        error = await execute_initial_render(attempt)
        if error:
            return error
        error = await journal_pending_figure(attempt)
        if error:
            return error
        await guard_attempt_await(
            attempt, self._notify("figure", attempt.figure.png_base64)
        )
        if review_callback:
            return await run_review_loop(attempt)
        publish_error = await finalize_render(attempt)
        if publish_error:
            return publish_error
        await self._emit_figure_record("figure_review", attempt.figure)
        return (
            f"Figure {figure_name} has been generated and saved.\n"
            f"JPG: {attempt.final_jpg_path}\nPDF: {attempt.final_pdf_path}"
            + (f"\n{attempt.staleness_note}" if attempt.staleness_note else "")
        )

    @staticmethod
    def _refresh_figure_integrity_hashes(figure) -> None:
        """Bind a record to its full render and compact persisted preview."""
        return _figure_persistence_owner.refresh_figure_integrity_hashes(
            figure, bindings=sys.modules[__name__]
        )

    @staticmethod
    def _clear_unpublished_figure_payload(figure) -> None:
        """Keep audit metadata while removing staged bytes and direct paths."""
        return _figure_persistence_owner.clear_unpublished_figure_payload(figure)

    async def _persist_figure_record_boundary(
        self, state: AnalysisState, figures_dir: Path
    ) -> bool:
        """Durably persist the record half for canonical session workspaces."""
        return await _figure_persistence_owner.persist_figure_record_boundary(
            self, state, figures_dir
        )

    async def _emit_figure_record(
        self,
        event_type: str,
        figure,
        *,
        review: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish the producer's authoritative post-gate figure state.

        Reviewer callbacks report an opinion.  Only this producer knows whether
        renderer measurements overruled it, whether a rollback happened, and
        which revision bytes are current.  Keep the event compact; the record is
        already upserted, so the image endpoint can serve the bound render.
        """
        return await _figure_persistence_owner.emit_figure_record(
            self,
            event_type,
            figure,
            review=review,
            bindings=sys.modules[__name__],
        )

    def _upsert_publication_figure(
        self, state, figure, *, refresh_integrity: bool = True,
    ):
        """Update one attempt and advance its published head only when eligible."""
        return _figure_persistence_owner.upsert_publication_figure(
            self,
            state,
            figure,
            refresh_integrity=refresh_integrity,
            bindings=sys.modules[__name__],
        )

    def _compress_image_to_base64(self, image_path: Path, max_width: int = 800) -> str:
        """
        Compress an image and convert to base64 for efficient transmission.

        Args:
            image_path: Path to the image file
            max_width: Maximum width for the compressed image

        Returns:
            Base64 encoded JPEG string
        """
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                # Calculate resize ratio
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    # Handle PIL version compatibility
                    try:
                        resample_method = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_method = Image.LANCZOS  # For older PIL versions
                    img = img.resize(new_size, resample_method)

                # Convert to RGB if necessary (handle RGBA, P modes)
                if img.mode in ('RGBA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        # Safely get alpha channel - check split() returns enough channels
                        channels = img.split()
                        if len(channels) >= 4:
                            background.paste(img, mask=channels[3])
                        else:
                            background.paste(img)
                    else:
                        background.paste(img)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            logger.exception("[Bioinformatics] Failed to compress image")
            # Fallback: read raw file and encode
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

    async def _revise_figure_code(
        self,
        original_code: str,
        feedback: str,
        suggestions: List[str],
        must_fix: List[str],
        figure_type: str,
        state: AnalysisState = None,
        image_base64: str = None,
        revision_history: List[Dict[str, Any]] = None,
        layout_violations: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate revised plotting code based on PI feedback.

        Args:
            original_code: The original plotting code
            feedback: Overall feedback from PI
            suggestions: List of improvement suggestions
            must_fix: List of critical issues that must be fixed
            figure_type: Type of figure
            state: Analysis state with treatment group info
            image_base64: Base64 encoded current figure image for visual review
            revision_history: Accumulated feedback from all revision rounds (text only)
            layout_violations: Renderer-measured collisions in the current image

        Returns:
            Revised Python plotting code
        """
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"
        must_fix_text = "\n".join(f"- {m}" for m in must_fix) if must_fix else "None"

        exp_design_constraint = ""
        if state and state.treatment_groups:
            group_names = ', '.join(g.name for g in state.treatment_groups)
            exp_design_constraint = f"""
**CRITICAL - Experimental Design Constraint:**
The actual experimental groups are: {group_names}
You must ONLY use data from these groups. Do NOT create or reference groups that don't exist.
If the PI feedback asks you to add groups not in the experimental design, IGNORE that part of the feedback and only plot data for the actual groups.
"""

        layout_section = _layout_feedback_block(layout_violations)
        from .figure_context import figure_input_context

        input_context = await figure_input_context(
            getattr(self, "sandbox", None), original_code,
        )

        # Build revision history section (show previous rounds, exclude current which is shown below)
        history_section = ""
        if revision_history and len(revision_history) > 1:
            past_rounds = revision_history[:-1]
            history_parts = []
            for entry in past_rounds:
                round_num = entry.get('round', '?')
                hist_feedback = entry.get('feedback', '')
                hist_suggestions = entry.get('suggestions', [])
                hist_must_fix = entry.get('must_fix', [])
                round_text = f"### Round {round_num}\n"
                round_text += f"- Feedback: {hist_feedback}\n"
                if hist_suggestions:
                    round_text += f"- Suggestions: {'; '.join(str(s) for s in hist_suggestions)}\n"
                if hist_must_fix:
                    round_text += f"- Must Fix: {'; '.join(str(m) for m in hist_must_fix)}\n"
                history_parts.append(round_text)
            history_section = (
                "**Previous Revision History (CRITICAL - read carefully):**\n"
                "The following issues were identified and some were FIXED in prior rounds.\n"
                "You MUST:\n"
                "1. Keep ALL previously applied fixes (do NOT revert them)\n"
                "2. Address the NEW issues from the current round\n"
                "3. Before submitting, mentally verify each previous fix is still present in your code\n\n"
                + "\n".join(history_parts) + "\n"
            )

        def build_prompt(with_image: bool) -> str:
            """Assemble the request with the invariant half in front.

            Prompt caching matches an exact token prefix, so a variable fragment
            near the top truncates the cache for everything behind it. The
            standing instructions remain byte-stable across revisions; code,
            feedback, history, and render diagnostics do not.
            """
            note = (
                "\n**Current Figure:** An image of the current figure is attached. "
                "Review it carefully to understand what needs to be changed.\n\n"
                if with_image else ""
            )
            return f"""Please revise the following plotting code based on PI feedback.

**Requirements:**
1. Fix every critical issue listed below under "Critical Issues (MUST FIX)"
2. Address as many of the improvement suggestions below as possible
3. CRITICAL: Do NOT revert previously applied fixes. If Round 1 changed grid lines to dashed, keep them dashed. If Round 2 added axis labels, keep them.
4. You MAY change the figure size, the GridSpec, the number of rows/columns, and where legends and annotations sit — resolving a collision usually requires exactly that. Do NOT change which data is plotted.
5. Reuse the original data variables and verify names and columns against the live input inventory below. Do not invent a new matrix or column to satisfy feedback. If a requested panel needs unavailable data or new analysis, preserve the existing valid panels and explain that limitation in the figure instead of fabricating data. Remove fabricated/dummy data for groups not in the experimental design.
6. Before calling set_xticklabels()/set_yticklabels(), ALWAYS call set_xticks()/set_yticks() first with matching count
7. For pandas Categorical columns, use .cat.add_categories() before assigning new category values
8. Use legend_.get_legend_handles_labels() instead of legend_.legendHandles (deprecated in newer matplotlib)
9. For ax.bar()/ax.barh(), the 3rd positional arg IS the bar thickness (width for bar, height for barh) — never also pass it as a width=/height= keyword, or you get "got multiple values for argument". Use the explicit keyword form only: ax.barh(y, values, height=0.7, ...)
10. Return ONLY the revised Python code, no explanations

**Layout rules (this revision is judged against them by the rendering engine, not by opinion):**
- Build the figure with a layout engine: `plt.figure(figsize=(...), layout='constrained')`. Never combine that with `tight_layout()` or `subplots_adjust()` — they fight each other and the result is the one you did not intend.
- A legend placed outside its axes needs space reserved for it. Give it its own GridSpec column or row, or use `fig.legend(...)` with the layout engine active. `bbox_to_anchor=(1.02, 0.5)` does NOT reserve space — it pushes the legend into whatever is drawn to the right, usually the neighbouring panel's labels.
- Any `ax.text` / `ax.annotate` placed outside the axes must either use `transform=fig.transFigure` or set `clip_on=True`. An unclipped annotation positioned from raw data values escapes the axes entirely and lands on the title.
- Panel letters: anchor with `va='bottom'` above the axes (`ax.text(-0.12, 1.02, 'A', transform=ax.transAxes, va='bottom', ha='right')`). With `va='top'` the glyph hangs downward into the first rows of tick labels on a short panel.
- A heatmap with many rows needs vertical room: roughly 0.18 inch per labelled row. If there is not enough, show fewer rows, shorten the labels, or split the panel — do not shrink the font below 6 pt.
- Keep tick labels human-readable: no raw variable names (`r5_mpt_mito_injury`), no words broken across lines.
- Every series in a legend needs a distinguishable colour. When one axes carries several artist types (e.g. stripplot + pointplot), select legend handles by label rather than slicing the handle list positionally.

**Other common fixes:**
- Add/improve axis labels and titles
- Adjust font sizes (plt.rcParams['font.size'] = 12)
- Use colorblind-friendly palettes
- Add scale bars or annotations
- Improve color contrast
- Remove unnecessary elements

---

**Original Code:**
```python
{original_code}
```

**Figure Type:** {figure_type}
**Measured live input inventory (bounded metadata, not scientific validation):**
{input_context}
Names missing from this inventory are not proof of absence because it is bounded;
check existing code bindings before use. Locally defined names in the plotting code
need not already exist in the kernel. An unavailable raw matrix must never be
replaced by normalized values and labeled as raw.
{exp_design_constraint}
{history_section}**PI Feedback (Current Round):** {feedback}
{layout_section}
**Improvement Suggestions (Current Round):**
{suggestions_text}

**Critical Issues (MUST FIX - Current Round):**
{must_fix_text}

Return only the Python code:{note}"""

        prompt = build_prompt(bool(image_base64))

        self._set_llm_context()
        # Use slightly lower temperature for code revision (needs precision)
        revision_temp = min(self.config.temperature, 0.2)
        if image_base64 and self.has_vision:
            # Use vision API so the agent can see the current figure
            try:
                vision_kwargs = dict(
                    model=self.model,
                    text_prompt=prompt,
                    images=[image_base64],
                    temperature=revision_temp,
                    max_tokens=NO_OUTPUT_TOKEN_CAP,
                    system_prompt=self.system_prompt,
                    prefer_stream=True,
                )
                cache_plan_for_history = getattr(
                    self.llm_client, "prompt_cache_plan_for_history", None
                )
                if callable(cache_plan_for_history):
                    candidate = cache_plan_for_history(self.model)
                    if isinstance(candidate, PromptCachePlan):
                        vision_kwargs["prompt_cache_plan"] = candidate
                response = await self._call_figure_revision_llm(
                    lambda: self.llm_client.chat_with_vision(**vision_kwargs)
                )
            except Exception as vision_err:
                _is_vision_unsupported = is_vision_capability_mismatch(vision_err)
                _is_payload_error = is_vision_request_shape_error(
                    vision_err, vision_request=True
                )
                if _is_vision_unsupported or _is_payload_error:
                    fallback_reason = (
                        "model vision capability mismatch"
                        if _is_vision_unsupported
                        else "vision request payload/protocol rejection"
                    )
                    logger.warning(
                        f"[Bioinformatics] Falling back to text-only figure revision "
                        f"after {fallback_reason}: {vision_err}"
                    )
                    if _is_vision_unsupported:
                        self.has_vision = False
                    prompt = prompt.replace(
                        "**Current Figure:** An image of the current figure is attached. "
                        "Review it carefully to understand what needs to be changed.\n\n",
                        "**Current Figure:** (Image not available for this request. "
                        "Revise code based on PI feedback text only.)\n\n",
                    )
                    text_kwargs = dict(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=revision_temp,
                        max_tokens=NO_OUTPUT_TOKEN_CAP,
                        prefer_stream=True,
                    )
                    response = await self._call_figure_revision_llm(
                        lambda: self._budgeted_text_call(
                            self.llm_client.chat, **text_kwargs
                        )
                    )
                else:
                    raise
        else:
            text_kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=revision_temp,
                max_tokens=NO_OUTPUT_TOKEN_CAP,
                prefer_stream=True,
            )
            response = await self._call_figure_revision_llm(
                lambda: self._budgeted_text_call(
                    self.llm_client.chat, **text_kwargs
                )
            )

        response_text = response.content if response and response.content else None
        if not response_text:
            logger.warning("[Bioinformatics] Empty response from LLM for code revision")
            return original_code  # Return unchanged code

        # Extract code from markdown response
        code = extract_code_from_markdown(response_text)

        return code.strip()
