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
from ...services.figure_layout import (
    blocking_violations,
    format_violations_for_prompt,
    lint_ran,
    parse_layout_violations,
    summarize_violations,
)

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


def _render_landed(output: str | None) -> bool:
    """Whether this execution actually replaced the figure on disk."""
    text = output or ""
    return _NO_FIGURE_MARKER not in text and _SAVE_FAILED_MARKER not in text


def _layout_feedback_block(violations) -> str:
    """Render measured layout violations as a feedback section, or ''."""
    body = format_violations_for_prompt(violations)
    if not body:
        return ""
    return (
        "\n**Measured layout defects (from the rendering engine, not an "
        "opinion — these are pixel coordinates of artists that were drawn on "
        "top of each other):**\n" + body + "\n"
    )


def _violation_must_fix_items(violations) -> List[str]:
    """One must-fix line per measured violation, for the reviewer's list."""
    body = format_violations_for_prompt(violations)
    if not body:
        return []
    items = []
    for line in body.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if cleaned:
            items.append(cleaned)
    return items


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
    MAX_WORKSPACE_H5AD_FILES,
    _TPL_SAFE_CLOSE_ALL,
    _TPL_PLT_RESTORE_BASELINE,
    _build_exec_autosave_wrapper,
    _build_publication_figure_wrapper,
    _static_h5ad_path,
    workspace_h5ad_write_guard,
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

    async def _handle_generate_publication_figure(
        self,
        code: str,
        figure_name: str,
        description: str,
        figure_type: str,
        state: AnalysisState,
        review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None
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

        Returns:
            Tool result message
        """
        logger.info(f"[Bioinformatics] Generating publication figure: {figure_name}")

        # Validate code before execution
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return f"Error: Code validation failed - {error_msg}"

        # Ensure figures directory exists
        figures_dir = Path(self.sandbox.working_dir) / "figures"

        # Clean up figure name
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in figure_name)
        jpg_path = figures_dir / f"{safe_name}.jpg"
        pdf_path = figures_dir / f"{safe_name}.pdf"

        # Wrap code to save both JPG and PDF; paths are repr()'d inside the
        # shared wrapper to prevent injection.
        wrapped_code = _build_publication_figure_wrapper(code, figures_dir, jpg_path, pdf_path)

        expression_specs = extract_sensitive_expression_inputs(code)
        representation_will_change = mutates_expression_representation(code)
        preexisting_artifact_versions = (
            workspace_artifact_versions(self.sandbox.working_dir)
            if representation_will_change else {}
        )
        if (
            expression_specs
            and not prepares_expression_before_sensitive_call(code)
        ):
            preflight = await self.sandbox.execute(
                build_expression_preflight_code(expression_specs),
                save_to_history=False,
                skip_safety_check=True,
            )
            if preflight.error:
                return f"Figure generation blocked: {strip_ansi(preflight.error)}"
            record_expression_contract(
                state, parse_expression_contracts(preflight.output)
            )

        # Execute the code
        result = await self.sandbox.execute(wrapped_code, skip_safety_check=True)

        if result.error:
            result.error = strip_ansi(result.error)
            error_msg = f"Figure generation failed: {result.error}"
            logger.error(f"[Bioinformatics] {error_msg}")
            await self._notify("error", error_msg)
            return error_msg

        record_representation_change(state, code)
        if representation_will_change:
            record_invalidated_file_stems(
                state,
                unchanged_artifact_stems(
                    self.sandbox.working_dir, preexisting_artifact_versions
                ),
            )

        # Check if files were created
        if not jpg_path.exists():
            error_msg = f"JPG file was not created: {jpg_path}"
            logger.error(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Detect blank figures (empty white JPG with bbox_inches='tight' is typically <2KB)
        if jpg_path.stat().st_size < 2000:
            error_msg = f"Figure appears to be blank (file size: {jpg_path.stat().st_size} bytes). The plotting code may not have generated visible content."
            logger.warning(f"[Bioinformatics] {error_msg}")
            return error_msg

        # Ask the rendering engine where the text actually landed. This is the
        # deterministic half of the figure gate: the vision reviewer can say a
        # legend overlaps something, only the renderer knows which artists, by
        # how much, and where.
        layout_violations = parse_layout_violations(result.output)
        if lint_ran(result.output):
            logger.info(
                f"[Bioinformatics] Figure {safe_name} layout lint: "
                f"{summarize_violations(layout_violations)}"
            )
        else:
            logger.warning(
                f"[Bioinformatics] Figure {safe_name}: layout lint did not "
                f"report — treating layout as unverified, not as clean"
            )

        # Read JPG and compress to base64
        png_b64 = self._compress_image_to_base64(jpg_path, max_width=800)

        # Create PublicationFigure object (png_path field stores JPG path for web display)
        sandbox_state = state.sandbox_state or {}
        figure = PublicationFigure(
            id=str(uuid.uuid4()),
            name=safe_name,
            description=description,
            figure_type=figure_type,
            png_path=str(jpg_path),
            pdf_path=str(pdf_path),
            png_base64=png_b64,
            layout_violations=layout_violations,
            review_status=FigureReviewStatus.PENDING,
            round_number=int(state.current_round or 0),
            analysis_generation=int(sandbox_state.get("analysis_generation", 0) or 0),
            expression_provenance=dict(
                sandbox_state.get("expression_contract_v1") or {}
            ),
        )

        # Send figure to frontend for display
        await self._notify("figure", png_b64)

        # If review callback provided, get PI review
        # Allow up to 3 revisions (4 reviews total: initial + 3 revisions)
        if review_callback:
            revision_count = 0
            current_code = code
            revision_history = []  # Accumulate all rounds of PI feedback
            # Highest-scoring rendered version seen so far, kept in memory so a
            # revision that makes the figure worse can be undone.
            best_snapshot = None

            def _record_unresolved(items) -> None:
                seen = []
                for item in items or []:
                    text = str(item).strip()
                    if text and text not in seen:
                        seen.append(text)
                figure.unresolved_must_fix = seen[:20]
                # A figure held back by measured collisions is dropped from
                # every report, and nothing else on the record would say why:
                # the review status reads "accepted with reservations" and the
                # UI shows it as reviewed. Put the reason where the figure
                # itself carries it.
                remaining = blocking_violations(figure.layout_violations)
                if remaining:
                    note = (
                        f"[Layout gate] {summarize_violations(remaining)} — text is "
                        f"covered and cannot be read, so this figure is held back "
                        f"from reports until it is redrawn."
                    )
                    existing = figure.pi_feedback or ""
                    if note not in existing:
                        figure.pi_feedback = (
                            f"{existing}\n\n{note}" if existing else note
                        )

            def _take_snapshot(score):
                try:
                    jpg_bytes = jpg_path.read_bytes() if jpg_path.exists() else None
                    pdf_bytes = pdf_path.read_bytes() if pdf_path.exists() else None
                except Exception as snap_err:
                    logger.debug(f"[Bioinformatics] Figure snapshot skipped: {snap_err}")
                    return None
                if len(jpg_bytes or b"") + len(pdf_bytes or b"") > _MAX_ROLLBACK_SNAPSHOT_BYTES:
                    return None
                return {
                    "score": score,
                    "code": current_code,
                    "jpg": jpg_bytes,
                    "pdf": pdf_bytes,
                    "b64": figure.png_base64,
                    "violations": list(figure.layout_violations or []),
                }

            def _restore_snapshot(snap) -> bool:
                if not snap:
                    return False
                try:
                    if snap.get("jpg"):
                        jpg_path.write_bytes(snap["jpg"])
                    if snap.get("pdf"):
                        pdf_path.write_bytes(snap["pdf"])
                except Exception as restore_err:
                    logger.warning(
                        f"[Bioinformatics] Could not roll back {figure_name} to the "
                        f"previous version: {restore_err}"
                    )
                    return False
                figure.png_base64 = snap.get("b64") or figure.png_base64
                figure.layout_violations = list(snap.get("violations") or [])
                return True

            while revision_count <= 3:
                logger.info(f"[Bioinformatics] Submitting figure for PI review (attempt {revision_count + 1})")
                await self._notify("status", f"PI reviewing figure: {figure_name}...")

                review = await review_callback(figure)
                quality_score = review.get('quality_score', 5)
                blocking = blocking_violations(figure.layout_violations)
                budget_left = revision_count < 3

                if not blocking and (
                    best_snapshot is None
                    or quality_score > best_snapshot.get("score", -1)
                ):
                    snapshot = _take_snapshot(quality_score)
                    if snapshot is not None:
                        best_snapshot = snapshot

                approved = bool(review.get('approved', False))

                if approved and blocking and budget_left:
                    # The reviewer signed off on a figure in which the renderer
                    # measured text drawn on top of other text. Measurement
                    # wins: the reviewer judges a downscaled JPEG, the renderer
                    # reports the artists themselves. This is not a second
                    # opinion about taste — the covered text cannot be read.
                    logger.warning(
                        f"[Bioinformatics] Figure {figure_name}: reviewer approved "
                        f"(score={quality_score}) but layout lint reports "
                        f"{summarize_violations(blocking)} — continuing revision"
                    )
                    approved = False
                    review = dict(review)
                    review['feedback'] = (
                        (review.get('feedback') or '')
                        + "\n\nThe reviewer approved this figure, but the rendering "
                        "engine measured text drawn on top of other text. Those "
                        "collisions must be resolved before the figure can be used."
                    )
                    review['must_fix'] = (
                        list(review.get('must_fix') or [])
                        + _violation_must_fix_items(blocking)
                    )

                if approved:
                    # Mark figures approved under leniency (after multiple revisions with non-perfect score)
                    if blocking or (revision_count >= 2 and quality_score < 8):
                        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
                        logger.info(f"[Bioinformatics] Figure {figure_name} accepted with reservations (score={quality_score}, revisions={revision_count})")
                    else:
                        figure.review_status = FigureReviewStatus.APPROVED
                        logger.info(f"[Bioinformatics] Figure {figure_name} approved by PI")
                    figure.revision_history = revision_history
                    _record_unresolved(_violation_must_fix_items(blocking))
                    self._upsert_publication_figure(state, figure)
                    await self._notify("status", f"Figure {figure_name} approved by PI")
                    layout_note = (
                        f"\nUnresolved layout defects: {summarize_violations(blocking)}. "
                        f"This figure is excluded from reports until they are fixed."
                        if blocking else ""
                    )
                    return (
                        f"Figure {figure_name} has been approved by PI and saved.\n"
                        f"Quality score: {quality_score}/10\n"
                        f"Feedback: {review.get('feedback', 'Approved')}{layout_note}"
                    )

                # PI requested revision
                feedback = review.get('feedback', '')
                suggestions = review.get('suggestions', [])
                must_fix = review.get('must_fix', [])

                # If PI flagged a data/analysis issue that can't be fixed by revising plot code,
                # break out of the revision loop and defer to the main analysis flow
                if review.get('requires_reanalysis', False):
                    logger.warning(f"[Bioinformatics] Figure {figure_name} requires upstream reanalysis, skipping revision loop")
                    await self._notify("status", f"Figure {figure_name}: PI identified data-level issue, deferring to analysis")
                    figure.review_status = FigureReviewStatus.REVISION_REQUESTED
                    figure.pi_feedback = f"[REQUIRES REANALYSIS] {feedback}"
                    figure.revision_history = revision_history
                    _record_unresolved(
                        list(must_fix) + _violation_must_fix_items(blocking)
                    )
                    self._upsert_publication_figure(state, figure)
                    # A data-level objection exits before the revision loop, so
                    # any layout feedback from the same review would otherwise
                    # be discarded with it. Carry the measured defects forward
                    # so the regenerated figure does not reproduce them.
                    layout_note = ""
                    if blocking:
                        layout_note = (
                            "\n\nSeparately from the data issue, the rendering engine "
                            "measured these layout defects. The regenerated figure must "
                            "avoid them too:\n"
                            + format_violations_for_prompt(blocking)
                        )
                    return (
                        f"Figure {figure_name} has a data-level issue that cannot be fixed by revising the plot code. "
                        f"PI feedback: {feedback}\n"
                        f"Please address this in the analysis pipeline and regenerate the figure later."
                        f"{layout_note}"
                    )

                # Record this round's feedback in cumulative history
                revision_history.append({
                    "round": revision_count + 1,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "must_fix": must_fix,
                    "quality_score": review.get('quality_score', None)
                })

                figure.pi_feedback = feedback
                figure.revision_history = revision_history

                if revision_count >= 3:
                    # Out of revision budget
                    break

                # Detect stalled revision loop: if the quality score hasn't
                # improved across the last 2 rounds, further revisions are
                # unlikely to help — accept the current version to avoid
                # burning time and tokens.
                #
                # Two limits on that shortcut. It is a judgement call about
                # diminishing returns, so it is unavailable while the renderer
                # still measures text drawn over other text: that is not a
                # matter of taste and the reviewer's opinion does not settle it.
                # And a score that went *down* is a regression, not a stall —
                # roll back to the best rendered version rather than shipping
                # the worse one, which the old `<=` comparison did.
                if len(revision_history) >= 2 and not blocking:
                    prev_score = revision_history[-2].get("quality_score")
                    curr_score = revision_history[-1].get("quality_score")
                    if prev_score is not None and curr_score is not None and curr_score <= prev_score:
                        regressed = curr_score < prev_score
                        # Only roll back to something actually better. The
                        # snapshot is skipped for figures over the size cap, so
                        # best_snapshot can hold a version older and worse than
                        # the one being discarded.
                        rollback_worth_it = (
                            regressed
                            and best_snapshot is not None
                            and best_snapshot.get("score", -1) > curr_score
                        )
                        rolled_back = _restore_snapshot(best_snapshot) if rollback_worth_it else False
                        logger.warning(
                            f"[Bioinformatics] Figure {figure_name}: quality score "
                            f"{'regressed' if regressed else 'stalled'} "
                            f"({prev_score} -> {curr_score})"
                            f"{', rolled back to the best version' if rolled_back else ''}"
                            f", accepting current version"
                        )
                        figure.review_status = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS
                        _record_unresolved(must_fix)
                        self._upsert_publication_figure(state, figure)
                        await self._notify("status", f"Figure {figure_name} accepted (revision stalled)")
                        rollback_note = (
                            f"\nThe last revision scored lower than the previous one, so the "
                            f"earlier version (score {best_snapshot.get('score')}) was restored."
                            if rolled_back else ""
                        )
                        return (
                            f"Figure {figure_name} accepted with reservations (quality score did not improve "
                            f"after revision: {prev_score} -> {curr_score}).{rollback_note}\n"
                            f"Last feedback: {feedback}\n"
                            f"Please move on to the next analysis step."
                        )

                # Try to revise the figure
                await self._notify("status", f"Revising figure based on PI feedback...")
                logger.info(f"[Bioinformatics] Revising figure {figure_name}: {feedback}")

                # Generate revised code with full revision history
                try:
                    revised_code = await self._revise_figure_code(
                        current_code, feedback, suggestions, must_fix, figure_type, state, figure.png_base64,
                        revision_history=revision_history,
                        layout_violations=figure.layout_violations,
                    )
                except UpstreamExhaustedError:
                    raise
                except Exception as revise_err:
                    logger.error(f"[Bioinformatics] _revise_figure_code failed: {revise_err}, keeping current version")
                    await self._notify("status", f"Figure revision failed ({revise_err}), keeping current version")
                    break

                # Only increment revision count after successful code generation
                revision_count += 1
                figure.revision_count = revision_count

                # Validate revised code before execution
                is_valid, error_msg = self._validate_code(revised_code)
                if not is_valid:
                    logger.warning(f"[Bioinformatics] Revised code failed validation: {error_msg}")
                    break

                # Re-execute with revised code (rebuild wrapper instead of str.replace to avoid corruption)
                revised_wrapped = _build_publication_figure_wrapper(revised_code, figures_dir, jpg_path, pdf_path)
                result = await self.sandbox.execute(
                    revised_wrapped, skip_safety_check=True
                )

                if not result.error and jpg_path.exists():
                    # Detect blank revised figure. The wrapper saves in a
                    # finally block, so the blank render has already replaced
                    # the good file on disk — restore it rather than leave the
                    # session holding a blank image with a good base64.
                    if jpg_path.stat().st_size < 2000:
                        logger.warning(f"[Bioinformatics] Revised figure appears blank (size: {jpg_path.stat().st_size} bytes)")
                        _restore_snapshot(best_snapshot)
                        break
                    # Revised code that raises no error but creates no figure --
                    # `if len(sig): plot() else: print(...)` is the common shape
                    # -- leaves the PREVIOUS file untouched, so jpg_path.exists()
                    # and the size check both pass on stale bytes. Without this
                    # the loop would credit the old render to the new code.
                    if not _render_landed(result.output):
                        logger.warning(
                            f"[Bioinformatics] Figure {figure_name}: revision did not "
                            f"replace the rendered figure, keeping the previous version"
                        )
                        break
                    # Update figure with new image and re-measure its layout
                    figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                    if lint_ran(result.output):
                        figure.layout_violations = parse_layout_violations(result.output)
                        logger.info(
                            f"[Bioinformatics] Figure {figure_name} revision "
                            f"{revision_count} layout lint: "
                            f"{summarize_violations(figure.layout_violations)}"
                        )
                    else:
                        # No verdict for this render. Clearing the previous
                        # measurement would read as "the revision fixed it" and
                        # re-open both the stall shortcut and report eligibility
                        # -- the exact path this change exists to close.
                        logger.warning(
                            f"[Bioinformatics] Figure {figure_name} revision "
                            f"{revision_count}: layout unverified, keeping the "
                            f"previous measurement"
                        )
                    await self._notify("figure", figure.png_base64)
                    current_code = revised_code
                else:
                    result.error = strip_ansi(result.error) if result.error else result.error
                    logger.warning(f"[Bioinformatics] Revision failed: {result.error}")
                    error_msg = str(result.error)[:1000]

                    # Technique Agent documentation lookup for technical errors
                    extra_context = ""
                    if self.technique_agent:
                        err_type = extract_error_type(error_msg)
                        if err_type in TECHNICAL_ERROR_TYPES:
                            logger.info(f"[Bioinformatics] Looking up docs for {err_type} error")
                            doc_hint = await self.technique_agent.lookup_from_error(
                                self.sandbox, error_msg, revised_code
                            )
                            if doc_hint:
                                extra_context = f"\n\n[API Documentation]\n{doc_hint}"
                                logger.info(f"[Bioinformatics] Got documentation hint for {err_type}")

                    # Auto-retry: feed error + docs back to LLM
                    logger.info(f"[Bioinformatics] Auto-retrying revision with error context")
                    try:
                        retry_code = await self._revise_figure_code(
                            revised_code,
                            f"The previous code revision caused a runtime error:\n{error_msg}\nFix this error while keeping ALL the style improvements from the revision.{extra_context}",
                            [f"Fix the runtime error: {error_msg}"],
                            ["The code must execute without errors"],
                            figure_type, state, figure.png_base64,
                            revision_history=revision_history,
                            layout_violations=figure.layout_violations,
                        )
                        is_valid_retry, err_retry = self._validate_code(retry_code)
                        if is_valid_retry:
                            retry_wrapped = _build_publication_figure_wrapper(retry_code, figures_dir, jpg_path, pdf_path)
                            retry_result = await self.sandbox.execute(
                                retry_wrapped, skip_safety_check=True
                            )
                            if (
                                not retry_result.error
                                and jpg_path.exists()
                                and jpg_path.stat().st_size >= 2000
                                and _render_landed(retry_result.output)
                            ):
                                logger.info(f"[Bioinformatics] Auto-retry succeeded for {figure_name}")
                                figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                                if lint_ran(retry_result.output):
                                    figure.layout_violations = parse_layout_violations(retry_result.output)
                                await self._notify("figure", figure.png_base64)
                                current_code = retry_code
                                continue  # Back to review loop
                            else:
                                logger.warning(f"[Bioinformatics] Auto-retry also failed: {retry_result.error}")
                    except UpstreamExhaustedError:
                        raise
                    except Exception as retry_err:
                        logger.warning(f"[Bioinformatics] Auto-retry exception: {retry_err}")
                    break  # Both attempts failed

            # After 3 attempts, save with revision_requested status
            figure.review_status = FigureReviewStatus.REVISION_REQUESTED
            remaining_blocking = blocking_violations(figure.layout_violations)
            _record_unresolved(
                list(must_fix or []) + _violation_must_fix_items(remaining_blocking)
            )
            self._upsert_publication_figure(state, figure)
            logger.warning(f"[Bioinformatics] Figure {figure_name} not fully approved after {revision_count} revisions")
            layout_note = (
                f"\nUnresolved layout defects: {summarize_violations(remaining_blocking)}."
                if remaining_blocking else ""
            )
            return (
                f"Figure {figure_name} saved after {revision_count} revision attempts. "
                f"Status: needs further revision.\n"
                f"Last feedback: {figure.pi_feedback}{layout_note}"
            )

        # No review callback, just save the figure
        self._upsert_publication_figure(state, figure)
        return f"Figure {figure_name} has been generated and saved.\nJPG: {jpg_path}\nPDF: {pdf_path}"

    def _upsert_publication_figure(self, state, figure):
        """Replace existing figure with same name, or append if new."""
        refreshed_stems = {
            Path(str(figure.name)).stem,
            Path(str(figure.png_path)).stem,
        }
        state.invalidated_artifact_stems = [
            stem for stem in (state.invalidated_artifact_stems or [])
            if stem not in refreshed_stems
        ]
        existing_idx = next(
            (i for i, f in enumerate(state.publication_figures) if f.name == figure.name),
            None
        )
        if existing_idx is not None:
            state.publication_figures[existing_idx] = figure
        else:
            state.publication_figures.append(figure)

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

        figure_context = "**Current Figure:** An image of the current figure is attached. Review it carefully to understand what needs to be changed.\n\n" if image_base64 else ""

        layout_section = _layout_feedback_block(layout_violations)

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

        prompt = f"""Please revise the following plotting code based on PI feedback.

{figure_context}**Original Code:**
```python
{original_code}
```

**Figure Type:** {figure_type}
{exp_design_constraint}
{history_section}**PI Feedback (Current Round):** {feedback}
{layout_section}
**Improvement Suggestions (Current Round):**
{suggestions_text}

**Critical Issues (MUST FIX - Current Round):**
{must_fix_text}

**Requirements:**
1. Fix all critical issues listed above
2. Address as many suggestions as possible
3. CRITICAL: Do NOT revert previously applied fixes. If Round 1 changed grid lines to dashed, keep them dashed. If Round 2 added axis labels, keep them.
4. You MAY change the figure size, the GridSpec, the number of rows/columns, and where legends and annotations sit — resolving a collision usually requires exactly that. Do NOT change which data is plotted.
5. Keep using the same data variables (assume they exist), but REMOVE any fabricated/dummy data for groups not in the experimental design
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

Return only the Python code:"""

        self._set_llm_context()
        # Use slightly lower temperature for code revision (needs precision)
        revision_temp = min(self.config.temperature, 0.2)
        if image_base64 and self.has_vision:
            # Use vision API so the agent can see the current figure
            try:
                response = await self.llm_client.chat_with_vision(
                    model=self.model,
                    text_prompt=prompt,
                    images=[image_base64],
                    temperature=revision_temp,
                    max_tokens=32768
                )
            except Exception as vision_err:
                _is_vision_unsupported = (
                    "NotFoundError" in type(vision_err).__name__
                    or "404" in str(vision_err)
                    or "1210" in str(vision_err)  # ZhipuAI: API params error (non-vision model)
                )
                if _is_vision_unsupported:
                    # Model does not support vision input — disable for this session and fall back
                    logger.warning(
                        f"[Bioinformatics] Model {self.model} does not support vision "
                        f"({vision_err}), disabling vision for this session"
                    )
                    self.has_vision = False
                    prompt = prompt.replace(
                        "**Current Figure:** An image of the current figure is attached. "
                        "Review it carefully to understand what needs to be changed.\n\n",
                        "**Current Figure:** (Image not available — model does not support vision. "
                        "Revise code based on PI feedback text only.)\n\n",
                    )
                    response = await self.llm_client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=revision_temp,
                        max_tokens=32768
                    )
                else:
                    raise
        else:
            response = await self.llm_client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=revision_temp,
                max_tokens=32768
            )

        response_text = response.content if response and response.content else None
        if not response_text:
            logger.warning("[Bioinformatics] Empty response from LLM for code revision")
            return original_code  # Return unchanged code

        # Extract code from markdown response
        code = extract_code_from_markdown(response_text)

        return code.strip()
