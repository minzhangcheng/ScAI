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
from ...utils.error_parsing import compact_traceback
from ...utils.mechanism_policy import MECHANISM_EVIDENCE_POLICY
from ...services.figure_layout import (
    blocking_violations,
    format_violations_for_prompt,
    lint_ran,
    parse_layout_violations,
    summarize_violations,
)
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
    attempt_dir = figures_dir / ".attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir, attempt_dir / f"{safe_name}.jpg", attempt_dir / f"{safe_name}.pdf"


def _discard_render_attempt(attempt_dir: Path) -> None:
    try:
        shutil.rmtree(attempt_dir)
        parent = attempt_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        logger.warning("Could not remove failed figure attempt directory %s", attempt_dir)


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
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    archive_dir: Path | None = None
    archive_paths: dict[str, Path] = {}
    try:
        final_jpg.parent.mkdir(parents=True, exist_ok=True)
        for final in (final_jpg, final_pdf):
            if final.exists():
                backup = attempt_dir / f"previous_{final.name}"
                shutil.copy2(final, backup)
                backups.append((backup, final))
        if archived_record is not None and backups:
            archive_dir = (
                final_jpg.parent
                / ".revisions"
                / str(getattr(archived_record, "id", "unknown"))
            )
            archive_dir.mkdir(parents=True, exist_ok=True)
            for backup, final in backups:
                archived_path = archive_dir / final.name
                shutil.copy2(backup, archived_path)
                archive_paths[final.suffix.lower()] = archived_path
        os.replace(staged_jpg, final_jpg)
        published.append(final_jpg)
        os.replace(staged_pdf, final_pdf)
        published.append(final_pdf)
    except OSError as exc:
        for path in published:
            try:
                path.unlink()
            except OSError:
                pass
        for backup, final in backups:
            try:
                shutil.copy2(backup, final)
            except OSError:
                logger.exception("Could not restore prior figure render %s", final)
        if archive_dir is not None:
            try:
                shutil.rmtree(archive_dir)
            except OSError:
                pass
        _discard_render_attempt(attempt_dir)
        return f"Could not publish validated figure render: {exc}"
    if archived_record is not None:
        archived_jpg = archive_paths.get(".jpg") or archive_paths.get(".jpeg")
        archived_pdf = archive_paths.get(".pdf")
        if archived_jpg is not None:
            archived_record.png_path = str(archived_jpg)
        if archived_pdf is not None:
            archived_record.pdf_path = str(archived_pdf)
    _discard_render_attempt(attempt_dir)
    return None


def _render_landed(output: str | None) -> bool:
    """Whether this execution actually replaced the figure on disk."""
    text = output or ""
    return (
        _JPG_SAVED_MARKER in text
        and _NO_FIGURE_MARKER not in text
        and _SAVE_FAILED_MARKER not in text
    )


def _publication_output_error(
    jpg_path: Path, pdf_path: Path, output: str | None
) -> str | None:
    """Explain why a promised JPG+PDF render is not publication-ready."""
    if not _render_landed(output):
        return (
            "Figure generation produced no new JPG render; an older file with "
            "the same name, if present, was not re-registered."
        )
    if not jpg_path.exists():
        return f"JPG file was not created: {jpg_path}"
    try:
        jpg_size = jpg_path.stat().st_size
    except OSError:
        jpg_size = 0
    if jpg_size < 2000:
        return (
            f"Figure appears to be blank (file size: {jpg_size} bytes). "
            "The plotting code may not have generated visible content."
        )
    if image_file_sha256(jpg_path) is None:
        return f"JPG render is corrupt or cannot be decoded: {jpg_path}"
    if (
        _PDF_SAVE_FAILED_MARKER in (output or "")
        or _PDF_SAVED_MARKER not in (output or "")
    ):
        return (
            "Figure generation did not produce a new publication PDF; an older "
            "same-name PDF, if present, was not re-registered."
        )
    if not publication_pdf_is_valid(pdf_path):
        return f"Publication PDF is missing or invalid: {pdf_path}"
    return None


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
    return {
        "kind": "lint_skipped",
        "severity": "advise",
        "text": "",
        "what": "layout",
        "detail": (
            "the renderer returned no verdict for this render, so its layout "
            "is unverified rather than known to be clean"
        ),
    }


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

    async def _handle_generate_publication_figure(
        self,
        code: str,
        figure_name: str,
        description: str,
        figure_type: str,
        state: AnalysisState,
        review_callback: Optional[Callable[[PublicationFigure], Awaitable[Dict[str, Any]]]] = None,
        diagnostic: bool = False,
        plan_step_id: Optional[int] = None,
        purpose: Optional[str] = None,
        logical_figure_id: Optional[str] = None,
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
        logger.info(f"[Bioinformatics] Generating publication figure: {figure_name}")

        # Validate code before execution
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return f"Error: Code validation failed - {error_msg}"

        # Ensure figures directory exists
        figures_dir = Path(self.sandbox.working_dir) / "figures"
        # A previous worker may have died after either public-file replace or
        # after the durable record write but before its commit marker. Recover
        # that transaction before resolving the current logical head.
        try:
            from ...config import get_session_workspace
            if figures_dir.resolve() == (
                get_session_workspace(state.id) / "figures"
            ).resolve():
                from ...services.session_store import (
                    recover_publication_transactions_for_state,
                )
                await recover_publication_transactions_for_state(state.id, state)
        except Exception:
            logger.exception(
                "Could not recover publication transactions for %s", state.id
            )
            return "Could not recover pending publication figure transactions"

        # Clean up figure name
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in figure_name)
        final_jpg_path = figures_dir / f"{safe_name}.jpg"
        final_pdf_path = figures_dir / f"{safe_name}.pdf"

        sandbox_state = state.sandbox_state or {}
        current_plan = getattr(state, "current_plan", None)
        current_plan_id = str(getattr(current_plan, "id", "") or "").strip() or None
        active_lineage = getattr(state, "restore_branch_id", None) or "main"
        logical_id = str(logical_figure_id or "").strip()[:240]
        if not logical_id:
            # Tool calls require an explicit semantic ID.  Direct/internal
            # callers that predate the schema remain usable, but their attempts
            # are deliberately independent rather than grouped by filename.
            logical_id = f"render:{uuid.uuid4()}"
        prior_attempts = [
            item for item in eligible_artifacts(state.publication_figures)
            if (getattr(item, "lineage_id", None) or "main") == active_lineage
            and str(getattr(item, "logical_figure_id", None) or "") == logical_id
        ]
        prior_attempt = prior_attempts[-1] if prior_attempts else None
        render_attempt_id = str(uuid.uuid4())

        instrumented_code, sensitive_call_count = (
            instrument_expression_contract_calls(code)
        )
        preinstall_expression_runtime = has_direct_expression_contract_call(code)
        guarded_code = (
            instrumented_code
            if preinstall_expression_runtime or not sensitive_call_count
            else build_expression_runtime_code() + "\n" + instrumented_code
        )
        if preinstall_expression_runtime:
            runtime_setup = await self.sandbox.execute(
                build_expression_runtime_code(),
                save_to_history=False,
                skip_safety_check=True,
            )
            if runtime_setup.error:
                return (
                    "Figure generation blocked: "
                    + compact_traceback(runtime_setup.error, max_chars=3000)
                )

        representation_will_change = mutates_expression_representation(code)
        preexisting_artifact_versions = (
            workspace_artifact_versions(self.sandbox.working_dir)
            if representation_will_change else {}
        )
        attempt_dir, staged_jpg_path, staged_pdf_path = _new_render_attempt_paths(
            figures_dir, safe_name, render_attempt_id
        )
        # Wrap code to save both JPG and PDF; paths are repr()'d inside the
        # shared wrapper to prevent injection.
        wrapped_code = _build_publication_figure_wrapper(
            guarded_code, attempt_dir, staged_jpg_path, staged_pdf_path
        )

        # Execute the code. A transport/runtime exception must not strand an
        # attempt directory even though no CodeExecutionResult was returned.
        staleness_note = ""
        user_code_execution_started = False
        try:
            try:
                # Conservatively treat a transport failure or cancellation after
                # dispatch as a possibly executed, non-transactional mutation.
                user_code_execution_started = True
                result = await self.sandbox.execute(
                    wrapped_code, skip_safety_check=True
                )
            finally:
                try:
                    (
                        _invalidated_count,
                        _stale_stems,
                        staleness_note,
                    ) = record_executed_representation_change(
                        state,
                        code,
                        self.sandbox.working_dir,
                        preexisting_artifact_versions,
                        execution_started=user_code_execution_started,
                    )
                except Exception as provenance_error:
                    logger.warning(
                        "[Bioinformatics] Could not record figure expression "
                        "representation invalidation: %s",
                        compact_traceback(str(provenance_error), max_chars=1000),
                    )
        except BaseException:
            _discard_render_attempt(attempt_dir)
            raise

        expression_contracts = parse_expression_contracts(result.output)
        if expression_contracts:
            record_expression_contract(state, expression_contracts)
        result.output = strip_expression_contract_markers(result.output)

        if result.error:
            _discard_render_attempt(attempt_dir)
            result.error = compact_traceback(result.error, max_chars=3000)
            if staleness_note:
                result.error = f"{result.error}\n{staleness_note}".strip()
            error_msg = f"Figure generation failed: {result.error}"
            logger.error(f"[Bioinformatics] {error_msg}")
            await self._notify("error", error_msg)
            return error_msg

        output_error = _publication_output_error(
            staged_jpg_path, staged_pdf_path, result.output
        )
        if output_error:
            _discard_render_attempt(attempt_dir)
            logger.warning("[Bioinformatics] %s", output_error)
            return output_error

        # Keep all review/revision work under the hidden attempt directory.  A
        # report or workspace scan must continue to see the prior published
        # bytes until this logical attempt reaches a terminal producer state.
        jpg_path = staged_jpg_path
        pdf_path = staged_pdf_path

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
            # "Unverified" has to be written down, not just logged. The revision
            # path keeps the previous measurement for exactly this reason; the
            # first render has none to keep, so it records the absence itself —
            # otherwise the figure carries [], which is what a clean measurement
            # carries, and the layout gate is the only check a diagnostic figure
            # is subject to at all.
            layout_violations = [_layout_unverified_marker()]
            logger.warning(
                f"[Bioinformatics] Figure {safe_name}: layout lint did not "
                f"report — treating layout as unverified, not as clean"
            )

        # Read JPG and compress to base64
        try:
            png_b64 = self._compress_image_to_base64(jpg_path, max_width=800)
        except BaseException:
            _discard_render_attempt(attempt_dir)
            raise

        # Create PublicationFigure object (png_path field stores JPG path for web display)
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
            diagnostic=diagnostic,
            plan_step_id=plan_step_id,
            plan_id=current_plan_id,
            purpose=purpose or ("diagnostic" if diagnostic else "result"),
            logical_figure_id=logical_id,
            render_attempt_id=render_attempt_id,
            producer_status="pending",
            revision_of_id=getattr(prior_attempt, "id", None),
            is_published_head=False,
            lineage_id=active_lineage,
            round_number=int(state.current_round or 0),
            analysis_generation=int(sandbox_state.get("analysis_generation", 0) or 0),
            expression_provenance=dict(
                sandbox_state.get("expression_contract_v1") or {}
            ),
        )
        self._refresh_figure_integrity_hashes(figure)

        # Register and durably journal the pending render before asking any
        # reviewer to inspect it. A process death during that await is recovered
        # as an abandoned audit record on the next load.
        try:
            pending_transaction = register_pending_figure_attempt(
                attempt_dir=attempt_dir,
                staged_jpg=staged_jpg_path,
                staged_pdf=staged_pdf_path,
                final_jpg=final_jpg_path,
                final_pdf=final_pdf_path,
                published_record=figure,
                session_id=state.id,
            )
        except Exception as exc:
            _discard_render_attempt(attempt_dir)
            return f"Could not journal publication figure attempt: {exc}"

        # Register the pending render before asking any reviewer to inspect it.
        # Review events intentionally omit the large preview and the frontend
        # fetches it through the figure API; without this upsert that request
        # races the review callback and returns 404 (or shows an older revision).
        self._upsert_publication_figure(
            state, figure, refresh_integrity=False,
        )
        try:
            await self._persist_figure_record_boundary(state, figures_dir)
        except Exception as exc:
            rollback_figure_publish_transaction(
                Path(pending_transaction["journal_path"]),
                state=state,
                producer_status="failed",
            )
            return f"Could not persist pending publication figure record: {exc}"

        render_published = False

        async def _persist_terminal_rollback() -> None:
            try:
                persisted = await self._persist_figure_record_boundary(
                    state, figures_dir
                )
                if persisted:
                    mark_figure_record_recovery_durable(
                        figures_dir,
                        [str(getattr(figure, "render_attempt_id", "") or "")],
                    )
            except Exception:
                # The terminal marker remains deliberately non-durable. The
                # next session load will reapply the rollback and retry the
                # lifecycle-locked session write.
                logger.exception(
                    "Could not persist terminal figure rollback for %s",
                    figure.id,
                )

        async def _finalize_render() -> str | None:
            """Commit this reviewed render once, preserving any path owner."""
            nonlocal render_published
            if render_published:
                return None
            lock_error = acquire_figure_publish_locks(
                figures_dir=figures_dir,
                final_jpg=final_jpg_path,
                published_record=figure,
            )
            if lock_error:
                rollback_figure_publish_transaction(
                    Path(pending_transaction["journal_path"]),
                    state=state,
                    producer_status="failed",
                )
                await _persist_terminal_rollback()
                return lock_error
            try:
                frozen_transaction = refreeze_pending_figure_attempt(
                    pending_transaction,
                    attempt_dir=attempt_dir,
                    staged_jpg=jpg_path,
                    staged_pdf=pdf_path,
                    final_jpg=final_jpg_path,
                    final_pdf=final_pdf_path,
                    published_record=figure,
                )
                pending_transaction.clear()
                pending_transaction.update(frozen_transaction)
            except Exception as exc:
                freeze_error = (
                    "Could not authenticate final reviewed figure bytes: "
                    f"{exc}"
                )
                rollback_figure_publish_transaction(
                    Path(pending_transaction["journal_path"]),
                    state=state,
                    producer_status="failed",
                )
                await _persist_terminal_rollback()
                return freeze_error
            archival_owner = next(
                (
                    item for item in reversed(state.publication_figures)
                    if item is not figure
                    and (getattr(item, "lineage_id", None) or "main")
                    == active_lineage
                    and str(getattr(item, "png_path", "")) == str(final_jpg_path)
                    and archival_render_available(item)
                ),
                None,
            )
            figure.producer_status = "publishing"
            self._upsert_publication_figure(state, figure)
            transaction = None
            try:
                transaction, publish_error = begin_figure_publish_transaction(
                    attempt_dir=attempt_dir,
                    staged_jpg=jpg_path,
                    staged_pdf=pdf_path,
                    final_jpg=final_jpg_path,
                    final_pdf=final_pdf_path,
                    published_record=figure,
                    archived_record=archival_owner,
                )
            except BaseException:
                figure.producer_status = "abandoned"
                self._clear_unpublished_figure_payload(figure)
                self._upsert_publication_figure(state, figure)
                rollback_figure_publish_transaction(
                    Path(pending_transaction["journal_path"]),
                    state=state,
                    producer_status="abandoned",
                )
                await _persist_terminal_rollback()
                raise
            if publish_error:
                figure.producer_status = "failed"
                self._clear_unpublished_figure_payload(figure)
                figure.review_status = FigureReviewStatus.REVISION_REQUESTED
                figure.pi_feedback = (
                    f"{figure.pi_feedback}\n\n{publish_error}"
                    if figure.pi_feedback else publish_error
                )
                self._upsert_publication_figure(state, figure)
                await _persist_terminal_rollback()
                return publish_error
            try:
                apply_publish_record_paths(transaction, figure, archival_owner)
                figure.producer_status = "published"
                self._refresh_figure_integrity_hashes(figure)
                self._upsert_publication_figure(state, figure)
                mark_publish_record_applied(
                    transaction,
                    published_record=figure,
                    archived_record=archival_owner,
                )
                # The transaction marker is written only after session.json has
                # crossed its own fsync boundary.  Tests and non-session direct
                # callers do not own that canonical workspace and safely skip
                # the persistence callback.
                await self._persist_figure_record_boundary(state, figures_dir)
                commit_figure_publish_transaction(transaction)
            except BaseException:
                if transaction is not None:
                    rollback_figure_publish_transaction(
                        Path(transaction["journal_path"]),
                        state=state,
                        producer_status="abandoned",
                    )
                else:
                    figure.producer_status = "abandoned"
                    self._clear_unpublished_figure_payload(figure)
                    self._upsert_publication_figure(state, figure)
                await _persist_terminal_rollback()
                raise
            render_published = True
            return None

        def _abandon_render() -> None:
            """Remove a non-terminal attempt without disturbing any old head."""
            if not render_published:
                rollback_figure_publish_transaction(
                    Path(pending_transaction["journal_path"]),
                    state=state,
                    producer_status="abandoned",
                )
            else:
                self._upsert_publication_figure(state, figure)

        async def _guard_attempt_await(awaitable):
            """Clean staging if an external await aborts the producer attempt."""
            try:
                return await awaitable
            except BaseException:
                _abandon_render()
                await _persist_terminal_rollback()
                raise

        # Send figure to frontend for display
        await _guard_attempt_await(self._notify("figure", png_b64))

        async def _execute_revision_candidate(source: str):
            nonlocal staleness_note
            candidate_id = str(uuid.uuid4())
            candidate_dir, candidate_jpg, candidate_pdf = _new_render_attempt_paths(
                figures_dir, safe_name, candidate_id
            )
            instrumented_source, candidate_sensitive_call_count = (
                instrument_expression_contract_calls(source)
            )
            candidate_preinstall_runtime = has_direct_expression_contract_call(
                source
            )
            guarded_source = (
                instrumented_source
                if candidate_preinstall_runtime or not candidate_sensitive_call_count
                else build_expression_runtime_code() + "\n" + instrumented_source
            )
            if candidate_preinstall_runtime:
                try:
                    runtime_setup = await self.sandbox.execute(
                        build_expression_runtime_code(),
                        save_to_history=False,
                        skip_safety_check=True,
                    )
                except BaseException:
                    _discard_render_attempt(candidate_dir)
                    raise
                if runtime_setup.error:
                    runtime_setup.error = compact_traceback(
                        runtime_setup.error, max_chars=3000
                    )
                    _discard_render_attempt(candidate_dir)
                    return runtime_setup, None

            candidate_representation_will_change = (
                mutates_expression_representation(source)
            )
            candidate_preexisting_versions = (
                workspace_artifact_versions(self.sandbox.working_dir)
                if candidate_representation_will_change else {}
            )
            candidate_wrapper = _build_publication_figure_wrapper(
                guarded_source, candidate_dir, candidate_jpg, candidate_pdf
            )
            candidate_staleness_note = ""
            candidate_execution_started = False
            try:
                try:
                    candidate_execution_started = True
                    candidate_result = await self.sandbox.execute(
                        candidate_wrapper, skip_safety_check=True
                    )
                finally:
                    try:
                        (
                            _candidate_invalidated_count,
                            _candidate_stale_stems,
                            candidate_staleness_note,
                        ) = record_executed_representation_change(
                            state,
                            source,
                            self.sandbox.working_dir,
                            candidate_preexisting_versions,
                            execution_started=candidate_execution_started,
                        )
                        if candidate_staleness_note:
                            staleness_note = candidate_staleness_note
                    except Exception as provenance_error:
                        logger.warning(
                            "[Bioinformatics] Could not record revised figure "
                            "expression representation invalidation: %s",
                            compact_traceback(
                                str(provenance_error), max_chars=1000
                            ),
                        )
            except BaseException:
                _discard_render_attempt(candidate_dir)
                raise
            candidate_contracts = parse_expression_contracts(
                candidate_result.output
            )
            if candidate_contracts:
                record_expression_contract(state, candidate_contracts)
            candidate_result.output = strip_expression_contract_markers(
                candidate_result.output
            )
            if candidate_result.error:
                candidate_result.error = compact_traceback(
                    candidate_result.error, max_chars=3000
                )
                if candidate_staleness_note:
                    candidate_result.error = (
                        f"{candidate_result.error}\n{candidate_staleness_note}"
                    ).strip()
                _discard_render_attempt(candidate_dir)
                return candidate_result, None
            candidate_error = _publication_output_error(
                candidate_jpg, candidate_pdf, candidate_result.output
            )
            if candidate_error:
                _discard_render_attempt(candidate_dir)
                return candidate_result, candidate_error
            candidate_error = _publish_render_attempt(
                candidate_dir, candidate_jpg, candidate_pdf, jpg_path, pdf_path
            )
            return candidate_result, candidate_error

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
                figure.review_status = FigureReviewStatus.PENDING
                self._upsert_publication_figure(state, figure)
                logger.info(f"[Bioinformatics] Submitting figure for PI review (attempt {revision_count + 1})")
                await _guard_attempt_await(
                    self._notify("status", f"PI reviewing figure: {figure_name}...")
                )
                review = await _guard_attempt_await(review_callback(figure))
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
                    figure.pi_feedback = review.get('feedback', '') or None
                    _record_unresolved(_violation_must_fix_items(blocking))
                    publish_error = await _finalize_render()
                    if publish_error:
                        return publish_error
                    await self._emit_figure_record(
                        "figure_approved", figure, review=review
                    )
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
                        + (f"\n{staleness_note}" if staleness_note else "")
                    )

                # PI requested revision
                feedback = review.get('feedback', '')
                suggestions = review.get('suggestions', [])
                must_fix = review.get('must_fix', [])

                # If PI flagged a data/analysis issue that can't be fixed by revising plot code,
                # break out of the revision loop and defer to the main analysis flow
                if review.get('requires_reanalysis', False):
                    logger.warning(f"[Bioinformatics] Figure {figure_name} requires upstream reanalysis, skipping revision loop")
                    await _guard_attempt_await(self._notify(
                        "status",
                        f"Figure {figure_name}: PI identified data-level issue, deferring to analysis",
                    ))
                    figure.review_status = FigureReviewStatus.REVISION_REQUESTED
                    figure.pi_feedback = f"[REQUIRES REANALYSIS] {feedback}"
                    figure.revision_history = revision_history
                    _record_unresolved(
                        list(must_fix) + _violation_must_fix_items(blocking)
                    )
                    publish_error = await _finalize_render()
                    if publish_error:
                        return publish_error
                    await self._emit_figure_record(
                        "figure_revision", figure, review=review
                    )
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
                        + (f"\n{staleness_note}" if staleness_note else "")
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
                figure.review_status = FigureReviewStatus.REVISION_REQUESTED
                self._upsert_publication_figure(state, figure)
                await _guard_attempt_await(self._emit_figure_record(
                    "figure_revision", figure, review=review
                ))

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
                        publish_error = await _finalize_render()
                        if publish_error:
                            return publish_error
                        await self._emit_figure_record(
                            "figure_approved", figure, review=review
                        )
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
                            + (f"\n{staleness_note}" if staleness_note else "")
                        )

                # Try to revise the figure
                await _guard_attempt_await(self._notify(
                    "status", "Revising figure based on PI feedback..."
                ))
                logger.info(f"[Bioinformatics] Revising figure {figure_name}: {feedback}")

                # Generate revised code with full revision history
                try:
                    revised_code = await self._revise_figure_code(
                        current_code, feedback, suggestions, must_fix, figure_type, state, figure.png_base64,
                        revision_history=revision_history,
                        layout_violations=figure.layout_violations,
                    )
                except UpstreamExhaustedError:
                    _abandon_render()
                    raise
                except Exception as revise_err:
                    revise_error = compact_traceback(
                        str(revise_err), max_chars=1000
                    )
                    logger.error(
                        "[Bioinformatics] _revise_figure_code failed: %s; "
                        "keeping current version",
                        revise_error,
                    )
                    await self._notify(
                        "status",
                        f"Figure revision failed ({revise_error}), keeping current version",
                    )
                    break

                # Only increment revision count after successful code generation
                revision_count += 1
                figure.revision_count = revision_count

                # Validate revised code before execution
                is_valid, error_msg = self._validate_code(revised_code)
                if not is_valid:
                    logger.warning(f"[Bioinformatics] Revised code failed validation: {error_msg}")
                    break

                # Keep the previous bytes for quality-score rollback. Physical
                # failures no longer need this snapshot: candidates stay hidden
                # in staging until their JPG/PDF pair has passed validation.
                pre_revision_snapshot = _take_snapshot(quality_score)

                try:
                    result, physical_error = await _execute_revision_candidate(revised_code)
                except BaseException:
                    _abandon_render()
                    raise
                if not result.error and not physical_error:
                    # Update figure with new image and re-measure its layout
                    figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                    revised_sandbox_state = state.sandbox_state or {}
                    figure.analysis_generation = int(
                        revised_sandbox_state.get("analysis_generation", 0) or 0
                    )
                    figure.expression_provenance = dict(
                        revised_sandbox_state.get("expression_contract_v1") or {}
                    )
                    # The previous bytes of this pending attempt were correctly
                    # invalidated when the revision mutated the canonical
                    # representation.  The replacement render was produced
                    # afterwards and is therefore current for the new
                    # generation.
                    figure.is_current = True
                    figure.invalidated_at = None
                    figure.invalidation_reason = None
                    self._refresh_figure_integrity_hashes(figure)
                    figure.review_status = FigureReviewStatus.PENDING
                    self._upsert_publication_figure(state, figure)
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
                    await _guard_attempt_await(
                        self._notify("figure", figure.png_base64)
                    )
                    current_code = revised_code
                elif physical_error:
                    logger.warning(
                        "[Bioinformatics] Figure %s revision rejected: %s",
                        figure_name,
                        physical_error,
                    )
                    _restore_snapshot(pre_revision_snapshot or best_snapshot)
                    break
                else:
                    error_msg = compact_traceback(str(result.error), max_chars=1000)
                    result.error = error_msg
                    logger.warning(
                        "[Bioinformatics] Revision failed: %s", error_msg
                    )

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
                            retry_result, retry_physical_error = (
                                await _execute_revision_candidate(retry_code)
                            )
                            if not retry_result.error and not retry_physical_error:
                                logger.info(f"[Bioinformatics] Auto-retry succeeded for {figure_name}")
                                figure.png_base64 = self._compress_image_to_base64(jpg_path, max_width=800)
                                self._refresh_figure_integrity_hashes(figure)
                                figure.review_status = FigureReviewStatus.PENDING
                                self._upsert_publication_figure(state, figure)
                                if lint_ran(retry_result.output):
                                    figure.layout_violations = parse_layout_violations(retry_result.output)
                                await _guard_attempt_await(
                                    self._notify("figure", figure.png_base64)
                                )
                                current_code = retry_code
                                continue  # Back to review loop
                            else:
                                logger.warning(
                                    "[Bioinformatics] Auto-retry also failed: %s",
                                    retry_result.error or retry_physical_error,
                                )
                    except UpstreamExhaustedError:
                        _abandon_render()
                        raise
                    except asyncio.CancelledError:
                        _abandon_render()
                        raise
                    except Exception as retry_err:
                        logger.warning(
                            "[Bioinformatics] Auto-retry exception: %s",
                            compact_traceback(str(retry_err), max_chars=1000),
                        )
                    _restore_snapshot(pre_revision_snapshot or best_snapshot)
                    break  # Both attempts failed

            # After 3 attempts, save with revision_requested status
            figure.review_status = FigureReviewStatus.REVISION_REQUESTED
            remaining_blocking = blocking_violations(figure.layout_violations)
            _record_unresolved(
                list(must_fix or []) + _violation_must_fix_items(remaining_blocking)
            )
            publish_error = await _finalize_render()
            if publish_error:
                return publish_error
            await self._emit_figure_record(
                "figure_revision", figure, review=review
            )
            logger.warning(f"[Bioinformatics] Figure {figure_name} not fully approved after {revision_count} revisions")
            layout_note = (
                f"\nUnresolved layout defects: {summarize_violations(remaining_blocking)}."
                if remaining_blocking else ""
            )
            return (
                f"Figure {figure_name} saved after {revision_count} revision attempts. "
                f"Status: needs further revision.\n"
                f"Last feedback: {figure.pi_feedback}{layout_note}"
                + (f"\n{staleness_note}" if staleness_note else "")
            )

        # No review callback, just save the figure
        publish_error = await _finalize_render()
        if publish_error:
            return publish_error
        await self._emit_figure_record("figure_review", figure)
        return (
            f"Figure {figure_name} has been generated and saved.\n"
            f"JPG: {final_jpg_path}\nPDF: {final_pdf_path}"
            + (f"\n{staleness_note}" if staleness_note else "")
        )

    @staticmethod
    def _refresh_figure_integrity_hashes(figure) -> None:
        """Bind a record to its full render and compact persisted preview."""
        figure.render_sha256 = image_file_sha256(figure.png_path)
        figure.preview_sha256 = preview_render_sha256(figure.png_base64)
        figure.pdf_sha256 = pdf_file_sha256(figure.pdf_path)

    @staticmethod
    def _clear_unpublished_figure_payload(figure) -> None:
        """Keep audit metadata while removing staged bytes and direct paths."""
        figure.png_base64 = None
        figure.png_path = ""
        figure.pdf_path = ""
        figure.render_sha256 = None
        figure.preview_sha256 = None
        figure.pdf_sha256 = None
        figure.is_published_head = False

    async def _persist_figure_record_boundary(
        self, state: AnalysisState, figures_dir: Path
    ) -> bool:
        """Durably persist the record half for canonical session workspaces."""
        try:
            from ...config import get_session_workspace

            canonical = (get_session_workspace(state.id) / "figures").resolve()
            actual = figures_dir.resolve()
        except Exception:
            return False
        if actual != canonical:
            return False
        from ...services.session_store import async_save_session

        await async_save_session(state.id, state, force=True)
        return True

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
        review = review or {}
        status = getattr(figure.review_status, "value", figure.review_status)
        await self._notify(
            event_type,
            {
                "figure_id": figure.id,
                "figure_name": figure.name,
                "approved": status in {
                    FigureReviewStatus.APPROVED.value,
                    FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS.value,
                },
                "quality_score": review.get("quality_score"),
                "feedback": figure.pi_feedback or review.get("feedback", ""),
                "suggestions": review.get("suggestions", []),
                "must_fix": review.get("must_fix", []),
                "figure": {
                    "id": figure.id,
                    "name": figure.name,
                    "description": figure.description,
                    "figure_type": figure.figure_type,
                    "png_path": figure.png_path,
                    "pdf_path": figure.pdf_path,
                    "review_status": status,
                    "pi_feedback": figure.pi_feedback,
                    "revision_count": figure.revision_count,
                    "plan_step_id": getattr(figure, "plan_step_id", None),
                    "plan_id": getattr(figure, "plan_id", None),
                    "purpose": getattr(figure, "purpose", "result") or "result",
                    "logical_figure_id": getattr(figure, "logical_figure_id", None),
                    "render_attempt_id": getattr(figure, "render_attempt_id", None),
                    "producer_status": getattr(
                        figure, "producer_status", "published"
                    ),
                    "revision_of_id": getattr(figure, "revision_of_id", None),
                    "supersedes_id": getattr(figure, "supersedes_id", None),
                    "is_published_head": bool(
                        getattr(figure, "is_published_head", False)
                    ),
                    "round": getattr(figure, "round_number", 0) or None,
                    "diagnostic": bool(getattr(figure, "diagnostic", False)),
                    "layout_blocking": bool(
                        blocking_violations(figure.layout_violations)
                    ),
                    "created_at": figure.created_at.isoformat(),
                },
            },
        )

    def _upsert_publication_figure(
        self, state, figure, *, refresh_integrity: bool = True,
    ):
        """Update one attempt and advance its published head only when eligible."""
        if refresh_integrity:
            self._refresh_figure_integrity_hashes(figure)
        existing_idx = next(
            (
                i for i, f in enumerate(state.publication_figures)
                if getattr(f, "id", None) == getattr(figure, "id", None)
            ),
            None
        )
        if existing_idx is not None:
            state.publication_figures[existing_idx] = figure
        else:
            state.publication_figures.append(figure)

        if not is_report_eligible(figure):
            return

        # A pending or failed redraw must never make an invalidated logical
        # artifact current again merely because it reused the old filename.
        refreshed_stems = {
            Path(str(figure.name)).stem,
            Path(str(figure.png_path)).stem,
        }
        state.invalidated_artifact_stems = [
            stem for stem in (state.invalidated_artifact_stems or [])
            if stem not in refreshed_stems
        ]

        family_key = logical_figure_key(figure)
        prior_heads = [
            item for item in state.publication_figures
            if item is not figure
            and logical_figure_key(item) == family_key
            and is_report_eligible(item)
        ]
        prior_head = next(
            (
                item for item in reversed(prior_heads)
                if getattr(item, "is_published_head", False)
            ),
            prior_heads[-1] if prior_heads else None,
        )
        if prior_head is not None and not figure.supersedes_id:
            figure.supersedes_id = prior_head.id
        for item in state.publication_figures:
            if item is not figure and logical_figure_key(item) == family_key:
                item.is_published_head = False
        figure.is_published_head = True

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
                    max_tokens=32768,
                    system_prompt=self.system_prompt,
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
