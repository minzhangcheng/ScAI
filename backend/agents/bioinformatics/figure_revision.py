"""Publication-figure revision rendering and retry phases."""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

from .figure_persistence import abandon_render, guard_attempt_await
from .figure_preflight import FigureAttemptState
from .shared import FigureReviewStatus, TECHNICAL_ERROR_TYPES, UpstreamExhaustedError


def take_snapshot(
    attempt: FigureAttemptState,
    score,
    *,
    review: dict | None = None,
    must_fix: list[str] | None = None,
) -> dict | None:
    try:
        jpg_bytes = attempt.jpg_path.read_bytes() if attempt.jpg_path.exists() else None
        pdf_bytes = attempt.pdf_path.read_bytes() if attempt.pdf_path.exists() else None
    except Exception as snap_err:
        attempt.deps.get("logger").debug(
            "[Bioinformatics] Figure snapshot skipped: %s", snap_err
        )
        return None
    if len(jpg_bytes or b"") + len(pdf_bytes or b"") > attempt.deps.get(
        "_MAX_ROLLBACK_SNAPSHOT_BYTES"
    ):
        return None
    return {
        "score": score,
        "code": attempt.current_code,
        "jpg": jpg_bytes,
        "pdf": pdf_bytes,
        "b64": attempt.figure.png_base64,
        "violations": list(attempt.figure.layout_violations or []),
        "review": dict(review or {}),
        "must_fix": list(must_fix or []),
    }


def restore_snapshot(attempt: FigureAttemptState, snapshot) -> bool:
    if not snapshot:
        return False
    try:
        if snapshot.get("jpg"):
            attempt.jpg_path.write_bytes(snapshot["jpg"])
        if snapshot.get("pdf"):
            attempt.pdf_path.write_bytes(snapshot["pdf"])
    except Exception as restore_err:
        attempt.deps.get("logger").warning(
            "[Bioinformatics] Could not roll back %s to the previous version: %s",
            attempt.request.figure_name,
            restore_err,
        )
        return False
    attempt.figure.png_base64 = snapshot.get("b64") or attempt.figure.png_base64
    attempt.figure.layout_violations = list(snapshot.get("violations") or [])
    return True


async def execute_revision_candidate(
    attempt: FigureAttemptState, source: str
):
    deps = attempt.deps
    agent = attempt.agent
    figure_followup = attempt.request.producer_run_kind == "figure_followup"
    if figure_followup:
        from ...services.figure_followup import validate_plotting_only_code

        capability_error = validate_plotting_only_code(source)
        if capability_error:
            return SimpleNamespace(
                output="",
                error=f"Figure follow-up revision rejected: {capability_error}",
            ), None

    candidate_id = str(uuid.uuid4())
    candidate_figures_dir = (
        Path(agent.sandbox.working_dir) / "figures"
        if figure_followup
        else attempt.figures_dir
    )
    candidate_dir, candidate_jpg, candidate_pdf = deps.get(
        "_new_render_attempt_paths"
    )(candidate_figures_dir, attempt.safe_name, candidate_id)
    if figure_followup:
        instrumented_source, candidate_sensitive_call_count = source, 0
        candidate_preinstall_runtime = False
    else:
        instrumented_source, candidate_sensitive_call_count = deps.get(
            "instrument_expression_contract_calls"
        )(source)
        candidate_preinstall_runtime = deps.get(
            "has_direct_expression_contract_call"
        )(source)
    runtime_code = deps.get("build_expression_runtime_code")
    guarded_source = (
        instrumented_source
        if candidate_preinstall_runtime or not candidate_sensitive_call_count
        else runtime_code() + "\n" + instrumented_source
    )
    if candidate_preinstall_runtime:
        try:
            runtime_setup = await agent.sandbox.execute(
                runtime_code(), skip_safety_check=True
            )
        except BaseException:
            deps.get("_discard_render_attempt")(candidate_dir)
            raise
        if runtime_setup.error:
            runtime_setup.error = deps.get("compact_traceback")(
                runtime_setup.error, max_chars=3000
            )
            deps.get("_discard_render_attempt")(candidate_dir)
            return runtime_setup, None

    representation_will_change = False
    if not figure_followup:
        representation_will_change = deps.get("mutates_expression_representation")(
            source
        )
    preexisting_versions = (
        deps.get("workspace_artifact_versions")(agent.sandbox.working_dir)
        if representation_will_change
        else {}
    )
    wrapper = deps.get("_build_publication_figure_wrapper")(
        guarded_source, candidate_dir, candidate_jpg, candidate_pdf
    )
    candidate_staleness_note = ""
    execution_started = False
    try:
        try:
            execution_started = True
            candidate_result = await agent.sandbox.execute(
                wrapper, skip_safety_check=True
            )
        finally:
            if not figure_followup:
                try:
                    (
                        _invalidated_count,
                        _stale_stems,
                        candidate_staleness_note,
                    ) = deps.get("record_executed_representation_change")(
                        attempt.state,
                        source,
                        agent.sandbox.working_dir,
                        preexisting_versions,
                        execution_started=execution_started,
                    )
                    if candidate_staleness_note:
                        attempt.staleness_note = candidate_staleness_note
                except Exception as provenance_error:
                    deps.get("logger").warning(
                        "[Bioinformatics] Could not record revised figure expression "
                        "representation invalidation: %s",
                        deps.get("compact_traceback")(
                            str(provenance_error), max_chars=1000
                        ),
                    )
    except BaseException:
        deps.get("_discard_render_attempt")(candidate_dir)
        raise
    if not figure_followup:
        contracts = deps.get("parse_expression_contracts")(candidate_result.output)
        if contracts:
            deps.get("record_expression_contract")(attempt.state, contracts)
        candidate_result.output = deps.get("strip_expression_contract_markers")(
            candidate_result.output
        )
    if candidate_result.error:
        candidate_result.error = deps.get("compact_traceback")(
            candidate_result.error, max_chars=3000
        )
        if candidate_staleness_note:
            candidate_result.error = (
                f"{candidate_result.error}\n{candidate_staleness_note}"
            ).strip()
        deps.get("_discard_render_attempt")(candidate_dir)
        return candidate_result, None
    candidate_error = deps.get("_publication_output_error")(
        candidate_jpg, candidate_pdf, candidate_result.output
    )
    if candidate_error:
        deps.get("_discard_render_attempt")(candidate_dir)
        return candidate_result, candidate_error
    if figure_followup:
        owner_dir = None
        try:
            owner_dir, owner_jpg, owner_pdf = deps.get(
                "_new_render_attempt_paths"
            )(attempt.figures_dir, attempt.safe_name, candidate_id)
            shutil.copyfile(candidate_jpg, owner_jpg)
            shutil.copyfile(candidate_pdf, owner_pdf)
            jpg_sha = deps.get("image_file_sha256")(candidate_jpg)
            pdf_sha = deps.get("pdf_file_sha256")(candidate_pdf)
            if (
                not jpg_sha
                or not pdf_sha
                or jpg_sha != deps.get("image_file_sha256")(owner_jpg)
                or pdf_sha != deps.get("pdf_file_sha256")(owner_pdf)
            ):
                raise OSError("validated renderer bytes changed while being staged")
        except OSError as exc:
            if owner_dir is not None:
                deps.get("_discard_render_attempt")(owner_dir)
            deps.get("_discard_render_attempt")(candidate_dir)
            return candidate_result, (
                f"Could not transfer validated figure revision from disposable renderer: {exc}"
            )
        deps.get("_discard_render_attempt")(candidate_dir)
        candidate_dir, candidate_jpg, candidate_pdf = owner_dir, owner_jpg, owner_pdf
    candidate_error = deps.get("_publish_render_attempt")(
        candidate_dir,
        candidate_jpg,
        candidate_pdf,
        attempt.jpg_path,
        attempt.pdf_path,
    )
    return candidate_result, candidate_error


def _accept_revision_output(attempt: FigureAttemptState, result, revised_code: str) -> None:
    figure = attempt.figure
    deps = attempt.deps
    figure.png_base64 = attempt.agent._compress_image_to_base64(
        attempt.jpg_path, max_width=800
    )
    from .receipt_lookup import artifact_provenance

    provenance, provenance_error = artifact_provenance(
        attempt.state,
        figure.source_execution_id,
        fallback_lineage=figure.lineage_id,
        fallback_round=figure.round_number,
        fallback_plan_id=figure.plan_id,
        current_receipts=getattr(attempt.agent, "code_history", None) or [],
    )
    if provenance_error:
        figure.producer_status = "failed"
        attempt.terminal_revision_reason = "source_execution_unresolved"
        return
    figure.lineage_id = provenance["lineage_id"]
    figure.round_number = provenance["round_number"]
    figure.plan_id = provenance["plan_id"] or figure.plan_id
    figure.analysis_generation = provenance["analysis_generation"]
    figure.cohort_generation = provenance["cohort_generation"]
    figure.cohort_digest = provenance["cohort_digest"]
    figure.expression_provenance = dict(
        (attempt.state.sandbox_state or {}).get("expression_contract_v1") or {}
    )
    figure.is_current = True
    figure.invalidated_at = None
    figure.invalidation_reason = None
    attempt.agent._refresh_figure_integrity_hashes(figure)
    figure.review_status = FigureReviewStatus.PENDING
    attempt.agent._upsert_publication_figure(attempt.state, figure)
    if deps.get("lint_ran")(result.output):
        figure.layout_violations = deps.get("parse_layout_violations")(result.output)
        deps.get("logger").info(
            "[Bioinformatics] Figure %s revision %s layout lint: %s",
            attempt.request.figure_name,
            attempt.revision_count,
            deps.get("summarize_violations")(figure.layout_violations),
        )
    else:
        deps.get("logger").warning(
            "[Bioinformatics] Figure %s revision %s: layout unverified, "
            "keeping the previous measurement",
            attempt.request.figure_name,
            attempt.revision_count,
        )
    attempt.current_code = revised_code


def _accept_retry_output(attempt: FigureAttemptState, result, retry_code: str) -> None:
    """Apply the historical, narrower successful auto-retry update."""
    figure = attempt.figure
    deps = attempt.deps
    figure.png_base64 = attempt.agent._compress_image_to_base64(
        attempt.jpg_path, max_width=800
    )
    attempt.agent._refresh_figure_integrity_hashes(figure)
    figure.review_status = FigureReviewStatus.PENDING
    attempt.agent._upsert_publication_figure(attempt.state, figure)
    if deps.get("lint_ran")(result.output):
        figure.layout_violations = deps.get("parse_layout_violations")(result.output)
    attempt.current_code = retry_code


async def perform_revision(
    attempt: FigureAttemptState,
    *,
    feedback: str,
    suggestions: list,
    must_fix: list[str],
    quality_score,
) -> str:
    """Run one revision and optional technical retry; return continue or break."""
    agent = attempt.agent
    deps = attempt.deps
    logger = deps.get("logger")
    await guard_attempt_await(
        attempt, agent._notify("status", "Revising figure based on PI feedback...")
    )
    logger.info(
        "[Bioinformatics] Revising figure %s: %s",
        attempt.request.figure_name,
        feedback,
    )
    attempt.revision_count += 1
    attempt.figure.revision_count = attempt.revision_count
    try:
        revised_code = await agent._revise_figure_code(
            attempt.current_code,
            feedback,
            suggestions,
            must_fix,
            attempt.request.figure_type,
            attempt.state,
            attempt.figure.png_base64,
            revision_history=attempt.revision_history,
            layout_violations=attempt.figure.layout_violations,
        )
    except UpstreamExhaustedError:
        abandon_render(attempt)
        raise
    except Exception as revise_err:
        revise_error = deps.get("compact_traceback")(
            str(revise_err), max_chars=1000
        )
        logger.error(
            "[Bioinformatics] _revise_figure_code failed: %s; keeping current version",
            revise_error,
        )
        await agent._notify(
            "status", f"Figure revision failed ({revise_error}), keeping current version"
        )
        attempt.terminal_revision_reason = "revision_failed"
        return "break"

    is_valid, error_msg = agent._validate_code(revised_code)
    if not is_valid:
        logger.warning(
            "[Bioinformatics] Revised code failed validation: %s", error_msg
        )
        attempt.terminal_revision_reason = "revision_failed"
        return "break"
    pre_revision_snapshot = take_snapshot(attempt, quality_score)
    try:
        result, physical_error = await execute_revision_candidate(
            attempt, revised_code
        )
    except BaseException:
        abandon_render(attempt)
        raise
    if not result.error and not physical_error:
        _accept_revision_output(attempt, result, revised_code)
        await guard_attempt_await(
            attempt, agent._notify("figure", attempt.figure.png_base64)
        )
        return "continue"
    if physical_error:
        logger.warning(
            "[Bioinformatics] Figure %s revision rejected: %s",
            attempt.request.figure_name,
            physical_error,
        )
        restore_snapshot(attempt, pre_revision_snapshot or attempt.best_snapshot)
        attempt.terminal_revision_reason = "revision_failed"
        return "break"

    error_msg = deps.get("compact_traceback")(str(result.error), max_chars=1000)
    result.error = error_msg
    logger.warning("[Bioinformatics] Revision failed: %s", error_msg)
    extra_context = ""
    api_hint = await agent._environment_api_hint(
        attempt.state, error_msg, revised_code
    )
    if api_hint:
        extra_context = api_hint
    elif agent.technique_agent:
        error_type = deps.get("extract_error_type")(error_msg)
        if error_type in TECHNICAL_ERROR_TYPES:
            logger.info(
                "[Bioinformatics] Looking up docs for %s error", error_type
            )
            doc_hint = await agent.technique_agent.lookup_from_error(
                agent.sandbox, error_msg, revised_code
            )
            if doc_hint:
                extra_context = f"\n\n[API Documentation]\n{doc_hint}"
                logger.info(
                    "[Bioinformatics] Got documentation hint for %s", error_type
                )

    logger.info("[Bioinformatics] Auto-retrying revision with error context")
    try:
        retry_code = await agent._revise_figure_code(
            revised_code,
            "The previous code revision caused a runtime error:\n"
            f"{error_msg}\nFix this error while keeping ALL the style improvements "
            f"from the revision.{extra_context}",
            [f"Fix the runtime error: {error_msg}"],
            ["The code must execute without errors"],
            attempt.request.figure_type,
            attempt.state,
            attempt.figure.png_base64,
            revision_history=attempt.revision_history,
            layout_violations=attempt.figure.layout_violations,
        )
        is_valid_retry, _retry_error = agent._validate_code(retry_code)
        if is_valid_retry:
            retry_result, retry_physical_error = await execute_revision_candidate(
                attempt, retry_code
            )
            if not retry_result.error and not retry_physical_error:
                logger.info(
                    "[Bioinformatics] Auto-retry succeeded for %s",
                    attempt.request.figure_name,
                )
                _accept_retry_output(attempt, retry_result, retry_code)
                await guard_attempt_await(
                    attempt, agent._notify("figure", attempt.figure.png_base64)
                )
                return "continue"
            logger.warning(
                "[Bioinformatics] Auto-retry also failed: %s",
                retry_result.error or retry_physical_error,
            )
    except UpstreamExhaustedError:
        abandon_render(attempt)
        raise
    except asyncio.CancelledError:
        abandon_render(attempt)
        raise
    except Exception as retry_err:
        logger.warning(
            "[Bioinformatics] Auto-retry exception: %s",
            deps.get("compact_traceback")(str(retry_err), max_chars=1000),
        )
    restore_snapshot(attempt, pre_revision_snapshot or attempt.best_snapshot)
    attempt.terminal_revision_reason = "revision_failed"
    return "break"
