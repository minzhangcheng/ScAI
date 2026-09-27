"""Kernel rendering and output landing for publication figures."""
from __future__ import annotations

import uuid
from pathlib import Path
import shutil

from .figure_preflight import FigureAttemptState
from .shared import FigureReviewStatus, PublicationFigure


def _discard_render_attempt(attempt_dir, *, bindings) -> None:
    try:
        bindings.shutil.rmtree(attempt_dir)
        parent = attempt_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        bindings.logger.warning(
            "Could not remove failed figure attempt directory %s", attempt_dir
        )


def _discard_attempt_staging(attempt: FigureAttemptState) -> None:
    """Remove disposable and owner-side staging without touching published files."""
    seen = set()
    for path in (attempt.render_attempt_dir, attempt.attempt_dir):
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        attempt.deps.get("_discard_render_attempt")(path)


def _render_paths(attempt: FigureAttemptState):
    return (
        attempt.render_attempt_dir or attempt.attempt_dir,
        attempt.render_jpg_path or attempt.staged_jpg_path,
        attempt.render_pdf_path or attempt.staged_pdf_path,
    )


def _land_validated_render(attempt: FigureAttemptState) -> str | None:
    """Copy only the validated JPG/PDF pair out of a disposable renderer."""
    render_dir, render_jpg, render_pdf = _render_paths(attempt)
    if render_dir == attempt.attempt_dir:
        return None
    try:
        attempt.staged_jpg_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(render_jpg, attempt.staged_jpg_path)
        shutil.copyfile(render_pdf, attempt.staged_pdf_path)
        if (
            attempt.deps.get("image_file_sha256")(render_jpg)
            != attempt.deps.get("image_file_sha256")(attempt.staged_jpg_path)
            or attempt.deps.get("pdf_file_sha256")(render_pdf)
            != attempt.deps.get("pdf_file_sha256")(attempt.staged_pdf_path)
        ):
            raise OSError("validated renderer bytes changed while being staged")
    except OSError as exc:
        _discard_attempt_staging(attempt)
        return f"Could not transfer validated figure bytes from disposable renderer: {exc}"
    attempt.deps.get("_discard_render_attempt")(render_dir)
    attempt.render_attempt_dir = None
    attempt.render_jpg_path = None
    attempt.render_pdf_path = None
    return None


def _publish_render_attempt(
    attempt_dir,
    staged_jpg,
    staged_pdf,
    final_jpg,
    final_pdf,
    *,
    archived_record=None,
    bindings,
) -> str | None:
    """Land a validated pair and preserve the prior registered render."""
    backups = []
    published = []
    archive_dir = None
    archive_paths = {}
    try:
        final_jpg.parent.mkdir(parents=True, exist_ok=True)
        for final in (final_jpg, final_pdf):
            if final.exists():
                backup = attempt_dir / f"previous_{final.name}"
                bindings.shutil.copy2(final, backup)
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
                bindings.shutil.copy2(backup, archived_path)
                archive_paths[final.suffix.lower()] = archived_path
        bindings.os.replace(staged_jpg, final_jpg)
        published.append(final_jpg)
        bindings.os.replace(staged_pdf, final_pdf)
        published.append(final_pdf)
    except OSError as exc:
        for path in published:
            try:
                path.unlink()
            except OSError:
                pass
        for backup, final in backups:
            try:
                bindings.shutil.copy2(backup, final)
            except OSError:
                bindings.logger.exception(
                    "Could not restore prior figure render %s", final
                )
        if archive_dir is not None:
            try:
                bindings.shutil.rmtree(archive_dir)
            except OSError:
                pass
        bindings._discard_render_attempt(attempt_dir)
        return f"Could not publish validated figure render: {exc}"
    if archived_record is not None:
        archived_jpg = archive_paths.get(".jpg") or archive_paths.get(".jpeg")
        archived_pdf = archive_paths.get(".pdf")
        if archived_jpg is not None:
            archived_record.png_path = str(archived_jpg)
        if archived_pdf is not None:
            archived_record.pdf_path = str(archived_pdf)
    bindings._discard_render_attempt(attempt_dir)
    return None


async def execute_initial_render(attempt: FigureAttemptState) -> str | None:
    """Render, validate, measure, and materialize the initial pending record."""
    request = attempt.request
    agent = attempt.agent
    deps = attempt.deps
    logger = deps.get("logger")
    figure_followup = request.producer_run_kind == "figure_followup"
    instrumented_code, sensitive_call_count = (
        (request.code, 0)
        if figure_followup
        else deps.get("instrument_expression_contract_calls")(request.code)
    )
    preinstall_expression_runtime = (
        False if figure_followup
        else deps.get("has_direct_expression_contract_call")(request.code)
    )
    runtime_code = deps.get("build_expression_runtime_code")
    guarded_code = request.code if figure_followup else (
        instrumented_code
        if preinstall_expression_runtime or not sensitive_call_count
        else runtime_code() + "\n" + instrumented_code
    )
    if preinstall_expression_runtime:
        runtime_setup = await agent.sandbox.execute(
            runtime_code(),
            skip_safety_check=True,
        )
        if runtime_setup.error:
            return "Figure generation blocked: " + deps.get("compact_traceback")(
                runtime_setup.error, max_chars=3000
            )

    if not figure_followup:
        representation_will_change = deps.get("mutates_expression_representation")(
            request.code
        )
        attempt.preexisting_artifact_versions = (
            deps.get("workspace_artifact_versions")(agent.sandbox.working_dir)
            if representation_will_change
            else {}
        )
    attempt.attempt_dir, attempt.staged_jpg_path, attempt.staged_pdf_path = deps.get(
        "_new_render_attempt_paths"
    )(attempt.figures_dir, attempt.safe_name, attempt.render_attempt_id)
    render_dir, render_jpg, render_pdf = _render_paths(attempt)
    attempt.guarded_code = guarded_code
    attempt.wrapped_code = deps.get("_build_publication_figure_wrapper")(
        guarded_code,
        render_dir,
        render_jpg,
        render_pdf,
    )

    user_code_execution_started = False
    try:
        try:
            user_code_execution_started = True
            result = await agent.sandbox.execute(
                attempt.wrapped_code, skip_safety_check=True
            )
        finally:
            if not figure_followup:
                try:
                    (
                        _invalidated_count,
                        _stale_stems,
                        attempt.staleness_note,
                    ) = deps.get("record_executed_representation_change")(
                        attempt.state,
                        request.code,
                        agent.sandbox.working_dir,
                        attempt.preexisting_artifact_versions,
                        execution_started=user_code_execution_started,
                    )
                except Exception as provenance_error:
                    logger.warning(
                        "[Bioinformatics] Could not record figure expression "
                        "representation invalidation: %s",
                        deps.get("compact_traceback")(
                            str(provenance_error), max_chars=1000
                        ),
                    )
    except BaseException:
        _discard_attempt_staging(attempt)
        raise

    if not figure_followup:
        expression_contracts = deps.get("parse_expression_contracts")(result.output)
        if expression_contracts:
            deps.get("record_expression_contract")(attempt.state, expression_contracts)
        result.output = deps.get("strip_expression_contract_markers")(result.output)
    attempt.result = result

    if result.error:
        _discard_attempt_staging(attempt)
        result.error = deps.get("compact_traceback")(result.error, max_chars=3000)
        if attempt.staleness_note:
            result.error = f"{result.error}\n{attempt.staleness_note}".strip()
        error_msg = f"Figure generation failed: {result.error}"
        api_hint = await agent._environment_api_hint(
            attempt.state, result.error, request.code
        )
        if api_hint:
            error_msg = f"{error_msg}{api_hint}"
        logger.error("[Bioinformatics] %s", error_msg)
        await agent._notify("error", error_msg)
        return error_msg

    output_error = deps.get("_publication_output_error")(
        render_jpg, render_pdf, result.output
    )
    if output_error:
        _discard_attempt_staging(attempt)
        logger.warning("[Bioinformatics] %s", output_error)
        return output_error

    transfer_error = _land_validated_render(attempt)
    if transfer_error:
        return transfer_error

    attempt.jpg_path = attempt.staged_jpg_path
    attempt.pdf_path = attempt.staged_pdf_path
    layout_violations = deps.get("parse_layout_violations")(result.output)
    if deps.get("lint_ran")(result.output):
        logger.info(
            "[Bioinformatics] Figure %s layout lint: %s",
            attempt.safe_name,
            deps.get("summarize_violations")(layout_violations),
        )
    else:
        layout_violations = [deps.get("_layout_unverified_marker")()]
        logger.warning(
            "[Bioinformatics] Figure %s: layout lint did not report — "
            "treating layout as unverified, not as clean",
            attempt.safe_name,
        )

    try:
        png_b64 = agent._compress_image_to_base64(attempt.jpg_path, max_width=800)
    except BaseException:
        _discard_attempt_staging(attempt)
        raise

    request = attempt.request
    from .receipt_lookup import artifact_provenance

    provenance, provenance_error = artifact_provenance(
        attempt.state,
        request.source_execution_id,
        fallback_lineage=attempt.active_lineage,
        fallback_round=int(attempt.state.current_round or 0),
        fallback_plan_id=attempt.current_plan_id,
        current_receipts=getattr(attempt.agent, "code_history", None) or [],
    )
    if provenance_error:
        _discard_attempt_staging(attempt)
        message = f"Figure generation failed: {provenance_error}"
        logger.error("[Bioinformatics] %s", message)
        await agent._notify("error", message)
        return message
    from .receipt_lookup import lookup_execution_receipt

    source_receipt = lookup_execution_receipt(
        attempt.state,
        request.source_execution_id,
        current_receipts=getattr(attempt.agent, "code_history", None) or [],
    )
    producer_run_kind = (
        "platform"
        if request.diagnostic or request.purpose in {"baseline", "diagnostic"}
        else request.producer_run_kind
    )
    plan_bound_result = bool(
        attempt.current_plan_id
        and request.plan_step_id is not None
        and (request.purpose or "result") == "result"
        and not request.diagnostic
    )
    if plan_bound_result and source_receipt is None:
        _discard_attempt_staging(attempt)
        return "Figure generation failed: the plan-bound figure source receipt is unavailable."
    figure = PublicationFigure(
        id=str(uuid.uuid4()),
        name=attempt.safe_name,
        description=request.description,
        analysis_metadata=request.analysis_metadata or {},
        figure_type=request.figure_type,
        png_path=str(attempt.jpg_path),
        pdf_path=str(attempt.pdf_path),
        png_base64=png_b64,
        layout_violations=layout_violations,
        review_status=FigureReviewStatus.PENDING,
        revision_count=attempt.revision_count,
        diagnostic=request.diagnostic,
        submitted_plan_step_id=request.submitted_plan_step_id,
        plan_step_id=request.plan_step_id,
        plan_id=attempt.current_plan_id,
        plan_requirement_key=request.plan_requirement_key,
        supervisor_binding_decision=request.supervisor_binding_decision,
        supervisor_binding_reason=request.supervisor_binding_reason,
        purpose=request.purpose or (
            "diagnostic" if request.diagnostic else "result"
        ),
        logical_figure_id=attempt.logical_id,
        render_attempt_id=attempt.render_attempt_id,
        producer_status="pending",
        revision_of_id=getattr(attempt.prior_attempt, "id", None),
        is_published_head=False,
        lineage_id=provenance["lineage_id"],
        round_number=provenance["round_number"],
        analysis_generation=provenance["analysis_generation"],
        cohort_generation=provenance["cohort_generation"],
        cohort_digest=provenance["cohort_digest"],
        source_execution_id=request.source_execution_id,
        provenance_schema_version=2 if plan_bound_result else None,
        producer_run_kind=producer_run_kind,
        producer_run_scope_id=(
            str(request.producer_run_scope_id or "") or None
        ),
        source_execution_run_kind=(
            str(getattr(source_receipt, "executor_run_kind", "") or "") or None
        ),
        source_execution_validated=True if plan_bound_result else None,
        source_preparation_id=(
            str(getattr(source_receipt, "adata_source_preparation_id", "") or "")
            or None
        ),
        figure_followup_task_id=(
            str(request.figure_followup_task_id or "") or None
        ),
        expression_provenance=dict(
            attempt.sandbox_state.get("expression_contract_v1") or {}
        ),
    )
    agent._refresh_figure_integrity_hashes(figure)
    if plan_bound_result:
        from ...services.figure_followup import validate_publication_figure_provenance

        reason = validate_publication_figure_provenance(
            attempt.state,
            figure,
            followup_scope=request.figure_followup_scope,
            current_receipts=getattr(attempt.agent, "code_history", None) or [],
        )
        if reason:
            figure.source_execution_validated = False
            figure.provenance_validation_reason = reason[:1000]
            _discard_attempt_staging(attempt)
            return f"Figure generation rejected: {reason}"
    attempt.figure = figure
    return None
