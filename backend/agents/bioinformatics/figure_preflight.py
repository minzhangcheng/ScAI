"""Publication-figure request validation and attempt construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .figure_dependencies import FigureDependencies
from .shared import AnalysisState, PublicationFigure


def _new_render_attempt_paths(figures_dir: Path, safe_name: str, attempt_id: str):
    """Return a hidden staging directory and its JPG/PDF paths."""
    attempt_dir = figures_dir / ".attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return (
        attempt_dir,
        attempt_dir / f"{safe_name}.jpg",
        attempt_dir / f"{safe_name}.pdf",
    )


def _render_landed(output: str | None, *, bindings) -> bool:
    """Whether this execution actually replaced the figure on disk."""
    text = output or ""
    return (
        bindings._JPG_SAVED_MARKER in text
        and bindings._NO_FIGURE_MARKER not in text
        and bindings._SAVE_FAILED_MARKER not in text
    )


def _publication_output_error(
    jpg_path: Path, pdf_path: Path, output: str | None, *, bindings
) -> str | None:
    """Explain why a promised JPG+PDF render is not publication-ready."""
    if not bindings._render_landed(output):
        return (
            "Figure generation produced no new JPG render; an older file with "
            "the same name, if present, was not re-registered."
        )
    content_error = bindings.empty_render_reason(output)
    if content_error:
        return content_error
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
    if bindings.image_file_sha256(jpg_path) is None:
        return f"JPG render is corrupt or cannot be decoded: {jpg_path}"
    if (
        bindings._PDF_SAVE_FAILED_MARKER in (output or "")
        or bindings._PDF_SAVED_MARKER not in (output or "")
    ):
        return (
            "Figure generation did not produce a new publication PDF; an older "
            "same-name PDF, if present, was not re-registered."
        )
    if not bindings.publication_pdf_is_valid(pdf_path):
        return f"Publication PDF is missing or invalid: {pdf_path}"
    return None


def _layout_unverified_marker() -> dict[str, Any]:
    """Return the advisory record for a render with no linter verdict."""
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


def _layout_feedback_block(violations, *, bindings) -> str:
    """Render measured layout violations as a feedback section, or ''."""
    body = bindings.format_violations_for_prompt(violations)
    if not body:
        return ""
    return (
        "\n**Measured layout defects (from the rendering engine, not an "
        "opinion — these are pixel coordinates of artists that were drawn on "
        "top of each other):**\n" + body + "\n"
    )


def _violation_must_fix_items(violations, *, bindings) -> list[str]:
    """One must-fix line per measured violation, for the reviewer's list."""
    body = bindings.format_violations_for_prompt(violations)
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


async def environment_api_hint(
    agent, state, error: str | None, failed_code: str = "", *, bindings
) -> str:
    """Return one bounded, session-scoped note for an API mismatch."""
    from backend.services.api_change_notes import (
        api_change_signature,
        format_api_note,
        record_api_note,
        recorded_api_note,
    )

    signature = api_change_signature(error)
    if not signature:
        return ""
    note = recorded_api_note(state, signature)
    if note:
        return format_api_note(signature, note)
    if not getattr(agent, "technique_agent", None):
        return ""
    try:
        note = await agent.technique_agent.lookup_from_error(
            agent.sandbox, str(error or ""), failed_code or ""
        )
    except Exception as lookup_error:
        bindings.logger.warning(
            "[Bioinformatics] Environment-API lookup failed for %s: %s",
            signature,
            lookup_error,
        )
        return ""
    if not str(note or "").strip():
        return ""
    record_api_note(state, signature, note)
    return format_api_note(signature, note)


@dataclass(frozen=True)
class FigureRequestContext:
    """One non-persisted publication-figure request."""

    agent: Any
    code: str
    figure_name: str
    description: str
    figure_type: str
    state: AnalysisState
    review_callback: Optional[Callable[[PublicationFigure], Awaitable[dict[str, Any]]]]
    diagnostic: bool
    plan_step_id: Optional[int]
    purpose: Optional[str]
    logical_figure_id: Optional[str]
    source_execution_id: Optional[str]
    submitted_plan_step_id: Optional[int]
    plan_requirement_key: Optional[str]
    supervisor_binding_decision: Optional[str]
    supervisor_binding_reason: Optional[str]
    dependencies: FigureDependencies


@dataclass
class FigureAttemptState:
    """Mutable, ephemeral state shared by figure workflow phases."""

    request: FigureRequestContext
    figures_dir: Path
    safe_name: str
    final_jpg_path: Path
    final_pdf_path: Path
    sandbox_state: dict[str, Any]
    current_plan_id: str | None
    active_lineage: str
    logical_id: str
    prior_attempt: Any
    render_attempt_id: str
    attempt_dir: Path | None = None
    staged_jpg_path: Path | None = None
    staged_pdf_path: Path | None = None
    jpg_path: Path | None = None
    pdf_path: Path | None = None
    guarded_code: str = ""
    wrapped_code: str = ""
    preexisting_artifact_versions: dict[str, Any] = field(default_factory=dict)
    staleness_note: str = ""
    result: Any = None
    figure: PublicationFigure | None = None
    pending_transaction: dict[str, Any] | None = None
    render_published: bool = False
    revision_count: int = 0
    current_code: str = ""
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    best_snapshot: dict[str, Any] | None = None
    last_review: dict[str, Any] = field(default_factory=dict)
    last_must_fix: list[str] = field(default_factory=list)
    terminal_revision_reason: str = "attempts_exhausted"

    @property
    def agent(self) -> Any:
        return self.request.agent

    @property
    def state(self) -> AnalysisState:
        return self.request.state

    @property
    def deps(self) -> FigureDependencies:
        return self.request.dependencies


async def prepare_figure_request(
    request: FigureRequestContext,
) -> tuple[FigureAttemptState | None, str | None]:
    """Validate one request, recover its workspace, and allocate identities."""
    agent = request.agent
    deps = request.dependencies
    logger = deps.get("logger")
    logger.info(
        "[Bioinformatics] Generating publication figure: %s", request.figure_name
    )
    is_valid, error_msg = agent._validate_code(request.code)
    if not is_valid:
        return None, f"Error: Code validation failed - {error_msg}"

    figures_dir = Path(agent.sandbox.working_dir) / "figures"
    try:
        from ...config import get_session_workspace

        if figures_dir.resolve() == (
            get_session_workspace(request.state.id) / "figures"
        ).resolve():
            from ...services.session_store import (
                recover_publication_transactions_for_state,
            )

            await recover_publication_transactions_for_state(
                request.state.id, request.state
            )
    except Exception:
        logger.exception(
            "Could not recover publication transactions for %s", request.state.id
        )
        return None, "Could not recover pending publication figure transactions"

    safe_name = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in request.figure_name
    )
    sandbox_state = request.state.sandbox_state or {}
    current_plan = getattr(request.state, "current_plan", None)
    current_plan_id = str(getattr(current_plan, "id", "") or "").strip() or None
    active_lineage = getattr(request.state, "restore_branch_id", None) or "main"
    logical_id = str(request.logical_figure_id or "").strip()[:240]
    if not logical_id:
        import uuid

        logical_id = f"render:{uuid.uuid4()}"
    prior_attempts = [
        item
        for item in deps.get("eligible_artifacts")(request.state.publication_figures)
        if (getattr(item, "lineage_id", None) or "main") == active_lineage
        and str(getattr(item, "logical_figure_id", None) or "") == logical_id
    ]
    from ...services.figure_revision_tasks import find_figure_revision_task

    revision_task = find_figure_revision_task(
        request.state,
        plan_id=current_plan_id,
        plan_requirement_key=request.plan_requirement_key,
    )
    starting_revision_count = (
        min(revision_task.attempt_count, revision_task.max_attempts)
        if revision_task is not None
        and revision_task.logical_figure_id == logical_id
        and revision_task.status in {"pending", "in_progress", "blocked"}
        else 0
    )
    import uuid

    return FigureAttemptState(
        request=request,
        figures_dir=figures_dir,
        safe_name=safe_name,
        final_jpg_path=figures_dir / f"{safe_name}.jpg",
        final_pdf_path=figures_dir / f"{safe_name}.pdf",
        sandbox_state=sandbox_state,
        current_plan_id=current_plan_id,
        active_lineage=active_lineage,
        logical_id=logical_id,
        prior_attempt=prior_attempts[-1] if prior_attempts else None,
        render_attempt_id=str(uuid.uuid4()),
        current_code=request.code,
        revision_count=starting_revision_count,
    ), None
