"""Adopt workspace renders into the plan figure contract.

See docs/HANDOVER.md for figure identity and review responsibilities.

The executor saves result figures with ``plt.savefig`` inside ``execute_code``.
Nothing turns those files into ``PublicationFigure`` records, and
``figure_contract._matches`` reads ``figure.plan_step_id`` -- never prose -- so
a requirement stays unmet however plainly its figure sits on disk. Measured on
session 3d675631: 37 image files, one ``generate_publication_figure`` call in
the whole round, 16 requirements at 0/1 eligible, every one of them already
rendered.

Adoption constructs the missing record. It does not render, review, or lint:

- Which candidate answers which requirement is decided upstream by the PI, which
  is the only participant holding both the plan and the execution history. This
  module validates that decision and refuses anything doubtful, because a figure
  filed under the wrong step is a silent error and strictly worse than the gap
  it would replace.
- Layout is recorded as *unverified*, never as clean. The in-kernel linter takes
  a live matplotlib figure and cannot run against a saved file, and an empty
  violation list is exactly what a measured-clean render carries.
- The record is ACCEPTED_WITH_RESERVATIONS, and the caller must not pass that
  unless the PI actually saw the image. ``is_report_eligible`` admits a
  non-diagnostic record only at APPROVED or ACCEPTED_WITH_RESERVATIONS, so PI
  inspection is not a quality nicety here -- it is the eligibility precondition.
  The reservation is literal and checkable: the layout was never measured.
  Marking these ``diagnostic`` to dodge the check would be false twice over --
  the platform did not render them, and that branch demands a measured layout
  these records by construction do not have.

Filenames are never parsed to decide precedence between candidates -- see the
``logical_figure_id`` note on ``PublicationFigure``. Suffix conventions are not
reliable; in the measured corpus a ``_fixed`` member was the *older* render.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.models.schema_parts.enums import FigureReviewStatus

#: Why an adopted record carries reservations rather than a clean approval.
ADOPTION_RESERVATION = (
    "Adopted from an existing workspace render after execution. The PI "
    "confirmed this image answers the plan requirement; the in-kernel layout "
    "lint never ran on it, so its layout is unverified."
)
from backend.models.schemas import PublicationFigure
from backend.services.figure_contract import normalize_figure_type
from backend.services.report.artifacts import (
    LAYOUT_UNVERIFIED_KIND,
    archival_path_matches_record,
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
)

logger = logging.getLogger(__name__)

#: Prefix for the semantic identity of an adopted record. Keyed by requirement
#: rather than by filename, so re-running the pass recognizes its own work.
ADOPTED_LOGICAL_PREFIX = "adopted"
_UNVERIFIED_PROVENANCE = (
    "Workspace render has no current execution association; retained for PI "
    "context only and cannot be adopted."
)


@dataclass(frozen=True)
class AdoptionCandidate:
    """One workspace render offered to the PI, with how it came to exist."""

    name: str
    path: str
    pdf_path: Optional[str] = None
    provenance: str = ""
    source_execution_id: Optional[str] = None


@dataclass(frozen=True)
class FigureBinding:
    """The PI's claim that ``filename`` answers one plan requirement."""

    plan_step_id: int
    figure_type: str
    filename: str
    caption: str = ""


def _layout_unverified() -> Dict[str, Any]:
    """Record that no verdict exists, which is not the same as no defect."""
    return {
        "kind": LAYOUT_UNVERIFIED_KIND,
        "severity": "advise",
        "text": "",
        "what": "layout",
        "detail": (
            "this render was adopted from the workspace after the fact, so the "
            "in-kernel layout lint never ran on it and its layout is "
            "unverified rather than known to be clean"
        ),
    }


@dataclass(frozen=True)
class _LoggedWriter:
    """The last analysis-log observation for one saved figure stem."""

    execution_id: Optional[str]
    description: str


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _active_receipt_identity(state: Any) -> Optional[Dict[str, Any]]:
    """Return the complete current technical identity, or no identity."""
    plan_id = str(
        getattr(getattr(state, "current_plan", None), "id", "") or ""
    ).strip()
    if not plan_id:
        return None
    try:
        return {
            "round_number": int(getattr(state, "current_round", 0) or 0),
            "plan_id": plan_id,
            "lineage_id": getattr(state, "restore_branch_id", None) or "main",
            "adata_source_preparation_id": getattr(
                state.current_plan, "adata_source_preparation_id", None
            ),
        }
    except (TypeError, ValueError):
        return None


def _current_source_receipt(state: Any, execution_id: Any) -> Any:
    """Resolve a log-linked receipt using technical identity fields only."""
    wanted = str(execution_id or "").strip()
    active = _active_receipt_identity(state)
    if not wanted or active is None:
        return None
    history = getattr(state, "code_history", None) or []
    receipt = next(
        (
            item for item in history
            if str(_entry_value(item, "id", "") or "").strip() == wanted
        ),
        None,
    )
    if receipt is None:
        return None
    if (
        _entry_value(receipt, "evidence_schema_version") != 2
        or _entry_value(receipt, "execution_status") != "succeeded"
        or _entry_value(receipt, "error")
    ):
        return None
    try:
        return receipt if (
            int(_entry_value(receipt, "round_number", 0) or 0)
            == active["round_number"]
            and str(_entry_value(receipt, "plan_id", "") or "")
            == active["plan_id"]
            and (_entry_value(receipt, "lineage_id", None) or "main")
            == active["lineage_id"]
            and (
                not active["adata_source_preparation_id"]
                or _entry_value(receipt, "adata_source_preparation_id")
                == active["adata_source_preparation_id"]
            )
        ) else None
    except (TypeError, ValueError):
        return None


def _logical_id(
    binding: FigureBinding, *, plan_id: str, round_number: int
) -> str:
    return (
        f"{ADOPTED_LOGICAL_PREFIX}:{plan_id}:{int(round_number)}:"
        f"{binding.plan_step_id}:"
        f"{normalize_figure_type(binding.figure_type)}"
    )


def _requirement_accepts(requirement: Any, binding: FigureBinding) -> bool:
    """Whether this binding's declared type is the one the step asked for."""
    if getattr(requirement, "plan_step_id", None) != binding.plan_step_id:
        return False
    declared = normalize_figure_type(binding.figure_type)
    if getattr(requirement, "wildcard_result_type", False):
        return declared not in {"qc", "diagnostic", "baseline"}
    return declared == normalize_figure_type(
        getattr(requirement, "figure_type", "other")
    )


def validate_bindings(
    bindings: Optional[Iterable[FigureBinding]],
    requirements: Optional[Iterable[Any]],
    candidates: Optional[Iterable[AdoptionCandidate]],
) -> List[FigureBinding]:
    """Keep only the bindings that can be written without guessing.

    Rejects a binding whose file was never offered, whose declared type is not
    the one the step asked for, whose step matches no open requirement, or whose
    file another surviving binding already claimed. A rejected binding leaves
    its requirement missing, which is the honest outcome: the audit then reports
    a real gap rather than filing a real figure under the wrong step.
    """
    known = {c.name for c in (candidates or [])}
    open_requirements = list(requirements or [])
    claimed_files: set[str] = set()
    consumed: set[int] = set()
    kept: List[FigureBinding] = []

    for binding in bindings or []:
        if binding.filename not in known:
            logger.warning(
                "[FigureAdoption] dropping binding for step %s: %r was not among "
                "the offered candidates", binding.plan_step_id, binding.filename,
            )
            continue
        if binding.filename in claimed_files:
            logger.warning(
                "[FigureAdoption] dropping binding for step %s: %r already "
                "answers another requirement",
                binding.plan_step_id, binding.filename,
            )
            continue
        match = next(
            (
                index
                for index, requirement in enumerate(open_requirements)
                if index not in consumed
                and _requirement_accepts(requirement, binding)
            ),
            None,
        )
        if match is None:
            logger.warning(
                "[FigureAdoption] dropping binding %r for step %s: no open "
                "requirement of that step and type",
                binding.filename, binding.plan_step_id,
            )
            continue
        consumed.add(match)
        claimed_files.add(binding.filename)
        kept.append(binding)
    return kept


def adopt_bindings(
    state: Any,
    bindings: Optional[Iterable[FigureBinding]],
    candidates: Optional[Iterable[AdoptionCandidate]],
    *,
    review_status: FigureReviewStatus = FigureReviewStatus.ACCEPTED_WITH_RESERVATIONS,
) -> List[Any]:
    """Append a PublicationFigure for each binding, and return what was added.

    ``review_status`` is the caller's attestation, not a formality. Pass the
    default only when the bindings came from a PI that was shown the images;
    a text-only binding call must pass ``FigureReviewStatus.PENDING``, which
    keeps the record out of ``is_report_eligible`` and therefore out of the
    contract. Claiming a review that did not happen is the one failure this
    module cannot detect for the caller.

    Best-effort throughout: a candidate whose bytes cannot be digested is
    skipped rather than recorded, because a record that cannot be resolved back
    to an image would satisfy the contract without being reportable. Nothing
    here removes or invalidates an existing record.
    """
    by_name = {c.name: c for c in (candidates or [])}
    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    existing = {
        (
            getattr(figure, "lineage_id", None) or "main",
            str(getattr(figure, "logical_figure_id", "") or ""),
        )
        for figure in (getattr(state, "publication_figures", None) or [])
    }

    added: List[Any] = []
    for binding in bindings or []:
        candidate = by_name.get(binding.filename)
        if candidate is None:
            continue
        source = _current_source_receipt(state, candidate.source_execution_id)
        if source is None:
            logger.info(
                "[FigureAdoption] candidate %r has no current execution "
                "receipt; leaving it unattributed",
                candidate.name,
            )
            continue
        plan_id = str(_entry_value(source, "plan_id", "") or "").strip()
        round_number = int(_entry_value(source, "round_number", 0) or 0)
        logical = _logical_id(
            binding, plan_id=plan_id, round_number=round_number
        )
        if (active_lineage, logical) in existing:
            continue
        record = _build_record(
            binding,
            candidate,
            logical,
            source_execution=source,
            review_status=review_status,
        )
        if record is None:
            continue
        state.publication_figures.append(record)
        existing.add((active_lineage, logical))
        added.append(record)
        logger.info(
            "[FigureAdoption] adopted %r for plan step %s (%s)",
            candidate.name, binding.plan_step_id,
            normalize_figure_type(binding.figure_type),
        )
    return added


def _build_record(
    binding: FigureBinding,
    candidate: AdoptionCandidate,
    logical: str,
    *,
    source_execution: Any,
    review_status: FigureReviewStatus,
) -> Optional[Any]:
    image = Path(candidate.path)
    if not image.is_file():
        logger.warning("[FigureAdoption] %r is not on disk; not adopted", candidate.path)
        return None

    render_digest = image_file_sha256(str(image))
    if not render_digest:
        logger.warning(
            "[FigureAdoption] %r did not digest as a usable render; not adopted",
            candidate.path,
        )
        return None

    pdf_path = str(candidate.pdf_path or "")
    pdf_digest = pdf_file_sha256(pdf_path) if pdf_path else None
    if not pdf_digest:
        logger.warning(
            "[FigureAdoption] %r has no valid companion PDF; not adopted",
            candidate.path,
        )
        return None

    try:
        from backend.utils.image_utils import compress_image_to_base64

        preview = compress_image_to_base64(str(image), max_width=800)
    except Exception:
        logger.warning(
            "[FigureAdoption] could not build a preview for %r; not adopted",
            candidate.path, exc_info=True,
        )
        return None

    try:
        receipt_round = int(
            _entry_value(source_execution, "round_number", 0) or 0
        )
        receipt_analysis_generation = int(
            _entry_value(source_execution, "analysis_generation", 0) or 0
        )
        receipt_cohort_generation = int(
            _entry_value(source_execution, "cohort_generation", 0) or 0
        )
    except (TypeError, ValueError):
        logger.warning(
            "[FigureAdoption] source receipt for %r has invalid identity; "
            "not adopted",
            candidate.path,
        )
        return None
    receipt_plan_id = str(_entry_value(source_execution, "plan_id", "") or "").strip()
    receipt_lineage = _entry_value(source_execution, "lineage_id", None) or "main"
    receipt_digest = str(
        _entry_value(source_execution, "cohort_digest", "") or ""
    ) or None

    record = PublicationFigure(
        id=str(uuid.uuid4()),
        name=candidate.name,
        description=binding.caption or candidate.provenance or candidate.name,
        figure_type=normalize_figure_type(binding.figure_type),
        png_path=str(image),
        pdf_path=pdf_path,
        png_base64=preview,
        render_sha256=render_digest,
        pdf_sha256=pdf_digest,
        preview_sha256=preview_render_sha256(preview),
        layout_violations=[_layout_unverified()],
        review_status=review_status,
        pi_feedback=ADOPTION_RESERVATION,
        diagnostic=False,
        purpose="result",
        plan_step_id=binding.plan_step_id,
        plan_id=receipt_plan_id or None,
        logical_figure_id=logical,
        render_attempt_id=str(uuid.uuid4()),
        producer_status="published",
        is_published_head=True,
        is_current=True,
        lineage_id=receipt_lineage,
        round_number=receipt_round,
        analysis_generation=receipt_analysis_generation,
        cohort_generation=receipt_cohort_generation,
        cohort_digest=receipt_digest,
        source_execution_id=str(
            _entry_value(source_execution, "id", "") or ""
        ).strip() or None,
    )
    if not record.preview_sha256:
        logger.warning(
            "[FigureAdoption] preview for %r did not digest; not adopted",
            candidate.path,
        )
        return None
    return record


def build_adoption_candidates(state: Any, workspace: Any) -> List[AdoptionCandidate]:
    """Enumerate workspace renders with the execution context that made them.

    The provenance line is what lets the PI tell a superseded draft from the
    render that replaced it. ``saved_figure_filenames`` carries the file-to-step
    link directly; the step's own description is the only plain-language account
    of what that execution was for.
    """
    from backend.services.report.figure_discovery import discover_figures_in_workspace

    workspace = Path(workspace)
    try:
        discovered = discover_figures_in_workspace(workspace)
    except Exception:
        logger.warning(
            "[FigureAdoption] workspace discovery failed; nothing to adopt",
            exc_info=True,
        )
        return []

    provenance = _provenance_by_stem(getattr(state, "analysis_log", None))
    verified: List[AdoptionCandidate] = []
    unverified: List[AdoptionCandidate] = []
    registered = list(getattr(state, "publication_figures", None) or [])
    for entry in discovered:
        name = str(entry.get("name") or "")
        path = str(entry.get("path") or "")
        if not name or not path:
            continue
        if any(
            _registered_render_matches_candidate(figure, path, name)
            for figure in registered
        ):
            logger.info(
                "[FigureAdoption] %r already matches a registered render; "
                "not offered for re-adoption",
                path,
            )
            continue
        pdf = entry.get("pdf_path")
        if not pdf:
            guess = Path(path).with_suffix(".pdf")
            pdf = str(guess) if guess.is_file() else None
        writer = provenance.get(name)
        source = _current_source_receipt(
            state, writer.execution_id if writer is not None else None
        )
        if writer is None:
            provenance_text = _UNVERIFIED_PROVENANCE
        elif source is None:
            provenance_text = (
                f"{writer.description}; no current execution association; "
                "retained for PI context only and cannot be adopted."
            )
        else:
            provenance_text = writer.description
        candidate = AdoptionCandidate(
            name=name,
            path=path,
            pdf_path=pdf,
            provenance=provenance_text,
            source_execution_id=(
                str(_entry_value(source, "id", "") or "").strip() or None
                if source is not None else None
            ),
        )
        (verified if source is not None else unverified).append(candidate)
    return [*verified, *unverified]


def _registered_render_matches_path(
    figure: Any, path: str, candidate_name: str
) -> bool:
    """Require a related path/stem before treating matching bytes as owned."""
    declared = str(getattr(figure, "png_path", "") or "")
    if declared:
        try:
            if Path(declared).resolve() == Path(path).resolve():
                return True
        except (OSError, RuntimeError):
            pass
        return Path(declared).stem == Path(path).stem == candidate_name
    record_name = str(getattr(figure, "name", "") or "")
    return bool(record_name) and Path(record_name).stem == candidate_name


def _registered_render_matches_candidate(
    figure: Any, path: str, candidate_name: str
) -> bool:
    if not _registered_render_matches_path(figure, path, candidate_name):
        return False
    try:
        return archival_path_matches_record(figure, path)
    except Exception:
        logger.warning(
            "[FigureAdoption] could not compare %r with registered render",
            path,
            exc_info=True,
        )
        return False


def _provenance_by_stem(analysis_log: Any) -> Dict[str, _LoggedWriter]:
    """Map each saved file's stem to its last logged writer."""
    out: Dict[str, _LoggedWriter] = {}
    for entry in analysis_log or []:
        names = _entry_value(entry, "saved_figure_filenames", None) or []
        if isinstance(names, (str, bytes)):
            names = [names]
        if not names:
            continue
        step = _entry_value(entry, "step_number", None)
        description = str(_entry_value(entry, "code_description", "") or "").strip()
        line = f"execute_code step {step}: {description}" if description else \
            f"execute_code step {step}"
        execution_id = str(
            _entry_value(entry, "execution_id", "") or ""
        ).strip() or None
        for filename in names:
            stem = Path(str(filename)).stem
            if stem:
                # analysis_log is chronological; the last observed writer is
                # the only safe association when a stem was reused.
                out[stem] = _LoggedWriter(
                    execution_id=execution_id,
                    description=line,
                )
    return out


__all__ = [
    "ADOPTED_LOGICAL_PREFIX",
    "ADOPTION_RESERVATION",
    "AdoptionCandidate",
    "FigureBinding",
    "adopt_bindings",
    "build_adoption_candidates",
    "validate_bindings",
]
