"""Adopt workspace renders into the plan figure contract.

Spec: docs/superpowers/specs/2026-08-27-figure-adoption-design.md

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
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
)

logger = logging.getLogger(__name__)

#: Prefix for the semantic identity of an adopted record. Keyed by requirement
#: rather than by filename, so re-running the pass recognizes its own work.
ADOPTED_LOGICAL_PREFIX = "adopted"


@dataclass(frozen=True)
class AdoptionCandidate:
    """One workspace render offered to the PI, with how it came to exist."""

    name: str
    path: str
    pdf_path: Optional[str] = None
    provenance: str = ""


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


def _logical_id(binding: FigureBinding) -> str:
    return (
        f"{ADOPTED_LOGICAL_PREFIX}:{binding.plan_step_id}:"
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
    existing = {
        str(getattr(figure, "logical_figure_id", "") or "")
        for figure in (getattr(state, "publication_figures", None) or [])
    }
    plan = getattr(state, "current_plan", None)
    plan_id = str(getattr(plan, "id", "") or "").strip() or None
    lineage = getattr(state, "restore_branch_id", None) or "main"
    sandbox_state = getattr(state, "sandbox_state", None) or {}

    added: List[Any] = []
    for binding in bindings or []:
        logical = _logical_id(binding)
        if logical in existing:
            continue
        candidate = by_name.get(binding.filename)
        if candidate is None:
            continue
        record = _build_record(
            binding, candidate, logical,
            plan_id=plan_id, lineage=lineage, state=state,
            sandbox_state=sandbox_state, review_status=review_status,
        )
        if record is None:
            continue
        state.publication_figures.append(record)
        existing.add(logical)
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
    plan_id: Optional[str],
    lineage: str,
    state: Any,
    sandbox_state: Dict[str, Any],
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
        plan_id=plan_id,
        logical_figure_id=logical,
        render_attempt_id=str(uuid.uuid4()),
        producer_status="published",
        is_published_head=True,
        is_current=True,
        lineage_id=lineage,
        round_number=int(getattr(state, "current_round", 0) or 0),
        analysis_generation=int(sandbox_state.get("analysis_generation", 0) or 0),
        cohort_generation=int(
            (sandbox_state.get("cohort_identity_v1") or {}).get(
                "cohort_generation", 0
            ) or 0
        ),
        cohort_digest=(
            str(
                (sandbox_state.get("cohort_identity_v1") or {}).get(
                    "cohort_digest", ""
                ) or ""
            ) or None
        ),
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
    candidates: List[AdoptionCandidate] = []
    for entry in discovered:
        name = str(entry.get("name") or "")
        path = str(entry.get("path") or "")
        if not name or not path:
            continue
        pdf = entry.get("pdf_path")
        if not pdf:
            guess = Path(path).with_suffix(".pdf")
            pdf = str(guess) if guess.is_file() else None
        candidates.append(AdoptionCandidate(
            name=name,
            path=path,
            pdf_path=pdf,
            provenance=provenance.get(name, ""),
        ))
    return candidates


def _provenance_by_stem(analysis_log: Any) -> Dict[str, str]:
    """Map each saved file's stem to the execution that wrote it."""
    out: Dict[str, str] = {}
    for entry in analysis_log or []:
        if not isinstance(entry, dict):
            continue
        names = entry.get("saved_figure_filenames") or []
        if not names:
            continue
        step = entry.get("step_number")
        description = str(entry.get("code_description") or "").strip()
        line = f"execute_code step {step}: {description}" if description else \
            f"execute_code step {step}"
        for filename in names:
            stem = Path(str(filename)).stem
            if stem and stem not in out:
                out[stem] = line
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
