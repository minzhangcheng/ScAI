"""Enforce figure placement against the report outline.

The report model is asked to put each figure in the section that owns its round,
but a prompt instruction is not a guarantee: the same instruction already said
"each figure appears exactly once" while figures were being emitted three times.
This module makes placement a property of the code rather than of the model's
compliance.

Three corrections are applied to the parsed sections:

* **Cross-round references are relocated.** A figure produced in round 3 cannot
  illustrate the quality-control narrative of round 1, whatever its filename
  suggests — filenames are written by the analysis agent and routinely describe
  the plot type rather than the stage, which is exactly how a stress/metal-axis
  panel named ``..._qc_overview`` ends up opening a report.
* **Repeat references are dropped**, keeping the first.
* **Unreferenced figures are appended** to the section owning their round, so
  nothing silently disappears into a supplementary dump.

Relocated and appended figures are ordered by production order within their
section; figures the model placed correctly keep the position it chose, so the
narrative it wrote around them still reads.

Keep this generic: no dataset-, tissue-, disease-, or organism-specific rules.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["FigurePlacementReport", "enforce_figure_placement"]

# [FIGURE: name] optionally followed by [FIGURE_CAPTION: text]
_FIGURE_MARKER = re.compile(
    r'\[FIGURE:\s*([^\]]+)\]\s*(?:\[FIGURE_CAPTION:\s*([^\]]*)\])?',
    re.IGNORECASE,
)


@dataclass
class FigurePlacementReport:
    """What the enforcement pass changed, for logging and tests."""

    relocated: List[Tuple[str, str, str]] = field(default_factory=list)  # (fig, from, to)
    duplicates_removed: List[Tuple[str, str]] = field(default_factory=list)  # (fig, section)
    unknown_removed: List[Tuple[str, str]] = field(default_factory=list)  # (fig, section)
    appended: List[Tuple[str, str]] = field(default_factory=list)  # (fig, section)
    unplaced: List[str] = field(default_factory=list)
    # Sections the outline says own figures, that ended up with none. Observation
    # only -- it never withholds or fails a report.
    figureless_sections: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"relocated={len(self.relocated)} duplicates={len(self.duplicates_removed)} "
            f"unknown={len(self.unknown_removed)} appended={len(self.appended)} "
            f"unplaced={len(self.unplaced)} "
            f"figureless={len(self.figureless_sections)}"
        )


def _marker(name: str, caption: str = "") -> str:
    if caption:
        return f"\n\n[FIGURE: {name}]\n[FIGURE_CAPTION: {caption}]\n"
    return f"\n\n[FIGURE: {name}]\n"


def _inline_marker(name: str, caption: str = "") -> str:
    """A marker rebuilt where it already stands, without opening a new block."""
    if caption:
        return f"[FIGURE: {name}]\n[FIGURE_CAPTION: {caption}]"
    return f"[FIGURE: {name}]"


def _alt_ext(name: str) -> str:
    """The same filename under the other image extension (.png ↔ .jpg)."""
    stem, _, ext = name.rpartition(".")
    if not stem:
        return name
    return f"{stem}.png" if ext.lower() in ("jpg", "jpeg") else f"{stem}.jpg"


def _resolve(fig: str, known: set) -> Optional[str]:
    """The known filename a marker refers to, or None if it refers to nothing.

    Publication figures are written as JPG while the report model routinely
    writes ``.png``; the HTML renderer has carried a .png↔.jpg fallback for
    exactly that. Placement runs first, so resolving only on an exact match here
    deletes the marker -- and its caption -- before the renderer ever sees it.
    """
    if fig in known:
        return fig
    alt = _alt_ext(fig)
    return alt if alt in known else None


def enforce_figure_placement(
    sections: List[Dict[str, Any]],
    outline: Sequence[Any],
    attribution: Dict[str, int],
    known_figures: Iterable[str],
    figure_order: Optional[Dict[str, float]] = None,
    figure_captions: Optional[Dict[str, str]] = None,
    protected_sections: Iterable[str] = (),
) -> FigurePlacementReport:
    """Rewrite ``sections`` in place so every figure sits in its round's section.

    Args:
        sections: Parsed report sections, each a dict with ``id`` and ``content``.
        outline: The :class:`OutlineSection` list the report was planned from.
        attribution: Figure filename → producing round number.
        known_figures: Filenames that actually exist on disk.
        figure_order: Optional sort key per figure (e.g. mtime) used to order
            relocated and appended figures within a section.
        figure_captions: Optional authoritative caption by exact filename.
            Registered publication metadata should be supplied here so a model
            cannot relabel a bar plot as a UMAP from filename guesswork.
        protected_sections: Saved chapters whose bound markers retain their order;
            unbound figures remain supplementary instead of being appended there.

    Returns:
        A :class:`FigurePlacementReport` describing the corrections applied.
    """
    report = FigurePlacementReport()
    known = {name for name in known_figures if name}
    authoritative_captions = figure_captions or {}
    protected_sections = set(protected_sections)
    if not sections:
        report.unplaced = sorted(known)
        return report

    # First-wins: ``align_sections_to_outline`` keeps the first copy of a
    # duplicated section in its outline position and appends the second to the
    # extras at the end of the list. Last-wins would aim every relocated and
    # appended figure at that trailing duplicate, printing them in a stray
    # repeated chapter after the conclusions while the real one stays empty.
    by_id: Dict[Any, Dict[str, Any]] = {}
    for sec in sections:
        by_id.setdefault(sec.get("id"), sec)

    # round number → section id that owns it, recorded only where exactly one
    # section claims the round. An outline may deliberately give several
    # sections the same rounds: the legacy (single-pass) outline organises by
    # analysis type, so quality control, clustering, differential expression and
    # key findings each declare every round. There the round number says nothing
    # about which section a figure belongs in, and picking the first claimant
    # would sweep every figure in the report into it.
    claimants: Dict[int, set] = {}
    for osec in outline:
        for rn in getattr(osec, "rounds", ()) or ():
            claimants.setdefault(int(rn), set()).add(osec.id)
    owner_of_round: Dict[int, str] = {
        rn: next(iter(ids)) for rn, ids in claimants.items() if len(ids) == 1
    }
    if len(owner_of_round) < len(claimants):
        logger.info(
            "Outline does not partition rounds by section; leaving the model's "
            "placement alone for round(s) %s",
            ", ".join(str(rn) for rn in sorted(set(claimants) - set(owner_of_round))),
        )

    # Sections that may hold figures at all, in outline order. Used as the
    # landing place for figures whose round has no section of its own.
    figure_section_ids = [
        osec.id for osec in outline
        if getattr(osec, "rounds", ()) and osec.id in by_id
    ]
    fallback_id = figure_section_ids[-1] if figure_section_ids else None

    placed: set = set()
    pending: Dict[str, List[Tuple[str, str]]] = {}  # section id → [(fig, caption)]

    def _target_for(fig: str) -> Optional[str]:
        rn = attribution.get(fig)
        if rn is None:
            return None
        sid = owner_of_round.get(rn)
        return sid if sid in by_id else None

    for sec in sections:
        sid = sec.get("id")
        content = sec.get("content", "") or ""

        def _replace(match: "re.Match") -> str:
            written = match.group(1).strip()
            caption = (match.group(2) or "").strip()
            fig = _resolve(written, known)
            if fig is None:
                report.unknown_removed.append((written, sid))
                return ""
            if fig in placed:
                report.duplicates_removed.append((fig, sid))
                return ""
            if sid in protected_sections:
                placed.add(fig)
                return match.group(0)
            authoritative_caption = authoritative_captions.get(fig) or ""
            caption = authoritative_caption or caption
            target = _target_for(fig)
            if target in protected_sections:
                return ""  # An unbound candidate must not be added to a saved chapter.
            if target is None or target == sid:
                # Correctly placed, or unattributable — leave it where the model
                # put it. An unattributable figure has no better home to move to.
                placed.add(fig)
                # Rewrite only if the name had to be resolved, so a marker the
                # model wrote correctly keeps its exact original text.
                return (match.group(0) if fig == written and not authoritative_caption
                        else _inline_marker(fig, caption))
            placed.add(fig)
            pending.setdefault(target, []).append((fig, caption))
            report.relocated.append((fig, sid, target))
            return ""

        sec["content"] = _FIGURE_MARKER.sub(_replace, content)

    # Figures the model never referenced still belong to their round.
    for fig in sorted(known - placed):
        target = _target_for(fig)
        if target is None:
            target = fallback_id
        if target is None or target in protected_sections:
            report.unplaced.append(fig)
            continue
        pending.setdefault(target, []).append(
            (fig, authoritative_captions.get(fig, ""))
        )
        report.appended.append((fig, target))

    order = figure_order or {}
    for sid, items in pending.items():
        items.sort(key=lambda item: order.get(item[0], float("inf")))
        sec = by_id[sid]
        sec["content"] = (sec.get("content", "") or "").rstrip() + "".join(
            _marker(fig, caption) for fig, caption in items
        )

    # A chapter that expects a result figure but shows nothing from its round is
    # the failure mode
    # that hides best: the prose reads as complete, and the gap is visible only by
    # counting. One shipped report carried a 3,700-character quality-control
    # chapter, full of the exact numbers a panel would have plotted, with zero
    # figures and zero tables, and nothing anywhere said so. Sections the outline
    # gives no rounds (Methods), plus sections explicitly marked text-only
    # (integration gates/final syntheses), are excluded. Those sections may own a
    # workflow round for ordering without claiming that the round produced a new
    # visual result.
    figureless_rounds: List[Tuple[str, Tuple[int, ...]]] = []
    for osec in outline:
        sid = osec.id
        expects_figures = getattr(osec, "expects_figures", None)
        if expects_figures is None:
            expects_figures = bool(getattr(osec, "rounds", ()))
        if not expects_figures:
            continue
        section = by_id.get(sid)
        # Absence is a stronger coverage gap than an existing prose-only
        # chapter.  Previously the ``sid not in by_id`` guard silently waived
        # exactly this case, allowing a report that omitted an ordinary Results
        # chapter wholesale to remain ``completed``.
        if section is None or not _FIGURE_MARKER.search(
            section.get("content", "") or ""
        ):
            report.figureless_sections.append(sid)
            figureless_rounds.append(
                (sid, tuple(int(r) for r in (getattr(osec, "rounds", ()) or ())))
            )
    if report.figureless_sections:
        logger.warning(
            "Report sections own a round but show no figure: %s",
            ", ".join(
                f"{sid} (round {', '.join(str(r) for r in rounds)})"
                if rounds
                else sid
                for sid, rounds in figureless_rounds
            ),
        )

    if report.relocated or report.duplicates_removed or report.appended:
        logger.info("Figure placement enforced: %s", report.summary())
    return report
