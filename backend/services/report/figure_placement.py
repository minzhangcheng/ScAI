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


def enforce_figure_placement(
    sections: List[Dict[str, Any]],
    outline: Sequence[Any],
    attribution: Dict[str, int],
    known_figures: Iterable[str],
    figure_order: Optional[Dict[str, float]] = None,
) -> FigurePlacementReport:
    """Rewrite ``sections`` in place so every figure sits in its round's section.

    Args:
        sections: Parsed report sections, each a dict with ``id`` and ``content``.
        outline: The :class:`OutlineSection` list the report was planned from.
        attribution: Figure filename → producing round number.
        known_figures: Filenames that actually exist on disk.
        figure_order: Optional sort key per figure (e.g. mtime) used to order
            relocated and appended figures within a section.

    Returns:
        A :class:`FigurePlacementReport` describing the corrections applied.
    """
    report = FigurePlacementReport()
    known = {name for name in known_figures if name}
    if not sections:
        report.unplaced = sorted(known)
        return report

    by_id = {sec.get("id"): sec for sec in sections}
    # round number → section id that owns it
    owner_of_round: Dict[int, str] = {}
    for osec in outline:
        for rn in getattr(osec, "rounds", ()) or ():
            owner_of_round.setdefault(int(rn), osec.id)

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
            fig = match.group(1).strip()
            caption = (match.group(2) or "").strip()
            if fig not in known:
                report.unknown_removed.append((fig, sid))
                return ""
            if fig in placed:
                report.duplicates_removed.append((fig, sid))
                return ""
            target = _target_for(fig)
            if target is None or target == sid:
                # Correctly placed, or unattributable — leave it where the model
                # put it. An unattributable figure has no better home to move to.
                placed.add(fig)
                return match.group(0)
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
        if target is None:
            report.unplaced.append(fig)
            continue
        pending.setdefault(target, []).append((fig, ""))
        report.appended.append((fig, target))

    order = figure_order or {}
    for sid, items in pending.items():
        items.sort(key=lambda item: order.get(item[0], float("inf")))
        sec = by_id[sid]
        sec["content"] = (sec.get("content", "") or "").rstrip() + "".join(
            _marker(fig, caption) for fig, caption in items
        )

    # A chapter that owns a round but shows nothing from it is the failure mode
    # that hides best: the prose reads as complete, and the gap is visible only by
    # counting. One shipped report carried a 3,700-character quality-control
    # chapter, full of the exact numbers a panel would have plotted, with zero
    # figures and zero tables, and nothing anywhere said so. Sections the outline
    # gives no rounds (Methods) are excluded -- they own no figures by design.
    for osec in outline:
        sid = osec.id
        if not getattr(osec, "rounds", ()) or sid not in by_id:
            continue
        if not _FIGURE_MARKER.search(by_id[sid].get("content", "") or ""):
            report.figureless_sections.append(sid)
    if report.figureless_sections:
        logger.warning(
            "Report sections own a round but show no figure: %s",
            ", ".join(report.figureless_sections),
        )

    if report.relocated or report.duplicates_removed or report.appended:
        logger.info("Figure placement enforced: %s", report.summary())
    return report
