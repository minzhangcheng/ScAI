"""Attribute analysis figures to the round that produced them.

The report outline binds every figure to the round it came from, so a figure can
never migrate into a section that describes a different stage of the analysis.
Round provenance is recovered from four sources, most reliable first:

1. **Current registered bytes** — an immutable render hash bound to the actual
   report path is authoritative, even when an older log reused its basename.
2. **Supervision records** — ``saved_figure_filenames`` on a record that also
   carries ``round_number``. Authoritative, but sparse in practice: the sandbox
   only emits the ``[Auto-saved]`` marker for figures still open when a cell
   finishes, so analysis code that closes its own figures leaves no record.
3. **``PublicationFigure.round_number``** — set when a figure is promoted; the
   stem-only fallback is retained for restored/legacy records without usable
   immutable hashes.
4. **File mtime against round boundaries** — complete coverage, lowest
   precision. A round owns the half-open interval that ends at its own
   ``timestamp``; the first round's interval is open at the start.

Because source 1 is unreliable, source 3 is not a rare fallback — it routinely
carries most of the figures. Treat all three as normal paths.

Keep this generic: no dataset-, tissue-, disease-, or organism-specific rules.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .artifacts import archival_path_matches_record, artifact_stems

logger = logging.getLogger(__name__)

__all__ = [
    "round_boundaries",
    "attribute_figures_to_rounds",
    "figures_by_round",
]


def _to_epoch(value: Any) -> Optional[float]:
    """Best-effort conversion of a timestamp field to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Accept both "...Z" and offset-less ISO strings.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def round_boundaries(state: Any) -> List[Tuple[int, float, float]]:
    """Return ``(round_number, start_epoch, end_epoch)`` per round, in round order.

    A round owns everything produced after the previous round finished, up to and
    including its own ``timestamp``. The earliest round starts at ``-inf`` and the
    latest ends at ``+inf`` so that figures written slightly after the final
    round result was recorded still land somewhere.
    """
    results = list(getattr(state, "round_results", None) or [])
    stamped: List[Tuple[int, Optional[float]]] = []
    for rr in results:
        rn = getattr(rr, "round_number", None)
        if rn is None:
            continue
        stamped.append((int(rn), _to_epoch(getattr(rr, "timestamp", None))))
    stamped.sort(key=lambda item: item[0])

    bounds: List[Tuple[int, float, float]] = []
    prev_end = float("-inf")
    for idx, (rn, end) in enumerate(stamped):
        is_last = idx == len(stamped) - 1
        if end is None:
            # Undated round: it cannot claim an interval, but it must not
            # swallow the ones after it either. Give it an empty span.
            bounds.append((rn, prev_end, prev_end))
            continue
        upper = float("inf") if is_last else end
        bounds.append((rn, prev_end, upper))
        prev_end = end
    return bounds


def _round_for_epoch(bounds: Sequence[Tuple[int, float, float]], ts: float) -> Optional[int]:
    for rn, start, end in bounds:
        if start < ts <= end:
            return rn
    return None


def _registered_path_matches(figure: Any, path: Any) -> bool:
    """Require a matching registered path/stem before using its bytes."""
    actual = Path(str(path or ""))
    if not str(path or ""):
        return False
    declared = str(getattr(figure, "png_path", "") or "")
    if declared:
        try:
            if Path(declared).resolve() == actual.resolve():
                return True
        except (OSError, RuntimeError):
            pass
        return Path(declared).stem == actual.stem
    record_name = str(getattr(figure, "name", "") or "")
    return bool(record_name) and Path(record_name).stem == actual.stem


def _has_usable_render_hashes(figure: Any) -> bool:
    """Whether a publication record has a digest that can bind its bytes."""
    for field in ("render_sha256", "preview_sha256"):
        value = str(getattr(figure, field, "") or "").strip().lower()
        if len(value) == 64 and all(char in "0123456789abcdef" for char in value):
            return True
    return False


def attribute_figures_to_rounds(
    figure_names: Iterable[str],
    state: Any = None,
    records: Optional[Sequence[Dict[str, Any]]] = None,
    figure_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Map each figure filename to the round number that produced it.

    Filenames with no recoverable round are omitted from the result rather than
    guessed at, so callers can tell "round 3" from "unknown".
    """
    names = [n for n in figure_names if n]
    if not names:
        return {}
    wanted = set(names)
    attributed: Dict[str, int] = {}

    # A current registered render with exact bytes is authoritative even when
    # an older supervision record reused the same basename. The shared helper
    # checks bytes; this owner also checks the related path/stem so another
    # filename containing identical bytes cannot borrow the record's round.
    registered_exact: set[str] = set()
    active_lineage = getattr(state, "restore_branch_id", None) or "main"
    for pf in getattr(state, "publication_figures", None) or []:
        if not bool(getattr(pf, "is_current", True)):
            continue
        if (getattr(pf, "lineage_id", None) or "main") != active_lineage:
            continue
        rn = getattr(pf, "round_number", 0) or 0
        if rn <= 0:
            continue
        for name in names:
            path = (figure_paths or {}).get(name)
            if path is None or not _registered_path_matches(pf, path):
                continue
            try:
                matches = archival_path_matches_record(pf, path)
            except Exception:
                matches = False
            if matches:
                attributed[name] = int(rn)
                registered_exact.add(name)

    # Source 1: supervision records (authoritative where present, and the last
    # logged writer wins for an unregistered repeated basename).
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        rn = rec.get("round_number")
        if rn is None:
            continue
        for fname in rec.get("saved_figure_filenames") or []:
            base = os.path.basename(str(fname))
            if base in wanted and base not in registered_exact:
                attributed[base] = int(rn)

    # Source 2: publication figures promoted during a known round. Keep this
    # stem-only fallback for restored/legacy records without usable hashes.
    for pf in getattr(state, "publication_figures", None) or []:
        rn = getattr(pf, "round_number", 0) or 0
        if rn <= 0:
            continue
        if _has_usable_render_hashes(pf):
            continue
        stems = artifact_stems([pf])
        for candidate in names:
            if candidate in attributed:
                continue
            if Path(candidate).stem in stems:
                attributed[candidate] = int(rn)

    # Source 3: file mtime against round boundaries.
    missing = [n for n in names if n not in attributed]
    if missing and state is not None:
        bounds = round_boundaries(state)
        if bounds:
            for name in missing:
                path = (figure_paths or {}).get(name)
                if path is None:
                    continue
                try:
                    mtime = Path(path).stat().st_mtime
                except OSError:
                    continue
                rn = _round_for_epoch(bounds, mtime)
                if rn is not None:
                    attributed[name] = rn

    unresolved = len(names) - len(attributed)
    if unresolved:
        logger.info(
            "Figure round attribution: %d/%d resolved, %d unresolved",
            len(attributed), len(names), unresolved,
        )
    return attributed


def figures_by_round(attribution: Dict[str, int]) -> Dict[int, List[str]]:
    """Invert an attribution map, preserving the input ordering within a round."""
    grouped: Dict[int, List[str]] = {}
    for name, rn in attribution.items():
        grouped.setdefault(rn, []).append(name)
    return grouped
