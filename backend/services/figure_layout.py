"""Renderer-level layout linting for publication figures.

The figure-quality gate in this project is an LLM vision review. That review
reliably *names* layout defects but cannot say where they are in pixels, so a
redraw prompt built from it is prose the next draft has to guess at. This
module supplies the missing deterministic half: after the plotting code has
run and the figure has been drawn, matplotlib's own renderer is asked where
every piece of text actually landed, and collisions are reported as
coordinates.

Two halves live here:

* ``LAYOUT_LINT_SRC`` — Python source injected into the sandbox kernel by
  ``kernel_templates._build_publication_figure_wrapper``. It runs *inside* the
  kernel because only the kernel holds a live ``Figure``; the backend process
  ever sees JPEG bytes. It prints one ``__SCAI_LAYOUT_LINT__`` line.
* the parsing / formatting helpers used by the backend to turn that line into
  a gate decision and into revision-prompt text.

Everything here is geometry and typography. No claim about what is plotted.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


LAYOUT_LINT_SENTINEL = "__SCAI_LAYOUT_LINT__"
LAYOUT_LINT_ERROR_SENTINEL = "__SCAI_LAYOUT_LINT_ERROR__"
LAYOUT_LINT_KINDS = frozenset(
    {
        "text_over_text",
        "text_over_foreign_axes",
        "tick_label_collision",
        "tiny_text",
        "lint_skipped",
    }
)
LAYOUT_LINT_SEVERITIES = frozenset({"block", "advise"})


# ---------------------------------------------------------------------------
# Sandbox-side source
# ---------------------------------------------------------------------------
# Injected verbatim into the publication-figure wrapper. Must be self-contained
# (no imports from this package), must never raise into the caller, and must
# never be able to prevent the figure from being saved. Measurement/serialization
# failures emit an error sentinel and no normal verdict, so the backend records
# the layout as unverified rather than measured-clean.
#
# Note on units: every bbox below is in display pixels at the figure's current
# dpi, which is what Text/Legend/Axes ``get_window_extent`` returns. The saved
# JPG uses dpi=150 while the kernel default is dpi=300, so pixel *magnitudes*
# here are not the magnitudes in the output file. The decisions are kept
# dpi-invariant deliberately: the fractional thresholds and the point-based font
# floor are invariant by construction, and the two absolute pixel tolerances are
# scaled from a 100-dpi baseline, because plotting code sets dpi freely and the
# detector's sensitivity must not move with it.
LAYOUT_LINT_SRC = '''
def _scai_lint_figure_layout(fig):
    """Measure where text actually landed and report collisions.

    Returns a list of violation dicts. Never raises.
    """
    # Absolute pixel tolerances have to be stated at some dpi, and plotting code
    # sets dpi freely (plt.figure(dpi=...), sc.settings.set_figure_params). Scale
    # them from a 100-dpi baseline so the detector's sensitivity is a property of
    # the layout rather than of a number the agent happened to pass.
    try:
        _scale = (float(fig.dpi) or 100.0) / 100.0
    except Exception:
        _scale = 1.0
    _PAD = 1.0 * _scale             # px slack: anti-aliased descenders touch harmlessly
    _MIN_OVERLAP_AREA = 12.0 * _scale * _scale  # px^2 below this is not a legibility problem
    _MIN_OVERLAP_FRAC = 0.06    # of the smaller box, so a clipped corner is ignored
    _MIN_FONT_PT = 6.0
    _MAX_ANNOTATIONS_PER_AXES = 60   # above this an axes is a value matrix, not labels
    _MAX_PAIRS = 400000              # hard ceiling on comparison work
    _MAX_PER_KIND = 8                # keep the emitted report small

    violations = []
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        # Returning [] here would print the clean sentinel and the backend
        # would read an unmeasurable figure as a measured-clean one. None is
        # the "could not measure" answer.
        return None

    # A layout engine re-runs on every draw and is not idempotent, so the draw
    # above has already moved things and the savefig that follows would move
    # them again — leaving the reported coordinates describing a layout the
    # delivered image does not have. Freeze the engine now: the figure keeps
    # the layout this draw produced, which is the one savefig would have
    # produced on its own, and the boxes reported below are the boxes saved.
    try:
        if fig.get_layout_engine() is not None:
            fig.set_layout_engine("none")
    except Exception:
        pass

    def _extent(artist):
        try:
            bb = artist.get_window_extent(renderer=renderer)
        except TypeError:
            try:
                bb = artist.get_window_extent()
            except Exception:
                return None
        except Exception:
            return None
        try:
            if bb is None or bb.width <= 0 or bb.height <= 0:
                return None
            return (float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1))
        except Exception:
            return None

    def _overlap(a, b):
        """Intersection area of two (x0, y0, x1, y1) boxes, shrunk by _PAD."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0) + _PAD
        iy0 = max(ay0, by0) + _PAD
        ix1 = min(ax1, bx1) - _PAD
        iy1 = min(ay1, by1) - _PAD
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def _area(b):
        return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

    def _significant(a, b):
        ov = _overlap(a, b)
        if ov < _MIN_OVERLAP_AREA:
            return 0.0
        smaller = min(_area(a), _area(b))
        if smaller <= 0:
            return 0.0
        if ov / smaller < _MIN_OVERLAP_FRAC:
            return 0.0
        return ov

    def _clean(s):
        try:
            s = str(s).replace("\\n", " ").strip()
        except Exception:
            return ""
        return s[:40]

    def _fontsize(artist):
        try:
            return float(artist.get_fontsize())
        except Exception:
            return None

    def _rotation(artist):
        try:
            return float(artist.get_rotation()) % 180.0
        except Exception:
            return 0.0

    def _usable_text(artist):
        try:
            if not artist.get_visible():
                return False
            txt = artist.get_text()
        except Exception:
            return False
        return bool(txt) and bool(str(txt).strip())

    # ---- collect axes ----------------------------------------------------
    axes_list = []
    for ax in fig.get_axes():
        try:
            if not ax.get_visible():
                continue
        except Exception:
            pass
        bb = _extent(ax)
        if bb is None:
            continue
        # An axes turned off with ax.axis('off') paints nothing: no frame, no
        # ticks, no labels. scanpy builds its legend columns as exactly that —
        # an axison=False container spanning a gridspec cell, with the real
        # legend axes nested inside — so treating the container as a panel
        # makes its own children's labels read as spilling into it, which is
        # how a stock dotplot produced eight blocking violations.
        axis_on = True
        try:
            axis_on = bool(getattr(ax, "axison", True))
        except Exception:
            pass
        # Tick-run adjacency is a claim about labels stacked along a straight
        # axis. On polar the labels follow a circle and on 3D they run
        # diagonally, so projecting them onto one axis puts non-neighbours next
        # to each other and every such figure reports a collision.
        projection = "rectilinear"
        try:
            projection = str(getattr(ax, "name", "rectilinear"))
        except Exception:
            pass
        axes_list.append({
            "ax": ax,
            "bbox": bb,
            "axis_on": axis_on,
            "rectilinear": projection == "rectilinear",
        })

    def _iou(a, b):
        inter = _overlap(a, b)
        union = _area(a) + _area(b) - inter
        return (inter / union) if union > 0 else 0.0

    def _contains(outer, inner, slack=1.0):
        return (
            outer[0] <= inner[0] + slack and outer[1] <= inner[1] + slack
            and outer[2] >= inner[2] - slack and outer[3] >= inner[3] - slack
        )

    # Axes that coincide (twinx, sharex) or that nest one inside the other
    # (an inset_axes colorbar, scanpy's legend container) are not neighbouring
    # panels, and text belonging to one is not spilling into the other. Union
    # them so the foreign-axes check speaks only about genuinely adjacent
    # panels.
    for i, entry in enumerate(axes_list):
        entry["group"] = i

    def _find(i):
        while axes_list[i]["group"] != i:
            axes_list[i]["group"] = axes_list[axes_list[i]["group"]]["group"]
            i = axes_list[i]["group"]
        return i

    def _union(i, j):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            axes_list[rj]["group"] = ri

    for i in range(len(axes_list)):
        for j in range(i + 1, len(axes_list)):
            a, b = axes_list[i]["bbox"], axes_list[j]["bbox"]
            if _iou(a, b) > 0.9 or _contains(a, b) or _contains(b, a):
                _union(i, j)
    for i in range(len(axes_list)):
        axes_list[i]["group"] = _find(i)

    # ---- collect text -----------------------------------------------------
    # role "structural": the frame of the figure — titles, axis labels, tick
    #   labels, legend entries. Always compared against everything.
    # role "annotation": free ax.text/annotate calls. Compared against
    #   structural text and against annotations of *other* axes, but not
    #   against siblings when an axes carries a whole matrix of them.
    entries = []
    seen_ids = set()

    def _add(artist, role, owner, label_hint):
        if artist is None or id(artist) in seen_ids:
            return
        if not _usable_text(artist):
            return
        bb = _extent(artist)
        if bb is None:
            return
        seen_ids.add(id(artist))
        entries.append({
            "bbox": bb,
            "role": role,
            "owner": owner,
            "text": _clean(artist.get_text()),
            "what": label_hint,
            "fontsize": _fontsize(artist),
        })

    for idx, entry in enumerate(axes_list):
        ax = entry["ax"]
        group = entry["group"]
        try:
            _add(ax.title, "structural", group, "title")
        except Exception:
            pass
        for attr, hint in (("xaxis", "xlabel"), ("yaxis", "ylabel")):
            try:
                _add(getattr(ax, attr).label, "structural", group, hint)
            except Exception:
                pass

        tick_groups = []
        axbb = entry["bbox"]
        # An axison=False axes renders no ticks at all, but every tick label
        # Text it holds still reports get_visible() is True. Collecting them
        # invents text at coordinates nothing was ever drawn at.
        for getter, hint, lo, hi in () if not entry["axis_on"] else (
            (ax.get_xticklabels, "xticklabel", 0, 2),
            (ax.get_yticklabels, "yticklabel", 1, 3),
        ):
            try:
                labels = [t for t in getter() if _usable_text(t)]
            except Exception:
                labels = []
            group_entries = []
            for t in labels:
                # get_*ticklabels() also hands back ticks that fall outside the
                # current view limits. Those Text artists are positioned off
                # the canvas but still report get_visible() is True, so without
                # this filter they show up as phantom out-of-canvas text and as
                # phantom neighbours in the adjacency check below. Test for
                # disjointness rather than for the box centre: ha='right' and
                # rotation displace the box off its tick by half its width, so
                # a centre test drops labels that are genuinely in view — and
                # a dropped label is invisible to every other check too.
                tbb = _extent(t)
                if tbb is None:
                    continue
                if tbb[hi] < axbb[lo] - 1.0 or tbb[lo] > axbb[hi] + 1.0:
                    continue
                before = len(entries)
                _add(t, "structural", group, hint)
                if len(entries) > before:
                    entries[-1]["rotation"] = _rotation(t)
                    group_entries.append(entries[-1])
            if group_entries:
                tick_groups.append((hint, group_entries))
        entry["tick_groups"] = tick_groups

        try:
            legend = ax.get_legend()
        except Exception:
            legend = None
        if legend is not None:
            try:
                if legend.get_visible():
                    for t in legend.get_texts():
                        _add(t, "structural", group, "legend entry")
                    lbb = _extent(legend)
                    if lbb is not None:
                        entry["legend_bbox"] = lbb
                        entry["legend_group"] = group
            except Exception:
                pass

        try:
            ax_texts = [t for t in ax.texts if _usable_text(t)]
        except Exception:
            ax_texts = []
        dense = len(ax_texts) > _MAX_ANNOTATIONS_PER_AXES
        entry["dense_annotations"] = dense
        for t in ax_texts:
            before = len(entries)
            _add(t, "annotation", group, "annotation")
            if len(entries) > before and dense:
                entries[-1]["dense"] = True

    # figure-level text: suptitle lives outside fig.texts in current matplotlib
    for attr in ("_suptitle", "_supxlabel", "_supylabel"):
        try:
            _add(getattr(fig, attr, None), "structural", -1, "figure title")
        except Exception:
            pass
    try:
        for t in fig.texts:
            _add(t, "structural", -1, "figure text")
    except Exception:
        pass
    try:
        for lg in getattr(fig, "legends", []) or []:
            if not lg.get_visible():
                continue
            for t in lg.get_texts():
                _add(t, "structural", -1, "figure legend entry")
    except Exception:
        pass

    def _worst_first(found):
        """Blocking hits first, then advisory, each in the order found.

        Every section below caps its report with a head slice, and a section
        that can emit both severities collects them in artist order — so an
        axes full of advisory crowding would push every blocking collision out
        of the report and, with it, off the gate that reads the report.
        """
        return sorted(
            found, key=lambda h: 0 if h["severity"] == "block" else 1
        )

    # ---- V1: text spilling onto a foreign axes' data area -----------------
    # "Foreign" means a genuinely separate panel: not the same axes, not a twin
    # or a nested child of it, and not a container that paints nothing.
    panels = [a for a in axes_list if a["axis_on"]]
    hits = []
    for e in entries:
        # Figure-level text — a suptitle, a fig.text, a fig.legend entry —
        # belongs to the whole figure and not to any panel, so sitting over a
        # panel's rectangle is where it was placed, not a spill out of
        # somewhere else. Handing a legend that has outgrown its axes to the
        # figure is the remedy the plotting instructions recommend, and
        # fig.legend's default location is inside the figure rectangle, i.e.
        # on top of a panel: blocking that makes the recommended fix
        # unpublishable. Report the overlap, but leave the verdict to V2 —
        # what makes text unreadable is other text covering it, and every
        # figure-level artist is compared against all of it below.
        figure_level = e["owner"] == -1
        for entry in panels:
            if entry["group"] == e["owner"]:
                continue
            ov = _significant(e["bbox"], entry["bbox"])
            if ov > 0:
                hits.append({
                    "kind": "text_over_foreign_axes",
                    "severity": "advise" if figure_level else "block",
                    "text": e["text"],
                    "what": e["what"],
                    "overlap_px2": round(ov, 1),
                    "bbox": [round(v, 1) for v in e["bbox"]],
                    "foreign_axes_bbox": [round(v, 1) for v in entry["bbox"]],
                })
                break
    violations.extend(_worst_first(hits)[:_MAX_PER_KIND])

    # legend frames get their own report: the frame, not just its text, is what
    # covers the neighbour
    lg_hits = []
    for entry in axes_list:
        lbb = entry.get("legend_bbox")
        if lbb is None:
            continue
        for other in panels:
            if other["group"] == entry.get("legend_group"):
                continue
            ov = _significant(lbb, other["bbox"])
            if ov > 0:
                lg_hits.append({
                    "kind": "text_over_foreign_axes",
                    "severity": "block",
                    "text": "<legend box>",
                    "what": "legend",
                    "overlap_px2": round(ov, 1),
                    "bbox": [round(v, 1) for v in lbb],
                    "foreign_axes_bbox": [round(v, 1) for v in other["bbox"]],
                })
                break
    violations.extend(lg_hits[:_MAX_PER_KIND])

    # ---- V2: text drawn on top of other text ------------------------------
    n = len(entries)
    if n * n // 2 <= _MAX_PAIRS:
        tt_hits = []
        for i in range(n):
            a = entries[i]
            for j in range(i + 1, n):
                b = entries[j]
                # a matrix of value annotations inside one axes is not a
                # legibility bug worth a redraw; tick-label runs are covered by
                # the dedicated adjacency check below
                if a["role"] == "annotation" and b["role"] == "annotation":
                    if a["owner"] == b["owner"] and (
                        a.get("dense") or b.get("dense")
                    ):
                        continue
                if (
                    a["role"] == "structural"
                    and b["role"] == "structural"
                    and a["owner"] == b["owner"]
                    and a["what"] == b["what"]
                    and a["what"] in ("xticklabel", "yticklabel")
                ):
                    continue
                ov = _significant(a["bbox"], b["bbox"])
                if ov > 0:
                    # Two free annotations inside one axes crowding each other
                    # is data density, not a layout fault — scanpy's
                    # legend_loc='on data' places cluster names at cluster
                    # centroids and they touch whenever two clusters are close.
                    # Nothing a redraw can reliably fix, so it advises.
                    crowding = (
                        a["role"] == "annotation" and b["role"] == "annotation"
                        and a["owner"] == b["owner"]
                    )
                    tt_hits.append({
                        "kind": "text_over_text",
                        "severity": "advise" if crowding else "block",
                        "text": a["text"],
                        "what": a["what"],
                        "collides_with": b["text"],
                        "collides_with_what": b["what"],
                        "overlap_px2": round(ov, 1),
                        "bbox": [round(v, 1) for v in a["bbox"]],
                    })
        violations.extend(_worst_first(tt_hits)[:_MAX_PER_KIND])
    else:
        violations.append({
            "kind": "lint_skipped",
            "severity": "advise",
            "text": "",
            "what": "text_over_text",
            "detail": "too many text artists (%d) to compare pairwise" % n,
        })

    # ---- V3: tick labels of one axis colliding with each other ------------
    # A tick-label run is one-dimensional: labels are stacked along a single
    # axis and what matters is whether consecutive labels overlap *along that
    # axis*. Judging them by intersection area instead would let a 36-row gene
    # column pass, because each pair only shares a few pixels of height while
    # being tens of pixels wide.
    tick_hits = []
    for entry in axes_list:
        # Only a straight axis: see the projection note where axes are collected.
        if not entry["rectilinear"]:
            continue
        for hint, group_entries in entry.get("tick_groups", []):
            if len(group_entries) < 2:
                continue
            # Rotating the labels is what one DOES about a crowded categorical
            # axis, and it is the remedy this very violation recommends. A
            # rotated label's axis-aligned box is far wider than its glyph run,
            # so the projection reports neighbours as colliding when they are
            # parallel diagonal strips with no shared ink. Judge only the
            # unrotated case, where the box is the ink.
            if any(
                min(e.get("rotation", 0.0), 180.0 - e.get("rotation", 0.0)) > 5.0
                for e in group_entries
            ):
                continue
            lo, hi = (0, 2) if hint == "xticklabel" else (1, 3)
            # A text box carries ascent/descent leading its glyphs do not fill,
            # roughly a fifth of the line box, so stacked labels' boxes start
            # intersecting before anything is actually unreadable. Along the
            # writing direction the box is the ink, so only a hair of slack.
            min_frac = 0.22 if hint == "yticklabel" else 0.05
            ordered = sorted(group_entries, key=lambda e: e["bbox"][lo])
            worst = None
            n_collisions = 0
            for k in range(len(ordered) - 1):
                a = ordered[k]["bbox"]
                b = ordered[k + 1]["bbox"]
                overlap = min(a[hi], b[hi]) - max(a[lo], b[lo]) - _PAD
                extent = min(a[hi] - a[lo], b[hi] - b[lo])
                if overlap > max(1.0, extent * min_frac):
                    n_collisions += 1
                    if worst is None or overlap > worst[0]:
                        worst = (overlap, ordered[k], ordered[k + 1])
            if worst is not None:
                tick_hits.append({
                    "kind": "tick_label_collision",
                    "severity": "block",
                    "text": worst[1]["text"],
                    "what": hint,
                    "collides_with": worst[2]["text"],
                    "n_labels": len(group_entries),
                    "n_collisions": n_collisions,
                    "overlap_px": round(worst[0], 1),
                    "axes_bbox": [round(v, 1) for v in entry["bbox"]],
                })
    violations.extend(tick_hits[:_MAX_PER_KIND])

    # There is deliberately no "text outside the figure canvas" check. Every
    # save here goes through bbox_inches='tight', which grows the canvas rather
    # than cropping, so nothing is lost by being outside it — and the nominal
    # fig.bbox is routinely exceeded by perfectly placed labels on figures
    # whose axes scanpy sizes independently, and by every 3D projection.
    # The check produced only noise.

    # ---- V4: text too small to read in print ------------------------------
    small = []
    for e in entries:
        fs = e.get("fontsize")
        if fs is not None and fs < _MIN_FONT_PT:
            small.append({
                "kind": "tiny_text",
                "severity": "advise",
                "text": e["text"],
                "what": e["what"],
                "fontsize_pt": round(fs, 1),
            })
    if small:
        smallest = sorted(small, key=lambda v: v["fontsize_pt"])
        violations.extend(smallest[:_MAX_PER_KIND])

    return violations


def _scai_emit_layout_lint(fig):
    """Run the linter and print the sentinel line. Never raises."""
    import json as _scai_lint_json
    try:
        _v = _scai_lint_figure_layout(fig)
    except Exception as _lint_err:
        print("__SCAI_LAYOUT_LINT_ERROR__ %s" % _lint_err)
        return
    if _v is None:
        print("__SCAI_LAYOUT_LINT_ERROR__ figure could not be measured")
        return
    try:
        print("__SCAI_LAYOUT_LINT__" + _scai_lint_json.dumps(_v, default=str))
    except Exception as _dump_err:
        print("__SCAI_LAYOUT_LINT_ERROR__ %s" % _dump_err)
'''


# ---------------------------------------------------------------------------
# Backend-side helpers
# ---------------------------------------------------------------------------

def _parse_layout_verdict(
    output: str | None,
) -> List[Dict[str, Any]] | None:
    """Return the last authoritative lint verdict, or None if it is invalid.

    The injected linter runs after user plotting code. Reading the last lint
    event prevents an earlier user-printed normal sentinel from masking a later
    renderer error. A syntactically valid payload must still be a JSON list;
    malformed, empty, or wrong-shaped payloads are not measurements.
    """
    if not output:
        return None
    for line in reversed(output.splitlines()):
        event = line.lstrip()
        if event.startswith(LAYOUT_LINT_ERROR_SENTINEL):
            return None
        if not event.startswith(LAYOUT_LINT_SENTINEL):
            continue
        payload = event[len(LAYOUT_LINT_SENTINEL):].strip()
        if not payload:
            logger.warning("[LayoutLint] sentinel present without a payload")
            return None
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            logger.warning("[LayoutLint] sentinel present but payload is not JSON")
            return None
        if not isinstance(parsed, list):
            logger.warning("[LayoutLint] verdict payload is not a list")
            return None
        if any(
            not isinstance(violation, dict)
            or violation.get("kind") not in LAYOUT_LINT_KINDS
            or violation.get("severity") not in LAYOUT_LINT_SEVERITIES
            for violation in parsed
        ):
            logger.warning("[LayoutLint] verdict contains an invalid violation")
            return None
        return parsed
    return None


def parse_layout_violations(output: str | None) -> List[Dict[str, Any]]:
    """Extract the violation list from sandbox stdout.

    Returns ``[]`` when the sentinel is absent or unparseable — a linter that
    could not run must never be read as "the figure is clean", so callers that
    need to distinguish the two should use :func:`lint_ran`.
    """
    verdict = _parse_layout_verdict(output)
    return verdict if verdict is not None else []


def lint_ran(output: str | None) -> bool:
    """Whether the sandbox linter produced a valid list-shaped verdict."""
    return _parse_layout_verdict(output) is not None


def blocking_violations(
    violations: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """The subset that makes a figure unreadable rather than imperfect.

    Filtered on the severity the linter assigned, not on the kind: the same
    kind can be either, because whether a collision is a fault depends on what
    collided. Two free annotations crowding inside one axes is data density
    (scanpy's ``legend_loc='on data'`` puts cluster names at centroids, and
    they touch when two clusters are close); the same kind between a legend and
    a neighbouring panel's tick labels is a layout fault.
    """
    return [
        v for v in (violations or [])
        if str(v.get("severity", "")) == "block"
    ]


def summarize_violations(violations: List[Dict[str, Any]] | None) -> str:
    """One-line count by kind, for logs and status messages."""
    counts: Dict[str, int] = {}
    for v in violations or []:
        kind = str(v.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    if not counts:
        return "no layout violations"
    return ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))


def format_violations_for_prompt(
    violations: List[Dict[str, Any]] | None,
    limit: int = 12,
) -> str:
    """Render violations as instructions a redraw can act on.

    Coordinates are included deliberately: the vision reviewer can say a legend
    overlaps something, but only the renderer knows which artist, by how much,
    and where.
    """
    items = list(violations or [])
    if not items:
        return ""

    order = {"block": 0, "advise": 1}
    items.sort(key=lambda v: order.get(str(v.get("severity")), 2))

    lines: List[str] = []
    for v in items[:limit]:
        kind = str(v.get("kind", ""))
        text = str(v.get("text", "") or "")
        what = str(v.get("what", "") or "element")
        # Label from the severity the linter assigned, never from the kind:
        # the same kind is blocking or advisory depending on what collided,
        # and the reviewer is told not to approve anything marked BLOCKING
        # "regardless of revision count". Calling advisory crowding blocking
        # spends the whole revision budget on a defect no redraw can clear and
        # then drops the figure from every report.
        tag = "BLOCKING" if str(v.get("severity", "")) == "block" else "advisory"
        if kind == "text_over_foreign_axes":
            lines.append(
                f"- {tag}: the {what} \"{text}\" is drawn on top of a "
                f"different panel's plotting area "
                f"(overlap {v.get('overlap_px2')} px^2; its box "
                f"{v.get('bbox')}, the other panel's box "
                f"{v.get('foreign_axes_bbox')}). Give it space of its own "
                f"instead of letting it spill sideways."
            )
        elif kind == "text_over_text":
            lines.append(
                f"- {tag}: the {what} \"{text}\" overlaps the "
                f"{v.get('collides_with_what', 'text')} "
                f"\"{v.get('collides_with', '')}\" "
                f"(overlap {v.get('overlap_px2')} px^2 at {v.get('bbox')}). "
                f"Both are unreadable where they cross."
            )
        elif kind == "tick_label_collision":
            lines.append(
                f"- {tag}: {what}s run into each other — "
                f"\"{text}\" overlaps \"{v.get('collides_with', '')}\" by "
                f"{v.get('overlap_px')} px, and {v.get('n_collisions')} of "
                f"{v.get('n_labels')} labels on that axis collide. Show fewer "
                f"ticks, shorten the labels, rotate them, or make that axes "
                f"taller/wider."
            )
        elif kind == "tiny_text":
            lines.append(
                f"- the {what} \"{text}\" renders at "
                f"{v.get('fontsize_pt')} pt, below the 6 pt floor for print."
            )
        elif kind == "lint_skipped":
            lines.append(
                f"- note: overlap checking was skipped ({v.get('detail', '')})."
            )
    remaining = len(items) - len(lines)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more layout violation(s).")
    return "\n".join(lines)
