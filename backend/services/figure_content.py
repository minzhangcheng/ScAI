"""Renderer facts about whether a figure contains drawable data."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

FIGURE_CONTENT_SENTINEL = "__SCAI_FIGURE_CONTENT__"
FIGURE_CONTENT_ERROR_SENTINEL = "__SCAI_FIGURE_CONTENT_ERROR__"

_FAILURE_PHRASES = (
    "not found",
    "not available",
    "no data",
    "unavailable",
    "missing",
    "could not",
    "cannot",
    "failed",
    "error",
    "please ensure",
    "please run",
)


FIGURE_CONTENT_SRC = r'''
def _scai_census_figure_content(fig):
    """Count drawable data artists and collect a bounded text sample."""
    _MAX_TEXTS = 24
    _MAX_TEXT_CHARS = 160
    try:
        _axes_list = list(fig.axes)
    except Exception:
        return None

    _counts = {"data_artists": 0, "axes": 0, "texts": []}

    def _is_bare_text(_artist):
        try:
            if _artist.__class__.__name__ not in ("Text", "Annotation"):
                return False
            if getattr(_artist, "arrow_patch", None) is not None:
                return False
            if getattr(_artist, "arrowprops", None):
                return False
        except Exception:
            return False
        return True

    def _record_text(_artist):
        try:
            if not _artist.get_visible():
                return
            _text = str(_artist.get_text() or "").strip()
        except Exception:
            return
        if _text and len(_counts["texts"]) < _MAX_TEXTS:
            _counts["texts"].append(_text[:_MAX_TEXT_CHARS])

    def _collection_has_geometry(_collection):
        try:
            _n_paths = len(_collection.get_paths())
            _offsets = _collection.get_offsets()
            _n_offsets = 0 if _offsets is None else len(_offsets)
        except Exception:
            return True
        return _n_paths > 0 and _n_offsets > 0

    def _count_free_artists(_container):
        _count = 0
        for _artist in list(_container or []):
            if _is_bare_text(_artist):
                _record_text(_artist)
            else:
                _count += 1
        return _count

    try:
        for _ax in _axes_list:
            _counts["axes"] += 1
            try:
                for _line in list(getattr(_ax, "lines", None) or []):
                    try:
                        if len(_line.get_xydata()) > 0:
                            _counts["data_artists"] += 1
                    except Exception:
                        _counts["data_artists"] += 1
                _counts["data_artists"] += len(list(getattr(_ax, "patches", None) or []))
                for _collection in list(getattr(_ax, "collections", None) or []):
                    if _collection_has_geometry(_collection):
                        _counts["data_artists"] += 1
                _counts["data_artists"] += len(list(getattr(_ax, "images", None) or []))
                _counts["data_artists"] += len(list(getattr(_ax, "tables", None) or []))
                _counts["data_artists"] += _count_free_artists(
                    getattr(_ax, "artists", None)
                )
                for _text in list(getattr(_ax, "texts", None) or []):
                    _record_text(_text)
                    if not _is_bare_text(_text):
                        _counts["data_artists"] += 1
                try:
                    _title = str(_ax.get_title() or "").strip()
                except Exception:
                    _title = ""
                if _title and len(_counts["texts"]) < _MAX_TEXTS:
                    _counts["texts"].append(_title[:_MAX_TEXT_CHARS])
            except Exception:
                continue

        _counts["data_artists"] += len(list(getattr(fig, "images", None) or []))
        _counts["data_artists"] += _count_free_artists(getattr(fig, "artists", None))
        for _text in list(getattr(fig, "texts", None) or []):
            _record_text(_text)
            if not _is_bare_text(_text):
                _counts["data_artists"] += 1
    except Exception:
        return None
    return _counts
'''


def parse_figure_content(output: str | None) -> dict[str, Any] | None:
    """Return the last valid renderer census, if the kernel emitted one."""
    if not output:
        return None
    for line in reversed(output.splitlines()):
        event = line.lstrip()
        if event.startswith(FIGURE_CONTENT_ERROR_SENTINEL):
            return None
        if not event.startswith(FIGURE_CONTENT_SENTINEL):
            continue
        payload = event[len(FIGURE_CONTENT_SENTINEL):].strip()
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        data_artists = parsed.get("data_artists")
        axes = parsed.get("axes")
        texts = parsed.get("texts")
        if (
            type(data_artists) is not int
            or data_artists < 0
            or type(axes) is not int
            or axes < 0
            or not isinstance(texts, list)
            or any(not isinstance(text, str) for text in texts)
        ):
            return None
        return parsed
    return None


def census_ran(output: str | None) -> bool:
    return parse_figure_content(output) is not None


def _failure_texts(census: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for text in census.get("texts") or []:
        if isinstance(text, str) and any(
            phrase in text.lower() for phrase in _FAILURE_PHRASES
        ):
            result.append(text.strip())
    return result


def empty_render_reason(output: str | None) -> str | None:
    """Return a reason only for an explicit measured zero-data census."""
    census = parse_figure_content(output)
    if census is None or census["data_artists"] != 0:
        return None
    axes = census["axes"]
    failure_texts = _failure_texts(census)
    if failure_texts:
        detail = (
            " The canvas carries only text, including "
            + "; ".join(f'"{text}"' for text in failure_texts[:3])
            + ", so this is a rendered failure message rather than a figure."
        )
    else:
        detail = " The canvas carries no plotted data at all."
    return (
        "Figure generation produced no plotted data: the renderer measured 0 "
        f"data artists across {axes} axes.{detail} Do not catch a missing "
        "variable, column or key and draw a message onto the figure — a "
        "figure that reports its own failure is indistinguishable from a "
        "result. Either plot the requested data from values that actually "
        "exist, or let the exception propagate so the failure is reported as "
        "an execution error."
    )
