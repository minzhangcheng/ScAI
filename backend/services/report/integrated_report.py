"""Integrated Report publication and Markdown boundary helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Sequence

from backend.models.schemas import IntegratedReportFigureRef


_ATX_HEADING_RE = re.compile(
    r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*?)(?:[ \t]+#+[ \t]*)?$"
)
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_METHOD_TITLES = {
    "methods",
    "methods summary",
    "methodology",
    "computational methods",
    "analysis methods",
    "materials and methods",
    "experimental methods",
    "experimental procedures",
}
_INTERNAL_DECLARATION_TITLES = {
    "authoritative declarations",
    "figure type declarations",
    "approved primary tables",
    "pi figure dispositions and reservations",
    "figure inventory",
    "figure inventories",
}
_DECLARATION_TITLES = {
    "declarations and metadata",
    "declarations",
    "metadata",
}


def integrated_report_catalog_fingerprint(
    refs: Sequence[IntegratedReportFigureRef],
) -> str:
    """Hash the ordered render identity, not mutable labels or descriptions."""
    payload = [
        [ref.number, ref.figure_id, ref.render_sha256, ref.pdf_sha256]
        for ref in refs
    ]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def integrated_report_source_fingerprint(state: object, lineage_id: str) -> str:
    """Identify report inputs without coupling retries to state-sync versions."""

    def value(item: object, name: str, default=None):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    def scalar(item: object):
        raw = getattr(item, "value", item)
        if hasattr(raw, "isoformat"):
            return raw.isoformat()
        return raw

    target = str(lineage_id or "main")
    round_results = []
    for result in value(state, "round_results", []) or []:
        if str(value(result, "lineage_id", None) or "main") != target:
            continue
        round_results.append([
            value(result, "result_identity_key", ""),
            value(result, "round_number", 0),
            scalar(value(result, "timestamp", None)),
        ])

    figures = []
    for figure in value(state, "publication_figures", []) or []:
        if str(value(figure, "lineage_id", None) or "main") != target:
            continue
        figures.append([
            str(value(figure, "id", "")),
            str(value(figure, "render_sha256", "") or ""),
            str(value(figure, "pdf_sha256", "") or ""),
            str(value(figure, "name", "") or ""),
            str(value(figure, "description", "") or ""),
            str(value(figure, "figure_type", "") or ""),
            bool(value(figure, "is_current", True)),
            bool(value(figure, "is_published_head", False)),
            scalar(value(figure, "review_status", "pending")),
            scalar(value(figure, "invalidated_at", None)),
            int(value(figure, "revision_count", 0) or 0),
            int(value(figure, "round_number", 0) or 0),
            bool(value(figure, "diagnostic", False)),
        ])

    tables = []
    for table in value(state, "publication_tables", []) or []:
        if str(value(table, "lineage_id", None) or "main") != target:
            continue
        tables.append([
            str(value(table, "id", "")),
            str(value(table, "csv_sha256", "") or ""),
            str(value(table, "name", "") or ""),
            str(value(table, "description", "") or ""),
            bool(value(table, "is_current", True)),
            scalar(value(table, "review_status", "pending")),
        ])

    legacy = []
    for artifact in value(state, "legacy_artifacts", []) or []:
        if str(value(artifact, "lineage_id", None) or "main") != target:
            continue
        legacy.append([
            str(value(artifact, "id", "")),
            str(value(artifact, "render_sha256", "") or ""),
        ])

    revision_notes = []
    for task in value(state, "figure_revision_tasks", []) or []:
        task_lineage = value(task, "lineage_id", None)
        if task_lineage is not None and str(task_lineage or "main") != target:
            continue
        revision_notes.append([
            str(value(task, "id", "")),
            str(value(task, "current_figure_id", "") or ""),
            str(value(task, "selected_figure_id", "") or ""),
            str(value(task, "status", "") or ""),
            str(value(task, "pi_report_note", "") or ""),
        ])

    payload = {
        "lineage_id": target,
        "current_round": int(value(state, "current_round", 0) or 0),
        "integration_cycle": int(value(state, "integration_cycle", 0) or 0),
        "round_results": round_results,
        "publication_figures": sorted(figures, key=lambda item: str(item[0])),
        "publication_tables": sorted(tables, key=lambda item: str(item[0])),
        "legacy_artifacts": sorted(legacy, key=lambda item: str(item[0])),
        "figure_revision_tasks": sorted(
            revision_notes, key=lambda item: str(item[0]),
        ),
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized_heading(value: str) -> str:
    value = re.sub(r"[*_`]+", "", value).strip().lower()
    # Section numbers are heading syntax, not part of the section's name.
    value = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", value)
    return re.sub(r"\s+", " ", value)


def _heading_at(lines: Sequence[str], index: int) -> tuple[int, str, int] | None:
    """Return ``(level, title, consumed_lines)`` for one Markdown heading."""
    match = _ATX_HEADING_RE.match(lines[index])
    if match:
        return len(match.group(1)), _normalized_heading(match.group(2)), 1
    if index + 1 >= len(lines) or not lines[index].strip():
        return None
    underline = _SETEXT_UNDERLINE_RE.match(lines[index + 1])
    if not underline:
        return None
    level = 1 if underline.group(1).startswith("=") else 2
    return level, _normalized_heading(lines[index]), 2


def _strip_forbidden_sections(markdown: str) -> str:
    """Remove forbidden structural sections without matching fenced examples.

    Model-authored declarations and inventories are not authoritative. Canonical
    PI reservations are rendered separately from structured state, so the whole
    forbidden section can be removed without losing those disclosures.
    """
    lines = markdown.splitlines()
    out: list[str] = []
    skip_level: int | None = None
    in_fence = False
    fence_marker = ""

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""

        heading = None if in_fence else _heading_at(lines, index)
        consumed = heading[2] if heading else 1
        if heading:
            level, title, _ = heading
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and (
                title in _METHOD_TITLES
                or title in _INTERNAL_DECLARATION_TITLES
                or title in _DECLARATION_TITLES
            ):
                skip_level = level
                index += consumed
                continue

        if skip_level is None:
            out.append(line)
            if consumed == 2 and heading:
                out.append(lines[index + 1])
        index += consumed

    # Avoid leaving a renamed empty wrapper after all internal subsections were
    # removed. This operates only on adjacent Markdown lines, not prose.
    compact: list[str] = []
    for index, line in enumerate(out):
        heading = _heading_at(out, index)
        if heading and heading[1] == "reporting limitations":
            level = heading[0]
            has_content = False
            for following in out[index + 1:]:
                next_heading = _ATX_HEADING_RE.match(following)
                if next_heading and len(next_heading.group(1)) <= level:
                    break
                if following.strip() and not next_heading:
                    has_content = True
                    break
            if not has_content:
                continue
        compact.append(line)
    return "\n".join(compact).strip()


def _normalize_figure_links_chunk(markdown: str, valid_numbers: set[int]) -> str:
    """Repair only figure citations; never ask an LLM to rewrite the report."""
    placeholders: dict[str, str] = {}

    def stash(value: str) -> str:
        token = f"\x00SCAI_FIGURE_{len(placeholders)}\x00"
        placeholders[token] = value
        return token

    def closing_delimiter(
        text: str, start: int, opener: str, closer: str,
    ) -> int | None:
        depth = 0
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return index
        return None

    def protect_markdown_nodes(text: str) -> str:
        """Protect Markdown nodes while canonicalizing figure-link labels."""
        out: list[str] = []
        index = 0
        figure_label = re.compile(r"\s*Figure\s+(\d+)\s*", re.IGNORECASE)
        while index < len(text):
            if text[index] == "`":
                run_end = index
                while run_end < len(text) and text[run_end] == "`":
                    run_end += 1
                marker = text[index:run_end]
                closing = text.find(marker, run_end)
                if closing >= 0:
                    end = closing + len(marker)
                    out.append(stash(text[index:end]))
                    index = end
                    continue

            image = text.startswith("![", index)
            if image or text[index] == "[":
                bracket = index + 1 if image else index
                label_end = closing_delimiter(text, bracket, "[", "]")
                if label_end is not None:
                    cursor = label_end + 1
                    while cursor < len(text) and text[cursor] in " \t":
                        cursor += 1
                    node_end = label_end + 1
                    has_target = False
                    if cursor < len(text) and text[cursor] == "(":
                        destination_end = closing_delimiter(text, cursor, "(", ")")
                        if destination_end is not None:
                            node_end = destination_end + 1
                            has_target = True
                    elif cursor < len(text) and text[cursor] == "[":
                        reference_end = closing_delimiter(text, cursor, "[", "]")
                        if reference_end is not None:
                            node_end = reference_end + 1
                            has_target = True

                    label = text[bracket + 1:label_end]
                    match = figure_label.fullmatch(label)
                    is_definition = cursor < len(text) and text[cursor] == ":"
                    if match and not image and not is_definition:
                        number = int(match.group(1))
                        replacement = (
                            f"[Figure {number}](#figure-{number})"
                            if number in valid_numbers else ""
                        )
                        out.append(stash(replacement))
                        index = node_end if has_target else label_end + 1
                        continue
                    out.append(stash(text[index:node_end]))
                    index = node_end
                    continue

            out.append(text[index])
            index += 1
        return "".join(out)

    text = protect_markdown_nodes(markdown)

    def figure_range(match: re.Match[str]) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        if end < start or end - start > 50:
            return ""
        numbers = [number for number in range(start, end + 1) if number in valid_numbers]
        if not numbers:
            return ""
        links = [f"[Figure {number}](#figure-{number})" for number in numbers]
        if len(links) == 1:
            return stash(links[0])
        return stash(", ".join(links[:-1]) + f", and {links[-1]}")

    text = re.sub(
        r"\bFigures?\s+(\d+)\s*[\u2013\u2014-]\s*(\d+)\b",
        figure_range,
        text,
        flags=re.IGNORECASE,
    )

    def multi_figures(match: re.Match[str]) -> str:
        raw = match.group(1)
        numbers = [int(value) for value in re.findall(r"\d+", raw)]
        numbers = [number for number in numbers if number in valid_numbers]
        if not numbers:
            return ""
        links = [f"[Figure {number}](#figure-{number})" for number in numbers]
        if len(links) == 1:
            return stash(links[0])
        return stash(", ".join(links[:-1]) + f", and {links[-1]}")

    text = re.sub(
        r"\bFigures\s+(\d+(?:\s*,\s*\d+)*(?:\s*,?\s+and\s+\d+))\b",
        multi_figures,
        text,
        flags=re.IGNORECASE,
    )

    def single_figure(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in valid_numbers:
            return ""
        return stash(f"[Figure {number}](#figure-{number})")

    text = re.sub(r"\bFigure\s+(\d+)\b", single_figure, text, flags=re.IGNORECASE)
    for token, value in placeholders.items():
        text = text.replace(token, value)

    # Citation removal must not leave visibly damaged punctuation.
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    return text


def _normalize_figure_links(markdown: str, valid_numbers: set[int]) -> str:
    """Normalize prose chunks while leaving fenced examples byte-for-byte."""
    chunks: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""
    current_is_fence = False
    for line in markdown.splitlines(keepends=True):
        fence = _FENCE_RE.match(line)
        if fence and not in_fence:
            if current:
                chunks.append((current_is_fence, current))
            current = [line]
            current_is_fence = True
            in_fence = True
            fence_marker = fence.group(1)[0]
            continue
        current.append(line)
        if fence and in_fence and fence.group(1)[0] == fence_marker:
            chunks.append((True, current))
            current = []
            current_is_fence = False
            in_fence = False
            fence_marker = ""
    if current:
        chunks.append((current_is_fence, current))
    return "".join(
        "".join(lines)
        if is_fence
        else _normalize_figure_links_chunk("".join(lines), valid_numbers)
        for is_fence, lines in chunks
    )


def sanitize_integrated_report(
    markdown: str,
    figure_refs: Sequence[IntegratedReportFigureRef],
    reporting_limitations: Iterable[str] = (),
) -> str:
    """Apply the Integrated Report structural and citation-only contract."""
    text = _strip_forbidden_sections(markdown or "")
    text = _normalize_figure_links(text, {ref.number for ref in figure_refs})
    limitations = [str(note).strip() for note in reporting_limitations if str(note).strip()]
    if limitations:
        block = "\n".join(f"- {note}" for note in limitations)
        lines = text.splitlines()
        limitation_heading = None
        index = 0
        while index < len(lines):
            heading = _heading_at(lines, index)
            if heading and heading[1] == "reporting limitations":
                limitation_heading = (index, heading[0], heading[2])
                break
            index += heading[2] if heading else 1
        if limitation_heading is None:
            text = f"{text.rstrip()}\n\n## Reporting Limitations\n\n{block}"
        else:
            heading_index, heading_level, heading_lines = limitation_heading
            insertion_index = len(lines)
            index = heading_index + heading_lines
            while index < len(lines):
                heading = _heading_at(lines, index)
                if heading and heading[0] <= heading_level:
                    insertion_index = index
                    break
                index += heading[2] if heading else 1
            lines[insertion_index:insertion_index] = ["", block, ""]
            text = "\n".join(lines)
    return text.strip()
