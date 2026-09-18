"""Assemble and reuse round chapters; scientific ordering belongs to the PI."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from backend.models.schema_parts.round_report import (
    RoundReportArtifact, RoundReportBlock, RoundReportChapter,
)
from .artifacts import archival_path_matches_record, artifact_stems, eligible_artifacts
from .report_outline import report_result_identity


def round_chapter_digest(result):
    """Identify the exact saved chapter version independently of its round ID."""
    chapter = getattr(result, "report_chapter", None)
    if chapter is None:
        return ""
    payload = json.dumps(chapter.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def round_artifact_catalog(state):
    """Offer registered artifacts of this round, retaining their exact identity."""
    catalog = []
    for kind, records, path_field in (
        ("figure", state.publication_figures, "png_path"),
        ("table", state.publication_tables, "csv_path"),
    ):
        for record in eligible_artifacts(records, state=state):
            if record.round_number != state.current_round:
                continue
            catalog.append(RoundReportArtifact(
                artifact_id=record.id,
                kind=kind,
                filename=Path(getattr(record, path_field) or record.name).name,
                caption=record.description or record.interpretation or "",
            ))
    return catalog


def chapter_prompt(catalog):
    return """
## Reusable Round Report Chapter
Alongside the synthesis JSON, return a `report_chapter` object:
{"title": "Scientific chapter title", "blocks": [
  {"text": "Report-ready Markdown explaining a finding and its limitations",
   "artifact_ids": ["exact catalog ID, in reading order"]}
], "conclusions": "This round's conclusions and remaining uncertainty"}.
Begin with this round's research question. Organize blocks by the reasoning a
reader needs, and place each supporting artifact immediately after the block
that explains it. Decide the order from the evidence and research question;
catalog order, filenames, creation times, and plot types are NOT reading order.
Use each offered artifact once. Explain auxiliary/diagnostic evidence where it
belongs. Do not invent IDs, numbers, observations, or missing analyses. Use plain
prose when no artifact supports a finding. Refer to artifacts only through
artifact_ids, never handwritten FIGURE/TABLE markers or image links. Existing
captions will be retained verbatim. Do not repeat generic methods or execution
logs. This chapter is saved at round completion and reused without rewriting in
the final report. Later integration may explicitly qualify earlier conclusions.
The host validates references only; you own scientific interpretation and order.

Artifact catalog (not ordered for presentation):
""" + json.dumps([item.model_dump() for item in catalog], ensure_ascii=False)


def assemble_round_chapter(raw, catalog, summary, round_number):
    """Normalize objective references without failing synthesis or losing evidence."""
    try:
        if not isinstance(raw, dict) or not raw.get("blocks"):
            raise ValueError("No chapter blocks")
        # Captions and identities come from the catalog, not model-written copies.
        chapter = RoundReportChapter(
            title=raw["title"], blocks=raw["blocks"],
            conclusions=raw.get("conclusions", summary), artifacts=catalog,
        )
        if not chapter.title.strip() or not any(b.text.strip() for b in chapter.blocks):
            raise ValueError("Empty chapter")
        if any(re.search(r"\[(?:FIGURE|TABLE)(?:_CAPTION)?:|!\[|<img\b", b.text, re.I)
               for b in chapter.blocks):
            raise ValueError("Unstructured image reference")
    except (ValidationError, ValueError, KeyError, TypeError):
        chapter = RoundReportChapter(
            title=f"Round {round_number} Findings",
            blocks=[RoundReportBlock(text=summary or "No round summary was recorded.")],
            artifacts=catalog, source="fallback",
            assembly_notes=[
                "PI chapter composition was unavailable; the retained round summary "
                "and artifact captions are shown without a new interpretation."
            ],
        )
    known = {item.artifact_id for item in catalog}
    seen = set()
    for block in chapter.blocks:
        resolved = []
        for aid in block.artifact_ids:
            if aid not in known:
                note = "An unavailable artifact reference in the PI chapter was omitted."
                if note not in chapter.assembly_notes:
                    chapter.assembly_notes.append(note)
            elif aid not in seen:
                resolved.append(aid)
                seen.add(aid)
        block.artifact_ids = resolved
    missing = [item.artifact_id for item in catalog if item.artifact_id not in seen]
    if missing:
        chapter.blocks.append(RoundReportBlock(
            text="### Additional Recorded Evidence\n\nThese artifacts were retained "
                 "but not positioned in the PI's narrative.",
            artifact_ids=missing,
        ))
    return chapter


def chapter_markdown(chapter, names):
    """Render saved prose/order using currently resolved exact artifact IDs."""
    artifacts = {a.artifact_id: a for a in chapter.artifacts}
    parts = []
    for note in chapter.assembly_notes:
        parts.append(f"> {note}")
    for block in chapter.blocks:
        parts.append(block.text)
        for aid in block.artifact_ids:
            artifact = artifacts.get(aid)
            if artifact is None:
                continue
            name = names.get(aid)
            if not name:
                parts.append(
                    f"> Recorded {artifact.kind} unavailable in this report: "
                    f"{artifact.filename}. {artifact.caption}"
                )
                continue
            marker = artifact.kind.upper()
            # Brackets cannot delimit the marker's body; render equivalently.
            caption = artifact.caption.replace("[", "(").replace("]", ")")
            parts.append(f"[{marker}: {name}]\n[{marker}_CAPTION: {caption}]")
    if chapter.conclusions:
        parts.append("### Round Conclusions\n\n" + chapter.conclusions)
    return "\n\n".join(parts)


def report_artifact_names(state, figure_map, tables):
    """Bind saved IDs to the report collector's resolved render names."""
    names = {}
    for figure in eligible_artifacts(state.publication_figures, state=state):
        stems = artifact_stems([figure])
        # Recovered previews can have a materialized suffix. Check exact bytes
        # even for matching stems: two records can have reused one filename.
        for name in figure_map:
            if Path(name).stem in stems and archival_path_matches_record(figure, figure_map[name]):
                names[figure.id] = name
                break
        if figure.id not in names:
            for name, path in figure_map.items():
                if archival_path_matches_record(figure, path):
                    names[figure.id] = name
                    break
        # Branch inheritance assigns a distinct API ID to the same frozen
        # render. Keep the saved chapter's source ID usable only for this
        # explicitly inherited record, never by a filename-only substitution.
        lineage = getattr(figure, "lineage_id", "main")
        if (getattr(figure, "inherited_from_lineage_id", None)
                and figure.id in names and figure.id.endswith(f":{lineage}")):
            names.setdefault(figure.id[:-(len(lineage) + 1)], names[figure.id])
    for table in eligible_artifacts(state.publication_tables, state=state):
        name = Path(table.csv_path).name
        if any(t.get("table_id") == table.id for t in tables):
            names[table.id] = name
    return names


def saved_figure_captions(state, figure_map, tables):
    names = report_artifact_names(state, figure_map, tables)
    return {
        names[a.artifact_id]: a.caption
        for r in state.round_results if getattr(r, "report_chapter", None)
        for a in r.report_chapter.artifacts
        if a.kind == "figure" and a.artifact_id in names
    }


def reusable_sections(state, outline, figure_map, tables):
    """Project saved chapters into the final outline without another LLM call."""
    names = report_artifact_names(state, figure_map, tables)
    rounds = {report_result_identity(r): r for r in state.round_results}
    sections = []
    for section in outline:
        # The final integration remains a global synthesis. Intermediate
        # integrations reuse the already retained PI report at their own position.
        if section.kind not in {"exploration", "direction", "integration_gate"}:
            continue
        keys = section.result_keys
        if not keys:
            # Compatibility for callers supplying an older outline: phase must
            # still match, and an ambiguous same-phase round is not guessed.
            phase = {"exploration": "exploration", "direction": "innovation",
                     "integration_gate": "integration"}[section.kind]
            candidates = [r for r in rounds.values()
                          if r.phase == phase and r.round_number in section.rounds]
            if len(candidates) != len(section.rounds):
                continue
            keys = tuple(report_result_identity(r) for r in candidates)
        if any(key not in rounds for key in keys):
            continue
        owned = [rounds[key] for key in keys]
        if not owned or not all(getattr(r, "report_chapter", None) for r in owned):
            continue
        sections.append({
            "id": section.id,
            "title": owned[0].report_chapter.title,
            "content": "\n\n".join(chapter_markdown(r.report_chapter, names) for r in owned),
            "analysis_type": section.analysis_type,
            "figures": [], "tables": [],
        })
    return sections


def reuse_context(sections):
    if not sections:
        return ""
    return (
        "\n\n## Saved PI Round Chapters (read-only context)\n"
        "These chapters are already written and will be inserted verbatim. "
        "Do not emit, paraphrase, or reorder them or their artifacts. Write only "
        "the requested global or missing legacy sections. Integrate their findings "
        "and explain any later qualification of earlier conclusions.\n\n"
        + "\n\n".join(f"## [{s['id']}] {s['title']}\n{s['content']}" for s in sections)
    )


def merge_reusable_sections(generated, saved, reserved_names):
    """Keep model output from replacing a saved chapter or claiming its assets."""
    saved_ids = {s["id"] for s in saved}
    reserved_stems = {Path(name).stem for name in reserved_names}
    merged = [s for s in generated if s.get("id") not in saved_ids]
    for section in merged:
        section["content"] = re.sub(
            r'\[(FIGURE|TABLE):\s*([^\]]+)\]\s*(?:\[\1_CAPTION:\s*[^\]]*\])?',
            lambda m: "" if Path(m.group(2).strip()).stem in reserved_stems else m.group(0),
            section.get("content", ""),
        )
    return merged + saved
