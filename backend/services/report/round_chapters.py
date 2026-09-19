"""Assemble and reuse round chapters; scientific ordering belongs to the PI."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import ValidationError

from backend.models.schema_parts.session import AnalysisPlan
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


def round_segment_index(state, phase, result):
    """Find a prior segment, adopting legacy ownership only when unambiguous."""
    if result is None:
        return None
    key = report_result_identity(result)
    for index, segment in enumerate(state.detailed_report_segments):
        if segment.phase == phase and segment.result_identity_key == key:
            return index
    candidates = [(index, segment) for index, segment in enumerate(state.detailed_report_segments)
                  if not segment.result_identity_key and segment.phase == phase
                  and segment.round_number == result.round_number]
    if len(candidates) != 1:
        return None
    index, segment = candidates[0]
    owners = {report_result_identity(r) for r in state.round_results
              if r.round_number == result.round_number and r.phase == result.phase}
    if owners - {key}:
        return None
    # A cloned report may still be owned by an ancestor. Missing legacy lineage
    # cannot be reconstructed from a matching round number alone.
    if result.inherited_from_lineage_id:
        return None
    if result.report_segment_id == segment.id:
        return index
    if result.lineage_id == "main" and not state.restore_branch_id:
        return index
    return None


def round_artifact_catalog(state):
    """Offer registered artifacts of this round, retaining their exact identity."""
    catalog = []
    visible_rounds = {state.current_round}
    if state.current_round == 1 and state.analysis_phase == "exploration":
        visible_rounds.add(0)
    for kind, records, path_field in (
        ("figure", state.publication_figures, "png_path"),
        ("table", state.publication_tables, "csv_path"),
    ):
        for record in eligible_artifacts(records, state=state):
            if record.round_number not in visible_rounds:
                continue
            catalog.append(RoundReportArtifact(
                artifact_id=record.id,
                kind=kind,
                filename=Path(getattr(record, path_field) or record.name).name,
                caption=record.description or record.interpretation or "",
            ))
    return catalog


def _chapter_plan_context(plan):
    return plan.model_dump(mode="json", include={
        "id": True, "round_number": True, "stage": True, "overview": True,
        "steps": {"__all__": {
            "id", "title", "method", "input", "expected_output",
            "rationale", "depends_on", "status",
        }},
    })


def _chapter_analysis_context(state, catalog):
    """Project recorded step and follow-up links without assigning narrative order."""
    artifact_ids = {item.artifact_id for item in catalog}
    records = [record for record in [*state.publication_figures, *state.publication_tables]
               if record.id in artifact_ids]
    tasks = {task.id: task for task in state.figure_revision_tasks}
    artifacts = []
    for record in records:
        item = record.model_dump(mode="json", include={
            "id", "plan_id", "plan_step_id", "plan_step_ids", "purpose", "report_role",
            "source_execution_id", "producer_run_kind", "source_execution_run_kind",
            "logical_figure_id", "revision_of_id", "supersedes_id",
            "figure_followup_task_id", "round_number",
        })
        task = tasks.get(getattr(record, "figure_followup_task_id", None))
        if task is not None:
            item["followup_task"] = task.model_dump(mode="json", include={
                "id", "plan_id", "plan_step_id", "reason", "pi_instruction",
                "pi_rationale", "pi_report_note", "status",
            })
        artifacts.append(item)
    execution_ids = {record.source_execution_id for record in records
                     if record.source_execution_id}
    executions = [execution.model_dump(mode="json", include={
        "id", "plan_id", "plan_step_ids", "plan_binding_status",
        "execution_purpose", "execution_status", "round_number",
    }) for execution in state.code_history if execution.id in execution_ids]
    plan = state.current_plan
    plan_context = None
    if plan is not None and plan.round_number in (0, state.current_round):
        plan_context = _chapter_plan_context(plan)
    missing_plan_ids = {record.plan_id for record in records if record.plan_id}
    if plan_context:
        missing_plan_ids.discard(plan_context["id"])
    historical_plans = []
    # The dialogue counter is not an analysis round. Match the stored plan ID
    # and the plan's own round, never step numbers or the dialogue counter.
    for entry in reversed(state.plan_dialogue_history):
        raw = entry.get("plan")
        if not isinstance(raw, dict) or raw.get("id") not in missing_plan_ids:
            continue
        try:
            historical = AnalysisPlan.model_validate(raw)
        except ValidationError:
            continue
        if historical.round_number not in (0, state.current_round):
            continue
        historical_plans.append({
            **_chapter_plan_context(historical), "context_source": "plan_dialogue_history",
        })
        missing_plan_ids.remove(historical.id)
    # Accepted foundations survive later planning dialogues. They retain only
    # completed steps, not the full plan or its dependency graph.
    for inventory in reversed(state.accepted_foundation_inventories):
        if (inventory.plan_id not in missing_plan_ids
                or inventory.round_number != state.current_round):
            continue
        historical_plans.append({
            "id": inventory.plan_id, "round_number": inventory.round_number,
            "stage": inventory.planning_stage,
            "context_source": "accepted_foundation_inventory",
            "inventory_id": inventory.inventory_id,
            "context_limitations": "Only retained completed steps are available; the full "
                                    "plan, inputs, and dependency graph are unavailable.",
            "steps": [{"id": step.plan_step_id, **step.model_dump(
                mode="json", exclude={"plan_step_id"},
            )} for step in inventory.steps],
        })
        missing_plan_ids.remove(inventory.plan_id)
    return {"current_plan": plan_context, "artifacts": artifacts,
            "historical_plans": historical_plans,
            "missing_plan_ids": sorted(missing_plan_ids),
            "source_executions": executions}


def chapter_prompt(catalog, *, state=None):
    return """
## Reusable Round Report Chapter
Return exactly one JSON object for the synthesis. Inside that same object,
include a `report_chapter` field with this value (never a separate JSON object
or code block):
{"title": "Scientific chapter title", "blocks": [
  {"text": "Report-ready Markdown explaining a finding and its limitations",
   "artifact_ids": ["exact catalog ID, in reading order"]}
], "conclusions": "This round's conclusions and remaining uncertainty"}.
Begin with this round's research question, inputs, and sample design. Organize
headed blocks in the logical order of the analysis actually performed, respecting
step dependencies: describe each step's recorded method, relevant parameters,
result, and limitations, then place its supporting artifacts immediately after
that explanation. Follow this study's actual workflow, not a fixed assay template.
Planned steps and expected outputs are not proof of execution. Use execution
records and reported findings to distinguish performed, partial, and unperformed
work; disclose missing methods or parameters instead of inventing them. Refer to
retained execution records for reproducibility without dumping code or tool logs.
Plan step IDs are local to their plan_id; never equate steps across different plans.
Match each artifact to current_plan or historical_plans by exact plan_id. Historical
contexts may be partial; respect their context_limitations. For missing_plan_ids,
state that step details are unavailable rather than borrowing another plan's steps.

Late-generated figures belong at the analysis step they support, not at the end:
- Insert supplementary validation and diagnostic plots beside the original result.
- Keep full-dataset confirmed annotation UMAP and annotation validation in the
  QC/annotation foundation section, including artifacts marked foundation_annotation,
  even when rendered during Stage 2. A later downstream subset analysis belongs
  with its own result; do not move every UMAP into the foundation section.
- Use explicit revision/supersession links to explain corrections; do not infer
  replacement from similar filenames or hide offered evidence as a duplicate.
- Place genuinely new analysis after its prerequisite steps and explain what it adds.
- Write the associated prose together with the figures: state what the supplement
  validates, qualifies, or changes, preserving unrelated findings and uncertainty.
  A figure-only follow-up is a visualization of retained results, not a new analysis.
For unbound artifacts, reason from captions and available evidence, disclose uncertain
placement, and retain them with an explanation; never fabricate a step association.
Do not create a trailing supplement merely because an artifact was generated later.
Scientific interpretation and placement remain your responsibility;
catalog order, filenames, creation times, and plot types are NOT reading order.
Use each offered artifact once. Explain auxiliary/diagnostic evidence where it
belongs. Do not invent IDs, numbers, observations, or missing analyses. Use plain
prose when no artifact supports a finding. Refer to artifacts only through
artifact_ids, never handwritten FIGURE/TABLE markers or image links. Existing
captions will be retained verbatim. Avoid generic textbook methods; include the
recorded study-specific choices needed to understand the results.
This chapter is saved at round completion and reused without rewriting in
the final report. Later integration may explicitly qualify earlier conclusions.
The host validates references only; you own scientific interpretation and order.

Artifact catalog (not ordered for presentation):
""" + json.dumps([item.model_dump() for item in catalog], ensure_ascii=False) + (
        "\n\nRecorded analysis and supplementary-figure context "
        "(plans are intentions; execution links are not scientific acceptance):\n"
        + json.dumps(_chapter_analysis_context(state, catalog), ensure_ascii=False)
        if state is not None else ""
    )


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
    # Neutralize unstructured media syntax without discarding PI prose.
    for owner, field in [(chapter, "title"), (chapter, "conclusions"),
                         *((block, "text") for block in chapter.blocks)]:
        text = getattr(owner, field)
        cleaned = re.sub(r"\[(?:FIGURE|TABLE)(?:_CAPTION)?\s*:[^\]]*\]", "", text, flags=re.I)
        cleaned = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
        cleaned = re.sub(r"<img\b[^>]*>", "", cleaned, flags=re.I)
        cleaned = cleaned.replace("![", "[")
        if cleaned != text:
            setattr(owner, field, cleaned)
            note = "Unstructured media markup was removed; PI prose and catalog references were retained."
            if note not in chapter.assembly_notes:
                chapter.assembly_notes.append(note)
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


def report_artifact_names(state, figure_map, tables, *, artifact_ids=None):
    """Bind saved IDs to the report collector's resolved render names."""
    names = {}
    for figure in eligible_artifacts(state.publication_figures, state=state):
        lineage = getattr(figure, "lineage_id", "main")
        source_id = (figure.id[:-(len(lineage) + 1)]
                     if getattr(figure, "inherited_from_lineage_id", None)
                     and figure.id.endswith(f":{lineage}") else figure.id)
        if artifact_ids is not None and not {figure.id, source_id} & artifact_ids:
            continue
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
        if artifact_ids is not None and table.id not in artifact_ids:
            continue
        name = Path(table.csv_path).name
        if any(t.get("table_id") == table.id for t in tables):
            names[table.id] = name
    return names


def saved_figure_captions(state, figure_map, tables, *, names=None):
    if names is None:
        names = report_artifact_names(state, figure_map, tables)
    return {
        names[a.artifact_id]: a.caption
        for r in state.round_results if getattr(r, "report_chapter", None)
        for a in r.report_chapter.artifacts
        if a.kind == "figure" and a.artifact_id in names
    }


def reusable_sections(state, outline, figure_map, tables, *, names=None):
    """Project saved chapters into the final outline without another LLM call."""
    if names is None:
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
            "saved_round_chapter": True,
            "artifact_references": [
                a.model_dump(exclude={"caption"})
                for r in owned for a in r.report_chapter.artifacts
            ],
            "figures": [], "tables": [],
        })
    return sections


def reuse_context(sections, *, max_chars=None):
    if not sections:
        return ""
    context = (
        "\n\n## Saved PI Round Chapters (read-only context)\n"
        "These chapters are already written and will be inserted verbatim. "
        "Do not emit, paraphrase, or reorder them or their artifacts. Write only "
        "the requested global or missing legacy sections. Integrate their findings "
        "and explain any later qualification of earlier conclusions.\n\n"
        + "\n\n".join(f"## [{s['id']}] {s['title']}\n{s['content']}" for s in sections)
    )

    if max_chars is None or len(context) <= max_chars:
        return context
    max_chars = max(0, max_chars)
    header = (
        "\n\n## Saved PI Round Chapters — Budgeted Excerpts\n"
        "Only excerpts are provided for global synthesis; missing text is not missing evidence. "
        "Do not invent omitted findings or claim a complete cross-round assessment. "
        "Complete saved chapters will be inserted verbatim; do not rewrite them.\n"
    )
    if len(header) > max_chars:
        return "[Saved chapter context omitted for budget; full chapters remain in the report.]"[:max_chars]
    # Equal space per chapter, preserving both its opening question and closing
    # conclusions. This is technical truncation, not a ranking of findings.
    per_section = max(0, (max_chars - len(header)) // len(sections))
    parts = [header]
    for section in sections:
        text = f"\n## [{section['id']}] {section['title']}\n{section['content']}"
        if len(text) > per_section:
            marker = "\n[... chapter excerpt omitted for context budget ...]\n"
            if per_section <= len(marker):
                text = text[:per_section]
            else:
                keep = per_section - len(marker)
                head = (keep + 1) // 2
                tail = keep // 2
                text = text[:head] + marker + (text[-tail:] if tail else "")
        parts.append(text)
    return "".join(parts)


def reserved_artifact_names(saved):
    """Protect saved references by kind, including references with missing bytes."""
    reserved = {"FIGURE": set(), "TABLE": set()}
    for section in saved:
        for ref in section.get("artifact_references", []):
            reserved[ref["kind"].upper()].add(ref["filename"])
        for kind, name in re.findall(r'\[(FIGURE|TABLE):\s*([^\]]+)\]', section["content"]):
            reserved[kind].add(name.strip())
    return reserved


def merge_reusable_sections(generated, saved, *, figure_names=(), table_names=()):
    """Keep model output from replacing a saved chapter or claiming its assets."""
    saved_ids = {s["id"] for s in saved}
    reserved = reserved_artifact_names(saved)
    reserved_stems = {kind: {Path(name).stem for name in names}
                      for kind, names in reserved.items()}
    known = {"FIGURE": set(figure_names), "TABLE": set(table_names)}
    def is_reserved(kind, name):
        # A real exact filename wins over extension fallback. An unknown alias
        # must still not steal a saved figure through the renderer fallback.
        return (name in reserved[kind] or (name not in known[kind]
                and Path(name).stem in reserved_stems[kind]))

    merged = [s for s in generated if s.get("id") not in saved_ids]
    for section in merged:
        section["content"] = re.sub(
            r'\[(FIGURE|TABLE):\s*([^\]]+)\]\s*(?:\[\1_CAPTION:\s*[^\]]*\])?',
            lambda m: "" if is_reserved(m.group(1).upper(), m.group(2).strip()) else m.group(0),
            section.get("content", ""), flags=re.I,
        )
    return merged + saved
