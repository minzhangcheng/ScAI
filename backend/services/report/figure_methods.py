"""Version-bound method notes; never register images or change eligibility."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.models.schema_parts.analysis import FigureMethodAnnotation
from .artifacts import image_file_sha256

logger = logging.getLogger(__name__)


def annotation_metadata(annotation):
    if annotation is None:
        return {}
    return {
        **annotation.analysis_metadata,
        "annotation_basis": annotation.basis,
        "annotation_author": annotation.author,
        "annotation_evidence": "; ".join(annotation.evidence_execution_ids),
        "annotation_notes": annotation.evidence_notes,
    }


def capture_figure_methods(execution, declarations, saved_names, workspace):
    """Bind producer declarations only to files saved by this successful cell."""
    if execution.error or not isinstance(declarations, list):
        return
    root = Path(workspace).resolve()
    for item in declarations:
        if not isinstance(item, dict):
            continue
        name = item.get("filename")
        if not isinstance(name, str) or name not in saved_names or Path(name).name != name:
            continue
        path = root / "figures" / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        details = item.get("analysis_metadata")
        if not isinstance(details, dict) or not all(isinstance(v, str) for v in details.values()):
            continue
        digest = image_file_sha256(path, min_bytes=1)
        if digest:
            execution.figure_method_annotations.append(FigureMethodAnnotation(
                relative_path=path.relative_to(root).as_posix(), render_sha256=digest,
                analysis_metadata=details, basis="execution_context", author="bioinformatics",
                evidence_execution_ids=[execution.id],
                evidence_notes="Recorded with the producing execution; methods are producer-declared, not independently verified.",
            ))


def find_figure_method_annotation(state, relative_path, digest):
    """A matching filename alone can never transfer an annotation."""
    lineage = (getattr(state, "sandbox_state", {}) or {}).get("_report_target_lineage_id") or getattr(state, "restore_branch_id", None) or "main"
    for execution in reversed(getattr(state, "code_history", []) or []):
        if execution.lineage_id != lineage:
            continue
        for note in execution.figure_method_annotations:
            if note.relative_path == relative_path and note.render_sha256 == digest:
                return note
    for record in getattr(state, "legacy_artifacts", []) or []:
        if record.lineage_id != lineage:
            continue
        note = record.method_annotation
        if note and note.relative_path == relative_path and note.render_sha256 == digest:
            return note
    return None


async def annotate_legacy_figure_methods(state, workspace, annotate):
    """Fill eligible legacy records once, using bounded historical evidence."""
    from .figure_discovery import ensure_legacy_artifact_inventory
    from .artifacts import resolve_legacy_artifact_render

    ensure_legacy_artifact_inventory(state, Path(workspace))
    pending = []
    updated = 0
    lineage = str(getattr(state, "restore_branch_id", None) or "main")
    for record in state.legacy_artifacts:
        if record.lineage_id != lineage or resolve_legacy_artifact_render(record, workspace) is None:
            continue
        existing = find_figure_method_annotation(state, record.relative_path, record.render_sha256)
        if existing:
            if record.method_annotation != existing:
                record.method_annotation = existing.model_copy(deep=True)
                updated += 1
            continue
        evidence = []
        filename = Path(record.relative_path).name
        for execution in reversed(state.code_history):
            if execution.lineage_id != lineage:
                continue
            if filename in execution.code or filename in (execution.output or ""):
                evidence.append({
                    "execution_id": execution.id, "code": execution.code[:4000],
                    "code_truncated": len(execution.code) > 4000,
                    "output": (execution.output or "")[-1000:],
                    "error": (execution.error or "")[-500:],
                })
            if len(evidence) == 3:
                break
        pending.append((record, {
            "figure_id": record.id, "relative_path": record.relative_path,
            "candidate_executions": evidence,
        }))
    for start in range(0, len(pending), 4):
        batch = pending[start:start + 4]
        try:
            response = await annotate([context for _, context in batch])
        except Exception:
            logger.warning("Legacy figure method annotation unavailable", exc_info=True)
            response = []
        answers = {item.get("figure_id"): item for item in response
                   if isinstance(item, dict) and isinstance(item.get("figure_id"), str)} if isinstance(response, list) else {}
        for record, context in batch:
            item = answers.get(record.id, {})
            details = item.get("analysis_metadata") or {}
            if not isinstance(details, dict) or not all(isinstance(v, str) for v in details.values()):
                details = {}
            offered = {e["execution_id"] for e in context["candidate_executions"]}
            refs = item.get("evidence_execution_ids", [])
            refs = [ref for ref in refs if isinstance(ref, str) and ref in offered] if isinstance(refs, list) else []
            # Historical filename matches support inference, not exact producer identity.
            basis = "inferred" if details and item.get("basis") == "inferred" else "unknown"
            notes = item.get("evidence_notes")
            record.method_annotation = FigureMethodAnnotation(
                relative_path=record.relative_path, render_sha256=record.render_sha256,
                analysis_metadata=details, basis=basis, author="pi",
                evidence_execution_ids=refs,
                evidence_notes=notes if isinstance(notes, str) else "Historical method provenance is unresolved.",
            )
            updated += 1
    return updated
