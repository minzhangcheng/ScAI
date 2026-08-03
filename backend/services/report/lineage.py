"""Lineage filtering helpers for report generation.

Legacy analysis logs and plain workspace images predate lineage provenance. Main
reports keep accepting those files for backwards compatibility, but a restored
branch must not guess that an unowned file belongs to it: the same workspace can
also contain outputs produced before or after a different restore. Branch
reports therefore admit images only when a target-lineage ``PublicationFigure``
record vouches for them.
"""

import hashlib
import os
import stat
import uuid
from pathlib import Path
from typing import Any

from backend.models.schemas import AnalysisState
from backend.services.report.artifacts import (
    artifact_stems,
    eligible_artifacts,
    has_frozen_publication_pair,
    resolve_archival_pdf,
    resolve_archival_render,
)
from backend.services.report.figure_discovery import (
    discover_figures_in_workspace,
    ensure_legacy_artifact_inventory,
)
from backend.workflow.direction_coverage import normalize_legacy_round_result


REPORT_TARGET_LINEAGE_STATE_KEY = "_report_target_lineage_id"


def _absolute(path: Any) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_workspace_regular_nofollow(
    path: Any,
    workspace: Any,
    *,
    label: str,
    require_single_link: bool = False,
) -> bytes:
    """Read one workspace file through no-follow directory and file FDs."""
    workspace_path = _absolute(workspace)
    candidate = _absolute(path)
    try:
        relative = candidate.relative_to(workspace_path)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside the session workspace") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{label} has an invalid workspace path")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        current_fd = os.open(workspace_path, directory_flags)
        descriptors.append(current_fd)
        for component in relative.parts[:-1]:
            current_fd = os.open(
                component,
                directory_flags,
                dir_fd=current_fd,
            )
            descriptors.append(current_fd)
        try:
            file_descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current_fd,
            )
        except OSError as exc:
            raise RuntimeError(f"{label} is missing, invalid, or a symlink") from exc
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        if require_single_link and before.st_nlink != 1:
            raise RuntimeError(f"{label} has an external hard-link alias")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 65_536)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(file_descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if before_identity != after_identity or len(payload) != before.st_size:
            raise RuntimeError(f"{label} changed while it was read")
        return payload
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _write_content_addressed_file(
    destination: Path,
    payload: bytes,
    expected_sha256: str,
) -> None:
    """Install immutable bytes without replacing an existing archive object."""
    destination = _absolute(destination)
    pair_dir = destination.parent
    lineage_root = pair_dir.parent
    figures_dir = lineage_root.parent
    workspace = figures_dir.parent
    if (
        lineage_root.name != ".lineage"
        or figures_dir.name != "figures"
        or len(pair_dir.name) != 64
        or any(char not in "0123456789abcdef" for char in pair_dir.name)
        or destination.name not in {"render.jpg", "render.png", "render.pdf"}
    ):
        raise RuntimeError("Lineage archive destination is outside its trusted layout")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("Lineage archive payload does not match its frozen digest")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    temporary_name = f".lineage-copy-{uuid.uuid4().hex}"
    try:
        workspace_fd = os.open(workspace, directory_flags)
        descriptors.append(workspace_fd)
        figures_fd = os.open("figures", directory_flags, dir_fd=workspace_fd)
        descriptors.append(figures_fd)
        try:
            os.mkdir(".lineage", mode=0o700, dir_fd=figures_fd)
        except FileExistsError:
            pass
        lineage_fd = os.open(".lineage", directory_flags, dir_fd=figures_fd)
        descriptors.append(lineage_fd)
        try:
            os.mkdir(pair_dir.name, mode=0o700, dir_fd=lineage_fd)
        except FileExistsError:
            pass
        pair_fd = os.open(pair_dir.name, directory_flags, dir_fd=lineage_fd)
        descriptors.append(pair_fd)

        def verify_existing() -> bool:
            try:
                existing_fd = os.open(
                    destination.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=pair_fd,
                )
            except FileNotFoundError:
                return False
            except OSError as exc:
                raise RuntimeError(
                    f"Lineage archive collision or symlink at {destination}"
                ) from exc
            try:
                metadata = os.fstat(existing_fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise RuntimeError(
                        f"Lineage archive collision at {destination}"
                    )
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(existing_fd, 65_536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                existing = b"".join(chunks)
                after = os.fstat(existing_fd)
                if (
                    after.st_dev != metadata.st_dev
                    or after.st_ino != metadata.st_ino
                    or after.st_size != metadata.st_size
                    or len(existing) != metadata.st_size
                    or hashlib.sha256(existing).hexdigest() != expected_sha256
                ):
                    raise RuntimeError(
                        f"Lineage archive collision at {destination}"
                    )
                return True
            finally:
                os.close(existing_fd)

        if verify_existing():
            return

        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=pair_fd,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(temporary_fd, view)
                if written <= 0:
                    raise OSError("short lineage archive write")
                view = view[written:]
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=pair_fd,
                dst_dir_fd=pair_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if not verify_existing():
                raise RuntimeError(f"Lineage archive collision at {destination}")
        os.unlink(temporary_name, dir_fd=pair_fd)
        os.fsync(pair_fd)
        if not verify_existing():
            raise RuntimeError(f"Lineage archive write failed at {destination}")
    finally:
        try:
            if descriptors:
                os.unlink(temporary_name, dir_fd=descriptors[-1])
        except OSError:
            pass
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _archive_inherited_figure(figure: Any, workspace_path: Any) -> dict[str, Any]:
    """Move a source head onto an immutable, content-addressed pair.

    A restored branch can render the same logical filename as its parent. Both
    records therefore have to stop pointing at ``figures/<name>`` before the
    branch is activated; hashes alone cannot recover the parent's full PDF once
    those mutable files are overwritten.
    """
    workspace = _absolute(workspace_path)
    archive_root = workspace / "figures" / ".lineage"
    if archive_root.exists() and archive_root.is_symlink():
        raise RuntimeError("Lineage archive root is a symlink")

    render_path = resolve_archival_render(figure, workspace)
    if render_path is None:
        raise RuntimeError("Inherited figure has no verifiable JPG/PDF pair")
    expected_render_digest = str(getattr(figure, "render_sha256", None) or "")
    if not expected_render_digest:
        raise RuntimeError("Inherited figure pair has no frozen JPG digest")
    render_payload = _read_workspace_regular_nofollow(
        render_path,
        workspace,
        label="Inherited figure render",
    )
    render_digest = hashlib.sha256(render_payload).hexdigest()
    if render_digest != expected_render_digest:
        raise RuntimeError("Inherited figure JPG does not match its frozen pair")

    pdf_path = resolve_archival_pdf(figure, workspace)
    expected_pdf_digest = str(getattr(figure, "pdf_sha256", None) or "")
    if pdf_path is None or not expected_pdf_digest:
        raise RuntimeError("Inherited figure requires a complete JPG/PDF pair")
    pdf_payload = _read_workspace_regular_nofollow(
        pdf_path,
        workspace,
        label="Inherited figure PDF",
    )
    pdf_digest = hashlib.sha256(pdf_payload).hexdigest()
    if pdf_digest != expected_pdf_digest:
        raise RuntimeError("Inherited figure PDF does not match its frozen pair")

    pair_digest = hashlib.sha256(
        f"{render_digest}:{pdf_digest}".encode("ascii")
    ).hexdigest()
    pair_dir = archive_root / pair_digest
    render_suffix = ".png" if render_payload.startswith(b"\x89PNG") else ".jpg"
    archived_render = pair_dir / f"render{render_suffix}"
    _write_content_addressed_file(
        archived_render, render_payload, render_digest
    )
    archived_render_payload = _read_workspace_regular_nofollow(
        archived_render,
        workspace,
        label="Archived lineage render",
        require_single_link=True,
    )
    if hashlib.sha256(archived_render_payload).hexdigest() != render_digest:
        raise RuntimeError("Archived lineage render failed record validation")

    archived_pdf = pair_dir / "render.pdf"
    _write_content_addressed_file(archived_pdf, pdf_payload, pdf_digest)
    archived_pdf_payload = _read_workspace_regular_nofollow(
        archived_pdf,
        workspace,
        label="Archived lineage PDF",
        require_single_link=True,
    )
    if hashlib.sha256(archived_pdf_payload).hexdigest() != pdf_digest:
        raise RuntimeError("Archived lineage PDF failed validation")

    return {
        "png_path": str(archived_render),
        "pdf_path": str(archived_pdf),
        "render_sha256": render_digest,
        "pdf_sha256": pdf_digest,
    }


def clone_publication_figures_for_branch(
    figures,
    *,
    branch_id: str,
    source_lineage_id: str = "main",
    workspace_path: Any = None,
):
    """Preserve source records and add branch-local copies of inherited figures."""
    target = branch_id or "main"
    source = source_lineage_id or "main"
    output = list(figures or [])
    if target == source:
        return output
    source_heads = [
        figure for figure in eligible_artifacts(output)
        if (getattr(figure, "lineage_id", None) or "main") == source
        and has_frozen_publication_pair(figure)
    ]
    if source_heads and workspace_path is None:
        raise ValueError(
            "Branch figure inheritance requires a workspace-bound JPG/PDF pair"
        )
    existing_by_id: dict[str, Any] = {}
    for figure in output:
        record_id = str(getattr(figure, "id", "") or "")
        if not record_id:
            continue
        if record_id in existing_by_id and existing_by_id[record_id] is not figure:
            raise RuntimeError(f"Publication figure ID collision for {record_id}")
        existing_by_id[record_id] = figure

    # Validate every target identity before writing archive objects or mutating
    # either lineage. An existing clone ID is reusable only when it denotes the
    # same source record, target lineage, logical family, and immutable pair.
    existing_clones: dict[str, Any] = {}
    clone_ids = {f"{original.id}:{target}" for original in source_heads}
    if len(clone_ids) != len(source_heads):
        raise RuntimeError("Source publication figure identity collision")
    for original in source_heads:
        clone_id = f"{original.id}:{target}"
        existing_clone = existing_by_id.get(clone_id)
        if existing_clone is None:
            logical_id = str(
                getattr(original, "logical_figure_id", None) or ""
            )
            if logical_id and any(
                (getattr(candidate, "lineage_id", None) or "main") == target
                and str(
                    getattr(candidate, "logical_figure_id", None) or ""
                ) == logical_id
                for candidate in output
            ):
                raise RuntimeError(
                    f"Branch logical figure identity collision for {logical_id}"
                )
            continue
        expected_identity = {
            "lineage_id": target,
            "logical_figure_id": str(
                getattr(original, "logical_figure_id", None) or ""
            ),
            "name": str(getattr(original, "name", "") or ""),
            "figure_type": str(getattr(original, "figure_type", "") or ""),
            "render_sha256": str(
                getattr(original, "render_sha256", None) or ""
            ),
            "pdf_sha256": str(getattr(original, "pdf_sha256", None) or ""),
        }
        actual_identity = {
            key: str(getattr(existing_clone, key, None) or "")
            for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"Branch clone identity collision for {clone_id}"
            )
        existing_clones[clone_id] = existing_clone

    # Archive every source first. The helper writes immutable objects but does
    # not mutate records, so a collision or I/O failure for a later source
    # leaves every source and target record unchanged.
    archive_updates: dict[str, dict[str, Any]] = {}
    for original in source_heads:
        archive_updates[str(original.id)] = _archive_inherited_figure(
            original, workspace_path
        )

    all_ids = set(existing_by_id) | clone_ids
    planned_new_clones: list[tuple[Any, Any]] = []
    planned_existing_updates: list[tuple[Any, dict[str, Any]]] = []
    for original in source_heads:
        clone_id = f"{original.id}:{target}"
        update = archive_updates.get(str(original.id), {})
        clone_source = original.model_copy(deep=True)
        for field, value in update.items():
            setattr(clone_source, field, value)
        existing_clone = existing_clones.get(clone_id)
        if existing_clone is not None:
            planned_existing_updates.append((existing_clone, update))
            continue
        clone = clone_source.model_copy(deep=True)
        clone.lineage_id = target
        # IDs are API/download identities. Keep inherited records addressable in
        # both lineages without making two records share one route key.
        clone.id = clone_id
        for field in ("revision_of_id", "supersedes_id"):
            source_id = str(getattr(original, field, None) or "")
            inherited_id = f"{source_id}:{target}" if source_id else ""
            setattr(
                clone,
                field,
                inherited_id if inherited_id in all_ids else None,
            )
        clone.is_published_head = True
        planned_new_clones.append((original, clone))

    # All fallible validation, archive I/O, and model construction completed.
    # Apply the prepared record mutations as one final in-memory boundary.
    for original in source_heads:
        for field, value in archive_updates.get(str(original.id), {}).items():
            setattr(original, field, value)
    for existing_clone, update in planned_existing_updates:
        for field, value in update.items():
            setattr(existing_clone, field, value)
    for original, clone in planned_new_clones:
        logical_id = str(getattr(original, "logical_figure_id", None) or "")
        if logical_id:
            for existing in output:
                if (
                    (getattr(existing, "lineage_id", None) or "main") == target
                    and str(getattr(existing, "logical_figure_id", None) or "")
                    == logical_id
                ):
                    existing.is_published_head = False
        output.append(clone)
    return output


def _restrict_unowned_branch_figures(
    report_state: AnalysisState,
    workspace_path: Any,
) -> None:
    """Exclude legacy image files that cannot be assigned to this branch.

    The exclusions live only on the deep-copied report state. They reuse the
    existing stale-stem reconciliation path, so a current target-lineage record
    still wins even when an old flat invalidation set contains the same stem.
    """
    workspace = Path(workspace_path) if workspace_path else None
    owned_stems = artifact_stems(report_state.publication_figures)
    unowned_stems = {
        Path(str(name)).stem
        for step in (report_state.analysis_log or [])
        for name in (step.saved_figure_filenames or [])
        if Path(str(name)).stem not in owned_stems
    }

    # A legacy log entry proves only that a filename was once saved. Retain a
    # name on a non-main branch only when a lineage-filtered artifact record owns
    # that stem; collectors may still recover the record's verified preview.
    for step in report_state.analysis_log or []:
        step.saved_figure_filenames = [
            name
            for name in (step.saved_figure_filenames or [])
            if Path(str(name)).stem in owned_stems
        ]

    if workspace is not None and workspace.exists():
        unowned_stems.update(
            str(candidate.get("name") or "")
            for candidate in discover_figures_in_workspace(workspace)
            if candidate.get("name")
            and str(candidate.get("name")) not in owned_stems
        )
    report_state.invalidated_artifact_stems = sorted(
        set(report_state.invalidated_artifact_stems or []) | unowned_stems
    )


def build_lineage_report_state(
    state: AnalysisState,
    target_lineage_id: str,
    *,
    workspace_path: Any = None,
) -> AnalysisState:
    """Return a report snapshot limited to one analysis lineage."""
    target = target_lineage_id or "main"
    if workspace_path:
        ensure_legacy_artifact_inventory(state, Path(workspace_path))
    report_state = state.model_copy(deep=True)
    sandbox_state = dict(report_state.sandbox_state or {})
    sandbox_state[REPORT_TARGET_LINEAGE_STATE_KEY] = target
    report_state.sandbox_state = sandbox_state
    normalized = [normalize_legacy_round_result(r) for r in report_state.round_results]
    report_state.round_results = [r for r in normalized if r.lineage_id == target]
    report_state.publication_figures = [
        figure
        for figure in report_state.publication_figures
        if (getattr(figure, "lineage_id", None) or "main") == target
    ]
    if target != "main":
        _restrict_unowned_branch_figures(report_state, workspace_path)
    return report_state
