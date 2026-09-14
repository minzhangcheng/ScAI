# Archive member builders for state, snapshots, blobs, and reproducibility.
# Moved verbatim out of backend/routes/export_import.py.
import asyncio
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from backend.config import (
    DATA_BASE_DIR, REPORTS_DIR, SESSION_DIR, STATE_DIR, UPLOAD_DIR, WORKSPACE_DIR,
    UPLOAD_CHUNK_SIZE, UPLOAD_TEMP_DIR,
    get_session_dir, get_session_logs, get_session_metadata_path, get_session_reports,
    get_session_states, get_session_uploads, get_session_workspace, resolve_session_metadata_path,
)
from backend.models.schemas import AnalysisState, APIResponse
from backend.services.reproducibility import REPRODUCIBILITY_DIR_NAME
from backend.services.report.output_generation import (
    CURRENT_GENERATION_POINTER,
    GENERATION_MODE_SENTINEL,
    GENERATIONS_DIR_NAME,
    report_publication_lock,
    resolve_current_report_dir,
)
from backend.services.session_store import (
    async_save_session, ensure_session_loaded,
    update_session_cache as _update_session_cache,
)
from backend.state import (
    import_jobs as _import_jobs,
    import_jobs_lock as _import_jobs_lock,
    prepared_exports as _prepared_exports,
    prepared_exports_lock as _prepared_exports_lock,
)
from backend.utils.auth import require_session_ownership
from backend.utils.blob_store import inline_blobs
from backend.utils.checkpoint import create_checkpoint_export_snapshot
from backend.utils.files import (
    legacy_session_state_paths,
    slugify_session_name,
    temp_file_on_error,
    validate_path_within,
    validate_session_id,
)
from backend.utils.secret_redaction import ENDPOINT_FIELD_NAMES, strip_analysis_state_secrets
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from .core import is_runtime_generation_name, is_runtime_h5ad_name

logger = logging.getLogger(__name__)


class _PaddedReader:
    """Serve exactly *size* bytes from *fh*, zero-padding a file that shrank.

    ``tarfile`` writes a member header announcing the size it measured and then
    copies that many bytes. A session being exported while its analysis is still
    running has files rewritten underneath the packer: a file that shrinks makes
    ``copyfileobj`` raise mid-member, which aborts the whole export *and* leaves
    a structurally broken archive behind; a file that grows is silently cut at
    the announced size. Padding keeps the archive readable no matter which
    happens, and the caller logs the affected path instead of failing the export.
    """

    def __init__(self, fh, size: int):
        self._fh = fh
        self._remaining = size
        self.padded = False

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        want = self._remaining if size is None or size < 0 else min(size, self._remaining)
        chunk = self._fh.read(want)
        if len(chunk) < want:
            chunk += b"\0" * (want - len(chunk))
            self.padded = True
        self._remaining -= len(chunk)
        return chunk


def _safe_add_file(tf, path: Path, arcname: str) -> bool:
    """Add one regular file to *tf*, tolerating files that vanish or change.

    The member is built from an already-open descriptor rather than from the
    path, which is what turns symlinks and hardlinks into ordinary members: the
    import side rejects every SYMTYPE/LNKTYPE entry outright (see
    ``_import_tar_gz``), so a single ``os.symlink`` left in a workspace by
    analysis code used to make the resulting archive permanently un-importable.

    Returns True when a member was written.
    """
    try:
        fh = open(path, "rb")
    except OSError as exc:
        logger.warning("Export: skipping %s (%s)", path, exc)
        return False
    try:
        info = tf.gettarinfo(arcname=arcname, fileobj=fh)
        info.name = arcname
        # gettarinfo copies the descriptor's type bits; force a regular file so
        # a device or fifo that slipped through cannot become a tar special.
        info.type = tarfile.REGTYPE
        reader = _PaddedReader(fh, info.size)
        tf.addfile(info, reader)
        if reader.padded:
            logger.warning(
                "Export: %s shrank while it was being packed; the archived copy "
                "was zero-padded to the announced size.",
                path,
            )
        return True
    except OSError as exc:
        logger.warning("Export: skipping %s (%s)", path, exc)
        return False
    finally:
        fh.close()


def _contained_files(root: Path):
    """Yield every regular file under *root* that does not escape it.

    ``_safe_add_file`` builds the member from an open descriptor, so a symlink is
    packed with its *target's* bytes rather than as a link. A link that escapes
    the tree would put out-of-session content in an archive whose scope is a
    single session, so it is dropped here.
    """
    if not root.exists():
        return
    try:
        entries = sorted(root.rglob("*"))
    except OSError as exc:
        logger.warning("Export: cannot walk %s (%s)", root, exc)
        return
    try:
        root_resolved = root.resolve()
    except OSError as exc:
        logger.warning("Export: cannot resolve %s (%s)", root, exc)
        return
    for file_path in entries:
        try:
            if not file_path.is_file():
                continue
            if (file_path.is_symlink()
                    and not file_path.resolve().is_relative_to(root_resolved)):
                logger.warning(
                    "Export: skipping %s (symlink escapes %s)", file_path, root
                )
                continue
        except OSError:
            continue
        yield file_path


def _add_tree_to_tar(tf, root: Path, arc_prefix: str) -> None:
    """Add every regular file under *root* as ``<arc_prefix>/<relative path>``."""
    for file_path in _contained_files(root):
        rel = file_path.relative_to(root)
        _safe_add_file(tf, file_path, f"{arc_prefix}/{rel.as_posix()}")


def _add_report_output_to_tar(
    tf,
    output_root: Path,
    arc_prefix: str,
    *,
    include_root_variants: bool,
) -> None:
    """Capture one lineage while holding the same lock as pointer publication."""
    with report_publication_lock(output_root):
        published = resolve_current_report_dir(output_root)
        generation_mode = published.parent.name == GENERATIONS_DIR_NAME

        if generation_mode:
            pointer = output_root / CURRENT_GENERATION_POINTER
            _safe_add_file(tf, pointer, f"{arc_prefix}/{CURRENT_GENERATION_POINTER}")
            sentinel = output_root / GENERATION_MODE_SENTINEL
            if sentinel.is_file() and not sentinel.is_symlink():
                _safe_add_file(tf, sentinel, f"{arc_prefix}/{GENERATION_MODE_SENTINEL}")
            _add_tree_to_tar(
                tf,
                published,
                f"{arc_prefix}/{GENERATIONS_DIR_NAME}/{published.name}",
            )
        else:
            for file_path in _contained_files(output_root):
                relative = file_path.relative_to(output_root)
                if relative.parts and relative.parts[0] == "branches":
                    continue
                if any(part.startswith(".report-") for part in relative.parts):
                    continue
                _safe_add_file(tf, file_path, f"{arc_prefix}/{relative.as_posix()}")

        if not include_root_variants or not generation_mode:
            return
        # Variant reports are still legacy root-owned files. Detailed-report
        # JSON/HTML/assets come exclusively from the selected generation.
        for file_path in _contained_files(output_root):
            relative = file_path.relative_to(output_root)
            if not relative.parts:
                continue
            if relative.parts[0] in {
                "branches",
                "figures",
                "tables",
                GENERATIONS_DIR_NAME,
            }:
                continue
            if any(part.startswith(".report-") for part in relative.parts):
                continue
            if relative.name in {"report.json", "report.html"}:
                continue
            _safe_add_file(tf, file_path, f"{arc_prefix}/{relative.as_posix()}")


def _add_reports_to_tar(tf, reports_root: Path, arc_prefix: str) -> None:
    """Export current main/branch generations without stale or staging trees."""
    _add_report_output_to_tar(
        tf,
        reports_root,
        arc_prefix,
        include_root_variants=True,
    )
    branches = reports_root / "branches"
    if not branches.is_dir() or branches.is_symlink():
        return
    for branch_root in sorted(branches.iterdir()):
        if branch_root.is_symlink() or not branch_root.is_dir():
            continue
        _add_report_output_to_tar(
            tf,
            branch_root,
            f"{arc_prefix}/branches/{branch_root.name}",
            include_root_variants=False,
        )


def _add_json_bytes(tf, arcname: str, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = 0o600
    tf.addfile(info, io.BytesIO(data))

def _add_portable_state_json(tf, source: Path, arcname: str) -> None:
    """Add state JSON without installation-bound or plaintext credentials."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    _add_json_bytes(tf, arcname, strip_analysis_state_secrets(payload))


def _add_adata_checkpoint_blobs_to_tar(
    tf, session_id: str, descriptor: dict | None = None,
) -> None:
    """Add Host-owned immutable checkpoint blobs to a resumable archive.

    Results-only export callers must not call this helper; their import path
    marks omitted blob records unavailable with reason ``not_materialized``.
    """
    if descriptor is None:
        blobs_dir = get_session_dir(session_id) / "adata-checkpoints" / "blobs"
        if blobs_dir.is_dir():
            _add_tree_to_tar(
                tf,
                blobs_dir,
                f"{session_id}/adata-checkpoints/blobs",
            )
        return
    declared = dict(descriptor.get("checkpoint_blobs") or {})
    for relative in sorted(declared):
        source = get_session_dir(session_id) / relative
        added = _safe_add_file(tf, source, f"{session_id}/{relative}")
        if not added:
            raise OSError(
                f"failed to add declared AnnData checkpoint blob {relative}"
            )


def _add_states_to_tar(
    tf,
    session_id: str,
    include_runtime: bool = False,
    descriptor: dict | None = None,
):
    """Add files under sessions/<id>/states/ to the tar.

    Skips the live sandbox pickle and LangGraph checkpoint unless
    ``include_runtime`` is true; notebooks remain portable and are always
    included.

    The checkpoint travels with the pkl rather than on its own because the two
    are only useful together: the checkpoint says which graph node to resume at,
    and the pkl holds the kernel values that node expects to find. Importing one
    without the other resumes into an empty kernel, which
    ``_mark_runtime_availability_after_import`` deliberately blocks.

    (Credentials sealed inside the checkpoint are no longer a reason to withhold
    it — :func:`backend.utils.secret_redaction.open_secret` degrades a value it
    cannot decrypt to empty instead of failing the load.)
    """
    declared_executors = set(
        ((descriptor or {}).get("executor_checkpoints") or {}).keys()
    )

    def _add(state_file: Path):
        # The dump, its manifest and that manifest's signature are one
        # generation; a portable archive drops all three together.
        if not include_runtime and is_runtime_generation_name(state_file.name):
            return
        from ...agents.bioinformatics.executor_checkpoint import (
            is_executor_checkpoint_name,
        )

        if is_executor_checkpoint_name(
            state_file.name, session_id=session_id,
        ):
            if (
                not include_runtime
                or (
                    descriptor is not None
                    and state_file.name not in declared_executors
                )
            ):
                return
        # A live SQLite database is never archived by path. Runtime exports
        # add exactly one verified standalone backup below, and portable
        # exports omit the database plus its WAL/SHM companions entirely.
        if state_file.name.startswith("checkpoints.sqlite"):
            return
        # The h5ad sidecar set is runtime too: a Snapshot V2 restore writes
        # `runtime_h5ad_manifest.json` plus the `<name>.h5ad` files it lists so a
        # kernel can come back holding `adata` when the dill residual cannot
        # carry it. Matching neither filter above, the pair used to travel in the
        # portable export that also declares the runtime absent — shipping a
        # multi-GB AnnData the importer then reports as gone.
        #
        # The two halves must be dropped together or not at all, so the rule is
        # shared with the import side rather than restated here.
        if not include_runtime and is_runtime_h5ad_name(state_file.name):
            return
        _safe_add_file(tf, state_file, f"{session_id}/states/{state_file.name}")

    states_dir = get_session_states(session_id)
    checkpoint_source = states_dir / "checkpoints.sqlite"
    checkpoint_snapshot: Path | None = None
    try:
        if include_runtime and (
            checkpoint_source.exists() or checkpoint_source.is_symlink()
        ):
            UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=".checkpoint-export-",
                suffix=".sqlite",
                dir=str(UPLOAD_TEMP_DIR),
            )
            os.close(fd)
            checkpoint_snapshot = Path(temporary_name)
            checkpoint_snapshot.unlink()
            create_checkpoint_export_snapshot(
                checkpoint_source,
                checkpoint_snapshot,
            )
            added = _safe_add_file(
                tf,
                checkpoint_snapshot,
                f"{session_id}/states/checkpoints.sqlite",
            )
            if not added:
                raise OSError(
                    f"failed to add verified checkpoint snapshot for {session_id}"
                )

        if states_dir.exists():
            for sf in sorted(states_dir.iterdir()):
                if sf.is_file() or sf.is_symlink():
                    _add(sf)
        # Legacy fallback (flat STATE_DIR layout)
        for sf in legacy_session_state_paths(STATE_DIR, session_id):
            if sf.is_file() or sf.is_symlink():
                _add(sf)
    finally:
        if checkpoint_snapshot is not None:
            try:
                checkpoint_snapshot.unlink()
            except FileNotFoundError:
                pass

def _add_session_blobs_to_tar(tf, session_id: str):
    """Add the session-level externalized blob store to tar archive.

    The blob_store offloads any string >= 1024 chars from session.json into
    <session_dir>/blobs/<sha256>.b64 sidecars. Without these files the
    imported session.json contains dangling "@scai-blob:<hash>" tokens that
    surface as raw text in the UI (e.g. inside the messages history).
    """
    blob_dir = get_session_dir(session_id) / "blobs"
    if not blob_dir.exists():
        return
    for blob_file in sorted(blob_dir.iterdir()):
        if blob_file.is_file():
            _safe_add_file(tf, blob_file, f"{session_id}/blobs/{blob_file.name}")

def _add_session_logs_to_tar(tf, session_id: str):
    """Add sessions/<id>/logs/ — the only per-call record of LLM usage.

    ``get_agent_llm_stats`` reads these files to fill the Agent Configuration
    chapter of the detailed report. A session migrated without them regenerates
    that report claiming every agent made zero calls and spent zero tokens,
    silently under-reporting the work it is describing, and nothing in
    ``AnalysisState`` can reconstruct the numbers.

    JSONL records are rewritten on the way in so the endpoint fields carry no
    value. ``session.json`` is already stripped of them by
    ``strip_analysis_state_secrets``; a log that keeps them defeats that for the
    same reason -- it points the destination's operator at a host only the
    source could reach, next to an api_key the archive deliberately dropped.
    """
    logs_dir = get_session_logs(session_id)
    # ``_contained_files`` drops links out of the tree; for a .jsonl such a link
    # would additionally be parsed and re-emitted as this session's log.
    for log_file in _contained_files(logs_dir):
        arcname = f"{session_id}/logs/{log_file.relative_to(logs_dir).as_posix()}"
        if log_file.suffix.casefold() != ".jsonl":
            _safe_add_file(tf, log_file, arcname)
            continue
        buf = io.BytesIO()
        try:
            mtime = log_file.stat().st_mtime
            with open(log_file, "rb") as fh:
                for raw in fh:
                    try:
                        record = json.loads(raw)
                    except ValueError:
                        # A torn final line from a concurrent write is still the
                        # user's data; pass it through rather than dropping it,
                        # so a truncated file never exports as an empty one.
                        buf.write(raw if raw.endswith(b"\n") else raw + b"\n")
                        continue
                    if isinstance(record, dict):
                        for key in ENDPOINT_FIELD_NAMES:
                            if key in record:
                                record[key] = ""
                    buf.write(
                        json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
                    )
        except OSError as exc:
            logger.warning("Export: skipping log %s (%s)", log_file, exc)
            continue
        info = tarfile.TarInfo(name=arcname)
        info.size = buf.tell()
        info.mode = 0o600
        info.mtime = int(mtime)
        buf.seek(0)
        tf.addfile(info, buf)

def _add_reproducibility_to_tar(tf, session_id: str):
    """Add reproducibility bundle files to tar archive."""
    ws_dir = get_session_workspace(session_id)
    if not ws_dir.exists():
        ws_dir = WORKSPACE_DIR / session_id
    repro_dir = ws_dir / REPRODUCIBILITY_DIR_NAME
    if repro_dir.exists():
        for file_path in sorted(repro_dir.iterdir()):
            if file_path.is_file():
                _safe_add_file(
                    tf,
                    file_path,
                    f"{session_id}/{REPRODUCIBILITY_DIR_NAME}/{file_path.name}",
                )
