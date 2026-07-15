"""Snapshot manager for V1 file-backed snapshots.

After the Batch 4 V1 restore cutover, HTTP restore always returns 410.
``SnapshotManager`` retains compatibility readers/deleters used by V1
list/preview/delete and by unit tests of the on-disk V1 format. Authoritative
restorable lifecycle is Snapshot V2 (``backend.services.snapshot_v2``).
"""

import asyncio
import fnmatch
import json
import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..models.schemas import AnalysisState, Snapshot
from ..config import (
    MAX_SESSION_SNAPSHOT_BYTES,
    MAX_SNAPSHOT_KERNEL_PKL_BYTES,
    SNAPSHOT_EXCLUDED_WORKSPACE_PATTERNS,
    get_session_dir,
    get_session_states,
)
from ..utils.blob_store import externalize_blobs, inline_blobs
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def should_include_runtime_state(size_bytes: int, max_bytes: int = MAX_SNAPSHOT_KERNEL_PKL_BYTES) -> tuple[bool, str]:
    if size_bytes > max_bytes:
        return False, f"runtime state size {size_bytes} exceeds limit {max_bytes}"
    return True, ""


def should_exclude_snapshot_file(filename: str, patterns=SNAPSHOT_EXCLUDED_WORKSPACE_PATTERNS) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


_DIRECTION_LIST_FIELDS = ("user_chosen_directions", "directions_queue", "active_direction_batch")


def _coerce_is_custom(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def normalize_direction_dicts(state_dict: dict) -> dict:
    """Coerce known scalar keys inside direction dicts to their intended types.

    Runs before model_validate. Fixes semantic drift (e.g. is_custom stored as the
    truthy string "false") that type-widening alone leaves in place. Operates per-key
    inside each list item; never drops or empties a direction list.
    """
    for field in _DIRECTION_LIST_FIELDS:
        items = state_dict.get(field)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if "is_custom" in item:
                item["is_custom"] = _coerce_is_custom(item["is_custom"])
            if item.get("candidate_index") is not None:
                try:
                    item["candidate_index"] = int(item["candidate_index"])
                except (TypeError, ValueError):
                    logger.warning(
                        "candidate_index %r could not be coerced to int; setting None",
                        item["candidate_index"],
                    )
                    item["candidate_index"] = None
    return state_dict


class SnapshotRestoreError(Exception):
    """Raised when a snapshot's state cannot be safely restored into an AnalysisState."""


# Positive allowlist: ONLY these fields may be blanked to their model default when they
# fail validation. They are regenerable, non-identity scalars whose loss does not corrupt
# the workflow. Anything not listed here fails closed — that includes identity fields,
# direction state, and every collection field (round_results, annotation records, ...),
# so a drifted collection is never silently wiped (spec §2b). Keep this set minimal.
RECOVERABLE_RESTORE_DEFAULT_FIELDS = frozenset({
    "integration_cycle",
})


def _validate_state_with_fallback(state_dict: dict, session_id: str) -> "AnalysisState":
    """Validate state_dict; blank only allowlisted drifted fields, else fail closed."""
    try:
        return AnalysisState.model_validate(state_dict)
    except ValidationError as exc:
        # loc[0] is the top-level field name even for nested errors
        # (e.g. round_results.0.round_number -> "round_results").
        offending = {err["loc"][0] for err in exc.errors() if err.get("loc")}
        unrecoverable = offending - RECOVERABLE_RESTORE_DEFAULT_FIELDS
        if unrecoverable:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            raise SnapshotRestoreError(
                f"non-recoverable field(s) {sorted(unrecoverable)} failed validation "
                f"at {loc}: {first.get('msg', 'validation error')}"
            ) from exc
        for field in offending:
            logger.warning(
                "[%s] Snapshot restore: degrading drifted field %r to its default",
                session_id, field,
            )
            state_dict.pop(field, None)
        try:
            return AnalysisState.model_validate(state_dict)
        except ValidationError as exc2:
            first = exc2.errors()[0] if exc2.errors() else {}
            raise SnapshotRestoreError(
                f"unrecoverable snapshot state: {first.get('msg', 'validation error')}"
            ) from exc2


class SnapshotManager:
    """Manages lightweight analysis state snapshots for debugging rollback."""

    @staticmethod
    def compute_snapshot_context(state: "AnalysisState", phase: str) -> dict:
        """Compute parent_id, branch_label, and name for the next auto-snapshot.

        Shared by both workflow and chat snapshot creation to avoid logic drift.

        Valid phase values:
            "pre_annotation"       - Before annotation HITL
            "pre_direction"        - Before direction selection HITL
            "pre_round_wait"       - Before round-end wait
            "pre_execution_stage1" - Before executing stage 1 plan
            "pre_execution_stage2" - Before executing stage 2 plan
            "pre_execution_full"   - Before executing full round plan
            "pre_major_revision"   - Before major plan revision dialogue
            "post_report"          - After final report
        """
        # P2-4: do NOT use snapshots[-1] as fallback — it picks by time, not by
        # branch, which mis-parents across branches. Instead chain onto the lineage
        # head (last_snapshot_id), which advances on every create and points at the
        # restore target after a restore. A fresh restore (restore provenance)
        # takes precedence so the new snapshot anchors on the restored node and forms
        # a diverging branch.
        from backend.workflow.provenance import restore_provenance_id

        restored_from = restore_provenance_id(state)
        parent_id = restored_from or state.last_snapshot_id

        if restored_from:
            # Diverging off a restore point: allocate a new branch label.
            existing_branches = sum(
                1 for s in state.snapshots
                if s.parent_id == restored_from
                and s.branch_label and s.branch_label != "main"
            )
            branch_letter = chr(ord('A') + existing_branches) if existing_branches < 26 else str(existing_branches)
            branch_label = f"branch-{branch_letter}"
        elif parent_id:
            # Forward continuation: stay on the parent's branch.
            parent = next((s for s in state.snapshots if s.id == parent_id), None)
            branch_label = parent.branch_label if parent and parent.branch_label else "main"
        else:
            branch_label = "main"

        name = f"Auto: {phase} (Round {state.current_round})"
        if branch_label != "main":
            name += f" [{branch_label}]"

        return {"parent_id": parent_id, "branch_label": branch_label, "name": name}

    @staticmethod
    async def create_and_register_snapshot(
        session_id: str,
        state: "AnalysisState",
        phase: str,
        *,
        notify_fn,
    ) -> "Snapshot":
        """Create a snapshot, append to state, clear restore marker, and notify.

        Shared by workflow and chat to prevent logic drift.
        notify_fn(session_id, "snapshot_count", {"count": len(state.snapshots)})
        """
        mgr = SnapshotManager(session_id)
        ctx = SnapshotManager.compute_snapshot_context(state, phase)
        snapshot = await mgr.create_snapshot(
            state, ctx["name"], phase,
            parent_id=ctx["parent_id"],
            branch_label=ctx["branch_label"],
        )
        state.snapshots.append(snapshot)
        # Advance the lineage head so the next forward snapshot chains onto this one.
        state.last_snapshot_id = snapshot.id
        # Do not consume restore provenance here. Parent selection still reads
        # restore_provenance_id() while the first restored round is in flight;
        # synthesis/integration consume only after the canonical RoundResult.
        # Forward chain uses last_snapshot_id once provenance is consumed.
        # Runtime consistency is decided by the producer that owns the runtime
        # save attempt. Registration alone cannot prove that snapshot runtime
        # belongs to the same logical state.
        await notify_fn(
            session_id,
            "snapshot_count",
            {"count": len(state.snapshots)},
        )
        logger.info(f"[{session_id}] Auto-snapshot ({phase}): {ctx['name']}")
        return snapshot

    @staticmethod
    def find_descendant_ids(all_snapshots: List["Snapshot"], root_id: str, include_root: bool = False) -> set:
        """BFS to find all descendant snapshot IDs of a given root.

        Args:
            all_snapshots: All snapshots in the session.
            root_id: The ID of the root snapshot to start from.
            include_root: Whether to include root_id in the result set.

        Returns:
            Set of descendant snapshot IDs (optionally including root_id).
        """
        # Build adjacency map: parent_id -> [child_ids]
        children_map: Dict[str, List[str]] = {}
        for s in all_snapshots:
            if s.parent_id:
                children_map.setdefault(s.parent_id, []).append(s.id)

        descendant_ids = set()
        queue = [root_id]
        while queue:
            parent = queue.pop(0)
            for child_id in children_map.get(parent, []):
                if child_id not in descendant_ids:
                    descendant_ids.add(child_id)
                    queue.append(child_id)

        if include_root:
            descendant_ids.add(root_id)
        return descendant_ids

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.snapshots_dir = get_session_dir(session_id) / "snapshots"

    def _ensure_dir(self):
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def scan_workspace_manifest(self) -> Dict[str, List[str]]:
        """Scan workspace directories and return relative file paths."""
        session_dir = get_session_dir(self.session_id)
        manifest: Dict[str, List[str]] = {}
        for subdir in ("figures", "tables", "reproducibility"):
            dir_path = session_dir / "workspace" / subdir
            files = []
            if dir_path.exists():
                for f in sorted(dir_path.iterdir()):
                    if f.is_file() and not should_exclude_snapshot_file(f.name):
                        files.append(f.name)
            manifest[subdir] = files
        # Also track files in workspace root (e.g. .h5ad, stray .csv)
        root_files = []
        workspace_dir = session_dir / "workspace"
        if workspace_dir.exists():
            for f in sorted(workspace_dir.iterdir()):
                if f.is_file() and not should_exclude_snapshot_file(f.name):
                    root_files.append(f.name)
        manifest["root"] = root_files
        return manifest

    async def create_snapshot(
        self,
        state: AnalysisState,
        name: str,
        phase: str,
        description: str = "",
        parent_id: Optional[str] = None,
        branch_label: Optional[str] = None,
    ) -> Snapshot:
        """Create a snapshot: AnalysisState JSON + kernel session pkl backup."""
        self._ensure_dir()

        snapshot = Snapshot(
            name=name,
            description=description,
            phase=phase,
            round_number=state.current_round,
            parent_id=parent_id,
            branch_label=branch_label,
        )
        snapshot.branch_id = state.restore_branch_id
        snapshot.state_file = f"{snapshot.id}.json"

        # Serialize state, excluding large/recursive fields
        state_dict = state.model_dump(mode="json")
        state_dict.pop("snapshots", None)
        state_dict.pop("realtime_messages", None)
        # Strip content_html from report segments (can be regenerated, contains base64 images)
        for seg in state_dict.get("detailed_report_segments", []):
            if isinstance(seg, dict):
                seg.pop("content_html", None)

        state_path = self.snapshots_dir / snapshot.state_file
        blob_dir = self.snapshots_dir / f"{snapshot.id}_blobs"
        tmp_path = state_path.with_suffix(".tmp")

        def _externalize_and_write() -> int:
            # Blocking blob writes + JSON serialization + atomic write,
            # all off the event loop to avoid blocking WebSocket / LLM streaming.
            try:
                externalized = externalize_blobs(state_dict, blob_dir)
                state_json = json.dumps(externalized, ensure_ascii=False, default=str)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(state_json)
                os.replace(str(tmp_path), str(state_path))
            except Exception:
                # Snapshot isn't registered on failure, so delete_snapshot would
                # never reach these. The blob dir is unique to this snapshot id,
                # so removing it (and the temp json) on failure is safe.
                shutil.rmtree(blob_dir, ignore_errors=True)
                Path(tmp_path).unlink(missing_ok=True)
                raise
            return len(state_json.encode("utf-8"))

        snapshot.size_bytes = await asyncio.to_thread(_externalize_and_write)

        # Compress kernel session pkl into the snapshot directory.
        # Compression runs in the backend process (not the kernel) on a worker
        # thread, so anndata-heavy kernel memory is never multiplied by the
        # compression window. Snapshot pkls live as .zst on disk.
        pkl_src = get_session_states(self.session_id) / f"{self.session_id}_session.pkl"
        pkl_dst = self.snapshots_dir / f"{snapshot.id}_session.pkl.zst"
        if pkl_src.exists():
            pkl_size = pkl_src.stat().st_size
            include_runtime, reason = should_include_runtime_state(pkl_size)
            if not include_runtime:
                snapshot.runtime_state_included = False
                snapshot.skipped_files.append({"path": str(pkl_src), "reason": reason, "size_bytes": str(pkl_size)})
                logger.warning("[%s] Snapshot runtime state skipped: %s", self.session_id, reason)
            elif snapshot.size_bytes + pkl_size > MAX_SESSION_SNAPSHOT_BYTES:
                reason = f"snapshot total would exceed limit {MAX_SESSION_SNAPSHOT_BYTES}"
                snapshot.runtime_state_included = False
                snapshot.skipped_files.append({"path": str(pkl_src), "reason": reason, "size_bytes": str(pkl_size)})
                logger.warning("[%s] Snapshot runtime state skipped: %s", self.session_id, reason)
            else:
                snapshot.runtime_state_included = True

                def _compress_to_snapshot():
                    import zstandard as zstd
                    cctx = zstd.ZstdCompressor(level=3, threads=-1)
                    tmp_dst = pkl_dst.with_suffix(pkl_dst.suffix + ".tmp")
                    try:
                        with open(pkl_src, "rb") as src_f, open(tmp_dst, "wb") as dst_f:
                            cctx.copy_stream(src_f, dst_f)
                        os.replace(tmp_dst, pkl_dst)
                    except Exception:
                        try:
                            tmp_dst.unlink()
                        except OSError:
                            pass
                        raise
                    return pkl_dst.stat().st_size

                snapshot.size_bytes += await asyncio.to_thread(_compress_to_snapshot)
                logger.info(
                    f"[{self.session_id}] Snapshot created: {name} ({phase}), "
                    f"size={snapshot.size_bytes} bytes (includes compressed pkl)"
                )
        else:
            logger.info(
                f"[{self.session_id}] Snapshot created: {name} ({phase}), "
                f"size={snapshot.size_bytes} bytes (no pkl found)"
            )

        # Copy notebook file for full state restoration
        nb_src = get_session_states(self.session_id) / f"{self.session_id}.ipynb"
        nb_dst = self.snapshots_dir / f"{snapshot.id}_notebook.ipynb"
        if nb_src.exists():
            await asyncio.to_thread(shutil.copy2, nb_src, nb_dst)
            snapshot.notebook_file = f"{snapshot.id}_notebook.ipynb"
            snapshot.size_bytes += nb_dst.stat().st_size

        # Record workspace file manifest for rollback cleanup
        snapshot.workspace_manifest = await asyncio.to_thread(self.scan_workspace_manifest)

        return snapshot

    async def restore_snapshot(self, snapshot: Snapshot, runtime_action: str = "apply") -> AnalysisState:
        """Load AnalysisState from a snapshot and restore kernel session pkl."""
        state_path = (self.snapshots_dir / snapshot.state_file).resolve()
        # Guard against path traversal
        if not state_path.is_relative_to(self.snapshots_dir.resolve()):
            raise ValueError(f"Invalid snapshot path: {snapshot.state_file}")
        if not state_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {state_path}")

        with open(state_path, "r", encoding="utf-8") as f:
            state_dict = json.load(f)

        blob_dir = (self.snapshots_dir / snapshot.state_file).parent / f"{snapshot.id}_blobs"
        state_dict = inline_blobs(state_dict, blob_dir)
        state_dict = normalize_direction_dicts(state_dict)

        restored = _validate_state_with_fallback(state_dict, self.session_id)

        # Materialize runtime through the journaled, re-signed apply path.
        # "apply": decompress the snapshot pkl to the live path and re-sign its
        # HMAC (fixes the stale-signature bug). "reset": quarantine the live
        # runtime so no mismatched kernel loads. "keep": leave the live runtime.
        from .runtime_apply import apply_runtime_pkl, quarantine_live_runtime

        pkl_backup = self.snapshots_dir / f"{snapshot.id}_session.pkl.zst"
        if runtime_action == "apply":
            if pkl_backup.exists():
                await asyncio.to_thread(apply_runtime_pkl, self.session_id, pkl_backup)
                logger.info(f"[{self.session_id}] Kernel session restored from snapshot {snapshot.name}")
            else:
                logger.warning(
                    f"[{self.session_id}] No pkl backup for snapshot {snapshot.name}; "
                    "runtime not rolled back"
                )
        elif runtime_action == "reset":
            await asyncio.to_thread(quarantine_live_runtime, self.session_id)
            logger.info(f"[{self.session_id}] Live runtime quarantined on reset restore")
        elif runtime_action == "keep":
            logger.info(f"[{self.session_id}] Live runtime kept (user override) on restore")

        logger.info(f"[{self.session_id}] Snapshot restored: {snapshot.name}")
        return restored

    async def restore_notebook(self, snapshot: Snapshot) -> bool:
        """Restore the notebook file from a snapshot. Returns True if restored."""
        if not snapshot.notebook_file:
            logger.warning(
                f"[{self.session_id}] No notebook backup in snapshot {snapshot.name}; "
                "notebook will not be rolled back"
            )
            return False
        nb_backup = self.snapshots_dir / snapshot.notebook_file
        if not nb_backup.exists():
            logger.warning(
                f"[{self.session_id}] Notebook backup file not found: {nb_backup}; "
                "notebook will not be rolled back"
            )
            return False
        nb_dst = get_session_states(self.session_id) / f"{self.session_id}.ipynb"
        await asyncio.to_thread(shutil.copy2, nb_backup, nb_dst)
        logger.info(f"[{self.session_id}] Notebook restored from snapshot {snapshot.name}")
        return True

    def cleanup_workspace_files(self, manifest: Dict[str, List[str]]) -> Dict[str, int]:
        """Delete workspace files NOT in the given manifest. Returns deletion counts."""
        session_dir = get_session_dir(self.session_id)
        deleted: Dict[str, int] = {}
        for subdir in ("figures", "tables", "reproducibility"):
            dir_path = session_dir / "workspace" / subdir
            keep_set = set(manifest.get(subdir, []))
            count = 0
            if dir_path.exists():
                for f in list(dir_path.iterdir()):
                    if f.is_file() and f.name not in keep_set:
                        f.unlink(missing_ok=True)
                        count += 1
            deleted[subdir] = count
        # Also clean workspace root files
        workspace_dir = session_dir / "workspace"
        root_keep = set(manifest.get("root", []))
        root_count = 0
        if workspace_dir.exists():
            for f in list(workspace_dir.iterdir()):
                if f.is_file() and f.name not in root_keep:
                    f.unlink(missing_ok=True)
                    root_count += 1
        deleted["root"] = root_count
        if any(deleted.values()):
            logger.info(
                f"[{self.session_id}] Workspace cleanup: deleted "
                + ", ".join(f"{k}={v}" for k, v in deleted.items() if v > 0)
            )
        return deleted

    def cleanup_reports(self):
        """Delete generated report files, preserving directory structure."""
        session_dir = get_session_dir(self.session_id)
        reports_dir = session_dir / "reports"
        if not reports_dir.exists():
            return
        for subdir in ("figures", "tables"):
            d = reports_dir / subdir
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink(missing_ok=True)
        # Delete all files in reports root by extension (html, json, csv, ipynb)
        for f in reports_dir.iterdir():
            if f.is_file() and f.suffix in (".html", ".json", ".csv", ".ipynb"):
                f.unlink(missing_ok=True)
        logger.info(f"[{self.session_id}] Reports directory cleaned")

    async def delete_branch(self, root_snapshot: Snapshot, all_snapshots: List[Snapshot]) -> List[str]:
        """Delete a snapshot and all its descendants. Returns list of deleted snapshot IDs."""
        ids_to_delete = self.find_descendant_ids(all_snapshots, root_snapshot.id, include_root=True)
        deleted = []
        for sid in ids_to_delete:
            snap = next((s for s in all_snapshots if s.id == sid), None)
            if snap:
                await self.delete_snapshot(snap)
                deleted.append(sid)
        return deleted

    async def delete_snapshot(self, snapshot: Snapshot):
        """Delete a snapshot file and its kernel session pkl backup."""
        state_file = str(snapshot.state_file or "").strip()
        if not state_file:
            raise ValueError("V1 snapshot state_file must be a non-empty filename")
        try:
            format_version = (
                int(snapshot.format_version)
                if snapshot.format_version is not None
                else None
            )
        except (TypeError, ValueError):
            format_version = None
        if format_version is not None and format_version >= 2:
            raise ValueError("V1 snapshot deletion cannot manage Snapshot V2")
        state_path = (self.snapshots_dir / state_file).resolve()
        if not state_path.is_relative_to(self.snapshots_dir.resolve()):
            raise ValueError(f"Invalid snapshot path: {state_file}")
        if state_path.is_dir():
            raise ValueError(f"V1 snapshot state_file is not a file: {state_file}")
        if state_path.exists():
            state_path.unlink()
        pkl = self.snapshots_dir / f"{snapshot.id}_session.pkl.zst"
        pkl.unlink(missing_ok=True)
        nb = self.snapshots_dir / f"{snapshot.id}_notebook.ipynb"
        nb.unlink(missing_ok=True)
        blob_dir = self.snapshots_dir / f"{snapshot.id}_blobs"
        if blob_dir.exists():
            shutil.rmtree(blob_dir, ignore_errors=True)
        logger.info(f"[{self.session_id}] Snapshot deleted: {snapshot.name}")

    async def delete_checkpoint(self):
        """Delete LangGraph checkpoint files so resume() starts fresh after restore."""
        states_dir = get_session_states(self.session_id)
        deleted = []
        for suffix in ("checkpoints.sqlite", "checkpoints.sqlite-wal", "checkpoints.sqlite-shm"):
            p = states_dir / suffix
            if p.exists():
                deleted.append(p.name)
            p.unlink(missing_ok=True)
        if deleted:
            logger.info(f"[{self.session_id}] Deleted checkpoint files: {', '.join(deleted)}")
