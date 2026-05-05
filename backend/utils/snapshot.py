"""Snapshot manager for saving and restoring analysis state at key points."""

import asyncio
import json
import os
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..models.schemas import AnalysisState, Snapshot
from ..config import get_session_dir, get_session_states

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages lightweight analysis state snapshots for debugging rollback."""

    @staticmethod
    def compute_snapshot_context(state: "AnalysisState", phase: str) -> dict:
        """Compute parent_id, branch_label, and name for the next auto-snapshot.

        Shared by both workflow and chat snapshot creation to avoid logic drift.
        """
        parent_id = state.last_restored_snapshot_id
        if parent_id is None and state.snapshots:
            parent_id = state.snapshots[-1].id

        if state.last_restored_snapshot_id:
            existing_branches = sum(
                1 for s in state.snapshots
                if s.parent_id == state.last_restored_snapshot_id
                and s.branch_label and s.branch_label != "main"
            )
            branch_letter = chr(ord('A') + existing_branches) if existing_branches < 26 else str(existing_branches)
            branch_label = f"branch-{branch_letter}"
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
        if state.last_restored_snapshot_id:
            state.last_restored_snapshot_id = None
        await notify_fn(session_id, "snapshot_count", {"count": len(state.snapshots)})
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
                    if f.is_file():
                        files.append(f.name)
            manifest[subdir] = files
        # Also track files in workspace root (e.g. .h5ad, stray .csv)
        root_files = []
        workspace_dir = session_dir / "workspace"
        if workspace_dir.exists():
            for f in sorted(workspace_dir.iterdir()):
                if f.is_file():
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
        state_json = json.dumps(state_dict, ensure_ascii=False, default=str)
        snapshot.size_bytes = len(state_json.encode("utf-8"))

        # Atomic write
        tmp_path = state_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(state_json)
        os.replace(str(tmp_path), str(state_path))

        # Copy kernel session pkl (large file — run in thread to avoid blocking event loop)
        pkl_src = get_session_states(self.session_id) / f"{self.session_id}_session.pkl"
        pkl_dst = self.snapshots_dir / f"{snapshot.id}_session.pkl"
        if pkl_src.exists():
            await asyncio.to_thread(shutil.copy2, pkl_src, pkl_dst)
            snapshot.size_bytes += pkl_dst.stat().st_size
            logger.info(
                f"[{self.session_id}] Snapshot created: {name} ({phase}), "
                f"size={snapshot.size_bytes} bytes (includes pkl)"
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

    async def restore_snapshot(self, snapshot: Snapshot) -> AnalysisState:
        """Load AnalysisState from a snapshot and restore kernel session pkl."""
        state_path = (self.snapshots_dir / snapshot.state_file).resolve()
        # Guard against path traversal
        if not state_path.is_relative_to(self.snapshots_dir.resolve()):
            raise ValueError(f"Invalid snapshot path: {snapshot.state_file}")
        if not state_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {state_path}")

        with open(state_path, "r", encoding="utf-8") as f:
            state_dict = json.load(f)

        restored = AnalysisState.model_validate(state_dict)

        # Restore kernel session pkl (large file — run in thread to avoid blocking event loop)
        pkl_backup = self.snapshots_dir / f"{snapshot.id}_session.pkl"
        if pkl_backup.exists():
            pkl_dst = get_session_states(self.session_id) / f"{self.session_id}_session.pkl"
            await asyncio.to_thread(shutil.copy2, pkl_backup, pkl_dst)
            logger.info(f"[{self.session_id}] Kernel session restored from snapshot {snapshot.name}")
        else:
            logger.warning(
                f"[{self.session_id}] No pkl backup found for snapshot {snapshot.name}; "
                "kernel state will not be rolled back"
            )

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
        state_path = (self.snapshots_dir / snapshot.state_file).resolve()
        if not state_path.is_relative_to(self.snapshots_dir.resolve()):
            raise ValueError(f"Invalid snapshot path: {snapshot.state_file}")
        if state_path.exists():
            state_path.unlink()
        pkl = self.snapshots_dir / f"{snapshot.id}_session.pkl"
        pkl.unlink(missing_ok=True)
        nb = self.snapshots_dir / f"{snapshot.id}_notebook.ipynb"
        nb.unlink(missing_ok=True)
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
