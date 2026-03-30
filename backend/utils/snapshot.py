"""Snapshot manager for saving and restoring analysis state at key points."""

import asyncio
import json
import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from ..models.schemas import AnalysisState, Snapshot
from ..config import get_session_dir, get_session_states

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages lightweight analysis state snapshots for debugging rollback."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.snapshots_dir = get_session_dir(session_id) / "snapshots"

    def _ensure_dir(self):
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    async def create_snapshot(
        self,
        state: AnalysisState,
        name: str,
        phase: str,
        description: str = "",
        parent_id: Optional[str] = None,
    ) -> Snapshot:
        """Create a snapshot: AnalysisState JSON + kernel session pkl backup."""
        self._ensure_dir()

        snapshot = Snapshot(
            name=name,
            description=description,
            phase=phase,
            round_number=state.current_round,
            parent_id=parent_id,
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

    async def delete_snapshot(self, snapshot: Snapshot):
        """Delete a snapshot file and its kernel session pkl backup."""
        state_path = (self.snapshots_dir / snapshot.state_file).resolve()
        if not state_path.is_relative_to(self.snapshots_dir.resolve()):
            raise ValueError(f"Invalid snapshot path: {snapshot.state_file}")
        if state_path.exists():
            state_path.unlink()
        pkl = self.snapshots_dir / f"{snapshot.id}_session.pkl"
        pkl.unlink(missing_ok=True)
        logger.info(f"[{self.session_id}] Snapshot deleted: {snapshot.name}")

    async def cleanup_old_snapshots(self, keep_count: int = 20) -> List[str]:
        """Remove oldest snapshots if count exceeds limit."""
        if not self.snapshots_dir.exists():
            return []
        files = sorted(
            self.snapshots_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        removed = []
        while len(files) > keep_count:
            old = files.pop(0)
            old.unlink()
            removed.append(old.name)
            # Remove corresponding pkl backup
            (self.snapshots_dir / f"{old.stem}_session.pkl").unlink(missing_ok=True)
        return removed

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
