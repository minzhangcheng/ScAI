"""
Generic checkpoint manager for step-level recovery.

Provides atomic save/load/clear operations for JSON-serializable checkpoint data.
Used by BioinformaticsAgent (analysis checkpoints) and DetailedReportGenerator
(report generation checkpoints).
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import aiofiles

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages step-level checkpoints with atomic file writes.
    Enables recovery from interruptions during long-running processes.
    """

    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N steps

    def __init__(self, session_id: str, state_dir: str, prefix: str = "bioinfo"):
        """
        Initialize checkpoint manager.

        Args:
            session_id: Session ID for this process
            state_dir: Directory to store checkpoint files
            prefix: Filename prefix to distinguish checkpoint types
                    (e.g. "bioinfo", "report")
        """
        self.session_id = session_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.state_dir / f"{session_id}_{prefix}_checkpoint.json"

    async def save(self, checkpoint: Dict[str, Any]):
        """
        Save checkpoint to disk atomically.

        Args:
            checkpoint: Checkpoint data to save
        """
        checkpoint['updated_at'] = datetime.now().isoformat()

        # Atomic write: write to temp file then rename
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(checkpoint, indent=2, default=str))
            tmp_path.replace(self.checkpoint_path)
            logger.info(f"[{self.session_id}] Checkpoint saved: {checkpoint.get('step_name', 'unknown')}")
        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to save checkpoint")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    async def load(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from disk.

        Returns:
            Checkpoint data or None if not found
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            async with aiofiles.open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.exception(f"[{self.session_id}] Failed to load checkpoint")
            return None

    def clear(self):
        """Clear checkpoint after successful completion."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info(f"[{self.session_id}] Checkpoint cleared")

    def should_checkpoint(self, step: int) -> bool:
        """
        Check if we should save a checkpoint at this step.

        Args:
            step: Current step number

        Returns:
            True if checkpoint should be saved
        """
        return step > 0 and step % self.CHECKPOINT_INTERVAL == 0
