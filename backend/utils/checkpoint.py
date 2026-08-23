"""Checkpoint persistence helpers and the generic step-level checkpoint file."""
import json
import logging
import os
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import aiofiles
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from backend.utils.websocket_delivery import strip_nested_binary_payload

logger = logging.getLogger(__name__)


class DurableJsonPlusSerializer(JsonPlusSerializer):
    """LangGraph serializer that strips binary output on a detached copy."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return super().dumps_typed(strip_nested_binary_payload(obj))


_RETENTION_TRIGGER_NAME = "scai_checkpoint_retention"


def _validated_keep(keep: int) -> int:
    value = int(keep)
    if value < 1:
        raise ValueError("checkpoint retention keep must be at least one")
    return value


def _checkpoint_retention_sql(keep: int) -> tuple[str, str]:
    """Return the shared checkpoint and write pruning statements."""
    keep = _validated_keep(keep)
    checkpoint_sql = f"""
        DELETE FROM checkpoints AS c
        WHERE c.checkpoint_id NOT IN (
            SELECT newest.checkpoint_id
            FROM checkpoints AS newest
            WHERE newest.thread_id = c.thread_id
              AND newest.checkpoint_ns = c.checkpoint_ns
            ORDER BY newest.checkpoint_id DESC
            LIMIT {keep}
        )
    """
    writes_sql = """
        DELETE FROM writes AS w
        WHERE NOT EXISTS (
            SELECT 1
            FROM checkpoints AS c
            WHERE c.thread_id = w.thread_id
              AND c.checkpoint_ns = w.checkpoint_ns
              AND (
                  c.checkpoint_id = w.checkpoint_id
                  OR c.parent_checkpoint_id = w.checkpoint_id
              )
        )
    """
    return checkpoint_sql, writes_sql


def _retention_trigger_sql(keep: int) -> str:
    keep = _validated_keep(keep)
    return f"""
        CREATE TRIGGER {_RETENTION_TRIGGER_NAME}
        AFTER INSERT ON checkpoints
        BEGIN
            DELETE FROM checkpoints
            WHERE thread_id = NEW.thread_id
              AND checkpoint_ns = NEW.checkpoint_ns
              AND checkpoint_id NOT IN (
                  SELECT newest.checkpoint_id
                  FROM checkpoints AS newest
                  WHERE newest.thread_id = NEW.thread_id
                    AND newest.checkpoint_ns = NEW.checkpoint_ns
                  ORDER BY newest.checkpoint_id DESC
                  LIMIT {keep}
              );
            DELETE FROM writes
            WHERE thread_id = NEW.thread_id
              AND checkpoint_ns = NEW.checkpoint_ns
              AND NOT EXISTS (
                  SELECT 1
                  FROM checkpoints AS retained
                  WHERE retained.thread_id = NEW.thread_id
                    AND retained.checkpoint_ns = NEW.checkpoint_ns
                    AND (
                        retained.checkpoint_id = writes.checkpoint_id
                        OR retained.parent_checkpoint_id = writes.checkpoint_id
                    )
              );
        END
    """


async def configure_checkpoint_retention(
    connection: Any, *, keep: int = 5
) -> None:
    """Prune current history, then replace the persistent retention trigger."""
    keep = _validated_keep(keep)
    checkpoint_sql, writes_sql = _checkpoint_retention_sql(keep)
    cursor = await connection.execute(checkpoint_sql)
    await cursor.close()
    cursor = await connection.execute(writes_sql)
    await cursor.close()
    drop = f"DROP TRIGGER IF EXISTS {_RETENTION_TRIGGER_NAME}"
    cursor = await connection.execute(drop)
    await cursor.close()
    cursor = await connection.execute(_retention_trigger_sql(keep))
    await cursor.close()
    await connection.commit()


async def compact_checkpoint_history(
    connection: Any, *, keep: int = 5, vacuum: bool = False
) -> None:
    """Retain resumable checkpoints and writes for every namespace."""
    keep = _validated_keep(keep)
    await configure_checkpoint_retention(connection, keep=keep)
    cursor = await connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    await cursor.close()
    if vacuum:
        cursor = await connection.execute("VACUUM")
        await cursor.close()


def _compact_checkpoint_history_sync(
    connection: sqlite3.Connection, *, keep: int = 5
) -> None:
    """Synchronous counterpart used for standalone export destinations."""
    keep = _validated_keep(keep)
    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "checkpoints" not in table_names or "writes" not in table_names:
        return
    checkpoint_sql, writes_sql = _checkpoint_retention_sql(keep)
    connection.execute(checkpoint_sql)
    connection.execute(writes_sql)


def create_checkpoint_export_snapshot(
    source: Path, destination: Path, *, keep: int = 5
) -> None:
    """Create one verified, standalone SQLite checkpoint snapshot."""
    source = Path(source)
    destination = Path(destination)
    _validated_keep(keep)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"checkpoint source is not a regular file: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint export destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(
            f"file:{source}?mode=ro", uri=True
        )
        destination_connection = sqlite3.connect(destination)
        source_connection.backup(destination_connection)
        destination_connection.execute(
            f"DROP TRIGGER IF EXISTS {_RETENTION_TRIGGER_NAME}"
        )
        _compact_checkpoint_history_sync(destination_connection, keep=keep)
        destination_connection.commit()
        journal_mode = destination_connection.execute(
            "PRAGMA journal_mode=DELETE"
        ).fetchone()
        if not journal_mode or str(journal_mode[0]).casefold() != "delete":
            raise sqlite3.DatabaseError(
                "checkpoint export did not enter delete mode"
            )
        result = destination_connection.execute(
            "PRAGMA quick_check(1)"
        ).fetchone()
        if not result or result[0] != "ok":
            raise sqlite3.DatabaseError("checkpoint export failed quick_check")
    except BaseException:
        if destination.exists() or destination.is_symlink():
            try:
                destination.unlink()
            except OSError:
                logger.warning("Failed to remove incomplete checkpoint export %s", destination)
        for suffix in ("-wal", "-shm"):
            sidecar = destination.with_name(destination.name + suffix)
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Failed to remove incomplete checkpoint sidecar %s", sidecar)
        raise
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()
    for suffix in ("-wal", "-shm"):
        sidecar = destination.with_name(destination.name + suffix)
        if sidecar.exists():
            raise sqlite3.DatabaseError(
                f"checkpoint export requires a sidecar: {sidecar.name}"
            )
    os.chmod(destination, 0o600)


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
        self._last_saved_step: Optional[str] = None  # Dedup: track last saved step name

    async def save(self, checkpoint: Dict[str, Any]):
        """
        Save checkpoint to disk atomically.

        Skips the write if the same step_name was just saved (dedup for
        figure-revision loops that decrement the step counter).

        Args:
            checkpoint: Checkpoint data to save
        """
        step_name = checkpoint.get('step_name', 'unknown')
        if step_name == self._last_saved_step:
            logger.debug(f"[{self.session_id}] Checkpoint for {step_name} already saved, skipping")
            return

        checkpoint['updated_at'] = datetime.now().isoformat()

        # Atomic write: write to temp file then rename
        tmp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(checkpoint, indent=2, default=str))
            tmp_path.replace(self.checkpoint_path)
            self._last_saved_step = step_name
            logger.info(f"[{self.session_id}] Checkpoint saved: {step_name}")
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
