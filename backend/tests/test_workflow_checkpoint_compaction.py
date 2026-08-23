import sqlite3
import pytest

from backend.workflow.core import SingleCellAnalysisWorkflow


class _AsyncCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    async def close(self):
        self._cursor.close()


class _AsyncSqliteShim:
    """Minimal async surface over sqlite3 for deterministic SQL testing."""
    def __init__(self, path):
        self.raw = sqlite3.connect(path)

    async def execute(self, sql, parameters=()):
        return _AsyncCursor(self.raw.execute(sql, parameters))

    async def commit(self):
        self.raw.commit()


@pytest.mark.asyncio
async def test_checkpoint_compaction_keeps_latest_per_namespace_and_pending_parent_writes(
    tmp_path,
):
    connection = _AsyncSqliteShim(tmp_path / "checkpoints.sqlite")
    connection.raw.executescript(
        """
        CREATE TABLE checkpoints (
            thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
            parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );
        CREATE TABLE writes (
            thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,
            task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        );
        """
    )
    checkpoint_rows = []
    write_rows = []
    for namespace in ("", "child"):
        for index in range(8):
            checkpoint_id = f"{index:02d}"
            parent = f"{index - 1:02d}" if index else None
            checkpoint_rows.append(
                ("thread", namespace, checkpoint_id, parent, "json", b"x", b"{}")
            )
            write_rows.append(
                ("thread", namespace, checkpoint_id, "task", 0, "channel", "json", b"x")
            )
    cursor = connection.raw.executemany(
        "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)", checkpoint_rows
    )
    cursor.close()
    cursor = connection.raw.executemany(
        "INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", write_rows
    )
    cursor.close()
    await connection.commit()

    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow._db_conn = connection
    workflow.session_id = "thread"
    await workflow._compact_checkpoints(keep=3)

    cursor = connection.raw.execute(
        "SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id "
        "FROM checkpoints ORDER BY checkpoint_ns, checkpoint_id"
    )
    rows = cursor.fetchall()
    cursor.close()
    cursor = connection.raw.execute("SELECT count(*) FROM writes")
    writes = cursor.fetchone()
    cursor.close()
    assert rows == [
        ("", "05", "04"), ("", "06", "05"), ("", "07", "06"),
        ("child", "05", "04"), ("child", "06", "05"), ("child", "07", "06"),
    ]
    assert writes[0] == 8
    connection.raw.close()
