"""F4 durable serializer, checkpoint retention, and SQLite export tests."""

from __future__ import annotations

import copy
import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import TypedDict

import aiosqlite
import pytest

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.routes.export_import_parts import tar_builders
from backend.utils.checkpoint import (
    configure_checkpoint_retention,
    create_checkpoint_export_snapshot,
)
from backend.workflow import core as workflow_core
from backend.workflow.core import SingleCellAnalysisWorkflow


_ALLOWED_JSON_MODULES = [
    ("backend.models.schemas", "AgentType"),
    ("backend.models.schemas", "AnalysisState"),
    ("backend.models.schemas", "AnalysisPhase"),
    ("backend.models.schemas", "RoundResult"),
    ("backend.models.schemas", "TaskStatus"),
    ("backend.models.schemas", "MessageRole"),
]


class _AsyncCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    async def close(self):
        self._cursor.close()


class _AsyncSqliteShim:
    def __init__(self, path: Path):
        self.raw = sqlite3.connect(path)

    async def execute(self, sql: str, parameters=()):
        return _AsyncCursor(self.raw.execute(sql, parameters))

    async def commit(self):
        self.raw.commit()


def _create_retention_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );
        CREATE TABLE writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT,
            type TEXT,
            value BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        );
        """
    )


def _seed_retention_rows(connection: sqlite3.Connection, count: int = 8) -> None:
    for index in range(1, count + 1):
        checkpoint_id = f"{index:04d}"
        parent = f"{index - 1:04d}" if index > 1 else None
        connection.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("thread", "", checkpoint_id, parent, "json", b"{}", b"{}"),
        )
        connection.execute(
            "INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("thread", "", checkpoint_id, "task", 0, "channel", "json", b"{}"),
        )
    connection.commit()


def test_checkpoint_serializer_strips_nested_binary_without_mutating_graph_state():
    serializer = workflow_core.JsonPlusSerializer(
        allowed_json_modules=_ALLOWED_JSON_MODULES
    )
    image = "iVBORw0KGgo" + ("A" * 2048)
    graph_state = {
        "nested": {"image_base64": image, "raw": b"raw-binary"},
        "values": [memoryview(b"memoryview")],
    }
    original = {
        "nested": dict(graph_state["nested"]),
        "values": [graph_state["values"][0]],
    }

    type_name, payload = serializer.dumps_typed(graph_state)

    assert type_name == "msgpack"
    assert image.encode("utf-8") not in payload
    assert b"raw-binary" not in payload
    assert b"memoryview" not in payload
    assert graph_state == original


@pytest.mark.asyncio
async def test_retention_replaces_stale_trigger_and_keeps_parent_writes(tmp_path):
    path = tmp_path / "checkpoints.sqlite"
    raw = sqlite3.connect(path)
    _create_retention_schema(raw)
    _seed_retention_rows(raw)
    raw.execute(
        "INSERT INTO writes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("thread", "", "orphan", "task", 0, "channel", "json", b"{}"),
    )
    raw.executescript(
        """
        CREATE TRIGGER scai_checkpoint_retention
        AFTER INSERT ON checkpoints
        BEGIN
            DELETE FROM checkpoints
            WHERE checkpoint_id = '0001';
        END;
        """
    )
    raw.commit()
    workflow = object.__new__(SingleCellAnalysisWorkflow)
    workflow._db_conn = _AsyncSqliteShim(path)
    workflow.session_id = "thread"

    await configure_checkpoint_retention(workflow._db_conn, keep=5)

    checkpoints = [
        row[0]
        for row in raw.execute(
            "SELECT checkpoint_id FROM checkpoints ORDER BY checkpoint_id"
        )
    ]
    writes = {
        row[0]
        for row in raw.execute("SELECT checkpoint_id FROM writes")
    }
    trigger_sql = raw.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
        ("scai_checkpoint_retention",),
    ).fetchone()[0]
    assert checkpoints == ["0004", "0005", "0006", "0007", "0008"]
    assert "0003" in writes
    assert "0001" not in writes
    assert "orphan" not in writes
    assert "0005" in trigger_sql or "5" in trigger_sql
    assert trigger_sql.count("scai_checkpoint_retention") <= 1

    raw.execute(
        "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("thread", "", "0009", "0008", "json", b"{}", b"{}"),
    )
    raw.commit()
    assert [
        row[0]
        for row in raw.execute(
            "SELECT checkpoint_id FROM checkpoints ORDER BY checkpoint_id"
        )
    ] == ["0005", "0006", "0007", "0008", "0009"]
    raw.close()


@pytest.mark.asyncio
async def test_backup_snapshot_includes_committed_wal_and_is_standalone(
    tmp_path, monkeypatch
):
    session_id = "checkpoint-backup"
    session_root = tmp_path / session_id
    states = session_root / "states"
    states.mkdir(parents=True)
    source = states / "checkpoints.sqlite"
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE values_table (id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO values_table(value) VALUES ('committed-wal')")
    connection.commit()
    assert source.with_name("checkpoints.sqlite-wal").exists()

    monkeypatch.setattr(tar_builders, "get_session_states", lambda _sid: states)
    monkeypatch.setattr(tar_builders, "legacy_session_state_paths", lambda *_args: [])

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tf:
        tar_builders._add_states_to_tar(tf, session_id, include_runtime=True)
    connection.close()

    archive.seek(0)
    with tarfile.open(fileobj=archive, mode="r") as tf:
        names = tf.getnames()
        assert names.count(f"{session_id}/states/checkpoints.sqlite") == 1
        assert f"{session_id}/states/checkpoints.sqlite-wal" not in names
        assert f"{session_id}/states/checkpoints.sqlite-shm" not in names
        extracted = tf.extractfile(f"{session_id}/states/checkpoints.sqlite").read()

    standalone = tmp_path / "standalone.sqlite"
    standalone.write_bytes(extracted)
    check = sqlite3.connect(standalone)
    assert check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert check.execute("SELECT value FROM values_table").fetchone()[0] == "committed-wal"
    assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    check.close()


class _ContinuationState(TypedDict, total=False):
    count: int
    sent_value: int


def _continuation_graph(saver, *, interrupt_before=()):
    graph = StateGraph(_ContinuationState)

    def increment(state: _ContinuationState):
        return {"count": state.get("count", 0) + 1}

    def route(state: _ContinuationState):
        if state.get("count", 0) < 7:
            return "increment"
        return [Send("apply", {"sent_value": state["count"]})]

    def apply(state: _ContinuationState):
        return {"sent_value": state.get("sent_value", 0) * 10}

    graph.add_node("increment", increment)
    graph.add_node("apply", apply)
    graph.add_edge(START, "increment")
    graph.add_conditional_edges("increment", route)
    graph.add_edge("apply", END)
    return graph.compile(
        checkpointer=saver,
        interrupt_before=list(interrupt_before),
    )


@pytest.mark.asyncio
async def test_retained_latest_checkpoint_resumes_after_older_window_is_pruned(
    tmp_path,
):
    path = tmp_path / "continuation.sqlite"
    connection = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(connection)
    await saver.setup()
    config = {"configurable": {"thread_id": "continuation", "checkpoint_ns": ""}}
    paused_app = _continuation_graph(saver, interrupt_before=("apply",))
    paused = await paused_app.ainvoke({"count": 0}, config)
    assert paused["count"] == 7
    rows = await connection.execute(
        "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints "
        "WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id",
        ("continuation", ""),
    )
    checkpoint_rows = await rows.fetchall()
    await rows.close()
    assert len(checkpoint_rows) > 5
    oldest_parent = checkpoint_rows[0][1]
    if oldest_parent:
        pending = await connection.execute(
            "SELECT 1 FROM writes WHERE thread_id=? AND checkpoint_ns=? "
            "AND checkpoint_id=? LIMIT 1",
            ("continuation", "", oldest_parent),
        )
        assert await pending.fetchone() is not None
        await pending.close()

    await connection.close()
    snapshot = tmp_path / "continuation-export.sqlite"
    create_checkpoint_export_snapshot(path, snapshot, keep=5)

    destination = await aiosqlite.connect(snapshot)
    destination_saver = AsyncSqliteSaver(destination)
    await destination_saver.setup()
    resumed_app = _continuation_graph(destination_saver)
    exported_rows = await destination.execute(
        "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints "
        "WHERE thread_id=? AND checkpoint_ns=? ORDER BY checkpoint_id",
        ("continuation", ""),
    )
    retained_rows = await exported_rows.fetchall()
    await exported_rows.close()
    assert len(retained_rows) == 5
    oldest_exported_parent = retained_rows[0][1]
    if oldest_exported_parent:
        pending = await destination.execute(
            "SELECT 1 FROM writes WHERE thread_id=? AND checkpoint_ns=? "
            "AND checkpoint_id=? LIMIT 1",
            ("continuation", "", oldest_exported_parent),
        )
        assert await pending.fetchone() is not None
        await pending.close()

    resumed = await resumed_app.ainvoke(None, config)
    assert resumed["sent_value"] == 70
    await destination.close()
    await connection.close()
