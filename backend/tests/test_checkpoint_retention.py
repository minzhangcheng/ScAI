"""F4 durable serializer, checkpoint retention, and SQLite export tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict

import aiosqlite
import pytest

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.utils.checkpoint import (
    create_checkpoint_export_snapshot,
)
from backend.workflow import core as workflow_core


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
