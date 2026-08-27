"""Generation-aware delivery for analysis and report WebSocket channels."""

from __future__ import annotations

import asyncio

import copy

from concurrent.futures import CancelledError as FutureCancelledError

from concurrent.futures import Future

from contextlib import asynccontextmanager

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, replace

from datetime import date, datetime, time

from enum import Enum

import logging

import math

from pathlib import Path

import re

import threading

from typing import Any, Literal

import weakref

from collections.abc import Mapping

from fastapi import WebSocketDisconnect

from backend.constants import STRIPPED_FIGURE_PLACEHOLDER

from backend.state import report_websockets, websockets

from backend.utils.asyncio_tasks import (
    await_with_deferred_cancellation,
    consume_current_cancellation_request,
)

logger = logging.getLogger(__name__)

Channel = Literal["analysis", "report"]

_WEBSOCKET_SEND_TIMEOUT_SECONDS = 10.0

_WEBSOCKET_CLOSE_TIMEOUT_SECONDS = 2.0

_WEBSOCKET_RETIRE_LOCK_TIMEOUT_SECONDS = 12.0

@dataclass(eq=False)
class WebSocketConnection:
    channel: Channel
    session_id: str
    generation: int
    websocket: Any
    owner_loop: weakref.ReferenceType[asyncio.AbstractEventLoop]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False
    socket_closed: bool = False
    transport_close_started: bool = False
    transport_close_abandoned: bool = False
    transport_close_task: asyncio.Task[Any] | None = None
    replacement_close_code: int | None = None

_LOOP_SLOTS_ATTR = "_scai_websocket_delivery_slots"

_tracked_loops: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, bool
] = weakref.WeakKeyDictionary()

_generations: dict[tuple[Channel, str], int] = {}

_current_generations: dict[tuple[Channel, str], int] = {}

_current_connections: dict[tuple[Channel, str], WebSocketConnection] = {}

_delivery_tails: dict[tuple[Channel, str], Future[None]] = {}

_registry_lock_guard = threading.Lock()

_registry_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    weakref.WeakValueDictionary[str, asyncio.Lock],
] = weakref.WeakKeyDictionary()

_abandoned_transport_tasks: set[asyncio.Task[Any]] = set()

_abandoned_transport_labels: dict[asyncio.Task[Any], str] = {}

_PAYLOAD_MAX_DEPTH = 64

_KNOWN_ENCODED_PREFIXES = (
    "iVBORw0KGgo",  # PNG
    "/9j/",         # JPEG
    "R0lGOD",       # GIF
    "UklGR",        # WEBP
    "JVBERi0",      # PDF
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_EXACT_BASE64_KEYS = {
    "base64",
    "base64data",
    "figurebase64",
    "gifbase64",
    "graphicbase64",
    "imagebase64",
    "jpegbase64",
    "jpgbase64",
    "pdfbase64",
    "plotbase64",
    "pngbase64",
    "previewbase64",
    "svgbase64",
    "thumbnailbase64",
    "webpbase64",
}

_DURABLE_REALTIME_MESSAGE_LIMIT = 1000

import sys
from .websocket_delivery_parts import connections as delivery_connections
from .websocket_delivery_parts import payloads as delivery_payloads
from .websocket_delivery_parts import queue as delivery_queue

def _normalized_payload_key(key: Any) -> str:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._normalized_payload_key(key)

def _is_media_value_key(key: Any) -> bool:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._is_media_value_key(key)

def _is_base64_key(key: Any) -> bool:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._is_base64_key(key)

def _is_encoded_media(value: str, *, key: Any=None, media_value: bool=False) -> bool:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._is_encoded_media(value, key=key, media_value=media_value)

def _json_key(key: Any) -> Any:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._json_key(key)

def _cycle_placeholder(json_mode: bool) -> Any:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._cycle_placeholder(json_mode)

def _copy_payload(value: Any, *, seen: set[int], depth: int, strip_binary: bool, json_mode: bool, key: Any=None, media_value: bool=False) -> Any:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads._copy_payload(value, seen=seen, depth=depth, strip_binary=strip_binary, json_mode=json_mode, key=key, media_value=media_value)

def strip_nested_binary_payload(value: Any) -> Any:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads.strip_nested_binary_payload(value)

def append_durable_realtime_message(target: Any, message: Any) -> None:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads.append_durable_realtime_message(target, message)

def json_safe_payload(value: Any, *, strip_binary: bool=True) -> Any:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads.json_safe_payload(value, strip_binary=strip_binary)

def serialize_realtime_message(message: Any, *, strip_binary: bool=True) -> dict[str, Any]:
    delivery_payloads.bind_facade(sys.modules[__name__])
    return delivery_payloads.serialize_realtime_message(message, strip_binary=strip_binary)

def _log_background_failure(label: str, exc: BaseException) -> None:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._log_background_failure(label, exc)

def _observe_abandoned_transport_task(task: asyncio.Task[Any]) -> None:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._observe_abandoned_transport_task(task)

def _track_abandoned_transport_task(task: asyncio.Task[Any], *, operation: str) -> None:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._track_abandoned_transport_task(task, operation=operation)

async def _await_task_with_hard_timeout(task: asyncio.Task[Any], timeout: float, *, operation: str) -> Any:
    delivery_queue.bind_facade(sys.modules[__name__])
    return await delivery_queue._await_task_with_hard_timeout(task, timeout, operation=operation)

async def _await_with_hard_timeout(awaitable: Any, timeout: float, *, operation: str) -> Any:
    delivery_queue.bind_facade(sys.modules[__name__])
    return await delivery_queue._await_with_hard_timeout(awaitable, timeout, operation=operation)

def _slots_for_running_loop() -> dict[tuple[Channel, str], WebSocketConnection]:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._slots_for_running_loop()

def _registry_lock_for_running_loop() -> asyncio.Lock:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._registry_lock_for_running_loop()

def _install_current_connection(key: tuple[Channel, str], websocket: Any, owner_loop: asyncio.AbstractEventLoop) -> tuple[WebSocketConnection, WebSocketConnection | None]:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._install_current_connection(key, websocket, owner_loop)

def _globally_current_connection(key: tuple[Channel, str]) -> WebSocketConnection | None:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._globally_current_connection(key)

def _is_globally_current(connection: WebSocketConnection) -> bool:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._is_globally_current(connection)

def _legacy_registry(channel: Channel) -> dict[str, Any]:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._legacy_registry(channel)

def _finish_delivery_turn(key: tuple[Channel, str], turn: Future[None]) -> None:
    delivery_queue.bind_facade(sys.modules[__name__])
    return delivery_queue._finish_delivery_turn(key, turn)

@asynccontextmanager
async def _delivery_turn(key: tuple[Channel, str]):
    delivery_queue.bind_facade(sys.modules[__name__])
    async with delivery_queue._delivery_turn(key) as value:
        yield value

async def register_websocket(channel: Channel, session_id: str, websocket: Any) -> WebSocketConnection:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.register_websocket(channel, session_id, websocket)

def _consume_thread_future(future: Future[Any]) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return delivery_connections._consume_thread_future(future)

async def _wait_thread_future(future: Future[Any]) -> Any:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections._wait_thread_future(future)

def _finalize_transport_close(connection: WebSocketConnection, task: asyncio.Task[Any]) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return delivery_connections._finalize_transport_close(connection, task)

async def _close_socket_locked(connection: WebSocketConnection, *, code: int) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections._close_socket_locked(connection, code=code)

async def _retire_connection_on_owner(connection: WebSocketConnection) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections._retire_connection_on_owner(connection)

async def _remove_if_current(connection: WebSocketConnection) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections._remove_if_current(connection)

async def _close_and_remove_on_owner(connection: WebSocketConnection, *, code: int) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections._close_and_remove_on_owner(connection, code=code)

async def unregister_websocket(connection: WebSocketConnection | None, *, close: bool=False, code: int=1000) -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.unregister_websocket(connection, close=close, code=code)

async def send_connection_json(connection: WebSocketConnection | None, payload: dict[str, Any]) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.send_connection_json(connection, payload)

async def _send_current_connection_json(connection: WebSocketConnection, payload: dict[str, Any]) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections._send_current_connection_json(connection, payload)

async def send_websocket_json(channel: Channel, session_id: str, payload: dict[str, Any]) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.send_websocket_json(channel, session_id, payload)

async def close_websocket(channel: Channel, session_id: str, *, code: int=1000) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.close_websocket(channel, session_id, code=code)

async def register_analysis_websocket(session_id: str, websocket: Any) -> WebSocketConnection:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.register_analysis_websocket(session_id, websocket)

async def register_report_websocket(session_id: str, websocket: Any) -> WebSocketConnection:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.register_report_websocket(session_id, websocket)

async def send_analysis_json(session_id: str, payload: dict[str, Any]) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.send_analysis_json(session_id, payload)

async def send_report_json(session_id: str, payload: dict[str, Any]) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.send_report_json(session_id, payload)

async def close_analysis_websocket(session_id: str, *, code: int=1000) -> bool:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.close_analysis_websocket(session_id, code=code)

def abandoned_transport_task_count() -> int:
    delivery_connections.bind_facade(sys.modules[__name__])
    return delivery_connections.abandoned_transport_task_count()

async def drain_abandoned_transport_tasks(*, timeout: float=2.0) -> int:
    delivery_connections.bind_facade(sys.modules[__name__])
    return await delivery_connections.drain_abandoned_transport_tasks(timeout=timeout)

def reset_websocket_delivery_for_tests() -> None:
    delivery_connections.bind_facade(sys.modules[__name__])
    return delivery_connections.reset_websocket_delivery_for_tests()
