"""File responses that preserve export cleanup ownership."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from starlette.responses import (
    FileResponse,
    Headers,
    MalformedRangeHeader,
    MutableHeaders,
    PlainTextResponse,
    RangeNotSatisfiable,
)

from backend.services.export_lifecycle_parts.archive import _unlink_owned_export_paths
from backend.services.export_lifecycle_parts.records import _begin_stream, cleanup_export_record
from backend.utils.asyncio_tasks import await_with_deferred_cancellation, run_in_owned_thread


class LifecycleFileResponse(FileResponse):
    """FileResponse whose send and unlink remain visible to session deletion."""

    def __init__(self, *args, export_info: dict[str, Any], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._export_info = export_info

    async def __call__(self, scope, receive, send) -> None:
        try:
            await _begin_stream(self._export_info)
            await super().__call__(scope, receive, send)
        finally:
            finalizer = asyncio.create_task(
                cleanup_export_record(self._export_info),
                name=(
                    "export-stream-cleanup:"
                    f"{self._export_info.get('session_id')}:"
                    f"{self._export_info.get('token')}"
                ),
            )
            await await_with_deferred_cancellation(finalizer)


class OwnedArchiveFileResponse(FileResponse):
    """Stream one already-open archive inode and retire only that inode."""

    def __init__(
        self,
        path: str | Path,
        *,
        file_fd: int,
        owned_identity: tuple[int, int],
        marker_owned: bool = False,
        **kwargs,
    ) -> None:
        self._owned_fd = int(file_fd)
        self._owned_identity = tuple(int(item) for item in owned_identity)
        self._marker_owned = bool(marker_owned)
        stat_result = os.fstat(self._owned_fd)
        super().__init__(path, stat_result=stat_result, **kwargs)

    async def _send_owned_body(
        self,
        send,
        *,
        start: int,
        end: int,
        final_more_body: bool = False,
    ) -> None:
        offset = int(start)
        limit = int(end)
        if offset >= limit:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": final_more_body,
            })
            return
        while offset < limit:
            try:
                block = os.pread(
                    self._owned_fd,
                    min(self.chunk_size, limit - offset),
                    offset,
                )
            except OSError as exc:
                raise RuntimeError("Owned archive became unreadable") from exc
            if not block:
                raise RuntimeError("Owned archive ended before the promised range")
            offset += len(block)
            await send({
                "type": "http.response.body",
                "body": block,
                "more_body": offset < limit or final_more_body,
            })

    async def _handle_owned_simple(self, send, send_header_only: bool) -> None:
        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        })
        if send_header_only:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return
        await self._send_owned_body(send, start=0, end=self.stat_result.st_size)

    async def _handle_owned_single_range(
        self,
        send,
        start: int,
        end: int,
        file_size: int,
        send_header_only: bool,
    ) -> None:
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send({
            "type": "http.response.start",
            "status": 206,
            "headers": headers.raw,
        })
        if send_header_only:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return
        await self._send_owned_body(send, start=start, end=end)

    async def _handle_owned_multiple_ranges(
        self,
        send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        boundary = os.urandom(13).hex()
        content_length, header_generator = self.generate_multipart(
            ranges,
            boundary,
            file_size,
            self.headers["content-type"],
        )
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(content_length)
        await send({
            "type": "http.response.start",
            "status": 206,
            "headers": headers.raw,
        })
        if send_header_only:
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            return
        for start, end in ranges:
            await send({
                "type": "http.response.body",
                "body": header_generator(start, end),
                "more_body": True,
            })
            await self._send_owned_body(
                send,
                start=start,
                end=end,
                final_more_body=True,
            )
            await send({
                "type": "http.response.body",
                "body": b"\r\n",
                "more_body": True,
            })
        await send({
            "type": "http.response.body",
            "body": f"--{boundary}--".encode("latin-1"),
            "more_body": False,
        })

    async def __call__(self, scope, receive, send) -> None:
        try:
            scope_type = scope["type"]
            send_header_only = (
                scope_type == "http" and scope["method"].upper() == "HEAD"
            )
            if scope_type == "websocket":
                send = self._wrap_websocket_denial_send(send)

            headers = Headers(scope=scope)
            http_range = headers.get("range")
            http_if_range = headers.get("if-range")
            if http_range is None or (
                http_if_range is not None and not self._should_use_range(http_if_range)
            ):
                await self._handle_owned_simple(send, send_header_only)
            else:
                try:
                    ranges = self._parse_range_header(
                        http_range,
                        self.stat_result.st_size,
                    )
                except MalformedRangeHeader as exc:
                    await PlainTextResponse(exc.content, status_code=400)(
                        scope,
                        receive,
                        send,
                    )
                except RangeNotSatisfiable as exc:
                    await PlainTextResponse(
                        status_code=416,
                        headers={"Content-Range": f"bytes */{exc.max_size}"},
                    )(scope, receive, send)
                else:
                    if not ranges:
                        await self._handle_owned_simple(send, send_header_only)
                    elif len(ranges) == 1:
                        start, end = ranges[0]
                        await self._handle_owned_single_range(
                            send,
                            start,
                            end,
                            self.stat_result.st_size,
                            send_header_only,
                        )
                    else:
                        await self._handle_owned_multiple_ranges(
                            send,
                            ranges,
                            self.stat_result.st_size,
                            send_header_only,
                        )

            if self.background is not None:
                await self.background()
        finally:
            file_fd = self._owned_fd
            self._owned_fd = -1
            try:
                if file_fd >= 0:
                    os.close(file_fd)
            finally:
                cleanup = asyncio.create_task(
                    run_in_owned_thread(
                        _unlink_owned_export_paths,
                        self.path,
                        self._owned_identity,
                        marker_owned=self._marker_owned,
                    ),
                    name=f"owned-archive-cleanup:{self.path}",
                )
                await await_with_deferred_cancellation(cleanup)
