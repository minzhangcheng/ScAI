"""Reject oversized multipart request bodies before form parsing."""

from __future__ import annotations

import logging

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger(__name__)

MULTIPART_ENVELOPE_ALLOWANCE = 1024 * 1024
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


class _BodyTooLarge(Exception):
    """Raised when an undeclared or untrusted body exceeds the limit."""


def _is_body_too_large(exc: BaseException) -> bool:
    if isinstance(exc, _BodyTooLarge):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_body_too_large(nested) for nested in exc.exceptions)
    return False


def _is_multipart(scope: Scope) -> bool:
    for name, value in scope.get("headers", ()):
        if name.lower() != b"content-type":
            continue
        media_type = value.decode("latin-1").split(";", 1)[0].strip().lower()
        return media_type == "multipart/form-data"
    return False


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name.lower() != b"content-length":
            continue
        try:
            declared = int(value)
        except ValueError:
            return None
        return declared if declared >= 0 else None
    return None


async def _disconnected() -> Message:
    return {"type": "http.disconnect"}


class UploadSizeLimitMiddleware:
    """Bound multipart bodies while leaving other request formats unchanged."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        advertised_bytes: int | None = None,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.advertised_bytes = advertised_bytes or max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method", "").upper() not in _BODY_METHODS
            or not _is_multipart(scope)
        ):
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            await self._refuse(scope, send, declared)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLarge(received)
            return message

        async def watching_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, watching_send)
        except BaseException as exc:
            if not _is_body_too_large(exc) or response_started:
                raise
            await self._refuse(scope, send, received)

    async def _refuse(self, scope: Scope, send: Send, size: int) -> None:
        logger.warning(
            "Multipart request body refused: %s %s carried %d bytes, over "
            "the %d-byte ceiling",
            scope.get("method", "?"),
            scope.get("path", "?"),
            size,
            self.max_body_bytes,
        )
        response = JSONResponse(
            status_code=413,
            content={
                "detail": (
                    "Multipart request body exceeds the maximum allowed size "
                    f"({self.advertised_bytes // (1024 * 1024)}MB)"
                )
            },
        )
        await response(scope, _disconnected, send)
