"""Root, frontend, and API metadata endpoints."""
import anyio
import os
import re
import stat
import weakref
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..constants import VERSION

router = APIRouter(tags=["meta"])
_HASHED_ASSET_RE = re.compile(
    r"^assets/.+-[A-Za-z0-9_-]{8,}\.[^/]+$"
)
_READ_ONLY_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_NO_FOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)


def _close_fd_quietly(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


class _OpenedFrontendFileResponse(FileResponse):
    """FileResponse whose bytes come from the descriptor validated up front."""

    def __init__(
        self,
        path: Path,
        fd: int,
        stat_result: os.stat_result,
        *,
        headers: dict[str, str],
    ) -> None:
        self._opened_fd = fd
        self._fd_closed = False
        self._fd_finalizer = weakref.finalize(self, _close_fd_quietly, fd)
        try:
            super().__init__(path, headers=headers, stat_result=stat_result)
            # Starlette versions differ on Range support.  This response does
            # not advertise it because accepting a Range through FileResponse
            # would hand control back to a path-based implementation.
            if "accept-ranges" in self.headers:
                del self.headers["accept-ranges"]
        except BaseException:
            self._close_opened_fd()
            raise

    def _close_opened_fd(self) -> None:
        if self._fd_closed:
            return
        self._fd_closed = True
        self._fd_finalizer.detach()
        _close_fd_quietly(self._opened_fd)

    async def _read_at(self, size: int, offset: int) -> bytes:
        return await anyio.to_thread.run_sync(
            os.pread, self._opened_fd, size, offset
        )

    async def __call__(self, scope, receive, send) -> None:
        """Serve the opened descriptor using only the public ASGI contract."""
        try:
            await send({
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.raw_headers,
            })
            if str(scope.get("method", "GET")).upper() == "HEAD":
                await send({
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                })
            else:
                file_size = int(self.stat_result.st_size)
                offset = 0
                while offset < file_size:
                    chunk = await self._read_at(
                        min(self.chunk_size, file_size - offset), offset
                    )
                    if not chunk:
                        break
                    offset += len(chunk)
                    await send({
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": offset < file_size,
                    })
                if offset == 0 or offset < file_size:
                    await send({
                        "type": "http.response.body",
                        "body": b"",
                        "more_body": False,
                    })
        finally:
            self._close_opened_fd()
        if self.background is not None:
            await self.background()


def _safe_relative_parts(relative: str) -> tuple[str, ...] | None:
    if "\x00" in relative:
        return None
    try:
        parts = tuple(Path(relative).parts)
    except (ValueError, OSError):
        return None
    if not parts or any(
        part in {"", ".", ".."} or "\x00" in part for part in parts
    ):
        return None
    return parts


def _open_regular_beneath(
    root: Path, parts: tuple[str, ...]
) -> tuple[int, os.stat_result] | None:
    """Open one regular file beneath *root* without following any symlink."""
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        if not root.is_absolute() or not root.anchor:
            return None
        root_fd = os.open(
            root.anchor,
            _READ_ONLY_FLAGS | _DIRECTORY_FLAG | _NO_FOLLOW_FLAG,
        )
        directory_fds.append(root_fd)
        parent_fd = root_fd
        # Open the configured build root from the filesystem root one
        # component at a time.  Holding each parent descriptor prevents a
        # rename/rebind of an already traversed ancestor from redirecting the
        # remainder of the lookup.
        for part in root.parts[1:]:
            parent_fd = os.open(
                part,
                _READ_ONLY_FLAGS | _DIRECTORY_FLAG | _NO_FOLLOW_FLAG,
                dir_fd=parent_fd,
            )
            directory_fds.append(parent_fd)
        for part in parts[:-1]:
            parent_fd = os.open(
                part,
                _READ_ONLY_FLAGS | _DIRECTORY_FLAG | _NO_FOLLOW_FLAG,
                dir_fd=parent_fd,
            )
            directory_fds.append(parent_fd)
        file_fd = os.open(
            parts[-1],
            _READ_ONLY_FLAGS | _NO_FOLLOW_FLAG,
            dir_fd=parent_fd,
        )
        stat_result = os.fstat(file_fd)
        if not stat.S_ISREG(stat_result.st_mode):
            return None
        opened_fd = file_fd
        file_fd = None
        return opened_fd, stat_result
    except (OSError, ValueError):
        return None
    finally:
        if file_fd is not None:
            _close_fd_quietly(file_fd)
        for directory_fd in reversed(directory_fds):
            _close_fd_quietly(directory_fd)


def _opened_frontend_response(
    dist: Path,
    parts: tuple[str, ...],
    display_path: Path,
    *,
    cache_control: str,
) -> FileResponse | None:
    opened = _open_regular_beneath(dist, parts)
    if opened is None:
        return None
    fd, stat_result = opened
    return _OpenedFrontendFileResponse(
        display_path,
        fd,
        stat_result,
        headers={"Cache-Control": cache_control},
    )


def _frontend_dist() -> Path:
    configured = os.getenv("SCAI_FRONTEND_DIST")
    if configured:
        return Path(os.path.abspath(configured))
    return (Path(__file__).resolve().parents[2] / "frontend" / "dist").resolve()


def frontend_file_response(path: str) -> FileResponse | None:
    """Resolve a built frontend file or an extensionless SPA route safely.

    API and WebSocket namespaces are never converted into the SPA index.  A
    missing filename with an extension is also a real 404, which prevents a
    JavaScript request from receiving HTML with status 200.  Every path
    component is opened relative to the build root without following symlinks,
    and the response streams that same validated descriptor.
    """
    relative = path.lstrip("/")
    first = relative.split("/", 1)[0]
    if first in {"api", "ws", "docs", "redoc", "openapi.json"}:
        return None

    dist = _frontend_dist()
    parts = _safe_relative_parts(relative) if relative else ()
    if relative and parts is None:
        return None

    if relative:
        cache_control = (
            "public, max-age=31536000, immutable"
            if _HASHED_ASSET_RE.match(relative)
            else "no-cache"
        )
        response = _opened_frontend_response(
            dist, parts, dist.joinpath(*parts), cache_control=cache_control
        )
        if response is not None:
            return response
    if relative and Path(relative).suffix:
        return None
    return _opened_frontend_response(
        dist,
        ("index.html",),
        dist / "index.html",
        cache_control="no-cache",
    )


@router.api_route("/", methods=["GET", "HEAD"])
async def root():
    frontend = frontend_file_response("")
    if frontend is not None:
        return frontend
    return {"message": "ScAI - Single Cell Sequencing Analysis Driven by AI", "version": VERSION, "docs": "/docs"}


@router.get("/api")
async def api_root():
    return {"message": "ScAI API", "version": VERSION}


@router.get("/api/health")
async def health():
    """Unauthenticated liveness probe for container/orchestrator health checks.

    Deliberately requires no auth so it keeps returning 200 after admin setup
    (unlike /api/config/status, which becomes protected once initialized).
    """
    return {"status": "ok", "version": VERSION}
