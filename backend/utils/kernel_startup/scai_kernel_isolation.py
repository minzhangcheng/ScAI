"""Early, process-level filesystem write isolation for Jupyter kernels.

This module is deliberately standalone: ``sitecustomize`` imports it before
IPython, the ScAI backend package, or scientific libraries are loaded.  On
Linux, Landlock provides the native-I/O boundary; the CPython audit hook adds
coverage for metadata operations that Landlock does not currently mediate and
provides a best-effort fallback on other platforms.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import sys
from typing import Iterable


WRITE_ROOTS_ENV = "SCAI_KERNEL_WRITE_ROOTS"
REQUIRE_LANDLOCK_ENV = "SCAI_KERNEL_REQUIRE_LANDLOCK"
LANDLOCK_ACTIVE_ENV = "SCAI_KERNEL_LANDLOCK_ACTIVE"

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_REMOVE_DIR = 1 << 4
_ACCESS_FS_REMOVE_FILE = 1 << 5
_ACCESS_FS_MAKE_CHAR = 1 << 6
_ACCESS_FS_MAKE_DIR = 1 << 7
_ACCESS_FS_MAKE_REG = 1 << 8
_ACCESS_FS_MAKE_SOCK = 1 << 9
_ACCESS_FS_MAKE_FIFO = 1 << 10
_ACCESS_FS_MAKE_BLOCK = 1 << 11
_ACCESS_FS_MAKE_SYM = 1 << 12
_ACCESS_FS_REFER = 1 << 13
_ACCESS_FS_TRUNCATE = 1 << 14

_BASE_WRITE_ACCESS = (
    _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_CHAR
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_MAKE_SOCK
    | _ACCESS_FS_MAKE_FIFO
    | _ACCESS_FS_MAKE_BLOCK
    | _ACCESS_FS_MAKE_SYM
)

_PROCESS_LAUNCH_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.exec",
        "os.spawn",
        "pty.spawn",
    }
)


class KernelIsolationError(RuntimeError):
    """The kernel process could not establish its required write boundary."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _syscall_numbers() -> tuple[int, int, int]:
    """Return Landlock syscall numbers for Linux's generic syscall table."""
    machine = os.uname().machine.lower()
    supported = {
        "aarch64", "arm64", "i386", "i486", "i586", "i686",
        "riscv64", "s390x", "x86_64", "amd64",
    }
    if machine not in supported:
        raise KernelIsolationError(
            f"unsupported architecture for Landlock syscalls: {machine}"
        )
    return 444, 445, 446


def _raise_errno(operation: str) -> None:
    error_number = ctypes.get_errno()
    raise KernelIsolationError(
        f"{operation} failed: {os.strerror(error_number)} ({error_number})"
    )


def _canonical_roots(roots: Iterable[str]) -> tuple[str, ...]:
    canonical = []
    for raw in roots:
        if not isinstance(raw, str) or not raw:
            raise KernelIsolationError("write roots must be non-empty strings")
        raw_path = Path(raw)
        if raw_path.is_symlink():
            raise KernelIsolationError(f"write root must not be a symlink: {raw}")
        resolved = str(raw_path.resolve(strict=True))
        if not Path(resolved).is_dir():
            raise KernelIsolationError(f"write root is not a directory: {raw}")
        if resolved == os.path.abspath(os.sep):
            raise KernelIsolationError("filesystem root cannot be a write root")
        if resolved not in canonical:
            canonical.append(resolved)
    if not canonical:
        raise KernelIsolationError("at least one write root is required")
    return tuple(canonical)


def _path_from_fd(fd: int) -> str | None:
    if sys.platform != "linux":
        return None
    try:
        return os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        return None


def _resolve_path(target, dir_fd=None) -> str | None:
    if isinstance(target, int):
        return _path_from_fd(target)
    try:
        raw = os.fspath(target)
    except TypeError:
        return None
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not os.path.isabs(raw) and dir_fd not in (
        None,
        -1,
        getattr(os, "AT_FDCWD", -100),
    ):
        base = _path_from_fd(int(dir_fd))
        if base is None:
            return None
        raw = os.path.join(base, raw)
    return os.path.realpath(os.path.abspath(raw))


def _is_allowed(path: str | None, roots: tuple[str, ...]) -> bool:
    if path is None:
        return False
    for root in roots:
        try:
            if os.path.commonpath((path, root)) == root:
                return True
        except (ValueError, OSError):
            continue
    return path == os.devnull


def _flags_request_write(flags) -> bool:
    if not isinstance(flags, int):
        return False
    access_mode = flags & getattr(os, "O_ACCMODE", 3)
    if access_mode != getattr(os, "O_RDONLY", 0):
        return True
    mutation_flags = (
        getattr(os, "O_CREAT", 0)
        | getattr(os, "O_TRUNC", 0)
        | getattr(os, "O_APPEND", 0)
        | getattr(os, "O_TMPFILE", 0)
    )
    return bool(flags & mutation_flags)


def _mode_requests_write(mode) -> bool:
    return isinstance(mode, str) and any(flag in mode for flag in "wax+")


def install_audit_guard(roots: Iterable[str]) -> tuple[str, ...]:
    """Install an irreversible CPython audit guard for write-like operations."""
    allowed_roots = _canonical_roots(roots)

    def require_allowed(target, *, dir_fd=None, operation="write"):
        resolved = _resolve_path(target, dir_fd)
        if not _is_allowed(resolved, allowed_roots):
            raise PermissionError(
                errno.EACCES,
                f"[Sandbox] {operation} blocked outside this session",
                os.fspath(target) if not isinstance(target, int) else target,
            )

    def audit(event, args):
        if event == "open" and len(args) >= 3:
            path, mode, flags = args[:3]
            if _mode_requests_write(mode) or _flags_request_write(flags):
                require_allowed(path, operation="write")
            return

        single_target_events = {
            "os.remove": (0, 1),
            "os.rmdir": (0, 1),
            "os.mkdir": (0, 2),
            "os.chmod": (0, 2),
            "os.chown": (0, 3),
            "os.utime": (0, 3),
            "os.truncate": (0, None),
            "os.setxattr": (0, None),
            "os.removexattr": (0, None),
            "os.chflags": (0, None),
        }
        if event in single_target_events:
            path_index, dir_fd_index = single_target_events[event]
            dir_fd = (
                args[dir_fd_index]
                if dir_fd_index is not None and len(args) > dir_fd_index
                else None
            )
            require_allowed(args[path_index], dir_fd=dir_fd, operation=event)
            return

        if event in {"os.rename", "os.link"} and len(args) >= 2:
            src_dir_fd = args[2] if len(args) > 2 else None
            dst_dir_fd = args[3] if len(args) > 3 else None
            require_allowed(args[0], dir_fd=src_dir_fd, operation=event)
            require_allowed(args[1], dir_fd=dst_dir_fd, operation=event)
            return

        if event == "os.symlink" and len(args) >= 2:
            dir_fd = args[2] if len(args) > 2 else None
            require_allowed(args[1], dir_fd=dir_fd, operation=event)
            return

        if event in {"shutil.copymode", "shutil.copystat"} and len(args) >= 2:
            require_allowed(args[1], operation=event)
            return

    sys.addaudithook(audit)
    return allowed_roots


def install_process_guard() -> None:
    """Irreversibly block process launch after the trusted launcher has execed.

    The first-stage launcher must perform exactly one ``os.execvpe`` to enter
    ipykernel, so this hook belongs in the second interpreter's trusted
    ``sitecustomize``. It is installed before IPython, workspace code, or
    scientific libraries are imported.
    """
    blocked_events = _PROCESS_LAUNCH_EVENTS
    permission_error = PermissionError
    access_denied = errno.EACCES

    def audit(event, args):
        if event in blocked_events:
            raise permission_error(
                access_denied,
                "[Sandbox] process launching is blocked in kernels; "
                "use a thread-based library backend",
            )

    sys.addaudithook(audit)


def apply_landlock(roots: Iterable[str]) -> int:
    """Restrict native filesystem content mutations to ``roots`` on Linux."""
    if sys.platform != "linux":
        raise KernelIsolationError("Landlock is only available on Linux")

    allowed_roots = _canonical_roots(roots)
    create_nr, add_nr, restrict_nr = _syscall_numbers()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    abi = int(
        libc.syscall(
            create_nr,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if abi < 0:
        _raise_errno("Landlock ABI query")
    if abi < 3:
        raise KernelIsolationError(
            f"Landlock ABI {abi} cannot restrict file truncation; ABI 3+ required"
        )

    handled_access = _BASE_WRITE_ACCESS | _ACCESS_FS_REFER | _ACCESS_FS_TRUNCATE
    ruleset_attr = _RulesetAttr(handled_access_fs=handled_access)
    ruleset_fd = int(
        libc.syscall(
            create_nr,
            ctypes.byref(ruleset_attr),
            ctypes.sizeof(ruleset_attr),
            ctypes.c_uint32(0),
        )
    )
    if ruleset_fd < 0:
        _raise_errno("Landlock ruleset creation")

    opened_fds: list[int] = []
    try:
        for root in allowed_roots:
            root_fd = os.open(
                root,
                getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0),
            )
            opened_fds.append(root_fd)
            path_attr = _PathBeneathAttr(
                allowed_access=handled_access,
                parent_fd=root_fd,
            )
            if libc.syscall(
                add_nr,
                ruleset_fd,
                ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(path_attr),
                ctypes.c_uint32(0),
            ) < 0:
                _raise_errno(f"Landlock rule for {root}")

        null_fd = os.open(
            os.devnull,
            getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0),
        )
        opened_fds.append(null_fd)
        null_attr = _PathBeneathAttr(
            allowed_access=_ACCESS_FS_WRITE_FILE | _ACCESS_FS_TRUNCATE,
            parent_fd=null_fd,
        )
        if libc.syscall(
            add_nr,
            ruleset_fd,
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(null_attr),
            ctypes.c_uint32(0),
        ) < 0:
            _raise_errno("Landlock rule for os.devnull")

        if libc.prctl(
            _PR_SET_NO_NEW_PRIVS,
            ctypes.c_ulong(1),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        ) != 0:
            _raise_errno("prctl(PR_SET_NO_NEW_PRIVS)")
        if libc.syscall(
            restrict_nr,
            ruleset_fd,
            ctypes.c_uint32(0),
        ) < 0:
            _raise_errno("Landlock restriction")
    finally:
        for fd in opened_fds:
            os.close(fd)
        os.close(ruleset_fd)

    os.environ[LANDLOCK_ACTIVE_ENV] = str(abi)
    return abi


def activate_from_environment() -> None:
    """Install the configured kernel boundary or terminate fail-closed."""
    raw_roots = os.environ.get(WRITE_ROOTS_ENV)
    if not raw_roots:
        return
    required = os.environ.get(REQUIRE_LANDLOCK_ENV) == "1"
    try:
        roots = json.loads(raw_roots)
        if not isinstance(roots, list):
            raise KernelIsolationError("write-root payload must be a JSON list")
        canonical_roots = install_audit_guard(roots)
        if sys.platform == "linux":
            apply_landlock(canonical_roots)
        elif required:
            raise KernelIsolationError("required Landlock boundary is unavailable")
    except BaseException as exc:
        message = f"[ScAI kernel isolation] startup failed: {exc}\n"
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
        finally:
            if required:
                os._exit(78)
        return
