"""Safe artifact and filesystem operations for frontend test evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ArtifactBytes:
    name: str
    payload: bytes
    size: int
    sha256: str


class FrontendError(RuntimeError):
    def __init__(self, label: str, message: str):
        super().__init__(message)
        self.label = label


def fail(label: str, message: str) -> None:
    raise FrontendError(label, message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
def read_fd_bytes(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def read_one_nofollow(path: Path) -> ArtifactBytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        fail("frontend evidence", f"cannot open artifact {path}: {exc}")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            fail("frontend evidence", f"unsafe artifact: {path}")
        payload = read_fd_bytes(fd)
        return ArtifactBytes(path.name, payload, len(payload), sha256_bytes(payload))
    finally:
        os.close(fd)

def read_direct_children_nofollow(root: Path, expected_names: set[str]) -> dict[str, ArtifactBytes]:
    if any(not name or "/" in name or name in {".", ".."} for name in expected_names):
        fail("frontend evidence", "artifact registration contains a non-direct name")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        fail("frontend evidence", f"result directory is not a safe directory: {exc}")
    try:
        actual_names = set(os.listdir(root_fd))
        if actual_names != expected_names:
            fail("frontend evidence", f"result artifact allowlist mismatch: {sorted(actual_names)}")
        result: dict[str, ArtifactBytes] = {}
        for name in sorted(expected_names):
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=root_fd,
                )
            except OSError as exc:
                fail("frontend evidence", f"cannot open result artifact {name}: {exc}")
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    fail("frontend evidence", f"unsafe result artifact: {name}")
                payload = read_fd_bytes(fd)
                result[name] = ArtifactBytes(name, payload, len(payload), sha256_bytes(payload))
            finally:
                os.close(fd)
        return result
    finally:
        os.close(root_fd)


def write_sealed_file(result_dir: Path, name: str, payload: bytes) -> ArtifactBytes:
    if not name or "/" in name or name in {".", ".."}:
        fail("frontend evidence", f"invalid sealed artifact name: {name!r}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(result_dir / name, flags, 0o644)
    except OSError as exc:
        fail("frontend evidence", f"cannot create sealed artifact {name}: {exc}")
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fchmod(fd, 0o644)
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(result_dir)
    loaded = read_one_nofollow(result_dir / name)
    if loaded.payload != payload or loaded.sha256 != sha256_bytes(payload):
        fail("frontend evidence", f"sealed artifact changed after write: {name}")
    return loaded


def lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def require_regular(path: Path, label: str, *, allow_symlink: bool = False) -> None:
    info = lstat_or_none(path)
    if info is None:
        fail(label, f"missing file: {path}")
    if stat.S_ISLNK(info.st_mode) and not allow_symlink:
        fail(label, f"symlink is not allowed: {path}")
    if not stat.S_ISREG(info.st_mode):
        fail(label, f"not a regular file: {path}")
    if info.st_nlink != 1:
        fail(label, f"hardlink is not allowed: {path}")


def require_directory(path: Path, label: str) -> None:
    info = lstat_or_none(path)
    if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        fail(label, f"directory is missing or unsafe: {path}")


def ensure_directory(path: Path, label: str, mode: int = 0o700) -> None:
    info = lstat_or_none(path)
    if info is None:
        try:
            path.mkdir(mode=mode)
        except FileExistsError:
            info = lstat_or_none(path)
        else:
            return
    if info is None or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        fail(label, f"unsafe directory: {path}")
    os.chmod(path, mode)


def path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def safe_lock(path: Path, label: str) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        fail(label, f"cannot open lock {path}: {exc}")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        fail(label, f"unsafe lock file: {path}")
    os.fchmod(fd, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError as exc:
        os.close(fd)
        fail(label, f"cannot acquire lock {path}: {exc}")
    return fd


def unlock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
