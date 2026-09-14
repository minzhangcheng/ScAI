"""Host-owned paths and validation helpers for AnnData checkpoints."""
from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from backend.config import get_session_adata_checkpoints_dir, get_session_dir


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOB_RELATIVE_PREFIX = "adata-checkpoints/blobs/"


@dataclass(frozen=True)
class BlobIdentity:
    sha256: str
    size: int
    relative_path: str
    absolute_path: Path

    @property
    def reused_digest(self) -> str:
        return self.sha256


def checkpoint_root(session_id: str) -> Path:
    """Return the Host-owned checkpoint root for one session."""
    return Path(get_session_adata_checkpoints_dir(str(session_id)))


def blob_dir(session_id: str, *, create: bool = False) -> Path:
    path = checkpoint_root(session_id) / "blobs"
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def staging_dir(session_id: str, *, create: bool = False) -> Path:
    path = checkpoint_root(session_id) / ".staging"
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def blob_relative_path(sha256: str) -> str:
    digest = validate_sha256(sha256)
    return f"{_BLOB_RELATIVE_PREFIX}{digest}.h5ad"


def blob_path(session_id: str, sha256: str, *, create_parent: bool = False) -> Path:
    return blob_dir(session_id, create=create_parent) / f"{validate_sha256(sha256)}.h5ad"


def validate_sha256(value: object) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError("expected a lowercase sha256 hex digest")
    return text


def canonical_blob_relative_path(value: object) -> str:
    text = str(value or "").strip()
    if not text or text != text.strip() or "\\" in text or "\x00" in text:
        raise ValueError("checkpoint blob path is not canonical")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError("checkpoint blob path escapes its session root")
    parts = pure.parts
    if len(parts) != 3 or parts[0] != "adata-checkpoints" or parts[1] != "blobs":
        raise ValueError("checkpoint blob path is outside the checkpoint layout")
    name = parts[2]
    if not name.endswith(".h5ad"):
        raise ValueError("checkpoint blob is not an h5ad file")
    digest = validate_sha256(name[: -len(".h5ad")])
    if text != blob_relative_path(digest):
        raise ValueError("checkpoint blob path does not match its digest")
    return text


def resolve_session_relative(session_id: str, relative_path: object) -> Path:
    """Resolve one validated checkpoint-relative path under *session_id*."""
    relative = canonical_blob_relative_path(relative_path)
    root = Path(get_session_dir(str(session_id))).resolve()
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("checkpoint path is outside its session root") from exc
    return candidate


def regular_file_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    """Return a stable identity tuple for a regular, non-symlink file."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(str(path), flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("checkpoint path is not a regular file")
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )
    finally:
        os.close(fd)


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(str(path), flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("checkpoint candidate is not a regular file")
        while True:
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def verify_blob(path: Path, *, expected_sha256: str, expected_size: Optional[int]) -> BlobIdentity:
    digest = validate_sha256(expected_sha256)
    if not path.is_file():
        raise FileNotFoundError("checkpoint blob is missing")
    size = path.stat().st_size
    if expected_size is not None and size != int(expected_size):
        raise ValueError("checkpoint blob size does not match its record")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError("checkpoint blob digest does not match its record")
    return BlobIdentity(
        sha256=actual,
        size=size,
        relative_path=canonical_blob_relative_path(
            f"{_BLOB_RELATIVE_PREFIX}{actual}.h5ad"
        ),
        absolute_path=path,
    )
