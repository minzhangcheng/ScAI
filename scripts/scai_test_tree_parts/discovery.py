"""Shared filesystem discovery and artifact primitives."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

class ContractError(ValueError):
    """A source, artifact, or outcome contract was not satisfied."""


@dataclass(frozen=True)
class ArtifactBytes:
    name: str
    payload: bytes
    size: int
    sha256: str


HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_STATES = {"modified", "untracked", "deleted", "added"}
SourceState = Literal["modified", "untracked", "deleted", "added"]


def _fail(message: str) -> None:
    raise ContractError(message)


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        _fail(
            "git command failed: "
            + " ".join(["git", *args])
            + ": "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout


def _relative_path(path: str) -> str:
    if not path or path.startswith("/") or "\x00" in path:
        _fail(f"invalid repository-relative path: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"non-canonical repository path: {path!r}")
    return path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_direct_children_nofollow(root: Path, expected_names: tuple[str, ...]) -> dict[str, ArtifactBytes]:
    """Read an exact direct-child file set once through no-follow descriptors."""
    if len(expected_names) != len(set(expected_names)) or any(
        not name or "/" in name or name in {".", ".."} for name in expected_names
    ):
        _fail("artifact child names are not exact direct names")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        _fail(f"artifact root is not a no-follow directory: {root}: {exc}")
    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            _fail(f"artifact root is not a directory: {root}")
        actual_names = set(os.listdir(root_fd))
        if actual_names != set(expected_names):
            _fail(f"artifact child allowlist mismatch: {sorted(actual_names)}")
        result: dict[str, ArtifactBytes] = {}
        for name in expected_names:
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=root_fd,
                )
            except OSError as exc:
                _fail(f"unsafe artifact child {name}: {exc}")
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    _fail(f"unsafe artifact child {name}: not a regular single-link file")
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                while True:
                    block = os.read(fd, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                    digest.update(block)
                payload = b"".join(chunks)
                result[name] = ArtifactBytes(name, payload, len(payload), digest.hexdigest())
            finally:
                os.close(fd)
        return result
    finally:
        os.close(root_fd)
