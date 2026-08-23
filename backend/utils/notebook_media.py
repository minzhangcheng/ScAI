"""Descriptor-pinned notebook media storage and standalone rehydration.

The notebook is host-owned state.  Large encoded image bodies are therefore
published once into a sibling sidecar directory and notebooks carry only a
small, content-addressed reference.  Every sidecar operation is relative to a
pinned directory descriptor so a later path replacement cannot redirect the
operation outside the session workspace.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import os
import secrets
import stat
import string
from pathlib import Path
from typing import Any, Mapping

import nbformat


NOTEBOOK_EXTERNAL_IMAGE_MIME = "application/vnd.scai.external-image+json"
_ALLOWED_MEDIA = {"image/png": ".png", "image/jpeg": ".jpg"}
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_INLINE_IMAGE_LIMIT_BYTES = 64 * 1024
_BASE64_BLOCK_CHARS = 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
EXTERNAL_IMAGE_PLACEHOLDER = "[external image unavailable]"


def _expected_basename(reference: Mapping[str, Any]) -> str:
    """Validate a raw reference path and derive its only permitted basename."""
    digest = str(reference.get("sha256") or "")
    media_type = str(reference.get("media_type") or "")
    raw_path = str(reference.get("path") or "")
    if (
        not raw_path
        or os.path.isabs(raw_path)
        or raw_path in {".", ".."}
        or "/" in raw_path
        or "\\" in raw_path
        or raw_path != os.path.basename(raw_path)
    ):
        raise ValueError("notebook media must be one raw direct-child name")
    if len(digest) != 64 or any(ch not in string.hexdigits for ch in digest):
        raise ValueError("invalid notebook media digest")
    suffix = _ALLOWED_MEDIA.get(media_type)
    if suffix is None:
        raise ValueError("unsupported notebook media type")
    if raw_path != f"{digest}{suffix}":
        raise ValueError("notebook media basename does not match digest")
    return raw_path


def _declared_size(reference: Mapping[str, Any]) -> int:
    value = reference.get("size")
    if isinstance(value, bool):
        raise ValueError("invalid notebook media size")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid notebook media size") from exc
    if size < 0:
        raise ValueError("invalid notebook media size")
    return size


def _workspace_for_source(source: Path) -> Path:
    """Resolve the fixed workspace sibling for a notebook or workspace path."""
    source = Path(source)
    if source.name == "workspace":
        return source
    # The retained layout is <session>/states/<session>.ipynb.
    if source.parent.name == "states":
        return source.parent.parent / "workspace"
    # Keep the helper useful to callers that pass a states directory.
    if source.name == "states":
        return source.parent / "workspace"
    return source.parent / "workspace"


def _pin_workspace_and_media(workspace: Path) -> tuple[int, int]:
    """Pin workspace and its sidecar directory without following final links."""
    workspace_fd = os.open(str(workspace), _DIRECTORY_FLAGS)
    try:
        try:
            os.mkdir(".notebook_media", 0o700, dir_fd=workspace_fd)
        except FileExistsError:
            pass
        media_fd = os.open(".notebook_media", _DIRECTORY_FLAGS, dir_fd=workspace_fd)
    except BaseException:
        os.close(workspace_fd)
        raise
    return workspace_fd, media_fd


def _regular_stat(media_fd: int, name: str) -> os.stat_result:
    info = os.stat(name, dir_fd=media_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("notebook media is not a regular file")
    return info


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _read_verified_from_fd(
    media_fd: int,
    name: str,
    reference: Mapping[str, Any],
) -> bytes:
    declared_size = _declared_size(reference)
    declared_digest = str(reference.get("sha256") or "").lower()
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=media_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != declared_size:
            raise ValueError("notebook media size or file type mismatch")
        digest = hashlib.sha256()
        parts: list[bytes] = []
        total = 0
        while True:
            block = os.read(fd, _COPY_CHUNK_BYTES)
            if not block:
                break
            total += len(block)
            if total > declared_size:
                raise ValueError("notebook media grew while being read")
            digest.update(block)
            parts.append(block)
        after = os.fstat(fd)
        if (
            _stat_identity(before) != _stat_identity(after)
            or total != declared_size
            or digest.hexdigest() != declared_digest
        ):
            raise ValueError("notebook media digest or identity mismatch")
        return b"".join(parts)
    finally:
        os.close(fd)


def read_verified_notebook_media(
    source: Path, reference: Mapping[str, Any]
) -> tuple[str, bytes]:
    """Read one verified direct-child sidecar and return ``(media_type, bytes)``."""
    name = _expected_basename(reference)
    workspace = _workspace_for_source(Path(source))
    workspace_fd, media_fd = _pin_workspace_and_media(workspace)
    try:
        _regular_stat(media_fd, name)
        raw = _read_verified_from_fd(media_fd, name, reference)
        return str(reference["media_type"]), raw
    finally:
        os.close(media_fd)
        os.close(workspace_fd)


def is_verified_notebook_media_reference(
    source: Path, reference: Mapping[str, Any]
) -> bool:
    try:
        read_verified_notebook_media(source, reference)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True


def _iter_decoded_base64(encoded: str):
    if not isinstance(encoded, str):
        raise ValueError("notebook image is not text")
    try:
        encoded.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("notebook image is not ASCII base64") from exc
    carry = ""
    for start in range(0, len(encoded), _BASE64_BLOCK_CHARS):
        block = carry + encoded[start : start + _BASE64_BLOCK_CHARS]
        aligned = len(block) - (len(block) % 4)
        if aligned:
            chunk = block[:aligned]
            carry = block[aligned:]
            try:
                decoded = base64.b64decode(chunk, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("invalid notebook image base64") from exc
            if decoded:
                yield decoded
        else:
            carry = block
    if carry:
        try:
            decoded = base64.b64decode(carry, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid notebook image base64") from exc
        if decoded:
            yield decoded


def externalize_notebook_image(
    workspace: Path,
    notebook_path: Path,
    encoded: Any,
    *,
    media_type: str,
) -> dict[str, Any] | None:
    """Publish one large encoded image as a verified content-addressed sidecar."""
    suffix = _ALLOWED_MEDIA.get(media_type)
    if suffix is None or not isinstance(encoded, str):
        return None
    workspace_fd = media_fd = None
    temp_name = None
    try:
        workspace_fd, media_fd = _pin_workspace_and_media(Path(workspace))
        temp_name = f".notebook-image-{secrets.token_hex(16)}.tmp"
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=media_fd,
        )
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb", closefd=True) as destination:
                for decoded in _iter_decoded_base64(encoded):
                    size += len(decoded)
                    digest.update(decoded)
                    destination.write(decoded)
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            # fdopen closes the descriptor on all normal/error exits.
            raise
        digest_hex = digest.hexdigest()
        target_name = f"{digest_hex}{suffix}"
        reference = {
            "path": target_name,
            "sha256": digest_hex,
            "size": size,
            "media_type": media_type,
        }
        try:
            existing = _regular_stat(media_fd, target_name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            # A same-name object is reusable only after the full verifier.
            _read_verified_from_fd(media_fd, target_name, reference)
            os.unlink(temp_name, dir_fd=media_fd)
            temp_name = None
            return reference
        os.replace(
            temp_name,
            target_name,
            src_dir_fd=media_fd,
            dst_dir_fd=media_fd,
        )
        temp_name = None
        # Verify the published object through the pinned descriptor before
        # handing a reference to notebook state.
        _regular_stat(media_fd, target_name)
        _read_verified_from_fd(media_fd, target_name, reference)
        return reference
    except (OSError, ValueError, TypeError, binascii.Error):
        return None
    finally:
        if temp_name is not None and media_fd is not None:
            try:
                os.unlink(temp_name, dir_fd=media_fd)
            except OSError:
                pass
        if media_fd is not None:
            os.close(media_fd)
        if workspace_fd is not None:
            os.close(workspace_fd)


def _rehydrate_output_mapping(
    value: Any,
    source: Path,
) -> tuple[Any, int]:
    """Copy a notebook object and inline verified private image references."""
    if isinstance(value, dict):
        copied: dict[Any, Any] = {}
        count = 0
        has_external = NOTEBOOK_EXTERNAL_IMAGE_MIME in value
        external = value.get(NOTEBOOK_EXTERNAL_IMAGE_MIME)
        for key, item in value.items():
            if key == NOTEBOOK_EXTERNAL_IMAGE_MIME:
                continue
            copied[key], child_count = _rehydrate_output_mapping(item, source)
            count += child_count
        if has_external:
            references = external if isinstance(external, list) else [external]
            unresolved: list[Any] = []
            for reference in references:
                if not isinstance(reference, Mapping):
                    unresolved.append(copy.deepcopy(reference))
                    continue
                try:
                    media_type, raw = read_verified_notebook_media(source, reference)
                except (OSError, ValueError, TypeError, KeyError):
                    unresolved.append(copy.deepcopy(reference))
                    continue
                copied[media_type] = base64.b64encode(raw).decode("ascii")
                count += 1
            if unresolved:
                copied[NOTEBOOK_EXTERNAL_IMAGE_MIME] = (
                    unresolved[0] if len(unresolved) == 1 else unresolved
                )
                plain = copied.get("text/plain")
                if not (
                    isinstance(plain, str)
                    and "unavailable" in plain.casefold()
                ):
                    copied["text/plain"] = EXTERNAL_IMAGE_PLACEHOLDER
            return copied, count
        return copied, count
    if isinstance(value, list):
        copied = []
        count = 0
        for item in value:
            child, child_count = _rehydrate_output_mapping(item, source)
            copied.append(child)
            count += child_count
        return copied, count
    return copy.deepcopy(value), 0


def export_notebook_with_inline_media(source: Path, destination: Path) -> int:
    """Write an atomic, standalone notebook with only verified media inlined."""
    source = Path(source)
    destination = Path(destination)
    with source.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    copied, count = _rehydrate_output_mapping(notebook, source)
    copied = nbformat.from_dict(copied)
    nbformat.validate(copied)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            nbformat.write(copied, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count
