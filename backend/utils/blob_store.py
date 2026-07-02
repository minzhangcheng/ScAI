"""Externalize large base64/blob strings out of session JSON.

The session state embeds base64 images (figures, png_base64, image_base64,
all_candidates, vision message content). Serializing them inline blows the
session.json up to hundreds of MB and rewrites the whole file every save.

This module walks a JSON-serializable structure and replaces any string
longer than MIN_BLOB_LEN with a reference token "@scai-blob:<sha256>",
writing the content to <blob_dir>/<sha256>.b64. Identical content is
deduplicated by hash, so repeated snapshots share one file. inline_blobs()
performs the exact inverse, yielding a lossless round-trip.
"""
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BLOB_REF_PREFIX = "@scai-blob:"
MIN_BLOB_LEN = 1024  # chars; below this, inlining is cheap and not worth a file
_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def externalize_blobs(obj: Any, blob_dir: Path) -> Any:
    """Return a copy of obj with long strings replaced by blob references.

    Files are written under blob_dir lazily (only created if a blob exists).
    """
    blob_dir = Path(blob_dir)

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            if len(node) >= MIN_BLOB_LEN and not node.startswith(BLOB_REF_PREFIX):
                digest = _hash(node)
                blob_dir.mkdir(parents=True, exist_ok=True)
                path = blob_dir / f"{digest}.b64"
                if not path.exists():
                    tmp = blob_dir / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                    try:
                        tmp.write_text(node, encoding="utf-8")
                        os.replace(tmp, path)
                    except Exception:
                        tmp.unlink(missing_ok=True)  # don't leave orphaned temp files on failure
                        raise
                return f"{BLOB_REF_PREFIX}{digest}"
            return node
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    return _walk(obj)


def inline_blobs(obj: Any, blob_dir: Path) -> Any:
    """Return a copy of obj with blob references replaced by stored content.

    A missing blob file is left as its reference token (logged) so a partially
    corrupt session can still load rather than crashing.
    """
    blob_dir = Path(blob_dir)

    def _walk(node: Any) -> Any:
        if isinstance(node, str) and node.startswith(BLOB_REF_PREFIX):
            digest = node[len(BLOB_REF_PREFIX):]
            if not _DIGEST_RE.match(digest):
                raise ValueError(f"Invalid blob digest: {digest!r}")
            path = blob_dir / f"{digest}.b64"
            if path.exists():
                return path.read_text(encoding="utf-8")
            logger.warning("Blob %s referenced but file missing in %s", digest, blob_dir)
            return node
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    return _walk(obj)
