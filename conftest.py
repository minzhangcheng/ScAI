"""Repository-root pytest conftest.

This file lives OUTSIDE the ``backend`` package so its top-level code runs
before pytest imports ``backend`` (whose ``backend/config/_config.py`` resolves
``SCAI_DATA_DIR`` and creates the data directory tree at import time). Guarding
``SCAI_DATA_DIR`` here — not in ``backend/tests/conftest.py`` — guarantees a
writable data dir before that import chain runs.

Snapshot V2 restore/session/journal tests perform blocking ``asyncio.to_thread``
disk writes under ``SCAI_DATA_DIR``. A pre-set but *unwritable* ``SCAI_DATA_DIR``
(e.g. a read-only CI mount) otherwise makes ``backend`` import fail or those
writes hang. Honor a usable pre-set value; otherwise fall back to a writable
temp dir.
"""

import os
import sys
import tempfile

_DEFAULT_DATA_DIR = "/tmp/scai_test_data"

# Snapshot regression tests exercise the explicitly enabled lifecycle. Default
# disabled behavior is covered in fresh subprocesses and targeted gate tests.
os.environ.setdefault("SCAI_ENABLE_SNAPSHOT_V2", "true")


def _is_writable_data_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".scai_write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def _ensure_writable_data_dir() -> None:
    preset = os.environ.get("SCAI_DATA_DIR")
    if preset:
        if _is_writable_data_dir(preset):
            return
        fallback = tempfile.mkdtemp(prefix="scai_test_data_")
        print(
            f"[conftest] SCAI_DATA_DIR={preset!r} is not writable; "
            f"using isolated {fallback}",
            file=sys.stderr,
        )
        os.environ["SCAI_DATA_DIR"] = fallback
        return
    os.environ["SCAI_DATA_DIR"] = _DEFAULT_DATA_DIR
    if not _is_writable_data_dir(_DEFAULT_DATA_DIR):
        os.environ["SCAI_DATA_DIR"] = tempfile.mkdtemp(prefix="scai_test_data_")


_ensure_writable_data_dir()
