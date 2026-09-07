# scripts/fetch_celltypist_models.py
"""Download CellTypist models into the runtime cache the kernels read from.

Destination is ``$SCAI_DATA_DIR/celltypist`` (see ``default_dest_dir``), which
the script exports as ``CELLTYPIST_FOLDER`` before importing celltypist, so the
package writes ``data/models/*.pkl`` and ``data/models/models.json`` there.
``backend/services/celltypist_models.py`` reads the same directory and links it
into every kernel's session folder; the cache is never inside the code tree.

By default every model in the CellTypist index is fetched (each is a few MB);
``--models`` restricts the download to named files. Files already present are
skipped unless ``--force`` is given, so the startup hook can call this
unconditionally. Failures are logged and reported through the exit status; a
missing cache only means kernels fall back to on-demand downloads.

CLI: python scripts/fetch_celltypist_models.py [--dest DIR] [--models A.pkl B.pkl] [--force]
Backend entry point: fetch(dest_dir, models=None, force=False) -> bool.

Models come from https://www.celltypist.org/models; the index records each
model's source and description. Terms of use are those published there.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: Written next to the cache so the provenance travels with the data.
NOTICE_TEXT = (
    "CellTypist models downloaded from https://www.celltypist.org/models\n"
    "(index: https://celltypist.cog.sanger.ac.uk/models/models.json).\n"
    "Each entry in data/models/models.json records the model's source and\n"
    "description; terms of use are those published by CellTypist and by the\n"
    "studies each model was trained on. Not redistributed with ScAI.\n"
)

_STALE_LOCK_SECONDS = 30 * 60


def default_dest_dir() -> Path:
    """``$SCAI_DATA_DIR/celltypist``, or ``<repo>/../data/celltypist`` when unset.

    Mirrors ``backend.config.DATA_BASE_DIR`` exactly. This module must stay
    importable without the backend package, so the rule is duplicated rather
    than imported, and a test pins the two together.
    """
    repo_root = Path(__file__).resolve().parent.parent
    base = Path(os.environ.get(
        "SCAI_DATA_DIR", str((repo_root.parent / "data").resolve())
    )).expanduser().resolve()
    return base / "celltypist"


def _acquire_lock(lock: Path):
    """O_CREAT|O_EXCL file lock; a stale one (>30 min) is taken over."""
    try:
        return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
            if age > _STALE_LOCK_SECONDS:
                logger.warning(f"stale celltypist fetch lock ({age:.0f}s); reclaiming")
                lock_tmp = lock.with_suffix(lock.suffix + ".tmp")
                fd = os.open(lock_tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.replace(lock_tmp, lock)
                return fd
        except OSError:
            pass
        return None


def fetch(dest_dir: Optional[Path] = None,
          models: Optional[Iterable[str]] = None,
          force: bool = False) -> bool:
    """Download the index and the requested models into ``dest_dir``.

    ``CELLTYPIST_FOLDER`` is exported before celltypist is imported because the
    package fixes its paths at import time. Returns False on failure rather than
    raising; the reason is in the log.
    """
    dest = Path(dest_dir) if dest_dir else default_dest_dir()
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["CELLTYPIST_FOLDER"] = str(dest)
    lock = dest / ".fetch.lock"
    fd = _acquire_lock(lock)
    if fd is None:
        logger.warning("celltypist fetch already in progress (lock present); skipping")
        return False
    try:
        from celltypist import models as ct_models  # noqa: WPS433 -- after the env export
        model_list = list(models) if models else None
        ct_models.download_models(force_update=force, model=model_list)
        models_dir = Path(ct_models.models_path)
        present = sorted(p.name for p in models_dir.glob("*.pkl"))
        if not present:
            logger.warning(f"celltypist fetch left no model under {models_dir}")
            return False
        (dest / "NOTICE").write_text(NOTICE_TEXT)
        logger.info(f"celltypist cache at {models_dir}: {len(present)} model(s)")
        return True
    except Exception as e:
        logger.warning(f"celltypist fetch failed: {type(e).__name__}: {e}")
        return False
    finally:
        os.close(fd)
        try:
            os.unlink(lock)
        except OSError:
            pass


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=None,
                    help="CELLTYPIST_FOLDER to fill (default: $SCAI_DATA_DIR/celltypist)")
    ap.add_argument("--models", nargs="+", default=None, metavar="FILE.pkl",
                    help="model files to fetch (default: every model in the index)")
    ap.add_argument("--force", action="store_true",
                    help="refresh the index and re-download models already present")
    ns = ap.parse_args()
    dest = ns.dest or default_dest_dir()
    logger.info(f"celltypist cache destination: {dest}")
    raise SystemExit(0 if fetch(dest, models=ns.models, force=ns.force) else 1)
