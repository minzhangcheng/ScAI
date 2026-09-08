"""CellTypist model cache: fetched once by the host, read by every kernel.

CellTypist resolves its model directory from ``CELLTYPIST_FOLDER`` (default
``~/.celltypist``) when it is imported and creates ``data/models`` under it, so
the folder has to sit inside one of the kernel's write roots or the import
itself is refused by the sandbox. The kernel launcher therefore points
``CELLTYPIST_FOLDER`` at ``<session_tmp>/celltypist`` and links every model
file the host has cached under ``$SCAI_DATA_DIR/celltypist`` into that view.

The cache lives on the data volume, outside the code tree, for the reason the
marker tables do (``marker_priors.runtime_markers_dir``): a file inside the
repository would be frozen into the image. The per-session folder is a view:
a kernel that downloads a model of its own writes a real file next to the
links and never touches the shared cache, and a link whose target has gone is
dropped at the next launch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

#: Directory under the data root that the host uses as ``CELLTYPIST_FOLDER``.
CACHE_DIRNAME = "celltypist"
#: CellTypist's own layout below ``CELLTYPIST_FOLDER``.
MODELS_SUBDIR = Path("data") / "models"
#: The model index CellTypist keeps next to the model files.
INDEX_FILENAME = "models.json"
MODEL_SUFFIX = ".pkl"
#: Name of the per-session view under the kernel's temporary root.
SESSION_VIEW_DIRNAME = "celltypist"


def runtime_cache_root() -> Path:
    """``$SCAI_DATA_DIR/celltypist`` -- the host's ``CELLTYPIST_FOLDER``."""
    from backend.config import DATA_BASE_DIR
    return Path(DATA_BASE_DIR) / CACHE_DIRNAME


def cached_models_dir() -> Path:
    """Where the cached ``*.pkl`` files and ``models.json`` are."""
    return runtime_cache_root() / MODELS_SUBDIR


def cached_model_files() -> List[Path]:
    """Model files and the index in the shared cache, sorted by name.

    Anything else in the directory (lock files, notices) is not a model and is
    not exposed to kernels. An absent or unreadable cache is an empty list.
    """
    root = cached_models_dir()
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    return [
        p for p in entries
        if p.is_file() and (p.suffix == MODEL_SUFFIX or p.name == INDEX_FILENAME)
    ]


def cached_model_names() -> List[str]:
    """File names of the cached models, without the index."""
    return [p.name for p in cached_model_files() if p.suffix == MODEL_SUFFIX]


def link_cache_into(folder: Path) -> List[str]:
    """Expose the shared cache inside ``folder``, the kernel's CELLTYPIST_FOLDER.

    Creates ``folder/data/models``, removes links left by an earlier launch,
    and symlinks every cached model file plus the index into it. Regular files
    already in the view -- models a kernel downloaded itself -- are kept and
    shadow the cache. Returns the names linked this time. Never raises: a
    missing optional resource must not stop a kernel from starting.
    """
    view = Path(folder) / MODELS_SUBDIR
    try:
        view.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkdir's mode covers the leaf only; the intermediate data/ directory
        # gets the same treatment as every other isolated kernel directory.
        view.parent.chmod(0o700)
        view.chmod(0o700)
        for entry in view.iterdir():
            if entry.is_symlink():
                entry.unlink()
    except OSError as e:
        logger.warning(f"celltypist view not prepared at {view}: {e}")
        return []
    linked: List[str] = []
    for src in cached_model_files():
        dst = view / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError as e:
            logger.warning(f"celltypist model {src.name} not linked: {e}")
            continue
        linked.append(src.name)
    if linked:
        logger.info(f"celltypist cache exposed at {view}: {len(linked)} file(s)")
    return linked
