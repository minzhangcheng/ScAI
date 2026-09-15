"""A generation's own copy of the workspace, and how a rollback uses it.

The dump, the notebook record, the graph checkpoint and the executor position
are captured together so a session can be returned to the moment they describe.
The workspace is the fourth thing a session is made of and it used to be left
wherever the crash left it: on the run that prompted this, eight files written
after the last save survived a rollback, among them an ``.h5ad`` intermediate. A
later step loading one by name would read a file produced by work the rollback
had discarded, and nothing downstream could tell.

Recording only a file list was the first attempt, and it cannot serve the case
that prompted the question: a list takes away what was added, but a file that
existed at the save and was *rewritten* since is gone either way. So each save
keeps the generation its own copy.

Copying a whole workspace at every save moves its full size every time.
Copying only what changed moves, over a run, one workspace in total. Measured
across five saves of a session whose workspace ended at 4.08 GB: 2.85 GB,
1.13 GB, 3 MB, 10 MB, 91 MB.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: Where a generation keeps its copy of the workspace, inside the pair.
WORKSPACE_COPY_DIRNAME = "workspace"

#: What that copy held, written *after* it, so its presence is the marker that
#: the copy beside it is complete.
WORKSPACE_INVENTORY_NAME = "workspace.json"

#: Bumped when the shape changes. A reader that does not recognise the version
#: restores nothing rather than guessing, because the consequence of guessing
#: here is deleting a file.
WORKSPACE_INVENTORY_SCHEMA = 1


def scan_workspace_inventory(workspace: Path) -> Dict[str, Dict[str, int]]:
    """Every regular file under *workspace*, by path relative to it.

    Measured at 0.9 ms for 116 files against a save that takes tens of seconds,
    so it is free at any workspace this platform has produced.
    """
    inventory: Dict[str, Dict[str, int]] = {}
    if not workspace.is_dir():
        return inventory
    for base, _dirs, files in os.walk(workspace):
        for name in files:
            path = Path(base) / name
            try:
                info = path.lstat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            try:
                relative = path.relative_to(workspace).as_posix()
            except ValueError:
                continue
            inventory[relative] = {
                "size": int(info.st_size),
                "mtime_ns": int(info.st_mtime_ns),
            }
    return inventory


def unchanged(live: Path, candidate: Path) -> bool:
    """Whether *candidate* is still the bytes *live* had when it was copied.

    ``shutil.copy2`` preserves mtime, so a previous generation's copy carries
    the modification time the live file had at that save. Same size and same
    mtime to the nanosecond means the live file has not been touched since, and
    the copy can be shared rather than made again.
    """
    try:
        a, b = live.lstat(), candidate.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(a.st_mode)
        and stat.S_ISREG(b.st_mode)
        and a.st_size == b.st_size
        and a.st_mtime_ns == b.st_mtime_ns
    )


def _link_source_for(generation_root: Path, exclude: Path) -> Optional[Path]:
    """The newest previous generation's workspace copy, if it has one."""
    if not generation_root.is_dir():
        return None
    best: Optional[tuple[int, Path]] = None
    for entry in generation_root.iterdir():
        if not entry.is_dir() or entry == exclude:
            continue
        try:
            revision = int(entry.name)
        except ValueError:
            continue
        copy_dir = entry / WORKSPACE_COPY_DIRNAME
        if not copy_dir.is_dir():
            continue
        if best is None or revision > best[0]:
            best = (revision, copy_dir)
    return best[1] if best else None


def stage_workspace_copy(
    workspace: Path, pair_dir: Path, generation_root: Path,
) -> Dict[str, int]:
    """Give this generation its own copy, sharing what has not changed.

    An unchanged file is hardlinked from the *previous generation's copy*, never
    from the live file. That distinction is the correctness of this scheme.
    Linking the live file would be free and wrong: h5py opens an existing path
    with ``w`` and truncates the same inode, so a link to a live ``.h5ad``
    follows the rewrite -- measured, not assumed. A link to our own copy cannot,
    because the live file became a different inode the moment it was first
    copied, and nothing outside this module ever writes here.
    """
    counts = {"linked": 0, "copied": 0, "bytes_copied": 0}
    if not workspace.is_dir():
        return counts

    incoming = pair_dir / f"{WORKSPACE_COPY_DIRNAME}.incoming"
    shutil.rmtree(incoming, ignore_errors=True)
    incoming.mkdir(parents=True, exist_ok=True)
    previous = _link_source_for(generation_root, exclude=pair_dir)

    for base, _dirs, files in os.walk(workspace):
        for name in files:
            live = Path(base) / name
            try:
                info = live.lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                relative = live.relative_to(workspace)
            except (OSError, ValueError):
                continue
            target = incoming / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shared = previous / relative if previous is not None else None
            if shared is not None and unchanged(live, shared):
                try:
                    os.link(shared, target)
                    counts["linked"] += 1
                    continue
                except OSError:
                    pass
            try:
                shutil.copy2(live, target)
            except OSError:
                logger.warning(
                    "Could not copy %s into the generation's workspace",
                    relative, exc_info=True,
                )
                continue
            counts["copied"] += 1
            counts["bytes_copied"] += int(info.st_size)

    published = pair_dir / WORKSPACE_COPY_DIRNAME
    shutil.rmtree(published, ignore_errors=True)
    incoming.rename(published)
    return counts


def write_workspace_inventory(
    pair_dir: Path, revision: int, inventory: Dict[str, Dict[str, int]],
) -> None:
    """Record what the workspace held when this generation was committed."""
    from backend.utils.files import atomic_write_json

    atomic_write_json(
        pair_dir / WORKSPACE_INVENTORY_NAME,
        {
            "schema_version": WORKSPACE_INVENTORY_SCHEMA,
            "revision": int(revision),
            "files": inventory,
        },
    )


def publish_workspace_generation(
    workspace: Path, pair_dir: Path, generation_root: Path, revision: int,
) -> None:
    """The copy first, its record second, and neither is worth failing a save.

    The record's presence is what tells a rollback the copy beside it is whole,
    so a crash between the two leaves a generation that restores nothing rather
    than one that restores half a workspace. A generation without a complete
    copy behaves exactly as every generation written before this existed did.
    """
    try:
        counts = stage_workspace_copy(workspace, pair_dir, generation_root)
        write_workspace_inventory(
            pair_dir, revision, scan_workspace_inventory(workspace),
        )
    except OSError:
        logger.warning(
            "Could not give generation %s a workspace copy", revision,
            exc_info=True,
        )
        return
    logger.info(
        "Generation %s keeps a workspace copy: %d file(s) shared with the "
        "previous generation, %d copied (%.1f MB)",
        revision, counts["linked"], counts["copied"],
        counts["bytes_copied"] / 1e6,
    )
