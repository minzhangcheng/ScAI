"""Workspace content capture for snapshot V2.

Capture scope (design contract):
- regular files directly under workspace root
- recursive regular files under figures/, tables/, reproducibility/

Unsafe or ambiguous symlinks fail capture rather than being skipped.
"""

import os
import stat
from pathlib import Path, PurePosixPath

from backend.services.snapshot_v2.errors import SnapshotCaptureError
from backend.services.snapshot_v2.models import WorkspaceEntry
from backend.services.snapshot_v2.object_store import ObjectStore

# Declared recursive subtrees under workspace root.
SCOPED_RECURSIVE_DIRS: frozenset[str] = frozenset({
    "figures",
    "tables",
    "reproducibility",
})


def _media_type(path: str) -> str:
    ext = PurePosixPath(path).suffix.lower()
    return {
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".txt": "text/plain",
        ".json": "application/json",
        ".yaml": "application/x-yaml",
        ".yml": "application/x-yaml",
        ".ipynb": "application/x-ipynb+json",
        ".py": "text/x-python",
        ".r": "text/x-r",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".gz": "application/gzip",
        ".tar": "application/x-tar",
        ".zip": "application/zip",
        ".h5ad": "application/x-hdf5",
        ".pkl": "application/octet-stream",
        ".pickle": "application/octet-stream",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".xml": "application/xml",
        ".toml": "application/toml",
        ".cfg": "text/x-config",
        ".ini": "text/x-config",
        ".log": "text/plain",
    }.get(ext, "application/octet-stream")


def _is_in_scope(relative: str) -> bool:
    """Return True if *relative* is inside the declared capture scope."""
    parts = PurePosixPath(relative).parts
    if not parts:
        return False
    if len(parts) == 1:
        # Root-level regular file
        return True
    return parts[0] in SCOPED_RECURSIVE_DIRS


class WorkspaceScope:
    """Shared scope rules for capture, apply, rollback, and forward recovery.

    Only root regular files and the three recursive roots may be created,
    replaced, or removed. Every other directory is left untouched.
    """

    SCOPED_RECURSIVE_DIRS = SCOPED_RECURSIVE_DIRS

    def is_in_scope(self, relative: str) -> bool:
        return _is_in_scope(relative)

    def list_live_scoped_paths(self, workspace_root: str | Path) -> set[str]:
        """Return relative paths of regular files currently under scope."""
        root = Path(workspace_root)
        if not root.is_dir():
            return set()
        found: set[str] = set()
        resolved = root.resolve()
        for abs_path in WorkspaceCapture()._walk_scope(resolved):
            try:
                relative = str(PurePosixPath(abs_path.relative_to(resolved)))
            except ValueError:
                continue
            if not self.is_in_scope(relative):
                continue
            try:
                st = abs_path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode):
                found.add(relative)
        return found

    def remove_added_files(
        self,
        workspace_root: str | Path,
        allowed_paths: set[str],
    ) -> list[str]:
        """Remove scoped live files not present in *allowed_paths*.

        Returns the list of relative paths removed. Out-of-scope paths are
        never touched.
        """
        root = Path(workspace_root)
        removed: list[str] = []
        for relative in sorted(self.list_live_scoped_paths(root)):
            if relative in allowed_paths:
                continue
            target = root / relative
            try:
                if target.is_file() and not target.is_symlink():
                    target.unlink()
                    removed.append(relative)
            except OSError:
                continue
        return removed


class WorkspaceCapture:
    """Scans a workspace directory and persists file content to the object store."""

    def capture(
        self, object_store: ObjectStore, workspace_root: str
    ) -> tuple[list[WorkspaceEntry], list[str]]:
        """Compatibility entry point — delegates to scoped capture."""
        return self.capture_scope(object_store, workspace_root)

    def capture_scope(
        self, object_store: ObjectStore, workspace_root: str
    ) -> tuple[list[WorkspaceEntry], list[str]]:
        """Capture only declared-scope entries; fail on unsafe symlinks.

        Returns ``(entries, warnings)``. Warnings are reserved for non-fatal
        notes; unsafe entries raise ``SnapshotCaptureError``.
        """
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            return [], [f"Workspace root does not exist: {workspace_root}"]

        entries: list[WorkspaceEntry] = []
        warnings: list[str] = []
        seen_relative: set[str] = set()

        for abs_path in self._walk_scope(root):
            try:
                relative = str(PurePosixPath(abs_path.relative_to(root)))
            except ValueError as e:
                raise SnapshotCaptureError(
                    "WORKSPACE_PATH_ESCAPE",
                    f"Path escaped workspace root: {abs_path}: {e}",
                ) from e

            if not _is_in_scope(relative):
                continue

            if relative in seen_relative:
                warnings.append(f"Duplicate path skipped: {relative}")
                continue
            seen_relative.add(relative)

            try:
                st = abs_path.lstat()
            except OSError as e:
                raise SnapshotCaptureError(
                    "WORKSPACE_STAT_FAILED",
                    f"Cannot stat {relative}: {e}",
                ) from e

            if stat.S_ISLNK(st.st_mode):
                raise SnapshotCaptureError(
                    "WORKSPACE_UNSAFE_SYMLINK",
                    f"Symlink in capture scope is not allowed: {relative}",
                    details={"path": relative},
                )

            if stat.S_ISDIR(st.st_mode):
                continue

            if not stat.S_ISREG(st.st_mode):
                raise SnapshotCaptureError(
                    "WORKSPACE_NON_REGULAR",
                    f"Non-regular file in capture scope: {relative}",
                    details={"path": relative},
                )

            try:
                put_result = object_store.put_file(abs_path, _media_type(relative))
                ref = put_result.ref
            except SnapshotCaptureError:
                raise
            except Exception as e:
                raise SnapshotCaptureError(
                    "WORKSPACE_STORE_FAILED",
                    f"Failed to store {relative}: {e}",
                    details={"path": relative},
                ) from e

            entries.append(
                WorkspaceEntry(
                    path=relative,
                    object_hash=ref.sha256,
                    size=ref.size,
                    mode=st.st_mode,
                    kind="file",
                )
            )

        return entries, warnings

    def _walk_scope(self, root: Path) -> list[Path]:
        """Collect scoped filesystem entries without following symlinks.

        Visits:
        - every direct child of *root* (files and dirs)
        - recursive contents of figures/, tables/, reproducibility/ only
        """
        result: list[Path] = []
        try:
            with os.scandir(root) as it:
                for entry in it:
                    entry_path = Path(entry.path)
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError as e:
                        raise SnapshotCaptureError(
                            "WORKSPACE_STAT_FAILED",
                            f"Cannot stat {entry.name}: {e}",
                        ) from e

                    # Always include the entry itself so root files and
                    # symlinks at root are examined by the caller.
                    result.append(entry_path)

                    if stat.S_ISLNK(st.st_mode):
                        # Symlink at root — caller will fail closed.
                        continue

                    if stat.S_ISDIR(st.st_mode) and entry.name in SCOPED_RECURSIVE_DIRS:
                        result.extend(self._walk_recursive(entry_path, root))
        except PermissionError as e:
            raise SnapshotCaptureError(
                "WORKSPACE_PERMISSION",
                f"Permission denied scanning workspace: {e}",
            ) from e
        except SnapshotCaptureError:
            raise
        except OSError as e:
            raise SnapshotCaptureError(
                "WORKSPACE_SCAN_FAILED",
                f"Failed to scan workspace: {e}",
            ) from e
        return result

    def _walk_recursive(self, start: Path, root: Path) -> list[Path]:
        """Recursively collect entries under *start* without following symlinks."""
        result: list[Path] = []
        stack = [start]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        entry_path = Path(entry.path)
                        try:
                            st = entry.stat(follow_symlinks=False)
                        except OSError as e:
                            raise SnapshotCaptureError(
                                "WORKSPACE_STAT_FAILED",
                                f"Cannot stat {entry_path.relative_to(root)}: {e}",
                            ) from e
                        result.append(entry_path)
                        if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
                            stack.append(entry_path)
                        elif stat.S_ISLNK(st.st_mode):
                            # Leave for caller to fail closed when in scope.
                            pass
            except PermissionError as e:
                raise SnapshotCaptureError(
                    "WORKSPACE_PERMISSION",
                    f"Permission denied scanning {current}: {e}",
                ) from e
            except SnapshotCaptureError:
                raise
            except OSError as e:
                raise SnapshotCaptureError(
                    "WORKSPACE_SCAN_FAILED",
                    f"Failed to scan {current}: {e}",
                ) from e
        return result
