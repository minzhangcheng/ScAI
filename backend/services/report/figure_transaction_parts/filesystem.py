"""Descriptor-relative figure transaction filesystem primitives."""
from __future__ import annotations

def _value(item: Any, key: str, default: Any=None, *, bindings) -> Any:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)

def _set(item: Any, key: str, value: Any, *, bindings) -> None:
    Any = getattr(bindings, 'Any')
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)

def _safe_component(value: Any, fallback: str, *, bindings) -> str:
    Any = getattr(bindings, 'Any')
    cleaned = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(value or "")
    ).strip("._")
    return cleaned[:180] or fallback

def _archive_record_component(record_id: Any, *, bindings) -> str:
    """Map an arbitrary record identity to one collision-resistant path part."""
    Any = getattr(bindings, 'Any')
    hashlib = getattr(bindings, 'hashlib')
    value = str(record_id or "")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _absolute(path: Path, *, bindings) -> Path:
    """Return a lexical absolute path without following its final component."""
    Path = getattr(bindings, 'Path')
    os = getattr(bindings, 'os')
    return Path(os.path.abspath(os.fspath(path)))

def _directory_flags(*, bindings) -> int:
    os = getattr(bindings, 'os')
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

def _open_directory_nofollow(path: Path, *, bindings) -> int:
    """Open one real directory, rejecting a file or symlink final component."""
    Path = getattr(bindings, 'Path')
    _directory_flags = getattr(bindings, '_directory_flags')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        descriptor = os.open(os.fspath(path), _directory_flags())
    except OSError as exc:
        raise ValueError(f"trusted directory is missing, invalid, or a symlink: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"trusted path is not a directory: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def _open_child_directory_nofollow(parent_fd: int, name: str, *, create: bool=False, bindings) -> int:
    """Open one direct child directory relative to an already trusted FD."""
    _directory_flags = getattr(bindings, '_directory_flags')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    if name in {"", ".", ".."} or "/" in name or os.sep in name:
        raise ValueError("invalid trusted directory component")
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(
            f"trusted directory component is missing, invalid, or a symlink: {name}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"trusted path component is not a directory: {name}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def _require_directory_entry_binding(parent_fd: int, name: str, child_fd: int, *, label: str, bindings) -> None:
    """Require a live parent entry to still name the opened directory inode."""
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} directory binding is missing or invalid") from exc
    opened = os.fstat(child_fd)
    if (
        not stat.S_ISDIR(entry.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError(f"{label} directory binding changed")

def _open_transaction_root(figures_dir: Path, *, create: bool=False, bindings) -> tuple[int, int]:
    """Open ``figures/.transactions`` without following either component."""
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _require_directory_entry_binding = getattr(bindings, '_require_directory_entry_binding')
    os = getattr(bindings, 'os')
    figures_fd = _open_directory_nofollow(_absolute(figures_dir))
    try:
        root_fd = _open_child_directory_nofollow(
            figures_fd, TRANSACTION_ROOT, create=create
        )
        _require_directory_entry_binding(
            figures_fd,
            TRANSACTION_ROOT,
            root_fd,
            label="publication transaction root",
        )
    except BaseException:
        os.close(figures_fd)
        raise
    return figures_fd, root_fd

def _open_transaction_directory(transaction_dir: Path, *, bindings) -> tuple[int, int, int]:
    """Open one direct transaction directory through trusted directory FDs."""
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_transaction_root = getattr(bindings, '_open_transaction_root')
    _require_directory_entry_binding = getattr(bindings, '_require_directory_entry_binding')
    os = getattr(bindings, 'os')
    transaction_dir = _absolute(transaction_dir)
    root = transaction_dir.parent
    if root.name != TRANSACTION_ROOT:
        raise ValueError("publication transaction is outside the transaction root")
    figures_fd, root_fd = _open_transaction_root(root.parent, create=False)
    try:
        transaction_fd = _open_child_directory_nofollow(
            root_fd, transaction_dir.name, create=False
        )
        _require_directory_entry_binding(
            figures_fd,
            TRANSACTION_ROOT,
            root_fd,
            label="publication transaction root",
        )
        _require_directory_entry_binding(
            root_fd,
            transaction_dir.name,
            transaction_fd,
            label="publication transaction",
        )
    except BaseException:
        os.close(root_fd)
        os.close(figures_fd)
        raise
    return figures_fd, root_fd, transaction_fd

def _open_attempt_directory(attempt_dir: Path, *, create: bool=False, bindings) -> tuple[int, int, int]:
    """Open ``figures/.attempts/<attempt>`` without following components."""
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _require_directory_entry_binding = getattr(bindings, '_require_directory_entry_binding')
    os = getattr(bindings, 'os')
    attempt_dir = _absolute(attempt_dir)
    attempts_root = attempt_dir.parent
    if attempts_root.name != ".attempts":
        raise ValueError("publication attempt is outside the attempts root")
    figures_fd = _open_directory_nofollow(attempts_root.parent)
    try:
        attempts_fd = _open_child_directory_nofollow(
            figures_fd, ".attempts", create=create
        )
        try:
            attempt_fd = _open_child_directory_nofollow(
                attempts_fd, attempt_dir.name, create=create
            )
            _require_directory_entry_binding(
                figures_fd,
                ".attempts",
                attempts_fd,
                label="publication attempts root",
            )
            _require_directory_entry_binding(
                attempts_fd,
                attempt_dir.name,
                attempt_fd,
                label="publication attempt",
            )
        except BaseException:
            os.close(attempts_fd)
            raise
    except BaseException:
        os.close(figures_fd)
        raise
    return figures_fd, attempts_fd, attempt_fd

def _open_revision_directory(revision_dir: Path, *, create: bool=False, bindings) -> tuple[int, int, int, int]:
    """Open ``figures/.revisions/<record>/<attempt>`` component by component."""
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _require_directory_entry_binding = getattr(bindings, '_require_directory_entry_binding')
    _safe_component = getattr(bindings, '_safe_component')
    os = getattr(bindings, 'os')
    revision_dir = _absolute(revision_dir)
    record_dir = revision_dir.parent
    revision_root = record_dir.parent
    if (
        revision_root.name != ".revisions"
        or len(record_dir.name) != 64
        or any(char not in "0123456789abcdef" for char in record_dir.name)
        or revision_dir.name != _safe_component(revision_dir.name, "attempt")
    ):
        raise ValueError("publication revision archive has an invalid layout")
    figures_fd = _open_directory_nofollow(revision_root.parent)
    try:
        revision_fd = _open_child_directory_nofollow(
            figures_fd, ".revisions", create=create
        )
        try:
            record_fd = _open_child_directory_nofollow(
                revision_fd, record_dir.name, create=create
            )
            try:
                attempt_fd = _open_child_directory_nofollow(
                    record_fd, revision_dir.name, create=create
                )
                _require_directory_entry_binding(
                    figures_fd,
                    ".revisions",
                    revision_fd,
                    label="publication revisions root",
                )
                _require_directory_entry_binding(
                    revision_fd,
                    record_dir.name,
                    record_fd,
                    label="publication revision record",
                )
                _require_directory_entry_binding(
                    record_fd,
                    revision_dir.name,
                    attempt_fd,
                    label="publication revision attempt",
                )
            except BaseException:
                os.close(record_fd)
                raise
        except BaseException:
            os.close(revision_fd)
            raise
    except BaseException:
        os.close(figures_fd)
        raise
    return figures_fd, revision_fd, record_fd, attempt_fd

def _require_transaction_directory_binding(figures_fd: int, root_fd: int, transaction_fd: int, attempt_id: str, *, bindings) -> None:
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _require_directory_entry_binding = getattr(bindings, '_require_directory_entry_binding')
    _require_directory_entry_binding(
        figures_fd,
        TRANSACTION_ROOT,
        root_fd,
        label="publication transaction root",
    )
    _require_directory_entry_binding(
        root_fd,
        attempt_id,
        transaction_fd,
        label="publication transaction",
    )

def _require_attempt_directory_binding(figures_fd: int, attempts_fd: int, attempt_fd: int, attempt_id: str, *, bindings) -> None:
    _require_directory_entry_binding = getattr(bindings, '_require_directory_entry_binding')
    _require_directory_entry_binding(
        figures_fd,
        ".attempts",
        attempts_fd,
        label="publication attempts root",
    )
    _require_directory_entry_binding(
        attempts_fd,
        attempt_id,
        attempt_fd,
        label="publication attempt",
    )

def _open_known_file_parent(path: Path, *, create: bool=False, bindings) -> tuple[int, list[int]]:
    """Open the trusted parent for public and hidden figure transaction files."""
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _open_attempt_directory = getattr(bindings, '_open_attempt_directory')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _open_revision_directory = getattr(bindings, '_open_revision_directory')
    _open_transaction_directory = getattr(bindings, '_open_transaction_directory')
    path = _absolute(path)
    parent = path.parent
    if parent.parent.name == TRANSACTION_ROOT:
        figures_fd, root_fd, parent_fd = _open_transaction_directory(parent)
        return parent_fd, [figures_fd, root_fd]
    if parent.parent.name == ".attempts":
        figures_fd, attempts_fd, parent_fd = _open_attempt_directory(
            parent, create=create
        )
        return parent_fd, [figures_fd, attempts_fd]
    if parent.parent.parent.name == ".revisions":
        figures_fd, revision_fd, record_fd, parent_fd = (
            _open_revision_directory(parent, create=create)
        )
        return parent_fd, [figures_fd, revision_fd, record_fd]
    return _open_directory_nofollow(parent), []

def _close_parent_chain(parent_fd: int, ancestors: Iterable[int], *, bindings) -> None:
    Iterable = getattr(bindings, 'Iterable')
    os = getattr(bindings, 'os')
    os.close(parent_fd)
    for descriptor in reversed(list(ancestors)):
        os.close(descriptor)

def _digest_regular_file_at(parent_fd: int, name: str, *, bindings) -> str:
    """Hash one stable regular direct child using one no-follow descriptor."""
    hashlib = getattr(bindings, 'hashlib')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ValueError("figure transaction file is missing or a symlink") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("figure transaction path is not a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or size != before.st_size:
            raise ValueError("figure transaction file changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)

def _digest_open_regular_file(descriptor: int, *, label: str, bindings) -> str:
    """Hash an already opened regular file and restore its seek position."""
    hashlib = getattr(bindings, 'hashlib')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or size != before.st_size:
            raise ValueError(f"{label} changed while it was read")
        return digest.hexdigest()
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)

def _open_frozen_staged_file(attempt_fd: int, name: str, expected_sha256: str, *, label: str, bindings) -> tuple[int, os.stat_result]:
    _digest_open_regular_file = getattr(bindings, '_digest_open_regular_file')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=attempt_fd,
        )
    except OSError as exc:
        raise ValueError(f"{label} is missing, invalid, or a symlink") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if _digest_open_regular_file(descriptor, label=label) != expected_sha256:
            raise ValueError(f"{label} does not match its frozen digest")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise

def _open_stable_staged_file(attempt_fd: int, name: str, *, label: str, bindings) -> tuple[int, os.stat_result, str]:
    """Open and hash one pending staged file without trusting its path again."""
    _digest_open_regular_file = getattr(bindings, '_digest_open_regular_file')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=attempt_fd,
        )
    except OSError as exc:
        raise ValueError(f"{label} is missing, invalid, or a symlink") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} is not a regular file")
        digest = _digest_open_regular_file(descriptor, label=label)
        return descriptor, metadata, digest
    except BaseException:
        os.close(descriptor)
        raise

def _require_regular_entry_binding(parent_fd: int, name: str, descriptor: int, *, label: str, bindings) -> None:
    """Require a direct filename to still name the opened regular-file inode."""
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} binding is missing or invalid") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(entry.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise ValueError(f"{label} binding changed")

def _regular_child_exists(parent_fd: int, name: str, *, label: str, bindings) -> bool:
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    return True

def _replace_staged_entry(attempt_fd: int, figures_fd: int, source_name: str, destination_name: str, *, bindings) -> None:
    os = getattr(bindings, 'os')
    os.replace(
        source_name,
        destination_name,
        src_dir_fd=attempt_fd,
        dst_dir_fd=figures_fd,
    )

def _verify_published_entry(figures_fd: int, name: str, staged_fd: int, staged_metadata: os.stat_result, expected_sha256: str, *, label: str, bindings) -> None:
    _digest_regular_file_at = getattr(bindings, '_digest_regular_file_at')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    current = os.stat(name, dir_fd=figures_fd, follow_symlinks=False)
    opened = os.fstat(staged_fd)
    expected_identity = (staged_metadata.st_dev, staged_metadata.st_ino)
    if (
        not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or (current.st_dev, current.st_ino) != expected_identity
        or (opened.st_dev, opened.st_ino) != expected_identity
        or opened.st_size != staged_metadata.st_size
    ):
        raise ValueError(f"{label} does not name the frozen staged inode")
    if _digest_regular_file_at(figures_fd, name) != expected_sha256:
        raise ValueError(f"{label} does not match its frozen digest")

def _read_regular_file_at(parent_fd: int, name: str, *, max_bytes: int, label: str, bindings) -> bytes:
    """Read and validate one regular file entirely through one no-follow FD."""
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"{label} is missing, unreadable, or a symlink") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        if before.st_size > max_bytes:
            raise ValueError(f"{label} is oversized")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(payload) > max_bytes:
            raise ValueError(f"{label} is oversized")
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before or len(payload) != before.st_size:
            raise ValueError(f"{label} changed while it was read")
        return payload
    finally:
        os.close(descriptor)

def _read_transaction_file(path: Path, *, max_bytes: int, label: str, bindings) -> bytes:
    Path = getattr(bindings, 'Path')
    _open_transaction_directory = getattr(bindings, '_open_transaction_directory')
    _read_regular_file_at = getattr(bindings, '_read_regular_file_at')
    os = getattr(bindings, 'os')
    figures_fd, root_fd, transaction_fd = _open_transaction_directory(Path(path).parent)
    try:
        return _read_regular_file_at(
            transaction_fd,
            Path(path).name,
            max_bytes=max_bytes,
            label=label,
        )
    finally:
        os.close(transaction_fd)
        os.close(root_fd)
        os.close(figures_fd)

def _fsync_file(path: Path, *, bindings) -> None:
    Path = getattr(bindings, 'Path')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"path is not a regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _fsync_dir(path: Path, *, bindings) -> None:
    Path = getattr(bindings, 'Path')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"path is not a directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _atomic_json(path: Path, payload: dict[str, Any], *, bindings) -> None:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _open_transaction_directory = getattr(bindings, '_open_transaction_directory')
    _require_transaction_directory_binding = getattr(bindings, '_require_transaction_directory_binding')
    json = getattr(bindings, 'json')
    os = getattr(bindings, 'os')
    uuid = getattr(bindings, 'uuid')
    path = _absolute(path)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8")
    figures_fd, root_fd, parent_fd = _open_transaction_directory(path.parent)
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short publication transaction metadata write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _require_transaction_directory_binding(
            figures_fd, root_fd, parent_fd, path.parent.name
        )
        # Keep metadata rename independent of failure injection for the public
        # JPG/PDF pair while retaining one trusted directory descriptor.
        os.rename(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        _require_transaction_directory_binding(
            figures_fd, root_fd, parent_fd, path.parent.name
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)
        os.close(root_fd)
        os.close(figures_fd)

def _copy_durable(source: Path, destination: Path, *, bindings) -> None:
    """Copy a regular file atomically without following hidden path entries."""
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _close_parent_chain = getattr(bindings, '_close_parent_chain')
    _open_known_file_parent = getattr(bindings, '_open_known_file_parent')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    uuid = getattr(bindings, 'uuid')
    source = _absolute(source)
    destination = _absolute(destination)
    source_parent_fd: int | None = None
    destination_parent_fd: int | None = None
    source_ancestors: list[int] = []
    destination_ancestors: list[int] = []
    source_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name = f".{destination.name}.copy-{uuid.uuid4().hex}"
    try:
        source_parent_fd, source_ancestors = _open_known_file_parent(
            source, create=False
        )
        destination_parent_fd, destination_ancestors = _open_known_file_parent(
            destination, create=True
        )
        source_fd = os.open(
            source.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=source_parent_fd,
        )
        source_before = os.fstat(source_fd)
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError("publication copy source is not a regular file")
        try:
            destination_metadata = os.stat(
                destination.name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(destination_metadata.st_mode):
                raise ValueError(
                    "publication copy destination is not a regular file"
                )
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=destination_parent_fd,
        )
        copied_size = 0
        while True:
            chunk = os.read(source_fd, 65_536)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(temporary_fd, view)
                if written <= 0:
                    raise OSError("short publication archive write")
                view = view[written:]
                copied_size += written
        source_after = os.fstat(source_fd)
        before_identity = (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        )
        after_identity = (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        )
        if before_identity != after_identity or copied_size != source_before.st_size:
            raise ValueError("publication copy source changed while it was read")
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.rename(
            temporary_name,
            destination.name,
            src_dir_fd=destination_parent_fd,
            dst_dir_fd=destination_parent_fd,
        )
        os.fsync(destination_parent_fd)
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if temporary_fd is not None:
            os.close(temporary_fd)
        if destination_parent_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=destination_parent_fd)
            except OSError:
                pass
            _close_parent_chain(
                destination_parent_fd, destination_ancestors
            )
        if source_parent_fd is not None:
            _close_parent_chain(source_parent_fd, source_ancestors)
