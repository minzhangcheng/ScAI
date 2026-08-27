"""Figure publication lock ownership primitives."""
from __future__ import annotations

def _publish_lock_keys(figures_dir: Path, final_jpg: Path, published_record: Any, *, bindings) -> list[str]:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _attempt_lock_key = getattr(bindings, '_attempt_lock_key')
    _safe_component = getattr(bindings, '_safe_component')
    _value = getattr(bindings, '_value')
    hashlib = getattr(bindings, 'hashlib')
    lineage = str(_value(published_record, "lineage_id", None) or "main")
    logical = str(_value(published_record, "logical_figure_id", None) or "")
    path_key = "path:" + str(_absolute(final_jpg))
    workspace_key = hashlib.sha256(
        str(_absolute(Path(figures_dir))).encode("utf-8")
    ).hexdigest()
    logical_key = "logical:" + workspace_key + ":" + lineage + ":" + logical
    keys = {path_key, logical_key}
    attempt_id = str(_value(published_record, "render_attempt_id", "") or "")
    if attempt_id and attempt_id == _safe_component(attempt_id, "attempt"):
        keys.add(_attempt_lock_key(figures_dir, attempt_id))
    return sorted(keys)

def _attempt_lock_key(figures_dir: Path, attempt_id: str, *, bindings) -> str:
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    hashlib = getattr(bindings, 'hashlib')
    workspace_key = hashlib.sha256(
        str(_absolute(Path(figures_dir))).encode("utf-8")
    ).hexdigest()
    return "attempt:" + workspace_key + ":" + str(attempt_id)

def _transaction_creation_is_live(figures_dir: Path, attempt_id: str, *, bindings) -> bool:
    """Probe the attempt-scoped flock before classifying a pre-journal directory."""
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _HELD_LOCKS = getattr(bindings, '_HELD_LOCKS')
    _absolute = getattr(bindings, '_absolute')
    _attempt_lock_key = getattr(bindings, '_attempt_lock_key')
    _lock_file = getattr(bindings, '_lock_file')
    _open_hidden_root = getattr(bindings, '_open_hidden_root')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    scope = str(_absolute(figures_dir))
    key = _attempt_lock_key(figures_dir, attempt_id)
    held = _HELD_LOCKS.get(key)
    if held is not None and held[0][0] == scope and held[0][2] == attempt_id:
        return True
    try:
        import fcntl
    except ImportError:  # pragma: no cover
        return False
    try:
        locks_fd, parents = _open_hidden_root(
            _absolute(figures_dir) / TRANSACTION_ROOT / ".locks",
            create=False,
        )
    except ValueError:
        return False
    descriptor: int | None = None
    try:
        lock_name = _lock_file(figures_dir, key).name
        descriptor = os.open(
            lock_name,
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=locks_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(locks_fd)
        for parent in reversed(parents):
            os.close(parent)

def _lock_file(figures_dir: Path, key: str, *, bindings) -> Path:
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    hashlib = getattr(bindings, 'hashlib')
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _absolute(figures_dir) / TRANSACTION_ROOT / ".locks" / f"{digest}.lock"

def _lock_owner(figures_dir: Path, session_id: str, attempt_id: str, keys: Iterable[str], *, bindings) -> _LockOwner:
    Iterable = getattr(bindings, 'Iterable')
    Path = getattr(bindings, 'Path')
    _LockOwner = getattr(bindings, '_LockOwner')
    _absolute = getattr(bindings, '_absolute')
    return (
        str(_absolute(figures_dir)),
        str(session_id or ""),
        str(attempt_id or ""),
        tuple(sorted(set(keys))),
    )

def _acquire_publish_locks(figures_dir: Path, attempt_id: str, keys: Iterable[str], *, session_id: str='', bindings) -> str | None:
    Iterable = getattr(bindings, 'Iterable')
    Path = getattr(bindings, 'Path')
    _HELD_LOCKS = getattr(bindings, '_HELD_LOCKS')
    _lock_file = getattr(bindings, '_lock_file')
    _lock_owner = getattr(bindings, '_lock_owner')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_transaction_root = getattr(bindings, '_open_transaction_root')
    _release_publish_locks = getattr(bindings, '_release_publish_locks')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        import fcntl
    except ImportError:  # pragma: no cover - ScAI's runtime is POSIX
        return "Cross-process figure publishing requires POSIX file locks"

    normalized_keys = tuple(sorted(set(keys)))
    owner = _lock_owner(figures_dir, session_id, attempt_id, normalized_keys)
    acquired: list[str] = []
    figures_fd: int | None = None
    root_fd: int | None = None
    locks_fd: int | None = None
    try:
        figures_fd, root_fd = _open_transaction_root(figures_dir, create=True)
        locks_fd = _open_child_directory_nofollow(root_fd, ".locks", create=True)
        for key in normalized_keys:
            held = _HELD_LOCKS.get(key)
            if held is not None:
                if held[0] == owner:
                    continue
                raise BlockingIOError(f"publication lock is held for {key}")
            path = _lock_file(figures_dir, key)
            descriptor = os.open(
                path.name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=locks_fd,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError("publication lock path is not a regular file")
            handle = os.fdopen(descriptor, "a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                handle.close()
                raise
            _HELD_LOCKS[key] = (owner, handle)
            acquired.append(key)
        return None
    except (BlockingIOError, OSError, ValueError) as exc:
        _release_publish_locks(
            attempt_id,
            keys=acquired,
            figures_dir=figures_dir,
            session_id=session_id,
        )
        return f"Another publication attempt is committing this figure: {exc}"
    finally:
        if locks_fd is not None:
            os.close(locks_fd)
        if root_fd is not None:
            os.close(root_fd)
        if figures_fd is not None:
            os.close(figures_fd)

def _release_publish_locks(attempt_id: str, *, keys: Iterable[str] | None=None, figures_dir: Path | None=None, session_id: str | None=None, bindings) -> None:
    Iterable = getattr(bindings, 'Iterable')
    Path = getattr(bindings, 'Path')
    _HELD_LOCKS = getattr(bindings, '_HELD_LOCKS')
    _LockOwner = getattr(bindings, '_LockOwner')
    _absolute = getattr(bindings, '_absolute')
    try:
        import fcntl
    except ImportError:  # pragma: no cover
        fcntl = None
    candidates = list(keys) if keys is not None else list(_HELD_LOCKS)
    scope = str(_absolute(figures_dir)) if figures_dir is not None else None
    matching_owners: list[_LockOwner] = []
    for key in candidates:
        held = _HELD_LOCKS.get(key)
        if held is None:
            continue
        owner = held[0]
        if owner[2] != attempt_id:
            continue
        if scope is not None and owner[0] != scope:
            continue
        if session_id is not None and owner[1] != str(session_id or ""):
            continue
        if owner not in matching_owners:
            matching_owners.append(owner)
    # A legacy unscoped release may see equal attempt IDs in two workspaces.
    # Release one complete ownership group, never every matching workspace.
    selected_owner = matching_owners[0] if matching_owners else None
    for key in candidates:
        held = _HELD_LOCKS.get(key)
        if held is None or held[0] != selected_owner:
            continue
        handle = held[1]
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            _HELD_LOCKS.pop(key, None)

def acquire_figure_publish_locks(*, figures_dir: Path, final_jpg: Path, published_record: Any, bindings) -> str | None:
    """Acquire path and semantic-lineage locks before choosing the old head."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _acquire_publish_locks = getattr(bindings, '_acquire_publish_locks')
    _journal_paths = getattr(bindings, '_journal_paths')
    _publish_lock_keys = getattr(bindings, '_publish_lock_keys')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _value = getattr(bindings, '_value')
    attempt_id = str(_value(published_record, "render_attempt_id", "") or "")
    try:
        _transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
        journal = _read_verified_journal(journal_path)
    except Exception as exc:
        return f"Could not verify the pending publication lock owner: {exc}"
    if str(journal.get("new_record_id") or "") != str(
        _value(published_record, "id", "") or ""
    ):
        return "Could not acquire a publication lock for a mismatched record"
    return _acquire_publish_locks(
        figures_dir,
        attempt_id,
        _publish_lock_keys(figures_dir, final_jpg, published_record),
        session_id=str(journal.get("session_id") or ""),
    )
