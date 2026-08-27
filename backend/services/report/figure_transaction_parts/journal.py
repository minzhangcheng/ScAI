"""Authenticated figure transaction journal primitives."""
from __future__ import annotations

def _canonical_json(payload: Mapping[str, Any], *, bindings) -> bytes:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    json = getattr(bindings, 'json')
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

def _journal_body(payload: Mapping[str, Any], *, bindings) -> bytes:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _RUNTIME_JOURNAL_FIELDS = getattr(bindings, '_RUNTIME_JOURNAL_FIELDS')
    _canonical_json = getattr(bindings, '_canonical_json')
    return _canonical_json({
        key: value
        for key, value in payload.items()
        if key != "journal_hmac" and key not in _RUNTIME_JOURNAL_FIELDS
    })

def _marker_body(payload: Mapping[str, Any], *, bindings) -> bytes:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _canonical_json = getattr(bindings, '_canonical_json')
    return _canonical_json({
        key: value for key, value in payload.items() if key != "marker_hmac"
    })

def _write_marker(path: Path, *, bindings) -> None:
    """Write an authenticated transaction marker bound to journal identity."""
    JOURNAL_FILENAME = getattr(bindings, 'JOURNAL_FILENAME')
    Path = getattr(bindings, 'Path')
    ROLLBACK_DURABLE_MARKER = getattr(bindings, 'ROLLBACK_DURABLE_MARKER')
    TRANSACTION_SCHEMA_VERSION = getattr(bindings, 'TRANSACTION_SCHEMA_VERSION')
    _TERMINAL_MARKERS = getattr(bindings, '_TERMINAL_MARKERS')
    _atomic_json = getattr(bindings, '_atomic_json')
    _marker_body = getattr(bindings, '_marker_body')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    sign_hmac_sha256 = getattr(bindings, 'sign_hmac_sha256')
    if path.name not in _TERMINAL_MARKERS | {ROLLBACK_DURABLE_MARKER}:
        raise ValueError("unknown publication transaction marker")
    journal = _read_verified_journal(path.parent / JOURNAL_FILENAME)
    payload = {
        "schema_version": TRANSACTION_SCHEMA_VERSION,
        "session_id": journal["session_id"],
        "attempt_id": journal["attempt_id"],
        "marker": path.name,
    }
    payload["marker_hmac"] = sign_hmac_sha256(_marker_body(payload))
    _atomic_json(path, payload)

def _authenticated_marker_exists(transaction_dir: Path, marker: str, journal: Mapping[str, Any], *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    Path = getattr(bindings, 'Path')
    ROLLBACK_DURABLE_MARKER = getattr(bindings, 'ROLLBACK_DURABLE_MARKER')
    TRANSACTION_SCHEMA_VERSION = getattr(bindings, 'TRANSACTION_SCHEMA_VERSION')
    _TERMINAL_MARKERS = getattr(bindings, '_TERMINAL_MARKERS')
    _marker_body = getattr(bindings, '_marker_body')
    _read_transaction_file = getattr(bindings, '_read_transaction_file')
    hmac = getattr(bindings, 'hmac')
    json = getattr(bindings, 'json')
    sign_hmac_sha256 = getattr(bindings, 'sign_hmac_sha256')
    marker_path = transaction_dir / marker
    try:
        if marker not in _TERMINAL_MARKERS | {ROLLBACK_DURABLE_MARKER}:
            return False
        raw = _read_transaction_file(
            marker_path,
            max_bytes=16_384,
            label="publication transaction marker",
        )
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return False
        signature = str(payload.get("marker_hmac") or "")
        if not signature or not hmac.compare_digest(
            signature, sign_hmac_sha256(_marker_body(payload))
        ):
            return False
        return bool(
            payload.get("schema_version") == TRANSACTION_SCHEMA_VERSION
            and str(payload.get("session_id") or "")
            == str(journal.get("session_id") or "")
            and str(payload.get("attempt_id") or "")
            == str(journal.get("attempt_id") or "")
            and payload.get("marker") == marker
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False

def _unlink_transaction_regular_file_if_present(transaction_dir: Path, name: str, *, bindings) -> None:
    """Unlink one direct transaction child only through a verified directory FD."""
    Path = getattr(bindings, 'Path')
    ROLLBACK_DURABLE_MARKER = getattr(bindings, 'ROLLBACK_DURABLE_MARKER')
    _TERMINAL_MARKERS = getattr(bindings, '_TERMINAL_MARKERS')
    _absolute = getattr(bindings, '_absolute')
    _open_transaction_directory = getattr(bindings, '_open_transaction_directory')
    _require_transaction_directory_binding = getattr(bindings, '_require_transaction_directory_binding')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    if name not in _TERMINAL_MARKERS | {ROLLBACK_DURABLE_MARKER}:
        raise ValueError("unknown publication transaction marker")
    figures_fd, root_fd, transaction_fd = _open_transaction_directory(
        transaction_dir
    )
    try:
        _require_transaction_directory_binding(
            figures_fd,
            root_fd,
            transaction_fd,
            _absolute(transaction_dir).name,
        )
        try:
            metadata = os.stat(
                name, dir_fd=transaction_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("publication transaction marker is not a regular file")
        os.unlink(name, dir_fd=transaction_fd)
        os.fsync(transaction_fd)
        _require_transaction_directory_binding(
            figures_fd,
            root_fd,
            transaction_fd,
            _absolute(transaction_dir).name,
        )
    finally:
        os.close(transaction_fd)
        os.close(root_fd)
        os.close(figures_fd)

def _sha256(path: Path, *, bindings) -> str | None:
    Path = getattr(bindings, 'Path')
    _close_parent_chain = getattr(bindings, '_close_parent_chain')
    _digest_regular_file_at = getattr(bindings, '_digest_regular_file_at')
    _open_known_file_parent = getattr(bindings, '_open_known_file_parent')
    parent_fd: int | None = None
    ancestors: list[int] = []
    try:
        parent_fd, ancestors = _open_known_file_parent(Path(path), create=False)
        return _digest_regular_file_at(parent_fd, Path(path).name)
    except (OSError, ValueError):
        return None
    finally:
        if parent_fd is not None:
            _close_parent_chain(parent_fd, ancestors)

def _snapshot_record(record: Any, *, include_preview: bool=False, bindings) -> dict[str, Any]:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    if record is None:
        return {}
    if hasattr(record, "model_dump"):
        dumped = record.model_dump(mode="json")
    elif isinstance(record, Mapping):
        dumped = dict(record)
    else:
        dumped = {
            key: value for key, value in vars(record).items()
            if not key.startswith("_")
        }
    if not include_preview:
        dumped["png_base64"] = None
    return dumped

def _find_record(records: Iterable[Any], record_id: str, *, bindings) -> Any | None:
    Any = getattr(bindings, 'Any')
    Iterable = getattr(bindings, 'Iterable')
    _value = getattr(bindings, '_value')
    return next(
        (record for record in records if str(_value(record, "id", "")) == record_id),
        None,
    )

def _journal_paths(figures_dir: Path, attempt_id: str, *, bindings) -> tuple[Path, Path]:
    JOURNAL_FILENAME = getattr(bindings, 'JOURNAL_FILENAME')
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _safe_component = getattr(bindings, '_safe_component')
    safe_attempt_id = _safe_component(attempt_id, "attempt")
    if str(attempt_id) != safe_attempt_id:
        raise ValueError("publication attempt ID is not a safe path component")
    transaction_dir = _absolute(figures_dir) / TRANSACTION_ROOT / safe_attempt_id
    return transaction_dir, transaction_dir / JOURNAL_FILENAME

def _require_within(path: Path, root: Path, *, label: str, bindings) -> Path:
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    resolved = _absolute(Path(path))
    root_resolved = _absolute(Path(root))
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} is outside its trusted root") from exc
    return resolved

def _validate_regular_child(parent_fd: int, name: str, *, label: str, bindings) -> None:
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"{label} is missing, invalid, or a symlink") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} is not a regular file")
    finally:
        os.close(descriptor)

def _validate_optional_regular_child(parent_fd: int, name: str, *, label: str, bindings) -> None:
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")

def _validate_registration_paths(*, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any, bindings) -> tuple[Path, str, str]:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _require_within = getattr(bindings, '_require_within')
    _safe_component = getattr(bindings, '_safe_component')
    _validate_optional_regular_child = getattr(bindings, '_validate_optional_regular_child')
    _validate_regular_child = getattr(bindings, '_validate_regular_child')
    _value = getattr(bindings, '_value')
    os = getattr(bindings, 'os')
    attempt_id = str(
        _value(published_record, "render_attempt_id", "") or Path(attempt_dir).name
    )
    if attempt_id != _safe_component(attempt_id, "attempt"):
        raise ValueError("publication attempt ID is not a safe path component")
    figure_name = str(_value(published_record, "name", "") or Path(final_jpg).stem)
    if figure_name != _safe_component(figure_name, "figure"):
        raise ValueError("publication figure name is not a safe path component")

    figures_dir = _absolute(Path(final_jpg).parent)
    expected_final_jpg = figures_dir / f"{figure_name}.jpg"
    expected_final_pdf = figures_dir / f"{figure_name}.pdf"
    expected_attempt_dir = figures_dir / ".attempts" / attempt_id
    if _absolute(Path(final_jpg)) != expected_final_jpg:
        raise ValueError("publication JPG is not the current public figure path")
    if _absolute(Path(final_pdf)) != expected_final_pdf:
        raise ValueError("publication PDF is not the current public figure path")
    if _absolute(Path(attempt_dir)) != expected_attempt_dir:
        raise ValueError("publication attempt directory is outside its trusted root")
    if _absolute(Path(staged_jpg)) != expected_attempt_dir / f"{figure_name}.jpg":
        raise ValueError("staged publication JPG does not match the attempt identity")
    if _absolute(Path(staged_pdf)) != expected_attempt_dir / f"{figure_name}.pdf":
        raise ValueError("staged publication PDF does not match the attempt identity")
    _require_within(expected_attempt_dir, figures_dir / ".attempts", label="attempt")
    figures_fd = _open_directory_nofollow(figures_dir)
    try:
        attempts_fd = _open_child_directory_nofollow(
            figures_fd, ".attempts", create=False
        )
        try:
            attempt_fd = _open_child_directory_nofollow(
                attempts_fd, attempt_id, create=False
            )
            try:
                _validate_regular_child(
                    attempt_fd, f"{figure_name}.jpg", label="staged publication JPG"
                )
                _validate_regular_child(
                    attempt_fd, f"{figure_name}.pdf", label="staged publication PDF"
                )
                _validate_optional_regular_child(
                    figures_fd, f"{figure_name}.jpg", label="publication JPG"
                )
                _validate_optional_regular_child(
                    figures_fd, f"{figure_name}.pdf", label="publication PDF"
                )
            finally:
                os.close(attempt_fd)
        finally:
            os.close(attempts_fd)
    finally:
        os.close(figures_fd)
    return figures_dir, attempt_id, figure_name

def _hydrate_verified_journal(payload: dict[str, Any], journal_path: Path, *, bindings) -> dict[str, Any]:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    TRANSACTION_SCHEMA_VERSION = getattr(bindings, 'TRANSACTION_SCHEMA_VERSION')
    _RUNTIME_JOURNAL_FIELDS = getattr(bindings, '_RUNTIME_JOURNAL_FIELDS')
    _absolute = getattr(bindings, '_absolute')
    _archive_record_component = getattr(bindings, '_archive_record_component')
    _publish_lock_keys = getattr(bindings, '_publish_lock_keys')
    _require_within = getattr(bindings, '_require_within')
    _safe_component = getattr(bindings, '_safe_component')
    if payload.get("schema_version") != TRANSACTION_SCHEMA_VERSION:
        raise ValueError("unsupported publication transaction journal schema")
    if any(field in payload for field in _RUNTIME_JOURNAL_FIELDS):
        raise ValueError("publication transaction journal contains persisted paths")

    session_id = str(payload.get("session_id") or "")
    attempt_id = str(payload.get("attempt_id") or "")
    figure_name = str(payload.get("figure_name") or "")
    if not session_id:
        raise ValueError("publication transaction journal has no session identity")
    if attempt_id != _safe_component(attempt_id, "attempt"):
        raise ValueError("publication transaction attempt identity is invalid")
    if figure_name != _safe_component(figure_name, "figure"):
        raise ValueError("publication transaction figure identity is invalid")

    transaction_root = journal_path.parent.parent
    if transaction_root.name != TRANSACTION_ROOT:
        raise ValueError("publication journal is outside the transaction root")
    figures_dir = _absolute(transaction_root.parent)
    expected_transaction_dir = transaction_root / attempt_id
    if journal_path.parent != expected_transaction_dir:
        raise ValueError("publication journal directory does not match its attempt")
    _require_within(journal_path.parent, transaction_root, label="transaction")

    new_record = payload.get("new_record")
    if not isinstance(new_record, dict):
        raise ValueError("publication journal has no record snapshot")
    new_record_id = str(payload.get("new_record_id") or "")
    if not new_record_id or str(new_record.get("id") or "") != new_record_id:
        raise ValueError("publication journal record identity is inconsistent")
    if str(new_record.get("render_attempt_id") or "") != attempt_id:
        raise ValueError("publication journal record attempt is inconsistent")
    if str(new_record.get("name") or "") != figure_name:
        raise ValueError("publication journal record figure name is inconsistent")

    old_record_id = str(payload.get("old_record_id") or "")

    attempt_dir = figures_dir / ".attempts" / attempt_id
    transaction_dir = transaction_root / attempt_id
    archive_dir = (
        figures_dir / ".revisions" / _archive_record_component(old_record_id) / attempt_id
        if old_record_id else None
    )
    hydrated = dict(payload)
    hydrated.update({
        "attempt_dir": str(attempt_dir),
        "staged_jpg": str(attempt_dir / f"{figure_name}.jpg"),
        "staged_pdf": str(attempt_dir / f"{figure_name}.pdf"),
        "final_jpg": str(figures_dir / f"{figure_name}.jpg"),
        "final_pdf": str(figures_dir / f"{figure_name}.pdf"),
        "backup_jpg": str(transaction_dir / "previous.jpg"),
        "backup_pdf": str(transaction_dir / "previous.pdf"),
        "archive_jpg": (
            str(archive_dir / f"{figure_name}.jpg") if archive_dir else ""
        ),
        "archive_pdf": (
            str(archive_dir / f"{figure_name}.pdf") if archive_dir else ""
        ),
        "journal_path": str(journal_path),
        "lock_keys": _publish_lock_keys(
            figures_dir, figures_dir / f"{figure_name}.jpg", new_record
        ),
    })
    return hydrated

def _read_verified_journal(journal_path: Path, *, bindings) -> dict[str, Any]:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _MAX_JOURNAL_BYTES = getattr(bindings, '_MAX_JOURNAL_BYTES')
    _absolute = getattr(bindings, '_absolute')
    _hydrate_verified_journal = getattr(bindings, '_hydrate_verified_journal')
    _journal_body = getattr(bindings, '_journal_body')
    _read_transaction_file = getattr(bindings, '_read_transaction_file')
    hmac = getattr(bindings, 'hmac')
    json = getattr(bindings, 'json')
    sign_hmac_sha256 = getattr(bindings, 'sign_hmac_sha256')
    journal_path = _absolute(Path(journal_path))
    raw = _read_transaction_file(
        journal_path,
        max_bytes=_MAX_JOURNAL_BYTES,
        label="publication transaction journal",
    )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("publication transaction journal is not an object")
    signature = str(payload.get("journal_hmac") or "")
    if not signature or not hmac.compare_digest(
        signature, sign_hmac_sha256(_journal_body(payload))
    ):
        raise ValueError("publication transaction journal HMAC is invalid")
    return _hydrate_verified_journal(payload, journal_path)

def _quarantine_transaction_dir(transaction_dir: Path, *, bindings) -> bool:
    """Isolate one untrusted root entry without following any journal path."""
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_transaction_root = getattr(bindings, '_open_transaction_root')
    _safe_component = getattr(bindings, '_safe_component')
    logger = getattr(bindings, 'logger')
    os = getattr(bindings, 'os')
    uuid = getattr(bindings, 'uuid')
    transaction_dir = _absolute(Path(transaction_dir))
    root = transaction_dir.parent
    if root.name != TRANSACTION_ROOT or transaction_dir.name.startswith("."):
        return False
    destination_name = (
        f"{_safe_component(transaction_dir.name, 'transaction')}-{uuid.uuid4().hex}"
    )
    figures_fd: int | None = None
    root_fd: int | None = None
    quarantine_fd: int | None = None
    try:
        figures_fd, root_fd = _open_transaction_root(root.parent, create=False)
        quarantine_fd = _open_child_directory_nofollow(
            root_fd, ".quarantine", create=True
        )
        os.rename(
            transaction_dir.name,
            destination_name,
            src_dir_fd=root_fd,
            dst_dir_fd=quarantine_fd,
        )
        os.fsync(root_fd)
        os.fsync(quarantine_fd)
        return True
    except (OSError, ValueError):
        logger.exception("Could not quarantine publication transaction %s", transaction_dir)
        return False
    finally:
        if quarantine_fd is not None:
            os.close(quarantine_fd)
        if root_fd is not None:
            os.close(root_fd)
        if figures_fd is not None:
            os.close(figures_fd)

def _persist_journal(journal_path: Path, journal: dict[str, Any], *, bindings) -> None:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _RUNTIME_JOURNAL_FIELDS = getattr(bindings, '_RUNTIME_JOURNAL_FIELDS')
    _atomic_json = getattr(bindings, '_atomic_json')
    _journal_body = getattr(bindings, '_journal_body')
    sign_hmac_sha256 = getattr(bindings, 'sign_hmac_sha256')
    payload = {
        key: value
        for key, value in journal.items()
        if key not in _RUNTIME_JOURNAL_FIELDS and key != "journal_hmac"
    }
    payload["journal_hmac"] = sign_hmac_sha256(_journal_body(payload))
    _atomic_json(journal_path, payload)
    journal["journal_hmac"] = payload["journal_hmac"]

def _complete_step(journal_path: Path, journal: dict[str, Any], step: str, *, status: str | None=None, bindings) -> None:
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _persist_journal = getattr(bindings, '_persist_journal')
    completed = list(journal.get("completed_steps") or [])
    if step not in completed:
        completed.append(step)
    journal["completed_steps"] = completed
    if status is not None:
        journal["status"] = status
    _persist_journal(journal_path, journal)
