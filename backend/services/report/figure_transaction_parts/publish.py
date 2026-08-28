"""Figure publication transaction lifecycle owner."""
from __future__ import annotations

def register_pending_figure_attempt(*, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any, session_id: str='', bindings) -> dict[str, Any]:
    """Persist the producer record while owning both publication locks."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    TRANSACTION_SCHEMA_VERSION = getattr(bindings, 'TRANSACTION_SCHEMA_VERSION')
    _PROCESS_TOKEN = getattr(bindings, '_PROCESS_TOKEN')
    _acquire_publish_locks = getattr(bindings, '_acquire_publish_locks')
    _fsync_dir = getattr(bindings, '_fsync_dir')
    _fsync_file = getattr(bindings, '_fsync_file')
    _hydrate_verified_journal = getattr(bindings, '_hydrate_verified_journal')
    _journal_body = getattr(bindings, '_journal_body')
    _journal_paths = getattr(bindings, '_journal_paths')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_transaction_root = getattr(bindings, '_open_transaction_root')
    _persist_journal = getattr(bindings, '_persist_journal')
    _publish_lock_keys = getattr(bindings, '_publish_lock_keys')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _release_publish_locks = getattr(bindings, '_release_publish_locks')
    _remove_transaction_dir = getattr(bindings, '_remove_transaction_dir')
    _sha256 = getattr(bindings, '_sha256')
    _snapshot_record = getattr(bindings, '_snapshot_record')
    _validate_registration_paths = getattr(bindings, '_validate_registration_paths')
    _value = getattr(bindings, '_value')
    os = getattr(bindings, 'os')
    sign_hmac_sha256 = getattr(bindings, 'sign_hmac_sha256')
    figures_dir, attempt_id, figure_name = _validate_registration_paths(attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record)
    if not str(session_id or ''):
        raise ValueError('publication transaction requires a session identity')
    transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
    lock_keys = _publish_lock_keys(figures_dir, Path(final_jpg), published_record)
    lock_error = _acquire_publish_locks(figures_dir, attempt_id, lock_keys, session_id=str(session_id))
    if lock_error:
        raise RuntimeError(lock_error)
    created_transaction_dir = False
    try:
        figures_fd, root_fd = _open_transaction_root(figures_dir, create=True)
        try:
            os.mkdir(attempt_id, mode=448, dir_fd=root_fd)
            created_transaction_dir = True
            transaction_fd = _open_child_directory_nofollow(root_fd, attempt_id, create=False)
            os.close(transaction_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
            os.close(figures_fd)
        _fsync_file(Path(staged_jpg))
        _fsync_file(Path(staged_pdf))
        _fsync_dir(Path(attempt_dir))
        new_render_sha256 = _sha256(Path(staged_jpg))
        new_pdf_sha256 = _sha256(Path(staged_pdf))
        if not new_render_sha256 or not new_pdf_sha256:
            raise ValueError('publication staged pair is not stable and regular')
        journal = {'schema_version': TRANSACTION_SCHEMA_VERSION, 'session_id': str(session_id), 'attempt_id': attempt_id, 'figure_name': figure_name, 'owner_pid': os.getpid(), 'owner_process_token': _PROCESS_TOKEN, 'status': 'REVIEW_PENDING', 'completed_steps': [], 'had_final_jpg': Path(final_jpg).exists(), 'had_final_pdf': Path(final_pdf).exists(), 'new_record_id': str(_value(published_record, 'id', '')), 'new_record': _snapshot_record(published_record), 'old_record_id': '', 'old_record_before': {}, 'new_render_sha256': new_render_sha256, 'new_pdf_sha256': new_pdf_sha256}
        journal['journal_hmac'] = sign_hmac_sha256(_journal_body(journal))
        runtime = _hydrate_verified_journal(journal, journal_path)
        _persist_journal(journal_path, runtime)
        return _read_verified_journal(journal_path)
    except BaseException:
        _release_publish_locks(attempt_id, keys=lock_keys, figures_dir=figures_dir, session_id=str(session_id))
        if created_transaction_dir:
            _remove_transaction_dir(transaction_dir)
        raise

def refreeze_pending_figure_attempt(pending_transaction: Mapping[str, Any], *, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any, bindings) -> dict[str, Any]:
    """Authenticate and freeze the final reviewed bytes of a pending attempt.

    Revisions replace the JPG/PDF inside the already-journaled attempt.  Only
    the process that owns the original authenticated transaction and all of its
    locks may advance the signed digests to those final candidate bytes.
    """
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    Path = getattr(bindings, 'Path')
    _HELD_LOCKS = getattr(bindings, '_HELD_LOCKS')
    _PROCESS_TOKEN = getattr(bindings, '_PROCESS_TOKEN')
    _TERMINAL_MARKERS = getattr(bindings, '_TERMINAL_MARKERS')
    _absolute = getattr(bindings, '_absolute')
    _authenticated_marker_exists = getattr(bindings, '_authenticated_marker_exists')
    _digest_open_regular_file = getattr(bindings, '_digest_open_regular_file')
    _journal_paths = getattr(bindings, '_journal_paths')
    _lock_owner = getattr(bindings, '_lock_owner')
    _open_attempt_directory = getattr(bindings, '_open_attempt_directory')
    _open_stable_staged_file = getattr(bindings, '_open_stable_staged_file')
    _persist_journal = getattr(bindings, '_persist_journal')
    _publish_lock_keys = getattr(bindings, '_publish_lock_keys')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _require_attempt_directory_binding = getattr(bindings, '_require_attempt_directory_binding')
    _require_regular_entry_binding = getattr(bindings, '_require_regular_entry_binding')
    _snapshot_record = getattr(bindings, '_snapshot_record')
    _validate_registration_paths = getattr(bindings, '_validate_registration_paths')
    _value = getattr(bindings, '_value')
    os = getattr(bindings, 'os')
    figures_dir, attempt_id, figure_name = _validate_registration_paths(attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record)
    _transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
    if _absolute(Path(str(pending_transaction.get('journal_path') or ''))) != journal_path:
        raise ValueError('pending publication transaction journal identity changed')
    journal = _read_verified_journal(journal_path)
    for field, expected in {'session_id': journal.get('session_id'), 'attempt_id': attempt_id, 'figure_name': figure_name, 'new_record_id': str(_value(published_record, 'id', '') or ''), 'journal_hmac': journal.get('journal_hmac')}.items():
        if str(pending_transaction.get(field) or '') != str(expected or ''):
            raise ValueError(f'pending publication transaction {field} identity changed')
    if str(journal.get('attempt_id') or '') != attempt_id or str(journal.get('figure_name') or '') != figure_name or str(journal.get('new_record_id') or '') != str(_value(published_record, 'id', '') or '') or (str(journal.get('status') or '') != 'REVIEW_PENDING') or list(journal.get('completed_steps') or []) or str(journal.get('old_record_id') or ''):
        raise ValueError('publication transaction is not an untouched pending attempt')
    if int(journal.get('owner_pid') or -1) != os.getpid() or str(journal.get('owner_process_token') or '') != _PROCESS_TOKEN:
        raise RuntimeError('publication transaction is not owned by this producer')
    if any((_authenticated_marker_exists(journal_path.parent, marker, journal) for marker in _TERMINAL_MARKERS)):
        raise RuntimeError('publication transaction is already terminal')
    lock_keys = _publish_lock_keys(figures_dir, Path(final_jpg), published_record)
    if list(journal.get('lock_keys') or []) != lock_keys:
        raise ValueError('publication transaction lock identity changed')
    lock_owner = _lock_owner(figures_dir, str(journal.get('session_id') or ''), attempt_id, lock_keys)
    if any((key not in _HELD_LOCKS or _HELD_LOCKS[key][0] != lock_owner for key in lock_keys)):
        raise RuntimeError('publication transaction locks are not held by this producer')
    figures_fd: int | None = None
    attempts_fd: int | None = None
    attempt_fd: int | None = None
    jpg_fd: int | None = None
    pdf_fd: int | None = None
    try:
        figures_fd, attempts_fd, attempt_fd = _open_attempt_directory(Path(attempt_dir), create=False)
        _require_attempt_directory_binding(figures_fd, attempts_fd, attempt_fd, attempt_id)
        jpg_name = f'{figure_name}.jpg'
        pdf_name = f'{figure_name}.pdf'
        jpg_fd, _jpg_metadata, render_digest = _open_stable_staged_file(attempt_fd, jpg_name, label='reviewed staged publication JPG')
        pdf_fd, _pdf_metadata, pdf_digest = _open_stable_staged_file(attempt_fd, pdf_name, label='reviewed staged publication PDF')
        _require_regular_entry_binding(attempt_fd, jpg_name, jpg_fd, label='reviewed staged publication JPG')
        _require_regular_entry_binding(attempt_fd, pdf_name, pdf_fd, label='reviewed staged publication PDF')
        journal.update({'owner_pid': os.getpid(), 'owner_process_token': _PROCESS_TOKEN, 'new_record': _snapshot_record(published_record), 'new_render_sha256': render_digest, 'new_pdf_sha256': pdf_digest})
        _persist_journal(journal_path, journal)
        _require_attempt_directory_binding(figures_fd, attempts_fd, attempt_fd, attempt_id)
        _require_regular_entry_binding(attempt_fd, jpg_name, jpg_fd, label='reviewed staged publication JPG')
        _require_regular_entry_binding(attempt_fd, pdf_name, pdf_fd, label='reviewed staged publication PDF')
        if _digest_open_regular_file(jpg_fd, label='reviewed staged publication JPG') != render_digest or _digest_open_regular_file(pdf_fd, label='reviewed staged publication PDF') != pdf_digest:
            raise ValueError('reviewed publication pair changed while it was frozen')
        verified = _read_verified_journal(journal_path)
        if str(verified.get('new_render_sha256') or '') != render_digest or str(verified.get('new_pdf_sha256') or '') != pdf_digest:
            raise ValueError('reviewed publication pair freeze was not durable')
        return verified
    finally:
        if pdf_fd is not None:
            os.close(pdf_fd)
        if jpg_fd is not None:
            os.close(jpg_fd)
        if attempt_fd is not None:
            os.close(attempt_fd)
        if attempts_fd is not None:
            os.close(attempts_fd)
        if figures_fd is not None:
            os.close(figures_fd)

def begin_figure_publish_transaction(*, attempt_dir: Path, staged_jpg: Path, staged_pdf: Path, final_jpg: Path, final_pdf: Path, published_record: Any, archived_record: Any=None, bindings) -> tuple[dict[str, Any] | None, str | None]:
    """Durably replace a validated pair, leaving its record uncommitted.

    The returned transaction remains recoverable until
    :func:`commit_figure_publish_transaction` writes its marker.  A normal I/O
    exception is rolled back here.  A process death or injected ``BaseException``
    is intentionally left to :func:`recover_figure_publish_transactions`.
    """
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _PROCESS_TOKEN = getattr(bindings, '_PROCESS_TOKEN')
    _TERMINAL_MARKERS = getattr(bindings, '_TERMINAL_MARKERS')
    _absolute = getattr(bindings, '_absolute')
    _acquire_publish_locks = getattr(bindings, '_acquire_publish_locks')
    _authenticated_marker_exists = getattr(bindings, '_authenticated_marker_exists')
    _complete_step = getattr(bindings, '_complete_step')
    _copy_durable = getattr(bindings, '_copy_durable')
    _digest_open_regular_file = getattr(bindings, '_digest_open_regular_file')
    _journal_paths = getattr(bindings, '_journal_paths')
    _open_attempt_directory = getattr(bindings, '_open_attempt_directory')
    _open_frozen_staged_file = getattr(bindings, '_open_frozen_staged_file')
    _path_from_journal = getattr(bindings, '_path_from_journal')
    _persist_journal = getattr(bindings, '_persist_journal')
    _publish_lock_keys = getattr(bindings, '_publish_lock_keys')
    _quarantine_transaction_dir = getattr(bindings, '_quarantine_transaction_dir')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _regular_child_exists = getattr(bindings, '_regular_child_exists')
    _release_publish_locks = getattr(bindings, '_release_publish_locks')
    _replace_staged_entry = getattr(bindings, '_replace_staged_entry')
    _require_attempt_directory_binding = getattr(bindings, '_require_attempt_directory_binding')
    _safe_component = getattr(bindings, '_safe_component')
    _snapshot_record = getattr(bindings, '_snapshot_record')
    _validate_registration_paths = getattr(bindings, '_validate_registration_paths')
    _value = getattr(bindings, '_value')
    _verify_published_entry = getattr(bindings, '_verify_published_entry')
    os = getattr(bindings, 'os')
    rollback_figure_publish_transaction = getattr(bindings, 'rollback_figure_publish_transaction')
    try:
        figures_dir, attempt_id, safe_name = _validate_registration_paths(attempt_dir=attempt_dir, staged_jpg=staged_jpg, staged_pdf=staged_pdf, final_jpg=final_jpg, final_pdf=final_pdf, published_record=published_record)
        transaction_dir, journal_path = _journal_paths(figures_dir, attempt_id)
        journal = _read_verified_journal(journal_path)
    except Exception as exc:
        figures_dir = _absolute(Path(final_jpg).parent)
        transaction_dir = figures_dir / TRANSACTION_ROOT / _safe_component(_value(published_record, 'render_attempt_id', ''), 'attempt')
        _quarantine_transaction_dir(transaction_dir)
        failed_attempt_id = str(_value(published_record, 'render_attempt_id', '') or '')
        _release_publish_locks(failed_attempt_id, figures_dir=figures_dir)
        return (None, f'Could not verify publication transaction journal: {exc}')
    if str(journal.get('new_record_id') or '') != str(_value(published_record, 'id', '') or '') or str(journal.get('attempt_id') or '') != attempt_id or str(journal.get('figure_name') or '') != safe_name:
        _quarantine_transaction_dir(transaction_dir)
        _release_publish_locks(attempt_id, keys=journal.get('lock_keys') or [], figures_dir=figures_dir, session_id=str(journal.get('session_id') or ''))
        return (None, 'Could not publish a mismatched figure transaction')
    if any((_authenticated_marker_exists(transaction_dir, marker, journal) for marker in _TERMINAL_MARKERS)):
        _release_publish_locks(attempt_id, keys=journal.get('lock_keys') or [], figures_dir=figures_dir, session_id=str(journal.get('session_id') or ''))
        return (None, 'Could not publish a terminalized figure attempt')
    lock_keys = _publish_lock_keys(figures_dir, Path(final_jpg), published_record)
    lock_error = _acquire_publish_locks(figures_dir, attempt_id, lock_keys, session_id=str(journal.get('session_id') or ''))
    if lock_error:
        return (None, lock_error)
    attempt_figures_fd: int | None = None
    attempts_fd: int | None = None
    attempt_fd: int | None = None
    staged_jpg_fd: int | None = None
    staged_pdf_fd: int | None = None
    try:
        attempt_figures_fd, attempts_fd, attempt_fd = _open_attempt_directory(Path(attempt_dir), create=False)
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
        expected_render_sha256 = str(journal.get('new_render_sha256') or '')
        expected_pdf_sha256 = str(journal.get('new_pdf_sha256') or '')
        if not expected_render_sha256 or not expected_pdf_sha256:
            raise ValueError('publication journal has no frozen pair digest')
        staged_jpg_fd, staged_jpg_metadata = _open_frozen_staged_file(attempt_fd, f'{safe_name}.jpg', expected_render_sha256, label='staged publication JPG')
        staged_pdf_fd, staged_pdf_metadata = _open_frozen_staged_file(attempt_fd, f'{safe_name}.pdf', expected_pdf_sha256, label='staged publication PDF')
        had_final_jpg = _regular_child_exists(attempt_figures_fd, f'{safe_name}.jpg', label='publication JPG')
        had_final_pdf = _regular_child_exists(attempt_figures_fd, f'{safe_name}.pdf', label='publication PDF')
        journal.update({'owner_pid': os.getpid(), 'owner_process_token': _PROCESS_TOKEN, 'status': 'PREPARED', 'completed_steps': [], 'had_final_jpg': had_final_jpg, 'had_final_pdf': had_final_pdf, 'new_record_id': str(_value(published_record, 'id', '')), 'new_record': _snapshot_record(published_record), 'old_record_id': str(_value(archived_record, 'id', '') or ''), 'old_record_before': _snapshot_record(archived_record)})
        _persist_journal(journal_path, journal)
        journal = _read_verified_journal(journal_path)
        backup_jpg = Path(journal['backup_jpg'])
        backup_pdf = Path(journal['backup_pdf'])
        archive_jpg = _path_from_journal(journal, 'archive_jpg')
        archive_pdf = _path_from_journal(journal, 'archive_pdf')
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
        if had_final_jpg:
            _copy_durable(Path(final_jpg), backup_jpg)
            _complete_step(journal_path, journal, 'backup_jpg')
        if had_final_pdf:
            _copy_durable(Path(final_pdf), backup_pdf)
            _complete_step(journal_path, journal, 'backup_pdf')
        if archived_record is not None:
            if had_final_jpg and archive_jpg is not None:
                _copy_durable(backup_jpg, archive_jpg)
                _complete_step(journal_path, journal, 'archive_jpg')
            if had_final_pdf and archive_pdf is not None:
                _copy_durable(backup_pdf, archive_pdf)
                _complete_step(journal_path, journal, 'archive_pdf')
        _complete_step(journal_path, journal, 'replace_jpg_started', status='JPG_REPLACE_INTENT')
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
        if _digest_open_regular_file(staged_jpg_fd, label='staged publication JPG') != expected_render_sha256:
            raise ValueError('staged publication JPG changed before publish')
        _replace_staged_entry(attempt_fd, attempt_figures_fd, f'{safe_name}.jpg', f'{safe_name}.jpg')
        _verify_published_entry(attempt_figures_fd, f'{safe_name}.jpg', staged_jpg_fd, staged_jpg_metadata, expected_render_sha256, label='published JPG')
        os.fsync(staged_jpg_fd)
        os.fsync(attempt_figures_fd)
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
        _complete_step(journal_path, journal, 'replace_jpg', status='JPG_REPLACED')
        _complete_step(journal_path, journal, 'replace_pdf_started', status='PDF_REPLACE_INTENT')
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
        if _digest_open_regular_file(staged_pdf_fd, label='staged publication PDF') != expected_pdf_sha256:
            raise ValueError('staged publication PDF changed before publish')
        _replace_staged_entry(attempt_fd, attempt_figures_fd, f'{safe_name}.pdf', f'{safe_name}.pdf')
        _verify_published_entry(attempt_figures_fd, f'{safe_name}.pdf', staged_pdf_fd, staged_pdf_metadata, expected_pdf_sha256, label='published PDF')
        os.fsync(staged_pdf_fd)
        os.fsync(attempt_figures_fd)
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
        _complete_step(journal_path, journal, 'replace_pdf', status='FILES_REPLACED')
        journal = _read_verified_journal(journal_path)
        _verify_published_entry(attempt_figures_fd, f'{safe_name}.jpg', staged_jpg_fd, staged_jpg_metadata, expected_render_sha256, label='published JPG')
        _verify_published_entry(attempt_figures_fd, f'{safe_name}.pdf', staged_pdf_fd, staged_pdf_metadata, expected_pdf_sha256, label='published PDF')
        _require_attempt_directory_binding(attempt_figures_fd, attempts_fd, attempt_fd, attempt_id)
    except Exception as exc:
        rollback_figure_publish_transaction(journal_path, state=None, producer_status='failed')
        return (None, f'Could not publish validated figure render: {exc}')
    finally:
        if staged_pdf_fd is not None:
            os.close(staged_pdf_fd)
        if staged_jpg_fd is not None:
            os.close(staged_jpg_fd)
        if attempt_fd is not None:
            os.close(attempt_fd)
        if attempts_fd is not None:
            os.close(attempts_fd)
        if attempt_figures_fd is not None:
            os.close(attempt_figures_fd)
    return (journal, None)

def apply_publish_record_paths(transaction: dict[str, Any], published_record: Any, archived_record: Any=None, *, bindings) -> None:
    """Apply the file half of a transaction to its in-memory records."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _set = getattr(bindings, '_set')
    _value = getattr(bindings, '_value')
    verified = _read_verified_journal(Path(transaction['journal_path']))
    if str(_value(published_record, 'id', '')) != str(verified.get('new_record_id') or ''):
        raise ValueError('published record does not match transaction identity')
    _set(published_record, 'png_path', verified['final_jpg'])
    _set(published_record, 'pdf_path', verified['final_pdf'])
    if archived_record is not None:
        if str(_value(archived_record, 'id', '')) != str(verified.get('old_record_id') or ''):
            raise ValueError('archived record does not match transaction identity')
        archive_jpg = str(verified.get('archive_jpg') or '')
        archive_pdf = str(verified.get('archive_pdf') or '')
        if archive_jpg:
            _set(archived_record, 'png_path', archive_jpg)
        if archive_pdf:
            _set(archived_record, 'pdf_path', archive_pdf)

def mark_publish_record_applied(transaction: dict[str, Any], *, published_record: Any, archived_record: Any=None, bindings) -> None:
    """Journal the exact record state that the session write must persist."""
    Any = getattr(bindings, 'Any')
    Path = getattr(bindings, 'Path')
    _complete_step = getattr(bindings, '_complete_step')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _snapshot_record = getattr(bindings, '_snapshot_record')
    _value = getattr(bindings, '_value')
    journal_path = Path(transaction['journal_path'])
    verified = _read_verified_journal(journal_path)
    if str(_value(published_record, 'id', '')) != str(verified.get('new_record_id') or ''):
        raise ValueError('published record does not match transaction identity')
    if archived_record is not None and str(_value(archived_record, 'id', '')) != str(verified.get('old_record_id') or ''):
        raise ValueError('archived record does not match transaction identity')
    verified['new_record'] = _snapshot_record(published_record)
    verified['old_record_after'] = _snapshot_record(archived_record)
    _complete_step(journal_path, verified, 'record_applied', status='RECORD_APPLIED')
    transaction.clear()
    transaction.update(_read_verified_journal(journal_path))

def commit_figure_publish_transaction(transaction: dict[str, Any], *, bindings) -> None:
    """Commit after the record boundary has been durably persisted."""
    Any = getattr(bindings, 'Any')
    COMMIT_MARKER = getattr(bindings, 'COMMIT_MARKER')
    Path = getattr(bindings, 'Path')
    _cleanup_attempt = getattr(bindings, '_cleanup_attempt')
    _persist_journal = getattr(bindings, '_persist_journal')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _release_journal_locks = getattr(bindings, '_release_journal_locks')
    _remove_transaction_dir = getattr(bindings, '_remove_transaction_dir')
    _write_marker = getattr(bindings, '_write_marker')
    logger = getattr(bindings, 'logger')
    journal_path = Path(transaction['journal_path'])
    verified = _read_verified_journal(journal_path)
    verified['status'] = 'RECORD_DURABLE'
    _persist_journal(journal_path, verified)
    _write_marker(journal_path.parent / COMMIT_MARKER)
    verified['status'] = 'COMMITTED'
    try:
        _persist_journal(journal_path, verified)
    except Exception:
        logger.exception('Publication transaction committed but final journal update failed: %s', journal_path)
    attempt_clean = _cleanup_attempt(verified)
    _release_journal_locks(verified)
    if attempt_clean:
        _remove_transaction_dir(journal_path.parent)

def _path_from_journal(journal: Mapping[str, Any], key: str, *, bindings) -> Path | None:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    Path = getattr(bindings, 'Path')
    value = str(journal.get(key) or '')
    return Path(value) if value else None

def _release_journal_locks(journal: Mapping[str, Any], *, bindings) -> None:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _path_from_journal = getattr(bindings, '_path_from_journal')
    _release_publish_locks = getattr(bindings, '_release_publish_locks')
    final_jpg = _path_from_journal(journal, 'final_jpg')
    if final_jpg is None:
        return
    _release_publish_locks(str(journal.get('attempt_id') or ''), keys=journal.get('lock_keys') or [], figures_dir=final_jpg.parent, session_id=str(journal.get('session_id') or ''))

def _cleanup_attempt(journal: Mapping[str, Any], *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _open_hidden_root = getattr(bindings, '_open_hidden_root')
    _path_from_journal = getattr(bindings, '_path_from_journal')
    _remove_bounded_root_entry = getattr(bindings, '_remove_bounded_root_entry')
    logger = getattr(bindings, 'logger')
    os = getattr(bindings, 'os')
    attempt_dir = _path_from_journal(journal, 'attempt_dir')
    if attempt_dir is None:
        return True
    attempts_root = attempt_dir.parent
    if attempts_root.name != '.attempts':
        return False
    try:
        root_fd, parents = _open_hidden_root(attempts_root, create=False)
    except ValueError:
        try:
            os.lstat(attempts_root)
        except FileNotFoundError:
            return True
        return False
    try:
        try:
            os.stat(attempt_dir.name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
    finally:
        os.close(root_fd)
        for descriptor in reversed(parents):
            os.close(descriptor)
    if _remove_bounded_root_entry(attempt_dir, attempts_root):
        try:
            root_fd, parents = _open_hidden_root(attempts_root, create=False)
        except ValueError:
            return True
        try:
            if not os.listdir(root_fd):
                os.close(root_fd)
                root_fd = -1
                os.rmdir(attempts_root.name, dir_fd=parents[-1])
                os.fsync(parents[-1])
        except OSError:
            pass
        finally:
            if root_fd >= 0:
                os.close(root_fd)
            for descriptor in reversed(parents):
                os.close(descriptor)
        return True
    else:
        logger.warning('Could not clean publication attempt %s', attempt_dir)
        return False

def _remove_transaction_dir(transaction_dir: Path, *, bindings) -> bool:
    Path = getattr(bindings, 'Path')
    _remove_bounded_root_entry = getattr(bindings, '_remove_bounded_root_entry')
    logger = getattr(bindings, 'logger')
    if _remove_bounded_root_entry(transaction_dir, transaction_dir.parent):
        return True
    else:
        logger.warning('Could not garbage-collect figure transaction %s', transaction_dir)
        return False

def _remove_revision_archive_dir(archive_dir: Path, *, bindings) -> bool:
    """Remove one derived revision attempt without following hidden roots."""
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _safe_component = getattr(bindings, '_safe_component')
    os = getattr(bindings, 'os')
    shutil = getattr(bindings, 'shutil')
    stat = getattr(bindings, 'stat')
    archive_dir = _absolute(archive_dir)
    record_dir = archive_dir.parent
    revision_root = record_dir.parent
    figures_dir = revision_root.parent
    if revision_root.name != '.revisions' or len(record_dir.name) != 64 or any((char not in '0123456789abcdef' for char in record_dir.name)) or (archive_dir.name != _safe_component(archive_dir.name, 'attempt')):
        return False
    figures_fd: int | None = None
    revision_fd: int | None = None
    record_fd: int | None = None
    try:
        figures_fd = _open_directory_nofollow(figures_dir)
        revision_fd = _open_child_directory_nofollow(figures_fd, '.revisions', create=False)
        record_fd = _open_child_directory_nofollow(revision_fd, record_dir.name, create=False)
        metadata = os.stat(archive_dir.name, dir_fd=record_fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            return False
        shutil.rmtree(archive_dir.name, dir_fd=record_fd)
        os.fsync(record_fd)
        if not os.listdir(record_fd):
            os.close(record_fd)
            record_fd = None
            os.rmdir(record_dir.name, dir_fd=revision_fd)
            os.fsync(revision_fd)
        if not os.listdir(revision_fd):
            os.close(revision_fd)
            revision_fd = None
            os.rmdir('.revisions', dir_fd=figures_fd)
            os.fsync(figures_fd)
        return True
    except (OSError, ValueError):
        return False
    finally:
        if record_fd is not None:
            os.close(record_fd)
        if revision_fd is not None:
            os.close(revision_fd)
        if figures_fd is not None:
            os.close(figures_fd)
