"""Figure publication recovery and bounded garbage collection owner."""
from __future__ import annotations

def _open_hidden_root(root: Path, *, create: bool=False, bindings) -> tuple[int, list[int]]:
    """Open one recognized hidden root and retain its trusted parent FDs."""
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_directory_nofollow = getattr(bindings, '_open_directory_nofollow')
    _open_transaction_root = getattr(bindings, '_open_transaction_root')
    os = getattr(bindings, 'os')
    root = _absolute(root)
    if root.name == '.attempts':
        figures_fd = _open_directory_nofollow(root.parent)
        try:
            root_fd = _open_child_directory_nofollow(figures_fd, root.name, create=create)
        except BaseException:
            os.close(figures_fd)
            raise
        return (root_fd, [figures_fd])
    if root.name == TRANSACTION_ROOT:
        figures_fd, root_fd = _open_transaction_root(root.parent, create=create)
        return (root_fd, [figures_fd])
    if root.name in {'.locks', '.quarantine'} and root.parent.name == TRANSACTION_ROOT:
        figures_fd, transaction_fd = _open_transaction_root(root.parent.parent, create=create)
        try:
            root_fd = _open_child_directory_nofollow(transaction_fd, root.name, create=create)
        except BaseException:
            os.close(transaction_fd)
            os.close(figures_fd)
            raise
        return (root_fd, [figures_fd, transaction_fd])
    raise ValueError('path is not a recognized hidden publication root')

def _remove_bounded_root_entry(path: Path, root: Path, *, bindings) -> bool:
    """Remove one direct child without following a symlink outside *root*."""
    Path = getattr(bindings, 'Path')
    _absolute = getattr(bindings, '_absolute')
    _open_hidden_root = getattr(bindings, '_open_hidden_root')
    os = getattr(bindings, 'os')
    shutil = getattr(bindings, 'shutil')
    stat = getattr(bindings, 'stat')
    path = _absolute(Path(path))
    root = _absolute(Path(root))
    if path.parent != root or path.name in {'', '.', '..'}:
        return False
    root_fd: int | None = None
    parents: list[int] = []
    try:
        root_fd, parents = _open_hidden_root(root, create=False)
        metadata = os.stat(path.name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
            os.unlink(path.name, dir_fd=root_fd)
        elif stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(path.name, dir_fd=root_fd)
        else:
            return False
        os.fsync(root_fd)
        return True
    except (OSError, ValueError):
        return False
    finally:
        if root_fd is not None:
            os.close(root_fd)
        for descriptor in reversed(parents):
            os.close(descriptor)

def garbage_collect_figure_transaction_roots(figures_dir: Path, *, budget: int | None=None, bindings) -> int:
    """Bound maintenance strictly to hidden publication transaction roots."""
    JOURNAL_FILENAME = getattr(bindings, 'JOURNAL_FILENAME')
    Path = getattr(bindings, 'Path')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _MAX_GC_ENTRIES = getattr(bindings, '_MAX_GC_ENTRIES')
    _QUARANTINE_RETENTION_SECONDS = getattr(bindings, '_QUARANTINE_RETENTION_SECONDS')
    _STALE_ENTRY_SECONDS = getattr(bindings, '_STALE_ENTRY_SECONDS')
    _absolute = getattr(bindings, '_absolute')
    _open_child_directory_nofollow = getattr(bindings, '_open_child_directory_nofollow')
    _open_hidden_root = getattr(bindings, '_open_hidden_root')
    _remove_bounded_root_entry = getattr(bindings, '_remove_bounded_root_entry')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    time = getattr(bindings, 'time')
    figures_dir = _absolute(Path(figures_dir))
    transaction_root = figures_dir / TRANSACTION_ROOT
    now = time.time()
    remaining = _MAX_GC_ENTRIES if budget is None else max(0, int(budget))
    try:
        transaction_fd, transaction_parents = _open_hidden_root(transaction_root, create=False)
    except ValueError:
        return remaining
    quarantine_root = transaction_root / '.quarantine'
    try:
        quarantine_fd, quarantine_parents = _open_hidden_root(quarantine_root, create=False)
    except ValueError:
        quarantine_fd = None
        quarantine_parents = []
    if quarantine_fd is not None:
        try:
            quarantine_names = sorted(os.listdir(quarantine_fd))
        finally:
            os.close(quarantine_fd)
            for descriptor in reversed(quarantine_parents):
                os.close(descriptor)
        for name in quarantine_names:
            if remaining <= 0:
                break
            entry = quarantine_root / name
            try:
                stale = now - entry.lstat().st_mtime >= _QUARANTINE_RETENTION_SECONDS
            except OSError:
                continue
            if stale and _remove_bounded_root_entry(entry, quarantine_root):
                remaining -= 1
    attempts_root = figures_dir / '.attempts'
    active_attempts: set[str] = set()
    for name in os.listdir(transaction_fd):
        if name.startswith('.'):
            continue
        try:
            child_fd = _open_child_directory_nofollow(transaction_fd, name, create=False)
        except ValueError:
            continue
        try:
            metadata = os.stat(JOURNAL_FILENAME, dir_fd=child_fd, follow_symlinks=False)
            if stat.S_ISREG(metadata.st_mode):
                active_attempts.add(name)
        except OSError:
            pass
        finally:
            os.close(child_fd)
    try:
        attempts_fd, attempts_parents = _open_hidden_root(attempts_root, create=False)
    except ValueError:
        attempts_fd = None
        attempts_parents = []
    if attempts_fd is not None:
        try:
            attempt_names = sorted(os.listdir(attempts_fd))
        finally:
            os.close(attempts_fd)
            for descriptor in reversed(attempts_parents):
                os.close(descriptor)
        for name in attempt_names:
            if remaining <= 0:
                break
            if name in active_attempts:
                continue
            entry = attempts_root / name
            try:
                stale = now - entry.lstat().st_mtime >= _STALE_ENTRY_SECONDS
            except OSError:
                continue
            if stale and _remove_bounded_root_entry(entry, attempts_root):
                remaining -= 1
    lock_root = transaction_root / '.locks'
    try:
        lock_fd, lock_parents = _open_hidden_root(lock_root, create=False)
    except ValueError:
        lock_fd = None
        lock_parents = []
    if lock_fd is not None:
        try:
            import fcntl
        except ImportError:
            os.close(lock_fd)
            for descriptor in reversed(lock_parents):
                os.close(descriptor)
            return remaining
        for name in sorted(os.listdir(lock_fd)):
            if remaining <= 0:
                break
            if Path(name).suffix != '.lock':
                continue
            try:
                metadata = os.stat(name, dir_fd=lock_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                if now - metadata.st_mtime < _STALE_ENTRY_SECONDS:
                    continue
                descriptor = os.open(name, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), dir_fd=lock_fd)
                with os.fdopen(descriptor, 'a+b') as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.unlink(name, dir_fd=lock_fd)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                os.fsync(lock_fd)
                remaining -= 1
            except (BlockingIOError, OSError):
                continue
        os.close(lock_fd)
        for descriptor in reversed(lock_parents):
            os.close(descriptor)
    os.close(transaction_fd)
    for descriptor in reversed(transaction_parents):
        os.close(descriptor)
    return remaining

def _restore_file(*, final_path: Path | None, backup_path: Path | None, was_replaced: bool, had_final: bool, new_digest: str | None, bindings) -> None:
    Path = getattr(bindings, 'Path')
    _close_parent_chain = getattr(bindings, '_close_parent_chain')
    _copy_durable = getattr(bindings, '_copy_durable')
    _fsync_dir = getattr(bindings, '_fsync_dir')
    _fsync_file = getattr(bindings, '_fsync_file')
    _open_known_file_parent = getattr(bindings, '_open_known_file_parent')
    _sha256 = getattr(bindings, '_sha256')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    uuid = getattr(bindings, 'uuid')
    if final_path is None or not was_replaced:
        return
    if had_final and backup_path is not None:
        temporary = final_path.with_name(f'.{final_path.name}.rollback-{uuid.uuid4().hex}')
        _copy_durable(backup_path, temporary)
        os.replace(temporary, final_path)
        _fsync_file(final_path)
        _fsync_dir(final_path.parent)
        return
    if new_digest and _sha256(final_path) == new_digest:
        parent_fd, ancestors = _open_known_file_parent(final_path, create=False)
        try:
            metadata = os.stat(final_path.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError('publication rollback target is not regular')
            os.unlink(final_path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            _close_parent_chain(parent_fd, ancestors)

def _restore_record(record: Any, snapshot: Mapping[str, Any], *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _TRANSACTION_RECORD_FIELDS = getattr(bindings, '_TRANSACTION_RECORD_FIELDS')
    _set = getattr(bindings, '_set')
    _value = getattr(bindings, '_value')
    logger = getattr(bindings, 'logger')
    mutated = False
    for key, value in snapshot.items():
        if key not in _TRANSACTION_RECORD_FIELDS:
            continue
        try:
            if _value(record, key) != value:
                _set(record, key, value)
                mutated = True
        except (AttributeError, TypeError, ValueError):
            logger.debug('Could not restore figure record field %s', key)
    return mutated

def _abandon_new_record(state: Any, journal: Mapping[str, Any], producer_status: str, *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _find_record = getattr(bindings, '_find_record')
    _set = getattr(bindings, '_set')
    _value = getattr(bindings, '_value')
    logger = getattr(bindings, 'logger')
    if state is None:
        return False
    records = list(_value(state, 'publication_figures', []) or [])
    record_id = str(journal.get('new_record_id') or '')
    record = _find_record(records, record_id)
    mutated = False
    if record is None and journal.get('new_record'):
        try:
            from backend.models.schemas import PublicationFigure
            snapshot = dict(journal['new_record'])
            snapshot['png_base64'] = None
            record = PublicationFigure.model_validate(snapshot)
            records.append(record)
            _set(state, 'publication_figures', records)
            mutated = True
        except Exception:
            logger.exception('Could not reconstruct abandoned figure record %s', record_id)
            return mutated
    if record is not None:
        for key, value in {'producer_status': producer_status, 'png_base64': None, 'png_path': '', 'pdf_path': '', 'render_sha256': None, 'preview_sha256': None, 'pdf_sha256': None, 'is_published_head': False}.items():
            if _value(record, key) != value:
                _set(record, key, value)
                mutated = True
    return mutated

def _restore_abandoned_records(state: Any, journal: Mapping[str, Any], producer_status: str, *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _abandon_new_record = getattr(bindings, '_abandon_new_record')
    _find_record = getattr(bindings, '_find_record')
    _restore_record = getattr(bindings, '_restore_record')
    _value = getattr(bindings, '_value')
    if state is None:
        return False
    records = list(_value(state, 'publication_figures', []) or [])
    old_id = str(journal.get('old_record_id') or '')
    old_record = _find_record(records, old_id) if old_id else None
    mutated = False
    if old_record is not None and journal.get('old_record_before'):
        mutated = _restore_record(old_record, journal['old_record_before']) or mutated
    return _abandon_new_record(state, journal, producer_status) or mutated

def rollback_figure_publish_transaction(journal_path: Path, *, state: Any=None, producer_status: str='abandoned', bindings) -> bool:
    """Rollback one non-committed journal and terminalize its producer record."""
    ABANDON_MARKER = getattr(bindings, 'ABANDON_MARKER')
    Any = getattr(bindings, 'Any')
    COMMIT_MARKER = getattr(bindings, 'COMMIT_MARKER')
    FAILURE_MARKER = getattr(bindings, 'FAILURE_MARKER')
    Path = getattr(bindings, 'Path')
    ROLLBACK_DURABLE_MARKER = getattr(bindings, 'ROLLBACK_DURABLE_MARKER')
    _absolute = getattr(bindings, '_absolute')
    _authenticated_marker_exists = getattr(bindings, '_authenticated_marker_exists')
    _cleanup_attempt = getattr(bindings, '_cleanup_attempt')
    _path_from_journal = getattr(bindings, '_path_from_journal')
    _persist_journal = getattr(bindings, '_persist_journal')
    _quarantine_transaction_dir = getattr(bindings, '_quarantine_transaction_dir')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _release_journal_locks = getattr(bindings, '_release_journal_locks')
    _release_publish_locks = getattr(bindings, '_release_publish_locks')
    _remove_revision_archive_dir = getattr(bindings, '_remove_revision_archive_dir')
    _restore_abandoned_records = getattr(bindings, '_restore_abandoned_records')
    _restore_file = getattr(bindings, '_restore_file')
    _unlink_transaction_regular_file_if_present = getattr(bindings, '_unlink_transaction_regular_file_if_present')
    _write_marker = getattr(bindings, '_write_marker')
    logger = getattr(bindings, 'logger')
    try:
        journal = _read_verified_journal(Path(journal_path))
    except Exception:
        logger.exception('Could not read publication transaction %s', journal_path)
        transaction_dir = _absolute(Path(journal_path).parent)
        _release_publish_locks(transaction_dir.name, figures_dir=transaction_dir.parent.parent)
        _quarantine_transaction_dir(transaction_dir)
        return False
    if _authenticated_marker_exists(Path(journal_path).parent, COMMIT_MARKER, journal):
        logger.error('Refusing to roll back committed publication transaction %s', journal_path)
        _release_journal_locks(journal)
        return False
    completed = set(journal.get('completed_steps') or [])
    try:
        _restore_file(final_path=_path_from_journal(journal, 'final_pdf'), backup_path=_path_from_journal(journal, 'backup_pdf'), was_replaced=bool({'replace_pdf_started', 'replace_pdf'} & completed), had_final=bool(journal.get('had_final_pdf')), new_digest=str(journal.get('new_pdf_sha256') or '') or None)
        _restore_file(final_path=_path_from_journal(journal, 'final_jpg'), backup_path=_path_from_journal(journal, 'backup_jpg'), was_replaced=bool({'replace_jpg_started', 'replace_jpg'} & completed), had_final=bool(journal.get('had_final_jpg')), new_digest=str(journal.get('new_render_sha256') or '') or None)
    except Exception:
        logger.exception('Could not roll back publication transaction %s', journal_path)
        _release_journal_locks(journal)
        return False
    mutated = False
    if state is not None:
        mutated = _restore_abandoned_records(state, journal, producer_status) or mutated
    try:
        archive_jpg = _path_from_journal(journal, 'archive_jpg')
        archive_dir = archive_jpg.parent if archive_jpg is not None else None
        if archive_dir is not None:
            _remove_revision_archive_dir(archive_dir)
        _cleanup_attempt(journal)
        journal['status'] = producer_status.upper()
        _persist_journal(journal_path, journal)
        marker = FAILURE_MARKER if producer_status == 'failed' else ABANDON_MARKER
        _unlink_transaction_regular_file_if_present(journal_path.parent, ROLLBACK_DURABLE_MARKER)
        _write_marker(journal_path.parent / marker)
    except Exception:
        logger.exception('Could not persist terminal publication rollback %s', journal_path)
    finally:
        _release_journal_locks(journal)
    return mutated

def _record_matches_committed_files(record: Any, journal: Mapping[str, Any], *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    Path = getattr(bindings, 'Path')
    _sha256 = getattr(bindings, '_sha256')
    _value = getattr(bindings, '_value')
    if record is None or str(_value(record, 'producer_status', '')) != 'published':
        return False
    return bool(str(_value(record, 'png_path', '')) == str(journal.get('final_jpg') or '') and str(_value(record, 'pdf_path', '')) == str(journal.get('final_pdf') or '') and (_sha256(Path(str(journal.get('final_jpg') or ''))) == str(journal.get('new_render_sha256') or '')) and (_sha256(Path(str(journal.get('final_pdf') or ''))) == str(journal.get('new_pdf_sha256') or '')))

def _apply_committed_records(state: Any, journal: Mapping[str, Any], *, bindings) -> bool:
    Any = getattr(bindings, 'Any')
    Mapping = getattr(bindings, 'Mapping')
    _find_record = getattr(bindings, '_find_record')
    _restore_record = getattr(bindings, '_restore_record')
    _set = getattr(bindings, '_set')
    _value = getattr(bindings, '_value')
    logger = getattr(bindings, 'logger')
    if state is None:
        return False
    records = list(_value(state, 'publication_figures', []) or [])
    new_id = str(journal.get('new_record_id') or '')
    new_record = _find_record(records, new_id)
    mutated = False
    if new_record is None and journal.get('new_record'):
        try:
            from backend.models.schemas import PublicationFigure
            new_record = PublicationFigure.model_validate(journal['new_record'])
            records.append(new_record)
            _set(state, 'publication_figures', records)
            mutated = True
        except Exception:
            logger.exception('Could not reconstruct committed figure record %s', new_id)
            return mutated
    if new_record is not None and journal.get('new_record'):
        mutated = _restore_record(new_record, journal['new_record']) or mutated
    old_id = str(journal.get('old_record_id') or '')
    old_record = _find_record(records, old_id) if old_id else None
    if old_record is not None and journal.get('old_record_after'):
        mutated = _restore_record(old_record, journal['old_record_after']) or mutated
    return mutated

def recover_figure_publish_transactions(state: Any, figures_dir: Path, *, force: bool=False, applied_transaction_ids: set[str] | None=None, bindings) -> bool:
    """Recover every publication transaction visible to this workspace.

    A commit marker is authoritative.  Without one, a record that is already
    durable and hash-bound to both final files can be forward-committed; every
    other state is rolled back and terminalized as ``abandoned``.
    """
    ABANDON_MARKER = getattr(bindings, 'ABANDON_MARKER')
    Any = getattr(bindings, 'Any')
    COMMIT_MARKER = getattr(bindings, 'COMMIT_MARKER')
    FAILURE_MARKER = getattr(bindings, 'FAILURE_MARKER')
    JOURNAL_FILENAME = getattr(bindings, 'JOURNAL_FILENAME')
    Path = getattr(bindings, 'Path')
    ROLLBACK_DURABLE_MARKER = getattr(bindings, 'ROLLBACK_DURABLE_MARKER')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _MAX_GC_ENTRIES = getattr(bindings, '_MAX_GC_ENTRIES')
    _PROCESS_TOKEN = getattr(bindings, '_PROCESS_TOKEN')
    _TERMINAL_MARKERS = getattr(bindings, '_TERMINAL_MARKERS')
    _absolute = getattr(bindings, '_absolute')
    _acquire_publish_locks = getattr(bindings, '_acquire_publish_locks')
    _apply_committed_records = getattr(bindings, '_apply_committed_records')
    _authenticated_marker_exists = getattr(bindings, '_authenticated_marker_exists')
    _cleanup_attempt = getattr(bindings, '_cleanup_attempt')
    _find_record = getattr(bindings, '_find_record')
    _open_hidden_root = getattr(bindings, '_open_hidden_root')
    _open_transaction_directory = getattr(bindings, '_open_transaction_directory')
    _persist_journal = getattr(bindings, '_persist_journal')
    _publish_lock_keys = getattr(bindings, '_publish_lock_keys')
    _quarantine_transaction_dir = getattr(bindings, '_quarantine_transaction_dir')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _record_matches_committed_files = getattr(bindings, '_record_matches_committed_files')
    _release_publish_locks = getattr(bindings, '_release_publish_locks')
    _remove_transaction_dir = getattr(bindings, '_remove_transaction_dir')
    _restore_abandoned_records = getattr(bindings, '_restore_abandoned_records')
    _transaction_creation_is_live = getattr(bindings, '_transaction_creation_is_live')
    _value = getattr(bindings, '_value')
    _write_marker = getattr(bindings, '_write_marker')
    garbage_collect_figure_transaction_roots = getattr(bindings, 'garbage_collect_figure_transaction_roots')
    logger = getattr(bindings, 'logger')
    os = getattr(bindings, 'os')
    rollback_figure_publish_transaction = getattr(bindings, 'rollback_figure_publish_transaction')
    stat = getattr(bindings, 'stat')
    figures_dir = _absolute(Path(figures_dir))
    root = figures_dir / TRANSACTION_ROOT
    try:
        root_fd, root_parents = _open_hidden_root(root, create=False)
    except ValueError:
        return False
    try:
        transaction_names = sorted(os.listdir(root_fd))
    finally:
        os.close(root_fd)
        for descriptor in reversed(root_parents):
            os.close(descriptor)
    mutated = False
    remaining_budget = _MAX_GC_ENTRIES
    for transaction_name in transaction_names:
        if transaction_name.startswith('.'):
            continue
        if remaining_budget <= 0:
            break
        remaining_budget -= 1
        transaction_dir = root / transaction_name
        journal_path = transaction_dir / JOURNAL_FILENAME
        try:
            _figures_fd, _root_fd, transaction_fd = _open_transaction_directory(transaction_dir)
            try:
                journal_metadata = os.stat(JOURNAL_FILENAME, dir_fd=transaction_fd, follow_symlinks=False)
                has_regular_journal = stat.S_ISREG(journal_metadata.st_mode)
            except OSError:
                has_regular_journal = False
            finally:
                os.close(transaction_fd)
                os.close(_root_fd)
                os.close(_figures_fd)
        except ValueError:
            has_regular_journal = False
        if not has_regular_journal:
            if _transaction_creation_is_live(figures_dir, transaction_name):
                continue
            _quarantine_transaction_dir(transaction_dir)
            continue
        try:
            journal = _read_verified_journal(journal_path)
        except Exception:
            logger.exception('Quarantining unreadable publication journal %s', journal_path)
            _quarantine_transaction_dir(transaction_dir)
            continue
        journal_session_id = str(journal.get('session_id') or '')
        state_session_id = str(_value(state, 'id', '') or '') if state else ''
        if journal_session_id and state_session_id and (journal_session_id != state_session_id):
            logger.error('Quarantining publication transaction for mismatched session %s', journal_session_id)
            _quarantine_transaction_dir(transaction_dir)
            continue
        if not force and str(journal.get('owner_process_token') or '') == _PROCESS_TOKEN and (not any((_authenticated_marker_exists(transaction_dir, marker, journal) for marker in _TERMINAL_MARKERS))):
            continue
        abandoned = _authenticated_marker_exists(transaction_dir, ABANDON_MARKER, journal)
        failed = _authenticated_marker_exists(transaction_dir, FAILURE_MARKER, journal)
        committed = _authenticated_marker_exists(transaction_dir, COMMIT_MARKER, journal)
        if abandoned or failed:
            mutated = _restore_abandoned_records(state, journal, 'failed' if failed else 'abandoned') or mutated
            rollback_durable = _authenticated_marker_exists(transaction_dir, ROLLBACK_DURABLE_MARKER, journal)
            if not rollback_durable:
                mutated = True
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(str(journal.get('attempt_id') or transaction_dir.name))
            attempt_clean = _cleanup_attempt(journal)
            if rollback_durable and attempt_clean:
                _remove_transaction_dir(transaction_dir)
            continue
        if committed:
            record_mutated = _apply_committed_records(state, journal)
            mutated = record_mutated or mutated
            attempt_clean = _cleanup_attempt(journal)
            if record_mutated:
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(str(journal.get('attempt_id') or transaction_dir.name))
            elif attempt_clean:
                _remove_transaction_dir(transaction_dir)
            continue
        attempt_id = str(journal.get('attempt_id') or '')
        lock_error = _acquire_publish_locks(figures_dir, attempt_id, _publish_lock_keys(figures_dir, Path(journal['final_jpg']), journal['new_record']), session_id=str(journal.get('session_id') or ''))
        if lock_error:
            continue
        records = list(_value(state, 'publication_figures', []) or []) if state else []
        record = _find_record(records, str(journal.get('new_record_id') or ''))
        if journal.get('status') in {'RECORD_APPLIED', 'RECORD_DURABLE', 'COMMITTED'} and _record_matches_committed_files(record, journal):
            _write_marker(transaction_dir / COMMIT_MARKER)
            journal['status'] = 'COMMITTED'
            _persist_journal(journal_path, journal)
            record_mutated = _apply_committed_records(state, journal)
            mutated = record_mutated or mutated
            attempt_clean = _cleanup_attempt(journal)
            _release_publish_locks(attempt_id, keys=journal.get('lock_keys') or [], figures_dir=figures_dir, session_id=str(journal.get('session_id') or ''))
            if record_mutated:
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(attempt_id)
            elif attempt_clean:
                _remove_transaction_dir(transaction_dir)
        else:
            rolled_back = rollback_figure_publish_transaction(journal_path, state=state, producer_status='abandoned')
            mutated = rolled_back or mutated
            refreshed = None
            try:
                refreshed = _read_verified_journal(journal_path)
            except Exception:
                pass
            if refreshed is not None and _authenticated_marker_exists(transaction_dir, ABANDON_MARKER, refreshed):
                if applied_transaction_ids is not None:
                    applied_transaction_ids.add(attempt_id)
                mutated = True
            else:
                _release_publish_locks(attempt_id, keys=journal.get('lock_keys') or [], figures_dir=figures_dir, session_id=str(journal.get('session_id') or ''))
    garbage_collect_figure_transaction_roots(figures_dir, budget=remaining_budget)
    return mutated

def mark_figure_record_recovery_durable(figures_dir: Path, transaction_ids: Iterable[str], *, bindings) -> None:
    """Mark terminal record rollbacks only after session.json was fsynced."""
    ABANDON_MARKER = getattr(bindings, 'ABANDON_MARKER')
    COMMIT_MARKER = getattr(bindings, 'COMMIT_MARKER')
    FAILURE_MARKER = getattr(bindings, 'FAILURE_MARKER')
    Iterable = getattr(bindings, 'Iterable')
    JOURNAL_FILENAME = getattr(bindings, 'JOURNAL_FILENAME')
    Path = getattr(bindings, 'Path')
    ROLLBACK_DURABLE_MARKER = getattr(bindings, 'ROLLBACK_DURABLE_MARKER')
    TRANSACTION_ROOT = getattr(bindings, 'TRANSACTION_ROOT')
    _absolute = getattr(bindings, '_absolute')
    _authenticated_marker_exists = getattr(bindings, '_authenticated_marker_exists')
    _cleanup_attempt = getattr(bindings, '_cleanup_attempt')
    _open_hidden_root = getattr(bindings, '_open_hidden_root')
    _open_transaction_directory = getattr(bindings, '_open_transaction_directory')
    _quarantine_transaction_dir = getattr(bindings, '_quarantine_transaction_dir')
    _read_verified_journal = getattr(bindings, '_read_verified_journal')
    _remove_transaction_dir = getattr(bindings, '_remove_transaction_dir')
    _safe_component = getattr(bindings, '_safe_component')
    _write_marker = getattr(bindings, '_write_marker')
    garbage_collect_figure_transaction_roots = getattr(bindings, 'garbage_collect_figure_transaction_roots')
    os = getattr(bindings, 'os')
    stat = getattr(bindings, 'stat')
    figures_dir = _absolute(Path(figures_dir))
    root = figures_dir / TRANSACTION_ROOT
    try:
        root_fd, parent_fds = _open_hidden_root(root, create=False)
    except ValueError:
        return
    try:
        transaction_names = set(os.listdir(root_fd))
    finally:
        os.close(root_fd)
        for descriptor in reversed(parent_fds):
            os.close(descriptor)
    for transaction_id in sorted({str(value) for value in transaction_ids if value}):
        if transaction_id != _safe_component(transaction_id, 'attempt'):
            continue
        if transaction_id not in transaction_names:
            continue
        transaction_dir = root / transaction_id
        try:
            figures_fd, transaction_root_fd, transaction_fd = _open_transaction_directory(transaction_dir)
            try:
                journal_metadata = os.stat(JOURNAL_FILENAME, dir_fd=transaction_fd, follow_symlinks=False)
                if not stat.S_ISREG(journal_metadata.st_mode):
                    raise ValueError('publication transaction journal is not regular')
            finally:
                os.close(transaction_fd)
                os.close(transaction_root_fd)
                os.close(figures_fd)
        except (OSError, ValueError):
            _quarantine_transaction_dir(transaction_dir)
            continue
        try:
            journal = _read_verified_journal(transaction_dir / JOURNAL_FILENAME)
        except Exception:
            _quarantine_transaction_dir(transaction_dir)
            continue
        terminal_rollback = any((_authenticated_marker_exists(transaction_dir, marker, journal) for marker in (ABANDON_MARKER, FAILURE_MARKER)))
        committed = _authenticated_marker_exists(transaction_dir, COMMIT_MARKER, journal)
        if not terminal_rollback and (not committed):
            continue
        if terminal_rollback:
            _write_marker(transaction_dir / ROLLBACK_DURABLE_MARKER)
        if _cleanup_attempt(journal):
            _remove_transaction_dir(transaction_dir)
    garbage_collect_figure_transaction_roots(Path(figures_dir))
