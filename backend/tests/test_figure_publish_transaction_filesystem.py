"""Split failure-injection contracts for publication transactions."""
from backend.tests.figure_transaction_support import *  # noqa: F401,F403


def test_authenticated_journal_and_marker_body_hashes_are_stable():
    journal_payload = {
        "schema_version": 2,
        "session_id": "session",
        "attempt_id": "attempt",
        "completed_steps": ["REGISTERED"],
        "journal_path": "/runtime-only",
        "journal_hmac": "ignored",
    }
    marker_payload = {
        "schema_version": 2,
        "session_id": "session",
        "attempt_id": "attempt",
        "marker": "COMMITTED",
        "marker_hmac": "ignored",
    }

    assert hashlib.sha256(tx._journal_body(journal_payload)).hexdigest() == (
        "96c2925360a4835174024184e4a697f2acc40ecc49b1c1f90db1c9773699d7c8"
    )
    assert hashlib.sha256(tx._marker_body(marker_payload)).hexdigest() == (
        "46f745749b13e759904c1167434cb87aeabd7f8be66fc51a4148d7e16bf81aeb"
    )


@pytest.mark.parametrize("replacement_kind", ["hardlink", "replace"])
def test_regular_entry_binding_rejects_replaced_child(tmp_path, replacement_kind):
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = os.open(original.name, os.O_RDONLY, dir_fd=parent_fd)
    try:
        if replacement_kind == "hardlink":
            original.unlink()
            os.link(replacement, original)
        else:
            os.replace(replacement, original)

        with pytest.raises(ValueError, match="publication file binding changed"):
            tx._require_regular_entry_binding(
                parent_fd,
                original.name,
                descriptor,
                label="publication file",
            )
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def test_regular_file_read_rejects_non_regular_and_oversized_children(tmp_path):
    (tmp_path / "directory").mkdir()
    (tmp_path / "oversized").write_bytes(b"12345")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="journal is not a regular file"):
            tx._read_regular_file_at(
                parent_fd, "directory", max_bytes=8, label="journal"
            )
        with pytest.raises(ValueError, match="journal is oversized"):
            tx._read_regular_file_at(
                parent_fd, "oversized", max_bytes=4, label="journal"
            )
    finally:
        os.close(parent_fd)


def test_regular_file_read_rejects_truncated_read(tmp_path, monkeypatch):
    child = tmp_path / "journal"
    child.write_bytes(b"complete")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_read = tx.os.read
    reads = 0

    def truncated_read(descriptor, size):
        nonlocal reads
        reads += 1
        if reads == 1:
            return real_read(descriptor, min(size, 3))
        return b""

    monkeypatch.setattr(tx.os, "read", truncated_read)
    try:
        with pytest.raises(ValueError, match="journal changed while it was read"):
            tx._read_regular_file_at(
                parent_fd, child.name, max_bytes=32, label="journal"
            )
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("hidden_root", ["quarantine", "attempts", "locks"])
def test_hidden_root_symlinks_are_never_followed_by_transaction_gc(
    tmp_path, monkeypatch, hidden_root,
):
    figures = tmp_path / "figures"
    transaction_root = figures / tx.TRANSACTION_ROOT
    transaction_root.mkdir(parents=True)
    outside = tmp_path / f"outside-{hidden_root}"
    outside.mkdir()
    victim = outside / ("victim.lock" if hidden_root == "locks" else "victim")
    victim.write_bytes(b"must survive")
    os.utime(victim, (1, 1))

    if hidden_root == "quarantine":
        (transaction_root / ".quarantine").symlink_to(
            outside, target_is_directory=True
        )
        monkeypatch.setattr(tx, "_QUARANTINE_RETENTION_SECONDS", 0)
    elif hidden_root == "attempts":
        (figures / ".attempts").symlink_to(outside, target_is_directory=True)
        monkeypatch.setattr(tx, "_STALE_ENTRY_SECONDS", 0)
    else:
        (transaction_root / ".locks").symlink_to(
            outside, target_is_directory=True
        )
        monkeypatch.setattr(tx, "_STALE_ENTRY_SECONDS", 0)

    tx.garbage_collect_figure_transaction_roots(figures)

    assert victim.read_bytes() == b"must survive"


def test_transaction_root_symlink_fails_closed_without_external_writes(tmp_path):
    figures = tmp_path / "figures"
    final_jpg = figures / "result.jpg"
    final_pdf = figures / "result.pdf"
    _write_pair(final_jpg, final_pdf, color=(1, 2, 3))
    attempt_id = "attempt"
    attempt_dir = figures / ".attempts" / attempt_id
    staged_jpg = attempt_dir / "result.jpg"
    staged_pdf = attempt_dir / "result.pdf"
    _write_pair(staged_jpg, staged_pdf, color=(3, 2, 1))
    outside = tmp_path / "outside-transactions"
    outside.mkdir()
    (figures / tx.TRANSACTION_ROOT).symlink_to(
        outside, target_is_directory=True
    )
    record = _record(
        record_id=attempt_id,
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
    )

    with pytest.raises(RuntimeError):
        tx.register_pending_figure_attempt(
            attempt_dir=attempt_dir,
            staged_jpg=staged_jpg,
            staged_pdf=staged_pdf,
            final_jpg=final_jpg,
            final_pdf=final_pdf,
            published_record=record,
            session_id="session",
        )

    assert list(outside.iterdir()) == []


def test_revision_root_symlink_fails_closed_without_external_writes(tmp_path):
    case = _setup_attempt(tmp_path)
    outside = tmp_path / "outside-revisions"
    outside.mkdir()
    (case["figures"] / ".revisions").symlink_to(
        outside, target_is_directory=True
    )

    transaction, error = _begin(case)

    assert transaction is None
    assert error
    assert list(outside.iterdir()) == []


def test_transaction_backup_symlink_never_overwrites_external_target(tmp_path):
    case = _setup_attempt(tmp_path)
    transaction_dir = Path(case["journal"]["journal_path"]).parent
    outside = tmp_path / "outside-previous.jpg"
    outside.write_bytes(b"must survive")
    (transaction_dir / "previous.jpg").symlink_to(outside)

    transaction, error = _begin(case)

    assert transaction is None
    assert error
    assert outside.read_bytes() == b"must survive"


def test_journal_and_marker_symlinks_fail_closed_at_rest(tmp_path):
    case = _setup_attempt(tmp_path)
    journal_path = Path(case["journal"]["journal_path"])
    journal_outside = tmp_path / "outside-journal.json"
    journal_outside.write_bytes(journal_path.read_bytes())
    journal_path.unlink()
    journal_path.symlink_to(journal_outside)
    try:
        with pytest.raises(ValueError):
            tx._read_verified_journal(journal_path)
    finally:
        tx._release_publish_locks(case["new"].render_attempt_id)

    marker_case = _setup_attempt(tmp_path / "marker")
    marker_journal = Path(marker_case["journal"]["journal_path"])
    marker_path = marker_journal.parent / tx.COMMIT_MARKER
    tx._write_marker(marker_path)
    marker_outside = tmp_path / "outside-marker.json"
    marker_outside.write_bytes(marker_path.read_bytes())
    marker_path.unlink()
    marker_path.symlink_to(marker_outside)
    try:
        assert not tx._authenticated_marker_exists(
            marker_journal.parent, tx.COMMIT_MARKER, marker_case["journal"]
        )
    finally:
        tx._release_publish_locks(marker_case["new"].render_attempt_id)


def test_verified_journal_read_does_not_use_path_read_text(tmp_path, monkeypatch):
    case = _setup_attempt(tmp_path)
    journal_path = Path(case["journal"]["journal_path"])
    real_read_text = Path.read_text

    def forbidden_journal_read_text(path, *args, **kwargs):
        if path == journal_path:
            raise AssertionError("journal verification must use its no-follow FD")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbidden_journal_read_text)
    try:
        verified = tx._read_verified_journal(journal_path)
        assert verified["attempt_id"] == case["new"].render_attempt_id
    finally:
        tx._release_publish_locks(case["new"].render_attempt_id)


def test_transaction_directory_swap_during_atomic_journal_write_fails_closed(
    tmp_path, monkeypatch,
):
    figures = tmp_path / "figures"
    final_jpg = figures / "result.jpg"
    final_pdf = figures / "result.pdf"
    _write_pair(final_jpg, final_pdf, color=(1, 2, 3))
    attempt_id = "journal-swap"
    attempt_dir = figures / ".attempts" / attempt_id
    staged_jpg = attempt_dir / "result.jpg"
    staged_pdf = attempt_dir / "result.pdf"
    _write_pair(staged_jpg, staged_pdf, color=(3, 2, 1))
    record = _record(
        record_id=attempt_id,
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
    )
    transaction_dir = figures / tx.TRANSACTION_ROOT / attempt_id
    detached = figures / tx.TRANSACTION_ROOT / f"{attempt_id}-detached"
    outside = tmp_path / "outside-journal-swap"
    outside.mkdir()
    real_rename = tx.os.rename
    swapped = False

    def swap_before_journal_rename(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and destination == tx.JOURNAL_FILENAME:
            swapped = True
            real_rename(transaction_dir, detached)
            transaction_dir.symlink_to(outside, target_is_directory=True)
        return real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(tx.os, "rename", swap_before_journal_rename)
    with pytest.raises((RuntimeError, ValueError)):
        tx.register_pending_figure_attempt(
            attempt_dir=attempt_dir,
            staged_jpg=staged_jpg,
            staged_pdf=staged_pdf,
            final_jpg=final_jpg,
            final_pdf=final_pdf,
            published_record=record,
            session_id="session",
        )

    assert swapped
    assert list(outside.iterdir()) == []
    assert not any(
        owner[0][2] == attempt_id for owner in tx._HELD_LOCKS.values()
    )


def test_attempt_root_swap_before_pair_replace_never_moves_external_files(
    tmp_path, monkeypatch,
):
    case = _setup_attempt(tmp_path)
    attempts_root = case["figures"] / ".attempts"
    detached = case["figures"] / ".attempts-detached"
    outside = tmp_path / "outside-attempts"
    outside_attempt = outside / case["attempt_dir"].name
    outside_attempt.mkdir(parents=True)
    outside_jpg = outside_attempt / case["staged_jpg"].name
    outside_pdf = outside_attempt / case["staged_pdf"].name
    outside_jpg.write_bytes(b"external JPG must survive")
    outside_pdf.write_bytes(b"external PDF must survive")
    real_replace = tx._replace_staged_entry
    swapped = False

    def swap_attempt_root(attempt_fd, figures_fd, source, destination):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.rename(attempts_root, detached)
            attempts_root.symlink_to(outside, target_is_directory=True)
        return real_replace(attempt_fd, figures_fd, source, destination)

    monkeypatch.setattr(tx, "_replace_staged_entry", swap_attempt_root)
    transaction, error = _begin(case)

    assert transaction is None and error
    assert outside_jpg.read_bytes() == b"external JPG must survive"
    assert outside_pdf.read_bytes() == b"external PDF must survive"
    assert case["final_jpg"].read_bytes() == case["old_jpg"]
    assert case["final_pdf"].read_bytes() == case["old_pdf"]
    assert not any(
        owner[0][2] == case["new"].render_attempt_id
        for owner in tx._HELD_LOCKS.values()
    )


def test_transaction_dir_swap_before_rollback_marker_cleanup_never_unlinks_outside(
    tmp_path, monkeypatch,
):
    case = _setup_attempt(tmp_path)
    journal_path = Path(case["journal"]["journal_path"])
    transaction_dir = journal_path.parent
    detached = transaction_dir.with_name(f"{transaction_dir.name}-detached")
    outside = tmp_path / "outside-rollback-marker"
    outside.mkdir()
    outside_marker = outside / tx.ROLLBACK_DURABLE_MARKER
    outside_marker.write_bytes(b"must survive")
    real_persist = tx._persist_journal
    swapped = False

    def swap_after_terminal_persist(path, journal):
        nonlocal swapped
        real_persist(path, journal)
        if not swapped and journal.get("status") == "ABANDONED":
            swapped = True
            os.rename(transaction_dir, detached)
            transaction_dir.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(tx, "_persist_journal", swap_after_terminal_persist)
    tx.rollback_figure_publish_transaction(
        journal_path, state=case["state"], producer_status="abandoned"
    )

    assert swapped
    assert outside_marker.read_bytes() == b"must survive"
    assert not any(
        owner[0][2] == case["new"].render_attempt_id
        for owner in tx._HELD_LOCKS.values()
    )


@pytest.mark.parametrize("mutated_suffix", [".jpg", ".pdf"])
def test_staged_pair_mutation_after_digest_before_replace_fails_closed(
    tmp_path, monkeypatch, mutated_suffix,
):
    case = _setup_attempt(tmp_path)
    real_replace = tx._replace_staged_entry
    mutated = False

    def mutate_after_digest(attempt_fd, figures_fd, source, destination):
        nonlocal mutated
        if not mutated and Path(source).suffix == mutated_suffix:
            mutated = True
            descriptor = os.open(
                source, os.O_WRONLY | os.O_TRUNC, dir_fd=attempt_fd
            )
            try:
                original = (
                    case["new_jpg"]
                    if mutated_suffix == ".jpg" else case["new_pdf"]
                )
                os.write(descriptor, b"X" * len(original))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return real_replace(attempt_fd, figures_fd, source, destination)

    monkeypatch.setattr(tx, "_replace_staged_entry", mutate_after_digest)
    transaction, error = _begin(case)

    assert mutated
    assert transaction is None and error
    assert case["final_jpg"].read_bytes() == case["old_jpg"]
    assert case["final_pdf"].read_bytes() == case["old_pdf"]
    assert not any(
        owner[0][2] == case["new"].render_attempt_id
        for owner in tx._HELD_LOCKS.values()
    )
