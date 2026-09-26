"""Split failure-injection contracts for publication transactions."""
from backend.tests.figure_transaction_support import *  # noqa: F401,F403


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
