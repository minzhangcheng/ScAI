"""Split publication transaction state-machine contracts."""
from backend.tests.figure_transaction_support import *  # noqa: F401,F403


EXPECTED_NEW_HEAD_PHASES = (
    "persist-prepared",
    "replace-jpg-intent",
    "replace-jpg",
    "replace-pdf-intent",
    "replace-pdf",
    "apply-record-paths",
    "mark-record-applied",
    "persist-record-boundary",
    "persist-record-durable",
    "write-commit-marker",
    "persist-committed",
    "cleanup-attempt",
    "release-locks",
    "remove-transaction-dir",
)

EXPECTED_EXISTING_HEAD_PHASES = (
    "persist-prepared",
    "backup-jpg",
    "backup-pdf",
    "archive-jpg",
    "archive-pdf",
    *EXPECTED_NEW_HEAD_PHASES[1:],
)


class _InjectedPhaseFailure(BaseException):
    pass


def _setup_new_head_attempt(tmp_path):
    figures = tmp_path / "figures"
    attempt_id = "attempt-new-head"
    attempt_dir = figures / ".attempts" / attempt_id
    staged_jpg = attempt_dir / "result.jpg"
    staged_pdf = attempt_dir / "result.pdf"
    new_jpg, new_pdf = _write_pair(
        staged_jpg, staged_pdf, color=(10, 80, 10)
    )
    record = _record(
        record_id=attempt_id,
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
    )
    state = AnalysisState(
        id="session-new-head",
        user_id="user",
        publication_figures=[record],
    )
    final_jpg = figures / "result.jpg"
    final_pdf = figures / "result.pdf"
    journal = tx.register_pending_figure_attempt(
        attempt_dir=attempt_dir,
        staged_jpg=staged_jpg,
        staged_pdf=staged_pdf,
        final_jpg=final_jpg,
        final_pdf=final_pdf,
        published_record=record,
        session_id=state.id,
    )
    return {
        "figures": figures,
        "attempt_dir": attempt_dir,
        "staged_jpg": staged_jpg,
        "staged_pdf": staged_pdf,
        "final_jpg": final_jpg,
        "final_pdf": final_pdf,
        "new_jpg": new_jpg,
        "new_pdf": new_pdf,
        "new": record,
        "state": state,
        "journal": journal,
    }


def _install_phase_recorder(monkeypatch, phases, *, fail_after=None):
    failed = False

    def record(phase):
        nonlocal failed
        phases.append(phase)
        if phase == fail_after and not failed:
            failed = True
            raise _InjectedPhaseFailure(phase)

    real_persist = tx._persist_journal

    def persist(journal_path, journal):
        real_persist(journal_path, journal)
        status = journal.get("status")
        if status == "PREPARED" and not journal.get("completed_steps"):
            record("persist-prepared")
        elif status == "RECORD_DURABLE":
            record("persist-record-durable")
        elif status == "COMMITTED":
            record("persist-committed")

    real_complete = tx._complete_step
    completed_phases = {
        "backup_jpg": "backup-jpg",
        "backup_pdf": "backup-pdf",
        "archive_jpg": "archive-jpg",
        "archive_pdf": "archive-pdf",
        "replace_jpg_started": "replace-jpg-intent",
        "replace_jpg": "replace-jpg",
        "replace_pdf_started": "replace-pdf-intent",
        "replace_pdf": "replace-pdf",
    }

    def complete(journal_path, journal, step, *, status=None):
        real_complete(journal_path, journal, step, status=status)
        if step in completed_phases:
            record(completed_phases[step])

    def wrap(name, phase, *, predicate=None):
        real = getattr(tx, name)

        def traced(*args, **kwargs):
            result = real(*args, **kwargs)
            if predicate is None or predicate(*args, **kwargs):
                record(phase)
            return result

        monkeypatch.setattr(tx, name, traced)

    monkeypatch.setattr(tx, "_persist_journal", persist)
    monkeypatch.setattr(tx, "_complete_step", complete)
    wrap("apply_publish_record_paths", "apply-record-paths")
    wrap("mark_publish_record_applied", "mark-record-applied")
    wrap(
        "_write_marker",
        "write-commit-marker",
        predicate=lambda path: Path(path).name == tx.COMMIT_MARKER,
    )
    wrap("_cleanup_attempt", "cleanup-attempt")
    wrap("_release_journal_locks", "release-locks")
    wrap("_remove_transaction_dir", "remove-transaction-dir")
    return record


def _publish_complete(case, record, *, archived_record=None):
    transaction, error = tx.begin_figure_publish_transaction(
        attempt_dir=case["attempt_dir"],
        staged_jpg=case["staged_jpg"],
        staged_pdf=case["staged_pdf"],
        final_jpg=case["final_jpg"],
        final_pdf=case["final_pdf"],
        published_record=case["new"],
        archived_record=archived_record,
    )
    assert error is None
    assert transaction is not None
    tx.apply_publish_record_paths(transaction, case["new"], archived_record)
    case["new"].producer_status = "published"
    case["new"].render_sha256 = image_file_sha256(case["new"].png_path)
    case["new"].pdf_sha256 = pdf_file_sha256(case["new"].pdf_path)
    case["new"].is_published_head = True
    if archived_record is not None:
        case["new"].supersedes_id = archived_record.id
        archived_record.is_published_head = False
    tx.mark_publish_record_applied(
        transaction,
        published_record=case["new"],
        archived_record=archived_record,
    )
    record("persist-record-boundary")
    tx.commit_figure_publish_transaction(transaction)
    return transaction


def test_new_and_existing_head_success_phase_traces_are_stable(
    tmp_path, monkeypatch,
):
    new_case = _setup_new_head_attempt(tmp_path / "new")
    new_phases = []
    new_record = _install_phase_recorder(monkeypatch, new_phases)
    _publish_complete(new_case, new_record)

    assert tuple(new_phases) == EXPECTED_NEW_HEAD_PHASES

    monkeypatch.undo()
    existing_case = _setup_attempt(tmp_path / "existing")
    existing_phases = []
    existing_record = _install_phase_recorder(monkeypatch, existing_phases)
    _publish_complete(
        existing_case,
        existing_record,
        archived_record=existing_case["old"],
    )

    assert tuple(existing_phases) == EXPECTED_EXISTING_HEAD_PHASES


@pytest.mark.parametrize("failed_phase", EXPECTED_NEW_HEAD_PHASES)
def test_failure_after_each_durable_phase_rolls_back_or_recovers_forward(
    tmp_path, monkeypatch, failed_phase,
):
    case = _setup_new_head_attempt(tmp_path)
    phases = []
    record = _install_phase_recorder(
        monkeypatch, phases, fail_after=failed_phase
    )

    try:
        _publish_complete(case, record)
    except _InjectedPhaseFailure:
        pass

    phase_index = EXPECTED_NEW_HEAD_PHASES.index(failed_phase)
    assert tuple(phases[: phase_index + 1]) == (
        EXPECTED_NEW_HEAD_PHASES[: phase_index + 1]
    )
    if phase_index < EXPECTED_NEW_HEAD_PHASES.index("persist-record-durable"):
        tx.rollback_figure_publish_transaction(
            Path(case["journal"]["journal_path"]),
            state=case["state"],
            producer_status="abandoned",
        )
        assert not case["final_jpg"].exists()
        assert not case["final_pdf"].exists()
        assert case["new"].producer_status == "abandoned"
    else:
        tx.recover_figure_publish_transactions(
            case["state"], case["figures"], force=True
        )
        assert case["final_jpg"].read_bytes() == case["new_jpg"]
        assert case["final_pdf"].read_bytes() == case["new_pdf"]
        assert case["new"].producer_status == "published"
        assert case["new"].is_published_head is True


def test_reviewed_revision_is_refrozen_before_pair_publication(tmp_path):
    case = _setup_attempt(tmp_path)
    revised_jpg, revised_pdf = _write_pair(
        case["staged_jpg"], case["staged_pdf"], color=(20, 30, 160)
    )
    case["new"].render_sha256 = image_file_sha256(case["staged_jpg"])
    case["new"].pdf_sha256 = pdf_file_sha256(case["staged_pdf"])

    case["journal"] = _refreeze(case)
    transaction, error = _begin(case)

    assert error is None
    assert transaction is not None
    assert transaction["new_render_sha256"] == hashlib.sha256(revised_jpg).hexdigest()
    assert transaction["new_pdf_sha256"] == hashlib.sha256(revised_pdf).hexdigest()
    assert case["final_jpg"].read_bytes() == revised_jpg
    assert case["final_pdf"].read_bytes() == revised_pdf
    tx.rollback_figure_publish_transaction(
        Path(transaction["journal_path"]), state=case["state"]
    )


def test_refreeze_rejects_forged_runtime_identity_without_rewriting_journal(
    tmp_path,
):
    case = _setup_attempt(tmp_path)
    original = tx._read_verified_journal(Path(case["journal"]["journal_path"]))
    _write_pair(case["staged_jpg"], case["staged_pdf"], color=(160, 30, 20))
    forged = dict(case["journal"])
    forged["journal_hmac"] = "0" * 64

    with pytest.raises(ValueError, match="journal_hmac identity changed"):
        _refreeze(case, forged)

    retained = tx._read_verified_journal(Path(case["journal"]["journal_path"]))
    assert retained["new_render_sha256"] == original["new_render_sha256"]
    assert retained["new_pdf_sha256"] == original["new_pdf_sha256"]
    tx.rollback_figure_publish_transaction(
        Path(case["journal"]["journal_path"]), state=case["state"]
    )


def test_refreeze_requires_the_original_transaction_locks(tmp_path):
    case = _setup_attempt(tmp_path)
    tx._release_publish_locks(
        case["new"].render_attempt_id,
        keys=case["journal"]["lock_keys"],
        figures_dir=case["figures"],
        session_id=case["state"].id,
    )

    with pytest.raises(RuntimeError, match="locks are not held"):
        _refreeze(case)

    tx.rollback_figure_publish_transaction(
        Path(case["journal"]["journal_path"]), state=case["state"]
    )


def test_archive_copy_failure_restores_pair_and_terminalizes_record(tmp_path, monkeypatch):
    case = _setup_attempt(tmp_path)
    real_copy = tx._copy_durable
    calls = 0

    def fail_first_archive(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:  # old JPG/PDF backups are calls 1/2
            raise OSError("archive copy fault")
        return real_copy(source, destination)

    monkeypatch.setattr(tx, "_copy_durable", fail_first_archive)
    transaction, error = _begin(case)
    assert transaction is None and "archive copy fault" in error
    tx.recover_figure_publish_transactions(case["state"], case["figures"], force=True)
    _assert_old_restored(case, terminal="failed")


@pytest.mark.parametrize("failed_suffix", [".jpg", ".pdf"])
def test_replace_failure_restores_the_whole_pair(
    tmp_path, monkeypatch, failed_suffix
):
    case = _setup_attempt(tmp_path)
    real_replace = tx._replace_staged_entry
    failed = False

    def injected_replace(attempt_fd, figures_fd, source, destination):
        nonlocal failed
        if not failed and Path(source).suffix == failed_suffix:
            failed = True
            raise OSError(f"{failed_suffix} replace fault")
        return real_replace(attempt_fd, figures_fd, source, destination)

    monkeypatch.setattr(tx, "_replace_staged_entry", injected_replace)
    transaction, error = _begin(case)
    assert transaction is None and "replace fault" in error
    tx.recover_figure_publish_transactions(case["state"], case["figures"], force=True)
    _assert_old_restored(case, terminal="failed")


def test_same_logical_and_filename_attempts_publish_serially_to_one_head(tmp_path):
    first = _setup_attempt(tmp_path, "attempt-one")
    figures = first["figures"]
    attempt_dir = figures / ".attempts" / "attempt-two"
    staged_jpg = attempt_dir / "result.jpg"
    staged_pdf = attempt_dir / "result.pdf"
    second_jpg, second_pdf = _write_pair(staged_jpg, staged_pdf, color=(10, 10, 90))
    second = _record(
        record_id="attempt-two",
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
        revision_of_id=first["old"].id,
    )
    first["state"].publication_figures.append(second)
    with pytest.raises(RuntimeError, match="publication lock is held"):
        tx.register_pending_figure_attempt(
            attempt_dir=attempt_dir,
            staged_jpg=staged_jpg,
            staged_pdf=staged_pdf,
            final_jpg=first["final_jpg"],
            final_pdf=first["final_pdf"],
            published_record=second,
            session_id=first["state"].id,
        )

    first_tx, error = _begin(first)
    assert error is None
    assert tx.acquire_figure_publish_locks(
        figures_dir=figures,
        final_jpg=first["final_jpg"],
        published_record=second,
    )
    _apply_record(first, first_tx)
    tx.commit_figure_publish_transaction(first_tx)

    tx.register_pending_figure_attempt(
        attempt_dir=attempt_dir,
        staged_jpg=staged_jpg,
        staged_pdf=staged_pdf,
        final_jpg=first["final_jpg"],
        final_pdf=first["final_pdf"],
        published_record=second,
        session_id=first["state"].id,
    )

    second_case = dict(first)
    second_case.update({
        "new": second,
        "attempt_dir": attempt_dir,
        "staged_jpg": staged_jpg,
        "staged_pdf": staged_pdf,
    })
    second_tx, error = _begin(second_case, archived=first["new"])
    assert error is None
    tx.apply_publish_record_paths(second_tx, second, first["new"])
    second.producer_status = "published"
    second.render_sha256 = image_file_sha256(second.png_path)
    second.pdf_sha256 = pdf_file_sha256(second.pdf_path)
    second.supersedes_id = first["new"].id
    second.is_published_head = True
    first["new"].is_published_head = False
    tx.mark_publish_record_applied(
        second_tx, published_record=second, archived_record=first["new"]
    )
    tx.commit_figure_publish_transaction(second_tx)

    assert first["final_jpg"].read_bytes() == second_jpg
    assert first["final_pdf"].read_bytes() == second_pdf
    assert [record.id for record in eligible_artifacts(first["state"].publication_figures)] == [
        second.id
    ]
    assert Path(first["new"].png_path).read_bytes() == first["new_jpg"]
