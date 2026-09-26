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
