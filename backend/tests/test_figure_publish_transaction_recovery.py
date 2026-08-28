"""Split publication transaction recovery contracts."""
from backend.tests.figure_transaction_support import *  # noqa: F401,F403


EXPECTED_ROLLBACK_PHASES = (
    "restore-pdf",
    "restore-jpg",
    "restore-records",
    "remove-revision-archive",
    "cleanup-attempt",
    "persist-terminal",
    "unlink-rollback-durable",
    "write-abandon-marker",
    "release-locks",
)

EXPECTED_RESTART_FORWARD_PHASES = (
    "acquire-locks",
    "match-committed-files",
    "write-commit-marker",
    "persist-committed",
    "apply-committed-records",
    "cleanup-attempt",
    "release-locks",
    "remove-transaction-dir",
    "garbage-collect",
)


def _trace_call(monkeypatch, phases, name, phase, *, predicate=None):
    real = getattr(tx, name)

    def traced(*args, **kwargs):
        result = real(*args, **kwargs)
        if predicate is None or predicate(*args, **kwargs):
            phases.append(phase)
        return result

    monkeypatch.setattr(tx, name, traced)


def test_rollback_phase_trace_is_stable(tmp_path, monkeypatch):
    case = _setup_attempt(tmp_path)
    transaction, error = _begin(case)
    assert error is None
    assert transaction is not None
    phases = []

    real_restore = tx._restore_file

    def restore(*args, **kwargs):
        result = real_restore(*args, **kwargs)
        phases.append(
            "restore-pdf"
            if Path(kwargs["final_path"]).suffix == ".pdf"
            else "restore-jpg"
        )
        return result

    real_persist = tx._persist_journal

    def persist(journal_path, journal):
        result = real_persist(journal_path, journal)
        if journal.get("status") == "ABANDONED":
            phases.append("persist-terminal")
        return result

    monkeypatch.setattr(tx, "_restore_file", restore)
    monkeypatch.setattr(tx, "_persist_journal", persist)
    _trace_call(monkeypatch, phases, "_restore_abandoned_records", "restore-records")
    _trace_call(
        monkeypatch, phases, "_remove_revision_archive_dir", "remove-revision-archive"
    )
    _trace_call(monkeypatch, phases, "_cleanup_attempt", "cleanup-attempt")
    _trace_call(
        monkeypatch,
        phases,
        "_unlink_transaction_regular_file_if_present",
        "unlink-rollback-durable",
        predicate=lambda transaction_dir, name: name == tx.ROLLBACK_DURABLE_MARKER,
    )
    _trace_call(
        monkeypatch,
        phases,
        "_write_marker",
        "write-abandon-marker",
        predicate=lambda path: Path(path).name == tx.ABANDON_MARKER,
    )
    _trace_call(monkeypatch, phases, "_release_journal_locks", "release-locks")

    tx.rollback_figure_publish_transaction(
        Path(transaction["journal_path"]), state=case["state"]
    )

    assert tuple(phases) == EXPECTED_ROLLBACK_PHASES
    _assert_old_restored(case, terminal="abandoned")


def test_restart_forward_recovery_phase_trace_is_stable(tmp_path, monkeypatch):
    case = _setup_attempt(tmp_path)
    transaction, error = _begin(case)
    assert error is None
    assert transaction is not None
    _apply_record(case, transaction)
    phases = []

    _trace_call(monkeypatch, phases, "_acquire_publish_locks", "acquire-locks")
    _trace_call(
        monkeypatch, phases, "_record_matches_committed_files", "match-committed-files"
    )
    _trace_call(
        monkeypatch,
        phases,
        "_write_marker",
        "write-commit-marker",
        predicate=lambda path: Path(path).name == tx.COMMIT_MARKER,
    )
    real_persist = tx._persist_journal

    def persist(journal_path, journal):
        result = real_persist(journal_path, journal)
        if journal.get("status") == "COMMITTED":
            phases.append("persist-committed")
        return result

    monkeypatch.setattr(tx, "_persist_journal", persist)
    _trace_call(
        monkeypatch, phases, "_apply_committed_records", "apply-committed-records"
    )
    _trace_call(monkeypatch, phases, "_cleanup_attempt", "cleanup-attempt")
    _trace_call(monkeypatch, phases, "_release_publish_locks", "release-locks")
    _trace_call(
        monkeypatch, phases, "_remove_transaction_dir", "remove-transaction-dir"
    )
    _trace_call(
        monkeypatch,
        phases,
        "garbage_collect_figure_transaction_roots",
        "garbage-collect",
    )

    tx.recover_figure_publish_transactions(
        case["state"], case["figures"], force=True
    )

    assert tuple(phases) == EXPECTED_RESTART_FORWARD_PHASES
    assert case["final_jpg"].read_bytes() == case["new_jpg"]
    assert case["final_pdf"].read_bytes() == case["new_pdf"]
    assert case["new"].producer_status == "published"


def test_restart_abandons_a_review_pending_attempt(tmp_path):
    case = _setup_attempt(tmp_path)
    applied: set[str] = set()

    assert tx.recover_figure_publish_transactions(
        case["state"], case["figures"], force=True,
        applied_transaction_ids=applied,
    )

    _assert_old_restored(case, terminal="abandoned")
    assert applied == {case["new"].render_attempt_id}


def test_restart_after_jpg_replace_rolls_back_pair_and_record(tmp_path, monkeypatch):
    case = _setup_attempt(tmp_path)
    real_replace = tx._replace_staged_entry

    def crash_before_pdf(attempt_fd, figures_fd, source, destination):
        if Path(source).suffix == ".pdf":
            raise SystemExit("process died after JPG replace")
        return real_replace(attempt_fd, figures_fd, source, destination)

    monkeypatch.setattr(tx, "_replace_staged_entry", crash_before_pdf)
    with pytest.raises(SystemExit):
        _begin(case)
    monkeypatch.setattr(tx, "_replace_staged_entry", real_replace)

    restart_state = case["state"].model_copy(deep=True)
    case["state"] = restart_state
    case["old"], case["new"] = restart_state.publication_figures
    tx.recover_figure_publish_transactions(
        restart_state, case["figures"], force=True
    )
    _assert_old_restored(case, terminal="abandoned")


def test_restart_after_pair_replace_before_record_rolls_back_pair_and_record(tmp_path):
    case = _setup_attempt(tmp_path)
    transaction, error = _begin(case)
    assert error is None
    assert transaction is not None
    assert case["final_jpg"].read_bytes() == case["new_jpg"]
    assert case["final_pdf"].read_bytes() == case["new_pdf"]

    restart_state = case["state"].model_copy(deep=True)
    case["state"] = restart_state
    case["old"], case["new"] = restart_state.publication_figures
    tx.recover_figure_publish_transactions(
        restart_state, case["figures"], force=True
    )

    _assert_old_restored(case, terminal="abandoned")


def test_record_durable_without_marker_forward_commits_on_restart(tmp_path):
    case = _setup_attempt(tmp_path)
    transaction, error = _begin(case)
    assert error is None
    _apply_record(case, transaction)
    durable_state = case["state"].model_copy(deep=True)

    tx.recover_figure_publish_transactions(
        durable_state, case["figures"], force=True
    )

    assert not Path(transaction["journal_path"]).parent.exists()
    head = eligible_artifacts(durable_state.publication_figures)
    assert [record.id for record in head] == [case["new"].id]
    assert case["final_jpg"].read_bytes() == case["new_jpg"]
    assert case["final_pdf"].read_bytes() == case["new_pdf"]


def test_commit_marker_failure_is_recoverable_from_durable_new_record(
    tmp_path, monkeypatch
):
    case = _setup_attempt(tmp_path)
    transaction, error = _begin(case)
    assert error is None
    _apply_record(case, transaction)
    durable_new_state = case["state"].model_copy(deep=True)
    real_marker = tx._write_marker

    def fail_commit_marker(path):
        if Path(path).name == tx.COMMIT_MARKER:
            raise OSError("commit marker fault")
        return real_marker(path)

    monkeypatch.setattr(tx, "_write_marker", fail_commit_marker)
    with pytest.raises(OSError, match="commit marker fault"):
        tx.commit_figure_publish_transaction(transaction)
    monkeypatch.setattr(tx, "_write_marker", real_marker)
    tx.rollback_figure_publish_transaction(
        Path(transaction["journal_path"]),
        state=case["state"],
        producer_status="abandoned",
    )

    # session.json could already contain the published record. The terminal
    # journal must repair that copy too and request a targeted durable rewrite.
    applied: set[str] = set()
    assert tx.recover_figure_publish_transactions(
        durable_new_state,
        case["figures"],
        force=True,
        applied_transaction_ids=applied,
    )
    case["state"] = durable_new_state
    case["old"], case["new"] = durable_new_state.publication_figures
    _assert_old_restored(case, terminal="abandoned")
    assert applied == {transaction["attempt_id"]}


def test_cleanup_failure_leaves_a_forward_recoverable_commit(tmp_path, monkeypatch):
    case = _setup_attempt(tmp_path)
    transaction, error = _begin(case)
    assert error is None
    _apply_record(case, transaction)
    real_cleanup = tx._cleanup_attempt
    monkeypatch.setattr(tx, "_cleanup_attempt", lambda _journal: None)
    tx.commit_figure_publish_transaction(transaction)
    assert case["attempt_dir"].exists()
    monkeypatch.setattr(tx, "_cleanup_attempt", real_cleanup)

    restart_state = case["state"].model_copy(deep=True)
    tx.recover_figure_publish_transactions(
        restart_state, case["figures"], force=True
    )
    assert not case["attempt_dir"].exists()
    assert [record.id for record in eligible_artifacts(restart_state.publication_figures)] == [
        case["new"].id
    ]


def test_terminal_transaction_gc_is_scoped_to_saved_transaction_ids(tmp_path):
    first = _setup_attempt(tmp_path / "one", "attempt-one")
    second = _setup_attempt(tmp_path / "two", "attempt-two")
    tx.rollback_figure_publish_transaction(
        Path(first["journal"]["journal_path"]), state=first["state"]
    )
    tx.rollback_figure_publish_transaction(
        Path(second["journal"]["journal_path"]), state=second["state"]
    )

    tx.mark_figure_record_recovery_durable(
        first["figures"], [first["new"].render_attempt_id]
    )

    first_dir = Path(first["journal"]["journal_path"]).parent
    second_dir = Path(second["journal"]["journal_path"]).parent
    assert not first_dir.exists()
    assert second_dir.exists()
    assert not (second_dir / tx.ROLLBACK_DURABLE_MARKER).exists()


def test_journal_is_authenticated_and_contains_no_executable_paths(tmp_path):
    case = _setup_attempt(tmp_path)
    payload = json.loads(
        Path(case["journal"]["journal_path"]).read_text(encoding="utf-8")
    )

    assert payload["journal_hmac"]
    assert not (set(payload) & tx._RUNTIME_JOURNAL_FIELDS)


def test_forged_journal_paths_are_quarantined_without_touching_sentinel(tmp_path):
    case = _setup_attempt(tmp_path)
    journal_path = Path(case["journal"]["journal_path"])
    sentinel = tmp_path / "outside-sentinel.jpg"
    sentinel.write_bytes(b"outside")
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    payload["final_jpg"] = str(sentinel)
    journal_path.write_text(json.dumps(payload), encoding="utf-8")
    tx._release_publish_locks(case["new"].render_attempt_id or "")

    assert not tx.recover_figure_publish_transactions(
        case["state"], case["figures"], force=True
    )
    assert sentinel.read_bytes() == b"outside"
    assert not journal_path.parent.exists()
    assert any(
        (case["figures"] / tx.TRANSACTION_ROOT / ".quarantine").iterdir()
    )


def test_bare_commit_marker_cannot_forward_commit_pending_record(tmp_path):
    case = _setup_attempt(tmp_path)
    transaction_dir = Path(case["journal"]["journal_path"]).parent
    (transaction_dir / tx.COMMIT_MARKER).write_bytes(b"1\n")
    tx._release_publish_locks(case["new"].render_attempt_id or "")

    tx.recover_figure_publish_transactions(
        case["state"], case["figures"], force=True
    )

    _assert_old_restored(case, terminal="abandoned")


def test_recovery_skips_review_pending_attempt_while_os_lock_is_live(tmp_path):
    case = _setup_attempt(tmp_path)
    # Hide this process's re-entrant bookkeeping while retaining the open flock,
    # which reproduces the view of an independently started worker.
    held = dict(tx._HELD_LOCKS)
    tx._HELD_LOCKS.clear()
    try:
        restart_state = case["state"].model_copy(deep=True)
        assert not tx.recover_figure_publish_transactions(
            restart_state, case["figures"], force=True
        )
        assert restart_state.publication_figures[-1].producer_status == "pending"
        assert Path(case["journal"]["journal_path"]).exists()
    finally:
        tx._HELD_LOCKS.update(held)
    tx.rollback_figure_publish_transaction(
        Path(case["journal"]["journal_path"]), state=case["state"]
    )


def test_two_stale_workers_merge_two_transactions_from_latest_session(
    tmp_path, monkeypatch
):
    from backend.config import _config
    from backend.services import session_lifecycle, session_store

    session_root = tmp_path / "sessions"
    monkeypatch.setattr(_config, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(session_lifecycle, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(session_lifecycle, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(session_lifecycle, "STATE_DIR", tmp_path / "states")
    session_lifecycle._locks.clear()

    workspace = _config.get_session_workspace("session-tx")
    first = _setup_attempt(workspace, "attempt-a", "alpha")
    tx._release_publish_locks(first["new"].render_attempt_id or "")
    second = _setup_attempt(workspace, "attempt-b", "beta")
    tx._release_publish_locks(second["new"].render_attempt_id or "")
    canonical = AnalysisState(
        id="session-tx",
        user_id="user",
        publication_figures=[
            first["old"], first["new"], second["old"], second["new"],
        ],
    )
    session_store.save_session_to_disk(canonical.id, canonical)
    stale_a = canonical.model_copy(deep=True)
    stale_a.publication_figures = stale_a.publication_figures[:2]
    stale_b = canonical.model_copy(deep=True)
    stale_b.publication_figures = stale_b.publication_figures[2:]
    monkeypatch.setattr(tx, "_PROCESS_TOKEN", "restarted-worker")

    assert session_store._recover_publication_transactions_under_file_lock(
        canonical.id, stale_a
    )
    assert not session_store._recover_publication_transactions_under_file_lock(
        canonical.id, stale_b
    )

    persisted = session_store.load_session_from_disk(canonical.id)
    assert persisted is not None
    status_by_id = {
        item.id: item.producer_status for item in persisted.publication_figures
    }
    assert status_by_id["attempt-a"] == "abandoned"
    assert status_by_id["attempt-b"] == "abandoned"
    assert status_by_id["old-head-alpha"] == "published"
    assert status_by_id["old-head-beta"] == "published"
    remaining = workspace / "figures" / tx.TRANSACTION_ROOT
    assert not [
        item for item in remaining.iterdir() if not item.name.startswith(".")
    ]


def test_disk_transaction_recovery_preserves_live_unsaved_invalidation(
    tmp_path, monkeypatch,
):
    """Recovery patches transaction fields without replacing live records."""
    from backend.config import _config
    from backend.services import session_lifecycle, session_store

    session_root = tmp_path / "sessions"
    monkeypatch.setattr(_config, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "SESSION_DIR", session_root)
    monkeypatch.setattr(session_lifecycle, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(session_lifecycle, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(session_lifecycle, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(session_lifecycle, "STATE_DIR", tmp_path / "states")
    session_lifecycle._locks.clear()

    workspace = _config.get_session_workspace("session-tx")
    case = _setup_attempt(workspace, "attempt-invalidation", "stale-input")
    tx._release_publish_locks(case["new"].render_attempt_id or "")
    session_store.save_session_to_disk(case["state"].id, case["state"])

    live = case["state"].model_copy(deep=True)
    live_old = next(item for item in live.publication_figures if item.id == case["old"].id)
    live_old.is_current = False
    live_old.invalidated_at = live_old.created_at
    live_old.invalidation_reason = "expression representation changed"
    monkeypatch.setattr(tx, "_PROCESS_TOKEN", "restarted-worker")

    assert session_store._recover_publication_transactions_under_file_lock(
        live.id, live
    )

    recovered_old = next(
        item for item in live.publication_figures if item.id == case["old"].id
    )
    recovered_attempt = next(
        item for item in live.publication_figures if item.id == case["new"].id
    )
    assert recovered_old.is_current is False
    assert recovered_old.invalidated_at == live_old.invalidated_at
    assert recovered_old.invalidation_reason == "expression representation changed"
    assert recovered_attempt.producer_status == "abandoned"
    assert recovered_attempt.png_path == ""

    persisted = session_store.load_session_from_disk(live.id)
    assert persisted is not None
    persisted_attempt = next(
        item for item in persisted.publication_figures if item.id == case["new"].id
    )
    assert persisted_attempt.producer_status == "abandoned"


def test_publication_recovery_merge_does_not_restore_unchanged_disk_only_record(
    tmp_path,
):
    """Only records created by recovery may be appended to the live list."""
    from backend.services.session_store import _merge_publication_recovery

    jpg = tmp_path / "figure.jpg"
    pdf = tmp_path / "figure.pdf"
    _write_pair(jpg, pdf, color=(10, 20, 30))
    disk_only = _record(
        record_id="removed-live-record",
        jpg=jpg,
        pdf=pdf,
        producer_status="published",
        head=True,
    )
    before = AnalysisState(
        id="session-tx",
        user_id="user",
        publication_figures=[disk_only],
    )
    recovered = before.model_copy(deep=True)
    live = AnalysisState(id="session-tx", user_id="user")

    _merge_publication_recovery(live, before, recovered)

    assert live.publication_figures == []


def test_durable_rollback_is_collected_after_a_later_cleanup_succeeds(
    tmp_path, monkeypatch,
):
    case = _setup_attempt(tmp_path)
    journal_path = Path(case["journal"]["journal_path"])
    tx.rollback_figure_publish_transaction(journal_path, state=case["state"])
    real_cleanup = tx._cleanup_attempt
    monkeypatch.setattr(tx, "_cleanup_attempt", lambda _journal: False)
    tx.mark_figure_record_recovery_durable(
        case["figures"], [case["new"].render_attempt_id]
    )
    assert journal_path.parent.is_dir()
    monkeypatch.setattr(tx, "_cleanup_attempt", real_cleanup)

    tx.recover_figure_publish_transactions(
        case["state"], case["figures"], force=True
    )

    assert not journal_path.parent.exists()


def test_recovery_and_quarantine_share_one_bounded_work_budget(tmp_path):
    figures = tmp_path / "figures"
    root = figures / tx.TRANSACTION_ROOT
    root.mkdir(parents=True)
    for index in range(tx._MAX_GC_ENTRIES + 25):
        (root / f"orphan-{index:03d}").mkdir()

    tx.recover_figure_publish_transactions(
        AnalysisState(id="session", user_id="u"), figures, force=True
    )

    quarantined = list((root / ".quarantine").iterdir())
    assert len(quarantined) <= tx._MAX_GC_ENTRIES
