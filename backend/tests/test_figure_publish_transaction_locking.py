"""Split failure-injection contracts for publication transactions."""
from backend.tests.figure_transaction_support import *  # noqa: F401,F403

def test_live_prejournal_directory_is_protected_by_attempt_flock(tmp_path):
    figures = tmp_path / "figures"
    final_jpg = figures / "live.jpg"
    final_pdf = figures / "live.pdf"
    _write_pair(final_jpg, final_pdf, color=(10, 20, 30))
    attempt_id = "live-attempt"
    attempt_dir = figures / ".attempts" / attempt_id
    staged_jpg = attempt_dir / "live.jpg"
    staged_pdf = attempt_dir / "live.pdf"
    _write_pair(staged_jpg, staged_pdf, color=(30, 20, 10))
    record = _record(
        record_id=attempt_id,
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
        name="live",
    )
    keys = tx._publish_lock_keys(figures, final_jpg, record)
    assert tx._acquire_publish_locks(
        figures, attempt_id, keys, session_id="live-session"
    ) is None
    transaction_dir = figures / tx.TRANSACTION_ROOT / attempt_id
    transaction_dir.mkdir(parents=True)
    state = AnalysisState(
        id="live-session", user_id="u", publication_figures=[record]
    )
    try:
        assert not tx.recover_figure_publish_transactions(
            state, figures, force=True
        )
        assert transaction_dir.is_dir()
    finally:
        tx._release_publish_locks(
            attempt_id,
            keys=keys,
            figures_dir=figures,
            session_id="live-session",
        )


def test_public_lock_acquisition_reuses_authenticated_session_owner(tmp_path):
    case = _setup_attempt(tmp_path)
    try:
        assert tx.acquire_figure_publish_locks(
            figures_dir=case["figures"],
            final_jpg=case["final_jpg"],
            published_record=case["new"],
        ) is None
    finally:
        tx._release_publish_locks(
            case["new"].render_attempt_id,
            figures_dir=case["figures"],
            session_id=case["state"].id,
        )


def _hold_cross_process_locks(figures, attempt_id, keys, ready, release):
    error = tx._acquire_publish_locks(
        Path(figures), attempt_id, keys, session_id="session"
    )
    ready.send(error)
    release.wait(10)
    tx._release_publish_locks(
        attempt_id,
        keys=keys,
        figures_dir=Path(figures),
        session_id="session",
    )


def test_cross_process_flock_serializes_and_releases(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    record = SimpleNamespace(lineage_id="main", logical_figure_id="logical")
    keys = tx._publish_lock_keys(figures, figures / "same.jpg", record)
    context = multiprocessing.get_context("fork")
    parent_ready, child_ready = context.Pipe(duplex=False)
    release = context.Event()
    child = context.Process(
        target=_hold_cross_process_locks,
        args=(str(figures), "owner", keys, child_ready, release),
    )
    child.start()
    try:
        assert parent_ready.poll(10)
        assert parent_ready.recv() is None
        assert tx._acquire_publish_locks(
            figures, "contender", keys, session_id="session"
        )
    finally:
        release.set()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(5)
    assert child.exitcode == 0
    assert tx._acquire_publish_locks(
        figures, "contender", keys, session_id="session"
    ) is None
    tx._release_publish_locks(
        "contender",
        keys=keys,
        figures_dir=figures,
        session_id="session",
    )


def test_unscoped_release_with_duplicate_attempt_id_releases_one_workspace(tmp_path):
    figures_a = tmp_path / "a" / "figures"
    figures_b = tmp_path / "b" / "figures"
    figures_a.mkdir(parents=True)
    figures_b.mkdir(parents=True)
    record = SimpleNamespace(lineage_id="main", logical_figure_id="logical")
    keys_a = tx._publish_lock_keys(figures_a, figures_a / "same.jpg", record)
    keys_b = tx._publish_lock_keys(figures_b, figures_b / "same.jpg", record)
    assert tx._acquire_publish_locks(
        figures_a, "same-attempt", keys_a, session_id="session"
    ) is None
    assert tx._acquire_publish_locks(
        figures_b, "same-attempt", keys_b, session_id="session"
    ) is None

    tx._release_publish_locks("same-attempt")
    try:
        assert all(key in tx._HELD_LOCKS for key in keys_b)
    finally:
        tx._release_publish_locks(
            "same-attempt",
            keys=keys_b,
            figures_dir=figures_b,
            session_id="session",
        )
