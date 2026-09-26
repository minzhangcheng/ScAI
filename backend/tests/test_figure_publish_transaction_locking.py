"""Split failure-injection contracts for publication transactions."""
from backend.tests.figure_transaction_support import *  # noqa: F401,F403


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
