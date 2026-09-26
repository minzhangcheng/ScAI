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
