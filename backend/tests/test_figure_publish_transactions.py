"""Failure-injection contracts for publication pair + record transactions."""
from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, FigureReviewStatus, PublicationFigure
from backend.services.report import figure_transactions as tx
from backend.services.report.artifacts import (
    eligible_artifacts,
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
)
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes


def _write_pair(jpg: Path, pdf: Path, *, color) -> tuple[bytes, bytes]:
    jpg.parent.mkdir(parents=True, exist_ok=True)
    jpg_bytes = valid_image_bytes("JPEG", color=color, min_bytes=4000)
    pdf_bytes = valid_pdf_bytes(color=color)
    jpg.write_bytes(jpg_bytes)
    pdf.write_bytes(pdf_bytes)
    return jpg_bytes, pdf_bytes


def _record(
    *,
    record_id: str,
    jpg: Path,
    pdf: Path,
    producer_status: str,
    head: bool,
    revision_of_id: str | None = None,
    name: str = "result",
) -> PublicationFigure:
    preview = base64.b64encode(jpg.read_bytes()).decode()
    return PublicationFigure(
        id=record_id,
        name=name,
        description=record_id,
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        png_base64=preview,
        render_sha256=image_file_sha256(jpg),
        preview_sha256=preview_render_sha256(preview),
        pdf_sha256=pdf_file_sha256(pdf),
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id=f"logical-{name}",
        render_attempt_id=record_id,
        producer_status=producer_status,
        revision_of_id=revision_of_id,
        is_published_head=head,
    )


def _setup_attempt(
    tmp_path: Path,
    attempt_id: str = "attempt-new",
    figure_name: str = "result",
):
    figures = tmp_path / "figures"
    final_jpg = figures / f"{figure_name}.jpg"
    final_pdf = figures / f"{figure_name}.pdf"
    old_jpg, old_pdf = _write_pair(final_jpg, final_pdf, color=(80, 10, 10))
    old = _record(
        record_id=f"old-head-{figure_name}",
        jpg=final_jpg,
        pdf=final_pdf,
        producer_status="published",
        head=True,
        name=figure_name,
    )
    attempt_dir = figures / ".attempts" / attempt_id
    staged_jpg = attempt_dir / f"{figure_name}.jpg"
    staged_pdf = attempt_dir / f"{figure_name}.pdf"
    new_jpg, new_pdf = _write_pair(staged_jpg, staged_pdf, color=(10, 80, 10))
    new = _record(
        record_id=attempt_id,
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
        revision_of_id=old.id,
        name=figure_name,
    )
    state = AnalysisState(
        id="session-tx",
        user_id="user",
        publication_figures=[old, new],
    )
    journal = tx.register_pending_figure_attempt(
        attempt_dir=attempt_dir,
        staged_jpg=staged_jpg,
        staged_pdf=staged_pdf,
        final_jpg=final_jpg,
        final_pdf=final_pdf,
        published_record=new,
        session_id=state.id,
    )
    return {
        "figures": figures,
        "final_jpg": final_jpg,
        "final_pdf": final_pdf,
        "staged_jpg": staged_jpg,
        "staged_pdf": staged_pdf,
        "attempt_dir": attempt_dir,
        "old_jpg": old_jpg,
        "old_pdf": old_pdf,
        "new_jpg": new_jpg,
        "new_pdf": new_pdf,
        "old": old,
        "new": new,
        "state": state,
        "journal": journal,
    }


def _begin(case, *, archived=None):
    return tx.begin_figure_publish_transaction(
        attempt_dir=case["attempt_dir"],
        staged_jpg=case["staged_jpg"],
        staged_pdf=case["staged_pdf"],
        final_jpg=case["final_jpg"],
        final_pdf=case["final_pdf"],
        published_record=case["new"],
        archived_record=case["old"] if archived is None else archived,
    )


def _refreeze(case, transaction=None):
    return tx.refreeze_pending_figure_attempt(
        transaction or case["journal"],
        attempt_dir=case["attempt_dir"],
        staged_jpg=case["staged_jpg"],
        staged_pdf=case["staged_pdf"],
        final_jpg=case["final_jpg"],
        final_pdf=case["final_pdf"],
        published_record=case["new"],
    )


def _apply_record(case, transaction, *, archived=None):
    old = case["old"] if archived is None else archived
    new = case["new"]
    tx.apply_publish_record_paths(transaction, new, old)
    new.producer_status = "published"
    new.render_sha256 = image_file_sha256(new.png_path)
    new.pdf_sha256 = pdf_file_sha256(new.pdf_path)
    new.supersedes_id = old.id
    new.is_published_head = True
    old.is_published_head = False
    tx.mark_publish_record_applied(
        transaction, published_record=new, archived_record=old
    )


def _assert_old_restored(case, *, terminal: str):
    assert case["final_jpg"].read_bytes() == case["old_jpg"]
    assert case["final_pdf"].read_bytes() == case["old_pdf"]
    assert case["old"].is_published_head is True
    assert case["new"].producer_status == terminal
    assert case["new"].png_base64 is None
    assert case["new"].png_path == ""
    assert eligible_artifacts(case["state"].publication_figures) == [case["old"]]
    assert not case["attempt_dir"].exists()


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


def test_restart_abandons_a_review_pending_attempt(tmp_path):
    case = _setup_attempt(tmp_path)
    applied: set[str] = set()

    assert tx.recover_figure_publish_transactions(
        case["state"], case["figures"], force=True,
        applied_transaction_ids=applied,
    )

    _assert_old_restored(case, terminal="abandoned")
    assert applied == {case["new"].render_attempt_id}


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
