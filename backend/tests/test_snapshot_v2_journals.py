"""Protocol tests for signed discriminated Snapshot V2 journals."""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.services.snapshot_v2.errors import SnapshotIntegrityError
from backend.services.snapshot_v2.recovery import RecoveryFailure


@pytest.fixture
def paths(tmp_path):
    class FakePaths:
        root = tmp_path / "v2"
        objects = tmp_path / "objects"
        staging = tmp_path / "staging"
        journals = tmp_path / "journals"
        integrity = tmp_path / "integrity"
        committed = tmp_path / "committed"

        def ensure_dirs(self):
            for d in [
                self.root,
                self.objects,
                self.staging,
                self.journals,
                self.integrity,
                self.committed,
            ]:
                d.mkdir(parents=True, exist_ok=True)

    p = FakePaths()
    p.ensure_dirs()
    return p


def make_envelope(
    *,
    kind: str,
    transaction_id: str,
    session_id: str = "sess-j",
    status: str = "STAGED",
    payload: dict[str, Any] | None = None,
):
    from backend.services.snapshot_v2.models import SnapshotJournalEnvelope

    body = payload if payload is not None else {"marker": f"{kind}-payload"}
    return SnapshotJournalEnvelope(
        schema_version=1,
        kind=kind,  # type: ignore[arg-type]
        transaction_id=transaction_id,
        session_id=session_id,
        status=status,
        payload=body,
        hmac="",
    )


def mutate_payload_without_resigning(path: Path) -> None:
    data = json.loads(path.read_text())
    payload = data.get("payload")
    if isinstance(payload, dict):
        payload["tampered"] = True
    else:
        data["status"] = "TAMPERED"
    path.write_text(json.dumps(data, sort_keys=True))


def capture_live_hashes(paths) -> dict[str, str]:
    """Hash all files under journals/objects/committed for mutation checks."""
    out: dict[str, str] = {}
    for root_name in ("journals", "objects", "committed", "staging"):
        root = getattr(paths, root_name)
        if not Path(root).exists():
            continue
        for f in sorted(Path(root).rglob("*")):
            if f.is_file():
                out[str(f.relative_to(paths.root.parent))] = hashlib.sha256(
                    f.read_bytes()
                ).hexdigest()
    return out


def write_unsigned_legacy_journal(paths, *, status: str = "COMMITTED") -> str:
    tx_id = "restore-unsigned-legacy"
    body = {
        "transaction_id": tx_id,
        "target_snapshot_id": "snap-x",
        "session_id": "sess-j",
        "recovery_mode": "rollback",
        "status": status,
        "pre_restore_ref": "pre-x",
        "applied_step_ledger": [],
    }
    jp = Path(paths.journals) / f"{tx_id}.json"
    jp.write_text(json.dumps(body, sort_keys=True))
    return tx_id


def write_hmac_valid_v0_restore(paths, *, status: str = "COMMITTED") -> str:
    """Write Task-1 legacy HMAC restore journal (no envelope fields)."""
    from backend.services.snapshot_v2.restore import _write_journal_hmac_signed

    tx_id = "restore-v0-legacy"
    body = {
        "transaction_id": tx_id,
        "target_snapshot_id": "snap-x",
        "session_id": "sess-j",
        "recovery_mode": "rollback",
        "status": status,
        "pre_restore_ref": "pre-x",
        "applied_step_ledger": [],
        "branch_id": "main",
        "continuation_kind": "idle",
        "_force_flat_legacy_v0": True,
    }
    _write_journal_hmac_signed(paths, tx_id, body)
    return tx_id


def read_envelope(paths, transaction_id: str | None = None):
    from backend.services.snapshot_v2.journal_store import JournalStore

    store = JournalStore(paths)
    if transaction_id is not None:
        return store.read(transaction_id)
    # first journal file
    for entry in sorted(Path(paths.journals).iterdir()):
        if entry.suffix == ".json" and entry.is_file():
            return store.read(entry.stem)
    raise AssertionError("no journal files")


@pytest.mark.parametrize(
    "kind",
    [
        "restore_apply",
        "restore_job",
        "delete",
        "import",
        "export_protection",
    ],
)
def test_every_journal_kind_round_trips_with_host_hmac(paths, kind):
    from backend.services.snapshot_v2.journal_store import JournalStore

    store = JournalStore(paths)
    expected = make_envelope(kind=kind, transaction_id=f"{kind}-1")
    store.write(expected)
    actual = store.read(expected.transaction_id)
    assert actual.kind == kind
    assert actual.payload == expected.payload
    assert actual.hmac
    assert actual.schema_version == 1


def test_tampered_journal_never_dispatches(paths):
    from backend.services.snapshot_v2.journal_store import JournalStore

    store = JournalStore(paths)
    store.write(make_envelope(kind="import", transaction_id="import-1"))
    mutate_payload_without_resigning(paths.journals / "import-1.json")
    with pytest.raises(SnapshotIntegrityError, match="JOURNAL_HMAC"):
        store.read("import-1")


@pytest.mark.asyncio
async def test_unsigned_legacy_journal_blocks_without_mutating_live_files(paths):
    from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

    write_unsigned_legacy_journal(paths, status="COMMITTED")
    # Snapshot after journal write: recovery must not mutate journal or live trees.
    before = capture_live_hashes(paths)
    result = await recover_snapshot_transactions(
        session_id="sess-j",
        paths=paths,
        session_dir=str(paths.root),
        workspace_root=str(paths.root / "ws"),
        states_root=str(paths.root / "states"),
    )
    assert capture_live_hashes(paths) == before
    assert (paths.journals / "restore-unsigned-legacy.json").exists()
    assert "restore-unsigned-legacy" in result.failed_transactions
    assert result.mutations_blocked is True
    assert any("LEGACY_UNSIGNED_JOURNAL" in e for e in result.errors)


@pytest.mark.asyncio
async def test_unsigned_journal_does_not_abort_sibling_recovery(paths):
    """One unsigned journal must not stop processing a later recoverable journal."""
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

    unsigned_id = write_unsigned_legacy_journal(paths, status="COMMITTED")
    # Lexicographically after the unsigned journal so scan would hit unsigned first
    # if recovery aborted mid-loop (restore-unsigned-legacy < restore-job-valid).
    store = JournalStore(paths)
    valid_tx = "z-restore-job-valid"
    store.write(
        make_envelope(
            kind="restore_job",
            transaction_id=valid_tx,
            session_id="sess-j",
            status="committed",
            payload={"job_id": valid_tx, "phase": "done"},
        )
    )
    assert unsigned_id < valid_tx  # scan order: unsigned first

    result = await recover_snapshot_transactions(
        session_id="sess-j",
        paths=paths,
        session_dir=str(paths.root),
        workspace_root=str(paths.root / "ws"),
        states_root=str(paths.root / "states"),
    )
    assert unsigned_id in result.failed_transactions
    assert (paths.journals / f"{unsigned_id}.json").exists()
    assert any("LEGACY_UNSIGNED_JOURNAL" in e for e in result.errors)
    # Terminal restore_job should still be cleaned even after unsigned failure.
    assert valid_tx in result.recovered_transactions
    assert not (paths.journals / f"{valid_tx}.json").exists()
    assert result.mutations_blocked is True
    assert result.recovered is True


@pytest.mark.asyncio
async def test_unverifiable_journal_defers_mutating_recovery(paths):
    """P1: an unverifiable journal must block mutating recovery of siblings.

    Fail-closed requires that a delete/restore/import transaction is NOT
    executed once any journal proves the session untrustworthy; it stays
    journaled and mutations are blocked.
    """
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2.recovery import recover_snapshot_transactions

    # Unsigned journal sorts first; a signed *mutating* delete journal sorts after.
    unsigned_id = write_unsigned_legacy_journal(paths, status="COMMITTED")
    store = JournalStore(paths)
    delete_tx = "z-delete-after-unsigned"
    store.write(
        make_envelope(
            kind="delete",
            transaction_id=delete_tx,
            session_id="sess-j",
            status="APPLYING",
            payload={"delete_set": ["snap-a"], "surviving_manifest_ids": []},
        )
    )
    assert unsigned_id < delete_tx

    with patch(
        "backend.services.snapshot_v2.recovery._recover_delete_envelope"
    ) as spy_delete:
        result = await recover_snapshot_transactions(
            session_id="sess-j",
            paths=paths,
            session_dir=str(paths.root),
            workspace_root=str(paths.root / "ws"),
            states_root=str(paths.root / "states"),
        )

    # Mutating delete recovery must NOT have run.
    spy_delete.assert_not_called()
    assert delete_tx not in result.recovered_transactions
    # Delete journal is preserved for a later, clean recovery.
    assert (paths.journals / f"{delete_tx}.json").exists()
    assert result.mutations_blocked is True
    assert unsigned_id in result.failed_transactions


def test_hmac_valid_restore_v0_is_the_only_legacy_shape_migrated(paths):
    import json

    from backend.services.snapshot_v2.journal_store import JournalStore

    tx_id = write_hmac_valid_v0_restore(paths, status="COMMITTED")
    # 1. On-disk JSON is flat v0 — no envelope fields
    jp = paths.journals / f"{tx_id}.json"
    disk = json.loads(jp.read_text())
    assert "schema_version" not in disk, disk
    assert "kind" not in disk, disk
    # 2. JournalStore projects into restore_apply envelope
    store = JournalStore(paths)
    env = store.read(tx_id)
    assert env.kind == "restore_apply"
    assert env.schema_version == 1
    assert env.payload.get("target_snapshot_id") == "snap-x"
    # 3. Persist as envelope so subsequent reads are schema-v1
    store.write(env)
    env2 = store.read(tx_id)
    assert env2.kind == "restore_apply"
    assert env2.schema_version == 1


def test_unsigned_delete_journal_is_legacy_not_restore_shape(paths):
    """Unsigned delete JSON is not silently treated as restore."""
    from backend.services.snapshot_v2.journal_store import JournalStore

    tx_id = "del-unsigned"
    (paths.journals / f"{tx_id}.json").write_text(
        json.dumps(
            {
                "transaction_id": tx_id,
                "session_id": "sess-j",
                "delete_set": ["s1"],
                "status": "STAGED",
            },
            sort_keys=True,
        )
    )
    store = JournalStore(paths)
    with pytest.raises(RecoveryFailure, match="LEGACY_UNSIGNED_JOURNAL"):
        store.read(tx_id)


@pytest.mark.asyncio
async def test_recovery_dispatches_only_matching_handler_per_kind(paths):
    """One valid journal of every kind → only its handler is invoked."""
    from backend.services.snapshot_v2.journal_store import JournalStore
    from backend.services.snapshot_v2 import recovery as recovery_mod

    store = JournalStore(paths)
    kinds = [
        "restore_apply",
        "restore_job",
        "delete",
        "import",
        "export_protection",
    ]
    for kind in kinds:
        store.write(
            make_envelope(
                kind=kind,
                transaction_id=f"{kind}-dispatch-1",
                session_id="sess-j",
                status="STAGED" if kind != "restore_job" else "queued",
                payload={
                    "target_snapshot_id": "t1",
                    "recovery_mode": "rollback",
                    "pre_restore_ref": "",
                    "applied_step_ledger": [],
                    "delete_set": [],
                    "surviving_manifest_ids": [],
                    "protected_snapshot_ids": [],
                },
            )
        )

    calls: list[str] = []

    async def _track(name):
        def _fn(*args, **kwargs):
            calls.append(name)
            return None

        return _fn

    # Patch handlers used by recover_snapshot_transactions dispatch
    with (
        patch.object(
            recovery_mod,
            "_recover_restore_envelope",
            side_effect=lambda *a, **k: calls.append("restore_apply"),
        ),
        patch.object(
            recovery_mod,
            "_recover_restore_job_envelope",
            side_effect=lambda *a, **k: calls.append("restore_job"),
        ),
        patch.object(
            recovery_mod,
            "_recover_delete_envelope",
            side_effect=lambda *a, **k: calls.append("delete"),
        ),
        patch.object(
            recovery_mod,
            "_recover_import_envelope",
            side_effect=lambda *a, **k: calls.append("import"),
        ),
        patch.object(
            recovery_mod,
            "_recover_export_protection_envelope",
            side_effect=lambda *a, **k: calls.append("export_protection"),
        ),
    ):
        await recovery_mod.recover_snapshot_transactions(
            session_id="sess-j",
            paths=paths,
            session_dir=str(paths.root),
            workspace_root=str(paths.root / "ws"),
            states_root=str(paths.root / "states"),
        )

    # Each kind exactly once
    assert sorted(calls) == sorted(kinds)
    assert len(calls) == len(kinds)
