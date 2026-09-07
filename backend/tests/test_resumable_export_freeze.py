"""Behavioral export races and validation of the captured generation."""
import asyncio
import hashlib
import tarfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# `isolated_data` is @pytest.fixture(autouse=True) over there; importing it
# is what applies it here. pyflakes reports it unused -- deleting it would
# silently remove this module's data isolation, not tidy an import.
from backend.tests.test_plain_persistence_repairs import isolated_data  # noqa: F401


def _committed(monkeypatch):
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState
    from backend.state import sessions
    from backend.utils.runtime_manifest import build_runtime_manifest, runtime_manifest_path, write_runtime_manifest
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    sid = str(uuid.uuid4())
    state = AnalysisState(id=sid, user_id="owner", name="export", current_round=2,
                          status="paused", state_version=7)
    monkeypatch.setitem(sessions, sid, state)
    root = get_session_states(sid)
    root.mkdir(parents=True, exist_ok=True)
    pkl = root / f"{sid}_session.pkl"
    pkl.write_bytes(b"generation-1")
    _write_pkl_signature(pkl, _integrity_dir(sid))
    digest = hashlib.sha256(pkl.read_bytes()).hexdigest()
    pairing = {"state_version": 7, "checkpoint": None}
    manifest = build_runtime_manifest(
        revision=1, state_revision=1, pkl_digest=digest, saved_names=["x"],
        skipped_names=[], serializer_version="dill-zstd-1", pairing=pairing,
    )
    write_runtime_manifest(runtime_manifest_path(sid), manifest)
    state.runtime_persistence = {
        "committed_revision": 1, "current_revision": 1, "saved_revision": 1,
        "last_attempt_revision": 1, "last_attempt_status": "saved",
        "committed_source": "regular_save", "committed_digest": digest,
        "committed_pairing": pairing,
    }
    return state, pkl


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [None, "absent", "hmac", "digest", "pairing"])
async def test_cold_export_validates_generation_and_reserves_start(monkeypatch, damage):
    from backend.services import resumable_export as export
    from backend.state import initializing_sessions
    from backend.utils.runtime_manifest import read_runtime_manifest, runtime_manifest_path, write_runtime_manifest
    from backend.utils.sandbox import _integrity_dir, _write_pkl_signature

    state, pkl = _committed(monkeypatch)
    if damage == "absent":
        pkl.unlink()
    elif damage in {"hmac", "digest"}:
        pkl.write_bytes(b"other-generation")
        if damage == "digest":
            _write_pkl_signature(pkl, _integrity_dir(state.id))
    elif damage == "pairing":
        path = runtime_manifest_path(state.id)
        manifest = read_runtime_manifest(path)
        manifest["pairing"]["state_version"] -= 1
        write_runtime_manifest(path, manifest)
    if damage:
        with pytest.raises(export.ResumableExportUnavailable):
            async with export.frozen_resumable_export(state.id, state):
                pytest.fail("invalid cold generation was accepted")
    else:
        async with export.frozen_resumable_export(state.id, state) as descriptor:
            assert state.id in initializing_sessions
            assert descriptor["manifest"]["pkl_sha256"] == hashlib.sha256(pkl.read_bytes()).hexdigest()
    assert state.id not in initializing_sessions


@pytest.mark.asyncio
async def test_valid_cold_archive_and_missing_required_member(monkeypatch):
    from backend.routes.export_import_parts import export_routes as route
    from backend.services import resumable_export as export
    from backend.services.export_lifecycle import reserve_export_record, abandon_export_record

    state, pkl = _committed(monkeypatch)
    info = await reserve_export_record(str(uuid.uuid4()), session_id=state.id,
                                       user_id=state.user_id, state=state, kind="prepared")
    try:
        path, _ = await route._pack_session_archive(
            state.id, state, True, prefix="scai-export-test-", export_info=info,
        )
        descriptor = export._capture_descriptor(state.id, state)
        export.verify_captured_archive(path, state.id, descriptor)
        broken = str(Path(path).with_name("missing-runtime.tar.gz"))
        with tarfile.open(path, "r:gz") as source, tarfile.open(broken, "w:gz") as dest:
            for member in source:
                if member.name.endswith(pkl.name):
                    continue
                dest.addfile(member, source.extractfile(member) if member.isfile() else None)
        with pytest.raises(export.ResumableExportUnavailable, match="complete runtime"):
            export.verify_captured_archive(broken, state.id, descriptor)
        Path(broken).unlink()
    finally:
        await abandon_export_record(info)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_archive_freeze_blocks_execute_save_and_shutdown_until_worker_drains(monkeypatch, cancel):
    from backend.config import get_session_states, get_session_workspace
    from backend.models.schemas import AnalysisState
    from backend.routes.export_import_parts import export_routes as route
    from backend.services.export_lifecycle import reserve_export_record, abandon_export_record
    from backend.state import sessions, chat_sandboxes
    from backend.utils.sandbox import LocalJupyterSandbox, CodeExecutionResult
    from backend.utils.sandbox_parts import kernel

    sid = str(uuid.uuid4())
    state = AnalysisState(id=sid, user_id="owner", name="race", current_round=2, state_version=7)
    monkeypatch.setitem(sessions, sid, state)
    from backend.services.session_lifecycle import register_new_generation
    register_new_generation(state)
    sb = LocalJupyterSandbox(session_id=sid, state_dir=str(get_session_states(sid)),
                              working_dir=str(get_session_workspace(sid)))
    sb.km = MagicMock()
    sb.kc = MagicMock()
    sb._initialized = True
    sb.runtime_revision = 1
    monkeypatch.setitem(chat_sandboxes, sid, sb)
    monkeypatch.setattr(kernel, "_shutdown_kernel_manager", AsyncMock())

    async def kernel_cell(code, **kwargs):
        sb._last_execution_reached_kernel = True
        if "SCAI_ZSTD_DILL_V1" in code:
            Path(str(sb.notebook_state.session_path) + ".tmp").write_bytes(
                f"generation-{sb.runtime_revision}".encode(),
            )
            return CodeExecutionResult(success=True, output=(
                'Session saved: 1 variables\n__SCAI_SAVED_NAMES__:["x"]\n'
                '__SCAI_EXCLUDED_KIND__:[]\n__SCAI_EXCLUDED_VOLATILE__:[]'
            ))
        return CodeExecutionResult(success=True)

    monkeypatch.setattr(sb, "_execute_unlocked", kernel_cell)
    entered, release = threading.Event(), threading.Event()
    build = route._build_session_tar

    def held_build(*args):
        entered.set()
        assert release.wait(5), "test failed to release archive worker"
        build(*args)

    monkeypatch.setattr(route, "_build_session_tar", held_build)
    info = await reserve_export_record(str(uuid.uuid4()), session_id=sid,
                                       user_id=state.user_id, state=state, kind="prepared")
    pack = asyncio.create_task(route._pack_session_archive(
        sid, state, True, prefix="scai-export-test-", export_info=info,
    ))
    queued = []
    try:
        for _ in range(500):
            if entered.is_set() or pack.done():
                break
            await asyncio.sleep(0.01)
        if pack.done():
            await pack
        assert entered.is_set()
        queued = [asyncio.create_task(sb.execute("x += 1")),
                  asyncio.create_task(sb.save_session()),
                  asyncio.create_task(sb.stop(save=False))]
        await asyncio.sleep(0.01)
        assert not any(task.done() for task in queued)
        if cancel:
            pack.cancel()
            await asyncio.sleep(0.01)
            assert not pack.done()
            assert not any(task.done() for task in queued)
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await pack
        else:
            path, _ = await pack
            with tarfile.open(path, "r:gz") as archive:
                assert archive.extractfile(f"{sid}/states/{sid}_session.pkl").read() == b"generation-1"
        await asyncio.gather(*queued)
        assert sb.runtime_revision == 2
        assert sb.notebook_state.session_path.read_bytes() == b"generation-2"
        assert sb.km is None
    finally:
        release.set()
        await asyncio.gather(pack, *queued, return_exceptions=True)
        await abandon_export_record(info)


@pytest.mark.asyncio
async def test_export_refuses_kernel_still_restoring(monkeypatch):
    from backend.services import resumable_export as export
    state, _ = _committed(monkeypatch)
    sandbox = SimpleNamespace(km=object(), _initialized=True, _starting=True,
                              _exec_lock=asyncio.Lock(), _save_session_locked=AsyncMock())
    monkeypatch.setattr(export, "live_sandbox", lambda _: sandbox)
    with pytest.raises(export.ResumableExportUnavailable) as refusal:
        async with export.frozen_resumable_export(state.id, state):
            pytest.fail("an incomplete startup was captured")
    assert refusal.value.reason == "runtime_busy"
    sandbox._save_session_locked.assert_not_awaited()
