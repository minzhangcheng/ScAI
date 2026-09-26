"""Failure cleanup and fail-closed checks for disposable figure runtimes."""
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services.figure_followup_runtime import FigureFollowupRuntime


def _install_generation(
    *, session_id, state_dir, integrity_dir, revision, payload,
):
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )
    from backend.utils.sandbox_parts.regular_save import _write_pkl_signature

    states = Path(state_dir)
    states.mkdir(parents=True, exist_ok=True)
    pkl = states / f"{session_id}_session.pkl"
    pkl.write_bytes(payload)
    manifest = build_runtime_manifest(
        revision=revision,
        state_revision=revision,
        pkl_digest=hashlib.sha256(payload).hexdigest(),
        saved_names=[],
        skipped_names=[],
        serializer_version="test-runtime",
    )
    write_runtime_manifest(
        runtime_manifest_path(session_id, states_dir=states), manifest
    )
    _write_pkl_signature(pkl, Path(integrity_dir))
    return pkl, manifest


class _RecordingLock:
    def __init__(self, events):
        self._lock = asyncio.Lock()
        self.events = events

    def locked(self):
        return self._lock.locked()

    async def __aenter__(self):
        await self._lock.acquire()
        self.events.append("lock_enter")

    async def __aexit__(self, exc_type, exc, tb):
        self.events.append("lock_exit")
        self._lock.release()


@pytest.mark.asyncio
async def test_runtime_saves_pins_verifies_then_copies_one_latest_generation(
    tmp_path, monkeypatch,
):
    from backend.utils.runtime_manifest import (
        read_runtime_manifest,
        runtime_manifest_path,
    )

    events = []
    owner_id = "owner"
    source_states = tmp_path / "source-states"
    integrity_root = tmp_path / "integrity"
    owner_integrity = integrity_root / owner_id
    source_pkl, initial_manifest = _install_generation(
        session_id=owner_id,
        state_dir=source_states,
        integrity_dir=owner_integrity,
        revision=1,
        payload=b"stale generation",
    )
    persistence = {
        "status": "saved",
        "committed_digest": initial_manifest["pkl_sha256"],
        "committed_revision": 1,
        "saved_revision": 1,
    }
    notebook_state = SimpleNamespace(
        session_path=str(source_pkl),
        state_dir=str(source_states),
        runtime_persistence=dict(persistence),
        has_session_dump=lambda: source_pkl.is_file(),
    )
    lock = _RecordingLock(events)
    state = SimpleNamespace(
        id=owner_id,
        runtime_persistence=dict(persistence),
        sandbox_state={},
    )
    contender_tasks = []
    latest_payload = b"fresh generation from locked save"

    async def save_locked():
        events.append("save")
        assert lock.locked()

        async def competing_runtime_operation():
            events.append("contender_wait")
            await lock._lock.acquire()
            try:
                events.append("contender_enter")
            finally:
                lock._lock.release()

        contender_tasks.append(asyncio.create_task(competing_runtime_operation()))
        await asyncio.sleep(0)
        assert "contender_enter" not in events
        _, saved_manifest = _install_generation(
            session_id=owner_id,
            state_dir=source_states,
            integrity_dir=owner_integrity,
            revision=2,
            payload=latest_payload,
        )
        persistence.update({
            "status": "saved",
            "committed_digest": saved_manifest["pkl_sha256"],
            "committed_revision": 2,
            "saved_revision": 2,
        })
        notebook_state.runtime_persistence = dict(persistence)
        return True

    def sync_runtime_state(target_state):
        target_state.runtime_persistence = dict(persistence)
        return dict(persistence)

    main_sandbox = SimpleNamespace(
        notebook_state=notebook_state,
        _exec_lock=lock,
        _save_session_locked=save_locked,
        sync_runtime_persistence_state=sync_runtime_state,
        timeout=1200,
        runtime_budgets={},
    )
    from backend.tests.split_facade import setattr_facade

    setattr_facade(
        monkeypatch,
        "backend.utils.sandbox",
        "_integrity_dir",
        lambda session_id: integrity_root / session_id,
    )
    original_read_manifest = read_runtime_manifest

    def tracked_read_manifest(path):
        events.append("verify")
        assert lock.locked()
        return original_read_manifest(path)

    monkeypatch.setattr(
        "backend.utils.runtime_manifest.read_runtime_manifest",
        tracked_read_manifest,
    )
    import backend.services.figure_followup_runtime as runtime_module

    original_copy2 = runtime_module.shutil.copy2
    runtime = FigureFollowupRuntime(main_sandbox, state)
    render_sandboxes = []

    class FakeRenderSandbox:
        def __init__(self, *, session_id, state_dir, working_dir, **_kwargs):
            self.session_id = session_id
            self.state_dir = Path(state_dir)
            self.working_dir = Path(working_dir)
            self.notebook_state = SimpleNamespace(
                session_path=str(self.state_dir / f"{session_id}_session.pkl"),
                has_session_dump=lambda: Path(
                    self.state_dir / f"{session_id}_session.pkl"
                ).is_file(),
            )
            render_sandboxes.append(self)

        async def start(self):
            events.append("render_start")
            await asyncio.sleep(0)
            return {"session_restored": True}

        async def stop(self, *, save=True):
            events.append(f"render_stop:{save}")

    def tracked_copy2(source, target, *args, **kwargs):
        source_path, target_path = Path(source), Path(target)
        if source_path.is_relative_to(Path(runtime._tempdir.name) / "pinned_generation"):
            events.append("copy")
            assert lock.locked()
        return original_copy2(source, target, *args, **kwargs)

    monkeypatch.setattr(runtime_module.shutil, "copy2", tracked_copy2)
    monkeypatch.setattr(
        "backend.utils.sandbox.LocalJupyterSandbox", FakeRenderSandbox
    )

    await runtime._start()
    await asyncio.gather(*contender_tasks)

    assert events.index("lock_enter") < events.index("save")
    assert events.index("save") < events.index("verify") < events.index("copy")
    assert events.index("copy") < events.index("lock_exit")
    assert events.index("lock_exit") < events.index("contender_enter")
    assert len(render_sandboxes) == 1
    render = render_sandboxes[0]
    render_pkl = Path(render.notebook_state.session_path)
    render_manifest = read_runtime_manifest(
        runtime_manifest_path(runtime.session_id, states_dir=render.state_dir)
    )
    assert render_pkl.read_bytes() == latest_payload
    assert render_manifest["revision"] == 2
    assert state.runtime_persistence["committed_revision"] == 2
    await runtime._cleanup()


@pytest.mark.asyncio
async def test_runtime_manifest_verification_failure_fails_closed_before_copy(
    tmp_path, monkeypatch,
):
    from backend.utils.runtime_manifest import RuntimeManifestError

    events = []
    owner_id = "owner"
    source_states = tmp_path / "source-states"
    integrity_root = tmp_path / "integrity"
    owner_integrity = integrity_root / owner_id
    source_pkl, manifest = _install_generation(
        session_id=owner_id,
        state_dir=source_states,
        integrity_dir=owner_integrity,
        revision=1,
        payload=b"committed generation",
    )
    persistence = {
        "committed_digest": manifest["pkl_sha256"],
        "committed_revision": 1,
        "saved_revision": 1,
    }
    notebook_state = SimpleNamespace(
        session_path=str(source_pkl),
        state_dir=str(source_states),
        runtime_persistence=dict(persistence),
        has_session_dump=lambda: True,
    )
    lock = _RecordingLock(events)

    async def save_locked():
        events.append("save")
        return True

    def sync_runtime_state(target_state):
        target_state.runtime_persistence = dict(persistence)
        return dict(persistence)

    main_sandbox = SimpleNamespace(
        notebook_state=notebook_state,
        _exec_lock=lock,
        _save_session_locked=save_locked,
        sync_runtime_persistence_state=sync_runtime_state,
    )
    runtime = FigureFollowupRuntime(
        main_sandbox,
        SimpleNamespace(id=owner_id, runtime_persistence=dict(persistence)),
    )
    from backend.tests.split_facade import setattr_facade

    setattr_facade(
        monkeypatch,
        "backend.utils.sandbox",
        "_integrity_dir",
        lambda session_id: integrity_root / session_id,
    )
    monkeypatch.setattr(
        "backend.utils.runtime_manifest.read_runtime_manifest",
        lambda _path: (_ for _ in ()).throw(
            RuntimeManifestError("signature verification failed")
        ),
    )
    import backend.services.figure_followup_runtime as runtime_module

    original_copy2 = runtime_module.shutil.copy2

    def tracked_copy2(source, target, *args, **kwargs):
        if Path(source).is_relative_to(Path(runtime._tempdir.name) / "pinned_generation"):
            events.append("copy")
        return original_copy2(source, target, *args, **kwargs)

    monkeypatch.setattr(runtime_module.shutil, "copy2", tracked_copy2)
    render_starts = []

    class NeverStartedSandbox:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            render_starts.append(True)
            return {"session_restored": True}

    monkeypatch.setattr(
        "backend.utils.sandbox.LocalJupyterSandbox", NeverStartedSandbox
    )

    with pytest.raises(RuntimeManifestError, match="signature verification failed"):
        await runtime.__aenter__()

    assert events[:2] == ["lock_enter", "save"]
    assert "copy" not in events
    assert render_starts == []
    assert runtime._tempdir is None
