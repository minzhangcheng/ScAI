"""Auto-snapshot generation consistency: pending-save barrier correctness.

Drives a real kernel session and verifies that ``_create_auto_snapshot``
waits for pending sandbox saves before capturing, and that the restored
kernel state matches the captured generation.
"""
import asyncio
import json
import subprocess

import pytest
import pytest_asyncio

from backend.config._config import (
    SystemConfig,
    ensure_session_dirs,
    get_session_dir,
    get_session_states,
    get_session_workspace,
)
from backend.state import sessions, workflows
from backend.services.snapshot_v2.safe_point import get_snapshot_safe_point_manager
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.service import SnapshotService
from backend.services.snapshot_v2.models import SnapshotRestoreRequest
from backend.utils.sandbox import LocalJupyterSandbox, _integrity_dir, _verify_pkl_integrity

_KERNEL_NAME = "python3"
_KERNEL_START_TIMEOUT_S = 30


def _sync_sandbox_state(state, sandbox) -> None:
    """Mirror production tracking of runtime generation on AnalysisState."""
    ss = dict(getattr(state, "sandbox_state", None) or {})
    ss["runtime_revision"] = int(getattr(sandbox, "runtime_revision", 0) or 0)
    ss["saved_runtime_revision"] = int(
        getattr(sandbox, "saved_runtime_revision", 0) or 0
    )
    state.sandbox_state = ss


def _kernelspec_diagnostics(kernel_name: str = _KERNEL_NAME) -> str:
    """Return jupyter kernelspec diagnostics for hard failures (never skip)."""
    lines: list[str] = []
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
        lines.append(f"available_kernelspecs={sorted(specs.keys())}")
        lines.append(f"requested_kernel={kernel_name!r} present={kernel_name in specs}")
        if kernel_name in specs:
            lines.append(f"kernel_path={specs[kernel_name]}")
    except Exception as exc:
        lines.append(f"KernelSpecManager error: {exc!r}")
    try:
        proc = subprocess.run(
            ["jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        lines.append(f"jupyter kernelspec list (rc={proc.returncode}):")
        lines.append((proc.stdout or "").strip() or "(empty stdout)")
        if proc.stderr:
            lines.append(f"stderr: {proc.stderr.strip()}")
    except Exception as exc:
        lines.append(f"jupyter kernelspec list failed: {exc!r}")
    return "\n".join(lines)


def _require_kernelspec(kernel_name: str = _KERNEL_NAME) -> None:
    """Fail with diagnostics if the configured kernelspec is missing."""
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
    except Exception as exc:
        raise AssertionError(
            f"Unable to query kernelspecs for {kernel_name!r}: {exc!r}\n"
            f"{_kernelspec_diagnostics(kernel_name)}"
        ) from exc
    if kernel_name not in specs:
        raise AssertionError(
            f"Required kernelspec {kernel_name!r} is not installed.\n"
            f"{_kernelspec_diagnostics(kernel_name)}"
        )


async def _hard_stop_kernel(sandbox) -> None:
    """Stop kernel channels without save_session overwriting restored pkl."""
    if getattr(sandbox, "kc", None) is not None:
        try:
            sandbox.kc.stop_channels()
        except Exception:
            pass
        sandbox.kc = None
    if getattr(sandbox, "km", None) is not None:
        try:
            sandbox.km.shutdown_kernel(now=True)
        except Exception:
            pass
        sandbox.km = None
        sandbox._initialized = False
        sandbox._session_restored = False
        sandbox._session_restore_failed = False


@pytest_asyncio.fixture
async def generation_env(request, tmp_path, monkeypatch):
    """Real-kernel session with SnapshotService for generation tests."""
    from backend.models.schemas import AnalysisState

    session_id = "gen-consistency-test"
    state = AnalysisState(id=session_id, user_id="u1", name="generation-test")
    sandbox = None
    env = None

    ensure_session_dirs(session_id)

    paths = SnapshotV2Paths.for_session(session_id)
    for d in [paths.root, paths.objects, paths.staging, paths.journals,
              paths.integrity, paths.committed]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        _require_kernelspec(_KERNEL_NAME)
        sandbox = LocalJupyterSandbox(
            session_id=session_id,
            state_dir=str(get_session_states(session_id)),
            working_dir=str(get_session_workspace(session_id)),
        )
        try:
            await asyncio.wait_for(sandbox.start(), timeout=_KERNEL_START_TIMEOUT_S)
        except Exception as exc:
            raise AssertionError(
                f"Real-kernel sandbox.start() failed within "
                f"{_KERNEL_START_TIMEOUT_S}s: {exc!r}\n"
                f"{_kernelspec_diagnostics(_KERNEL_NAME)}"
            ) from exc

        from backend.workflow.core import SingleCellAnalysisWorkflow as Workflow
        config = SystemConfig()
        workflow = Workflow(config=config, session_id=session_id)
        workflow.sandbox = sandbox

        service = SnapshotService(session_id, paths)
        sessions[session_id] = state
        workflows[session_id] = workflow

        env = DummyEnv(
            session_id=session_id,
            state=state,
            sandbox=sandbox,
            workflow=workflow,
            service=service,
            paths=paths,
        )
        yield env
    finally:
        # Always hard-stop any partially created kernel manager, including a
        # replacement sandbox assigned by the test body after restore.
        stop_targets = []
        if sandbox is not None:
            stop_targets.append(sandbox)
        if env is not None and getattr(env, "sandbox", None) is not None:
            if env.sandbox is not sandbox:
                stop_targets.append(env.sandbox)
        for live in stop_targets:
            try:
                await _hard_stop_kernel(live)
                if hasattr(live, "stop"):
                    await live.stop()
            except Exception:
                pass
        sessions.pop(session_id, None)
        workflows.pop(session_id, None)
        import shutil
        shutil.rmtree(get_session_dir(session_id), ignore_errors=True)


class DummyEnv:
    def __init__(self, session_id, state, sandbox, workflow, service, paths):
        self.session_id = session_id
        self.state = state
        self.sandbox = sandbox
        self.workflow = workflow
        self.service = service
        self.paths = paths


@pytest.mark.asyncio
async def test_auto_snapshot_waits_for_pending_save(generation_env, monkeypatch):
    """Auto-snapshot blocks on await_pending_saves and captures right generation."""
    env = generation_env
    sandbox = env.sandbox

    # Canonical runtime artifact so required-set capture is non-empty.
    first = await sandbox.execute("results = 41", save_to_history=True)
    second = await sandbox.execute("results += 1", save_to_history=True)
    assert first.success and second.success
    assert sandbox.runtime_revision == 2
    _sync_sandbox_state(env.state, sandbox)

    entered = asyncio.Event()
    release = asyncio.Event()
    original = sandbox.await_pending_saves

    async def gated_await_pending_saves(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(sandbox, "await_pending_saves", gated_await_pending_saves)

    capture = asyncio.create_task(
        env.workflow._create_auto_snapshot(
            env.state,
            "pre_round_wait",
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=10)
    assert not capture.done()

    release.set()
    record = await asyncio.wait_for(capture, timeout=60)
    assert record is not None and record.failure is None
    _sync_sandbox_state(env.state, sandbox)

    drift = await sandbox.execute("results = 99", save_to_history=True)
    assert drift.success
    _sync_sandbox_state(env.state, sandbox)

    safe_points = get_snapshot_safe_point_manager()
    request = await safe_points.request_safe_point(
        env.session_id,
        "generation-consistency-owner",
    )
    assert await safe_points.acknowledge_safe_point_request(
        env.session_id,
        request.generation,
        {"kernel_present": True, "next_node": "idle"},
        role="workflow",
    )
    try:
        restore_record = await env.service.restore_snapshot(
            SnapshotRestoreRequest(
                snapshot_id=record.snapshot_id,
                current_state_choice="discard_current",
                discard_confirmation=True,
            ),
            transaction_id="generation-consistency-restore",
        )
    finally:
        assert await safe_points.release(
            env.session_id,
            "generation-consistency-owner",
        )
    assert restore_record.status == "committed"

    # Restore replaces on-disk residual pkl; rehydrate via fresh kernel start.
    states = get_session_states(env.session_id)
    pkl = states / f"{env.session_id}_session.pkl"
    assert pkl.is_file()
    assert _verify_pkl_integrity(pkl, _integrity_dir(env.session_id)) is True

    import dill
    restored_vars = {}
    with open(pkl, "rb") as f:
        while True:
            try:
                name, obj = dill.load(f)
            except EOFError:
                break
            restored_vars[name] = obj
    assert restored_vars.get("results") == 42

    await _hard_stop_kernel(sandbox)
    sb2 = LocalJupyterSandbox(
        session_id=env.session_id,
        state_dir=str(states),
        working_dir=str(get_session_workspace(env.session_id)),
    )
    try:
        start_res = await asyncio.wait_for(sb2.start(), timeout=_KERNEL_START_TIMEOUT_S)
    except Exception as exc:
        await _hard_stop_kernel(sb2)
        raise AssertionError(
            f"Post-restore sandbox.start() failed within "
            f"{_KERNEL_START_TIMEOUT_S}s: {exc!r}\n"
            f"{_kernelspec_diagnostics(_KERNEL_NAME)}"
        ) from exc
    assert start_res.get("session_restored") is True, start_res
    env.sandbox = sb2
    env.workflow.sandbox = sb2

    runtime_check = await sb2.execute(
        "assert results == 42",
        save_to_history=False,
        skip_safety_check=True,
    )
    assert runtime_check.success

    restored_state = sessions[env.session_id]
    assert restored_state.sandbox_state["runtime_revision"] == 2

    # Restore and the restarted sandbox must use the same canonical notebook.
    notebook_path = sb2.notebook_state.notebook_path
    assert notebook_path == states / f"{env.session_id}.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code_sources = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    assert "results += 1" in code_sources
