"""Contracts for fail-closed Jupyter kernel bootstrap."""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest

from backend.agents.bioinformatics.kernel_templates import (
    _build_publication_figure_wrapper,
)
from backend.services.figure_layout import (
    blocking_violations,
    lint_ran,
    parse_layout_violations,
)
from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox


def _sandbox(tmp_path: Path, *, workspace_name: str = "workspace") -> LocalJupyterSandbox:
    return LocalJupyterSandbox(
        session_id=f"kernel-bootstrap-{uuid.uuid4().hex[:8]}",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / workspace_name),
    )


@pytest.mark.asyncio
async def test_bootstrap_injects_repr_safe_paths_before_workspace_use(tmp_path):
    workspace_name = "workspace with 'quote\nand newline"
    sandbox = _sandbox(tmp_path, workspace_name=workspace_name)
    sandbox._execute_unlocked = AsyncMock(
        side_effect=[
            CodeExecutionResult(success=True),
            CodeExecutionResult(success=True),
        ]
    )

    await sandbox._initialize_kernel()

    init_source = sandbox._execute_unlocked.await_args_list[0].args[0]
    path_source = sandbox._execute_unlocked.await_args_list[1].args[0]
    compile(init_source, "<kernel-init>", "exec")
    compile(path_source, "<kernel-path-setup>", "exec")

    workspace = str((tmp_path / workspace_name).absolute())
    assert repr(workspace) in init_source
    assert repr(str(sandbox._session_tmp)) in init_source
    assert init_source.index("_WORKSPACE_DIR = Path(") < init_source.index(
        "_FIGURES_DIR = _WORKSPACE_DIR"
    )
    assert "__SCAI_WORKSPACE_REPR__" not in init_source
    assert "__SCAI_SESSION_TMP_REPR__" not in init_source
    assert "os.chdir(" not in init_source
    assert "_sandboxed_open" not in init_source
    assert "scai_kernel_trust.capture_runtime" in init_source
    assert "_scai_threaded_local_cluster_init" in init_source
    trusted_startup = Path(
        "backend/utils/kernel_startup/sitecustomize.py"
    ).read_text(encoding="utf-8")
    assert "import scai_pydeseq2_compat" in trusted_startup
    # Installed through one loop now rather than three literal calls; the
    # fail-closed property is asserted structurally in
    # test_bootstrap_adapter_installation_is_fail_closed.
    assert '"scai_pydeseq2_compat"' in init_source
    assert "_scai_compat.install()" in init_source
    assert "import scai_scanpy_compat" in trusted_startup
    assert '"scai_scanpy_compat"' in init_source
    assert "import scai_arboreto_compat" in trusted_startup
    assert '"scai_arboreto_compat"' in init_source
    assert "Set num_workers=0" in init_source
    assert "os.chdir(_WORKSPACE_DIR)" in path_source
    assert "_sandboxed_open" in path_source
    assert sandbox._initialized is True

    normalized_common = init_source.replace(
        repr(str(sandbox.working_dir.absolute())), repr("<workspace>")
    ).replace(repr(str(sandbox._session_tmp)), repr("<session-tmp>"))
    normalized_path = path_source.replace(
        repr(str(sandbox.working_dir.absolute())), repr("<workspace>")
    ).replace(
        repr(str(sandbox._session_tmp)), repr("<session-tmp>")
    ).replace(
        repr(str(sandbox.notebook_state.state_dir)), repr("<state-dir>")
    )
    assert hashlib.sha256(normalized_common.encode()).hexdigest() == (
        "e4c59cf90d24b18b37a6aa7403488d92548a6fe83134b614b791bf173e6501b2"
    )
    assert hashlib.sha256(normalized_path.encode()).hexdigest() == (
        "03701773662b9ac96077db49e3ee5e37558053c21601e64d5ed94a571cde9e86"
    )


@pytest.mark.asyncio
async def test_bootstrap_adapter_installation_is_fail_closed(tmp_path):
    """A missing compat adapter must abort the bootstrap.

    These three patch library defects, so a kernel that runs without them
    returns wrong numbers rather than an error -- the failure mode a test cannot
    see afterwards. The one exception is a replay, which declares itself by
    setting SCAI_REPLAY_WORKSPACE and is reported as degraded; every other
    kernel raises.

    Asserted on the structure rather than on one spelling of the call. The
    previous version matched `sys.modules["name"].install()` literally and
    required no enclosing try -- a proxy that a rewrite could satisfy while
    losing the property, or break while keeping it.
    """
    sandbox = _sandbox(tmp_path)
    sandbox._execute_unlocked = AsyncMock(
        side_effect=[
            CodeExecutionResult(success=True),
            CodeExecutionResult(success=True),
        ]
    )

    await sandbox._initialize_kernel()
    init_source = sandbox._execute_unlocked.await_args_list[0].args[0]
    tree = ast.parse(init_source)

    adapter_names = {
        "scai_pydeseq2_compat",
        "scai_scanpy_compat",
        "scai_arboreto_compat",
    }
    named = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in adapter_names
    }
    assert named == adapter_names, named

    installs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "install"
    ]
    assert installs, "no adapter is installed"

    # Every ImportError handler that guards an adapter import must re-raise,
    # and the only way past it must be the declared-replay flag.
    adapter_handlers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        imports_adapter = any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "__import__"
            for inner in ast.walk(node)
        )
        if not imports_adapter:
            continue
        for handler in node.handlers:
            if isinstance(handler.type, ast.Name) and handler.type.id in (
                "ImportError", "ModuleNotFoundError"
            ):
                adapter_handlers.append(handler)

    assert adapter_handlers, "the adapter import is not guarded at all"
    for handler in adapter_handlers:
        raises = [n for n in ast.walk(handler) if isinstance(n, ast.Raise)]
        assert raises, "an adapter ImportError is swallowed without re-raising"
        guards = [
            n for n in ast.walk(handler)
            if isinstance(n, ast.If)
            and any(
                isinstance(sub, ast.Name) and sub.id == "_scai_is_replay"
                for sub in ast.walk(n.test)
            )
        ]
        assert guards, (
            "the re-raise is not conditioned on a declared replay, so either "
            "production degrades silently or a replay cannot run"
        )


@pytest.mark.asyncio
async def test_bootstrap_trust_module_is_fail_closed(tmp_path):
    """A missing `scai_kernel_trust` must abort the bootstrap too.

    It holds the publication-finalization capability and installs the figure
    capture, so a kernel that comes up without it looks healthy and produces
    nothing -- the same invisible failure mode as a missing compat adapter, and
    it gets the same treatment. A replay, which declares itself, still degrades.
    """
    sandbox = _sandbox(tmp_path)
    sandbox._execute_unlocked = AsyncMock(
        side_effect=[
            CodeExecutionResult(success=True),
            CodeExecutionResult(success=True),
        ]
    )
    await sandbox._initialize_kernel()
    tree = ast.parse(sandbox._execute_unlocked.await_args_list[0].args[0])

    handlers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        imports_trust = any(
            isinstance(inner, ast.Import)
            and any(alias.name == "scai_kernel_trust" for alias in inner.names)
            for inner in ast.walk(node)
        )
        if not imports_trust:
            continue
        handlers.extend(
            handler for handler in node.handlers
            if isinstance(handler.type, ast.Name)
            and handler.type.id in ("ImportError", "ModuleNotFoundError")
        )

    assert handlers, "the trust import is not guarded at all"
    for handler in handlers:
        assert [n for n in ast.walk(handler) if isinstance(n, ast.Raise)], (
            "the trust module's absence is swallowed without re-raising"
        )
        assert [
            n for n in ast.walk(handler)
            if isinstance(n, ast.If)
            and any(
                isinstance(sub, ast.Name) and sub.id == "_scai_is_replay"
                for sub in ast.walk(n.test)
            )
        ], (
            "the re-raise is not conditioned on a declared replay, so either "
            "production degrades silently or a replay cannot run"
        )


def test_kernel_launch_rejects_workspace_symlink_outside_session(tmp_path):
    session_root = tmp_path / "session"
    external = tmp_path / "external"
    session_root.mkdir()
    external.mkdir()
    workspace = session_root / "workspace"
    workspace.symlink_to(external, target_is_directory=True)
    sandbox = LocalJupyterSandbox(
        session_id="symlink-workspace",
        state_dir=str(session_root / "states"),
        working_dir=str(workspace),
    )

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        sandbox._kernel_launch_environment()


class _FakeKernelClient:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start_channels(self):
        self.started = True

    def stop_channels(self):
        self.stopped = True

    def get_iopub_msg(self, *, timeout):
        from queue import Empty

        raise Empty


class _FakeKernelManager:
    def __init__(self, *, kernel_name):
        self.kernel_name = kernel_name
        self.kernel_spec = SimpleNamespace(
            argv=["python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        )
        self._client = _FakeKernelClient()
        self.shutdown_called = False

    def start_kernel(self, *, cwd, env):
        self.cwd = cwd
        self.env = env

    def client(self):
        return self._client

    def restart_kernel(self):
        return None

    def shutdown_kernel(self, *, now):
        assert now is True
        self.shutdown_called = True

    def is_alive(self):
        return True


def _skip_restart_settle_delay(monkeypatch, kernel_module) -> None:
    """Short-circuit only the restart delay, not asyncio process-wide."""
    monkeypatch.setattr(
        kernel_module,
        "asyncio",
        SimpleNamespace(
            sleep=AsyncMock(),
            wait_for=asyncio.wait_for,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["manager", "process", "client", "channels"]
)
async def test_start_cleans_every_partially_acquired_kernel_resource(
    tmp_path, monkeypatch, failure
):
    import backend.utils.sandbox_parts.kernel as kernel_module

    sandbox = _sandbox(tmp_path)
    holder = {}

    def manager_factory(*, kernel_name):
        if failure == "manager":
            raise RuntimeError("manager acquisition failed")
        manager = _FakeKernelManager(kernel_name=kernel_name)
        holder["manager"] = manager
        if failure == "process":
            manager.start_kernel = lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("process acquisition failed")
            )
        elif failure == "client":
            manager.client = lambda: (_ for _ in ()).throw(
                RuntimeError("client acquisition failed")
            )
        elif failure == "channels":
            manager._client.start_channels = lambda: (_ for _ in ()).throw(
                RuntimeError("channel acquisition failed")
            )
        return manager

    monkeypatch.setattr(kernel_module, "KernelManager", manager_factory)

    with pytest.raises(RuntimeError, match="acquisition failed"):
        await sandbox.start()

    assert sandbox._initialized is False
    assert sandbox.km is None
    assert sandbox.kc is None
    if failure != "manager":
        assert holder["manager"].shutdown_called is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["readiness", "common_init", "path_init", "restore"]
)
async def test_restart_discards_failed_or_partially_restored_kernel(
    tmp_path, monkeypatch, failure
):
    import backend.utils.sandbox_parts.kernel as kernel_module

    sandbox = _sandbox(tmp_path)
    manager = _FakeKernelManager(kernel_name="python3")
    sandbox.km = manager
    sandbox.kc = manager.client()
    sandbox._initialized = True
    _skip_restart_settle_delay(monkeypatch, kernel_module)
    sandbox._wait_for_ready = AsyncMock(return_value=None)
    sandbox._initialize_kernel = AsyncMock(return_value=None)
    sandbox._restore_session = AsyncMock(return_value=None)

    if failure == "readiness":
        sandbox._wait_for_ready.side_effect = RuntimeError("readiness failed")
    elif failure == "common_init":
        sandbox._initialize_kernel.side_effect = RuntimeError(
            "Kernel common initialization failed"
        )
    elif failure == "path_init":
        sandbox._initialize_kernel.side_effect = RuntimeError(
            "Kernel path/sandbox initialization failed"
        )
    else:
        sandbox._restore_session.side_effect = RuntimeError("partial restore")

    await sandbox._restart_kernel()

    assert sandbox._kernel_dead is True
    assert sandbox._initialized is False
    assert sandbox.km is None
    assert sandbox.kc is None
    assert manager.shutdown_called is True
    assert sandbox._restarting is False
    if failure != "restore":
        sandbox._restore_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_ready_poll_is_drained_before_cancellation_returns(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    class _BlockingReadyClient:
        def kernel_info(self):
            return None

        def get_shell_msg(self, *, timeout):
            assert timeout == 5
            entered.set()
            release.wait(timeout=5)
            exited.set()
            return {"msg_type": "kernel_info_reply"}

    sandbox = _sandbox(tmp_path)
    sandbox.kc = _BlockingReadyClient()
    task = asyncio.create_task(sandbox._wait_for_ready())
    assert await asyncio.to_thread(entered.wait, 1)

    task.cancel()
    await asyncio.sleep(0)
    assert exited.is_set() is False
    assert task.done() is False

    task.cancel()
    await asyncio.sleep(0)
    assert exited.is_set() is False
    assert task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert exited.is_set() is True


@pytest.mark.asyncio
async def test_execute_reader_is_drained_before_kernel_teardown(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    class _BlockingExecutionClient(_FakeKernelClient):
        def execute(self, _code):
            return "execution-id"

        def get_iopub_msg(self, *, timeout):
            assert timeout > 0
            entered.set()
            release.wait(timeout=5)
            exited.set()
            return {
                "parent_header": {"msg_id": "execution-id"},
                "msg_type": "status",
                "content": {"execution_state": "idle"},
            }

    class _OrderingManager(_FakeKernelManager):
        def shutdown_kernel(self, *, now):
            assert exited.is_set(), "kernel was stopped while IOPub reader ran"
            super().shutdown_kernel(now=now)

    sandbox = _sandbox(tmp_path)
    manager = _OrderingManager(kernel_name="python3")
    client = _BlockingExecutionClient()
    manager._client = client
    sandbox.km = manager
    sandbox.kc = client
    sandbox._initialized = True

    task = asyncio.create_task(
        sandbox._execute_unlocked(
            "pass", records_history=False, skip_safety_check=True
        )
    )
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    assert exited.is_set() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await sandbox.stop(save=False)
    assert manager.shutdown_called is True


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_live_kernel_bootstrap_helpers_and_figure_wrappers(tmp_path):
    sandbox = _sandbox(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()

    try:
        await sandbox.start()
        probe = await sandbox.execute(
            """
import json
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class _MatrixBox:
    pass

_dense = _MatrixBox()
_dense.X = np.array([[np.nan, np.inf, -np.inf, 1.0]])
_dense_count = int(check_and_fix_nan(_dense))
_dense_unchanged = bool(
    np.isnan(_dense.X[0, 0])
    and np.isposinf(_dense.X[0, 1])
    and np.isneginf(_dense.X[0, 2])
)
check_and_fix_nan(_dense, fix=True)
_dense_fixed_only_nan = bool(
    _dense.X[0, 0] == 0.0
    and np.isposinf(_dense.X[0, 1])
    and np.isneginf(_dense.X[0, 2])
)

_sparse = _MatrixBox()
_sparse.X = sp.csr_matrix([[np.nan, np.inf, -np.inf, 1.0]])
_sparse_count = int(check_and_fix_nan(_sparse))
_sparse_unchanged = bool(
    np.isnan(_sparse.X.data[0])
    and np.isposinf(_sparse.X.data[1])
    and np.isneginf(_sparse.X.data[2])
)
check_and_fix_nan(_sparse, fix=True)
_sparse_fixed_only_nan = bool(
    _sparse.X.data[0] == 0.0
    and np.isposinf(_sparse.X.data[1])
    and np.isneginf(_sparse.X.data[2])
)

_fig = plt.figure()
plt.plot([0, 1], [0, 1])
plt.savefig('kernel_bootstrap.png')
plt.close(_fig)

print('__SCAI_BOOTSTRAP__' + json.dumps({
    'workspace': str(_WORKSPACE_DIR),
    'seed': get_seed(),
    'dense_count': _dense_count,
    'dense_unchanged': _dense_unchanged,
    'dense_fixed_only_nan': _dense_fixed_only_nan,
    'sparse_count': _sparse_count,
    'sparse_unchanged': _sparse_unchanged,
    'sparse_fixed_only_nan': _sparse_fixed_only_nan,
    'scanpy_figdir': str(sc.settings.figdir) if 'sc' in globals() else None,
}))
""",
            records_history=False,
            skip_safety_check=True,
        )
        assert probe.success, probe.error
        payload_line = next(
            line for line in probe.output.splitlines()
            if line.startswith("__SCAI_BOOTSTRAP__")
        )
        payload = json.loads(payload_line.removeprefix("__SCAI_BOOTSTRAP__"))
        assert payload == {
            "workspace": str(sandbox.working_dir),
            "seed": 42,
            "dense_count": 1,
            "dense_unchanged": True,
            "dense_fixed_only_nan": True,
            "sparse_count": 1,
            "sparse_unchanged": True,
            "sparse_fixed_only_nan": True,
            "scanpy_figdir": str(sandbox.working_dir / "figures"),
        }
        assert (sandbox.working_dir / "figures" / "kernel_bootstrap.jpg").is_file()
        assert not (sandbox.working_dir / "kernel_bootstrap.png").exists()

        show = await sandbox.execute(
            """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([0, 1], [1, 0])
plt.show()
print(f'open_figures={len(plt.get_fignums())}')
""",
            records_history=False,
            skip_safety_check=True,
        )
        assert show.success, show.error
        assert len(show.figures) == 1
        assert "open_figures=0" in show.output

        denied = await sandbox.execute(
            f"""
import matplotlib.pyplot as plt
_outside = {str(outside / 'blocked.png')!r}
plt.figure()
plt.plot([0, 1], [0, 1])
try:
    plt.savefig(_outside)
except PermissionError:
    print('outside_save=blocked')
finally:
    plt.close('all')
""",
            records_history=False,
            skip_safety_check=True,
        )
        assert denied.success, denied.error
        assert "outside_save=blocked" in denied.output
        assert not (outside / "blocked.jpg").exists()

        async def assert_wrappers_are_single_layer():
            identities = await sandbox.execute(
                """
import json
import matplotlib.figure as _mf
import matplotlib.pyplot as plt
print('__SCAI_WRAPPERS__' + json.dumps({
    'close': plt.close is _original_plt_close,
    'show': plt.show is _capturing_show,
    'savefig': _mf.Figure.savefig is _combined_savefig,
    'close_noop': bool(getattr(plt.close, '_scai_noop', False)),
    'show_noop': bool(getattr(plt.show, '_scai_noop', False)),
}))
""",
                records_history=False,
                skip_safety_check=True,
            )
            assert identities.success, identities.error
            line = next(
                item for item in identities.output.splitlines()
                if item.startswith("__SCAI_WRAPPERS__")
            )
            assert json.loads(line.removeprefix("__SCAI_WRAPPERS__")) == {
                "close": True,
                "show": True,
                "savefig": True,
                "close_noop": False,
                "show_noop": False,
            }

        persisted = await sandbox.execute(
            "user_value = {'restored': 7}\nprint('user_value_saved')",
            skip_safety_check=True,
        )
        assert persisted.success, persisted.error
        assert await sandbox.save_session() is True

        await assert_wrappers_are_single_layer()
        sandbox._kernel_dead = True
        await sandbox._restart_kernel()
        assert sandbox._kernel_dead is False
        await assert_wrappers_are_single_layer()
        restored = await sandbox.execute(
            "print(f\"restored_value={user_value['restored']}\")",
            records_history=False,
            skip_safety_check=True,
        )
        assert restored.success, restored.error
        assert "restored_value=7" in restored.output

        await sandbox.stop(save=False)
        sandbox._kernel_dead = True
        await sandbox.start()
        await assert_wrappers_are_single_layer()
        restored_again = await sandbox.execute(
            "print(f\"restored_again={user_value['restored']}\")",
            records_history=False,
            skip_safety_check=True,
        )
        assert restored_again.success, restored_again.error
        assert "restored_again=7" in restored_again.output
    finally:
        await sandbox.stop(save=False)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_live_kernel_trust_module_cannot_be_shadowed_by_workspace(tmp_path):
    sandbox = _sandbox(tmp_path)
    shadow = sandbox.working_dir / "scai_kernel_trust.py"
    shadow.write_text(
        "raise RuntimeError('workspace trust module was imported')\n",
        encoding="utf-8",
    )

    try:
        await sandbox.start()
        origin = await sandbox.execute(
            """
import scai_kernel_trust
print('__SCAI_TRUST_ORIGIN__' + str(scai_kernel_trust.__file__))
""",
            records_history=False,
            skip_safety_check=True,
        )
        assert origin.success, origin.error
        origin_line = next(
            line for line in origin.output.splitlines()
            if line.startswith("__SCAI_TRUST_ORIGIN__")
        )
        assert origin_line.removeprefix("__SCAI_TRUST_ORIGIN__") != str(shadow)
        assert origin_line.endswith("kernel_startup/scai_kernel_trust.py")

        wrapper = _build_publication_figure_wrapper(
            """
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 3))
fig.text(0.5, 0.5, 'OVERLAP', fontsize=12)
fig.text(0.5, 0.5, 'OVERLAP', fontsize=12)
""",
            str(sandbox.working_dir / "figures"),
            str(sandbox.working_dir / "figures" / "shadow-proof.jpg"),
            str(sandbox.working_dir / "figures" / "shadow-proof.pdf"),
        )
        rendered = await sandbox.execute(
            wrapper,
            records_history=False,
            skip_safety_check=True,
        )
        assert rendered.success, rendered.error
        assert lint_ran(rendered.output) is True
        assert blocking_violations(parse_layout_violations(rendered.output))
    finally:
        await sandbox.stop(save=False)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_live_publication_lint_reparents_stdout_after_prior_cell_pollution(
    tmp_path,
):
    sandbox = _sandbox(tmp_path)

    try:
        await sandbox.start()
        polluted = await sandbox.execute(
            """
import sys

class UserStdout:
    def write(self, value):
        return len(value)

    def flush(self):
        return None

sys.stdout = UserStdout()
""",
            records_history=False,
        )
        assert polluted.success, polluted.error

        wrapper = _build_publication_figure_wrapper(
            """
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 3))
fig.text(0.5, 0.5, 'OVERLAP', fontsize=12)
fig.text(0.5, 0.5, 'OVERLAP', fontsize=12)
""",
            str(sandbox.working_dir / "figures"),
            str(sandbox.working_dir / "figures" / "reparented.jpg"),
            str(sandbox.working_dir / "figures" / "reparented.pdf"),
        )
        rendered = await sandbox.execute(
            wrapper,
            records_history=False,
            skip_safety_check=True,
        )
        assert rendered.success, rendered.error
        assert lint_ran(rendered.output) is True
        assert blocking_violations(parse_layout_violations(rendered.output))
        assert rendered.output.rstrip().splitlines()[-1].startswith(
            "__SCAI_LAYOUT_LINT__"
        )
    finally:
        await sandbox.stop(save=False)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_live_kernel_uses_thread_backends_without_shared_posix_semaphores(
    tmp_path,
):
    sandbox = _sandbox(tmp_path)
    sandbox.timeout = 120

    try:
        await sandbox.start()
        probe = await sandbox.execute(
            """
import json
import numpy as np
import pandas as pd
import torch
from arboreto.algo import grnboost2
from distributed import LocalCluster, Nanny

loader_zero = list(
    torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(4)),
        batch_size=2,
        num_workers=0,
    )
)
try:
    torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.arange(4)),
        batch_size=2,
        num_workers=2,
    )
except RuntimeError as exc:
    loader_error = str(exc)
else:
    loader_error = "NOT_BLOCKED"

cluster = LocalCluster(
    n_workers=1,
    threads_per_worker=1,
    processes=True,
    worker_class=Nanny,
    dashboard_address=None,
)
try:
    local_processes = cluster.processes
    worker_types = sorted(type(worker).__name__ for worker in cluster.workers.values())
finally:
    cluster.close()

# Exercise distributed.LocalCluster's complete positional signature too. The
# worker_class slot is positional index 21 (excluding self), so guarding only
# keyword arguments would leave this valid construction process-backed.
positional_cluster = LocalCluster(
    None, 1, 1, True, None, None, None, None, 0, 30, None, None, None,
    None, None, None, False, None, None, None, None, Nanny,
)
try:
    positional_processes = positional_cluster.processes
    positional_worker_types = sorted(
        type(worker).__name__ for worker in positional_cluster.workers.values()
    )
finally:
    positional_cluster.close()

rng = np.random.default_rng(42)
expression = pd.DataFrame(
    rng.normal(size=(24, 5)),
    columns=["TF1", "TF2", "G1", "G2", "G3"],
)
network = grnboost2(
    expression,
    tf_names=["TF1", "TF2"],
    client_or_address="local",
    verbose=False,
)

print("__SCAI_THREAD_BACKENDS__" + json.dumps({
    "loader_batches": len(loader_zero),
    "loader_error": loader_error,
    "local_processes": local_processes,
    "worker_types": worker_types,
    "positional_processes": positional_processes,
    "positional_worker_types": positional_worker_types,
    "network_shape": list(network.shape),
}))
""",
            records_history=False,
        )
        assert probe.success, probe.error
        payload_line = next(
            line
            for line in probe.output.splitlines()
            if line.startswith("__SCAI_THREAD_BACKENDS__")
        )
        payload = json.loads(
            payload_line.removeprefix("__SCAI_THREAD_BACKENDS__")
        )
        assert payload["loader_batches"] == 2
        assert "Set num_workers=0" in payload["loader_error"]
        assert payload["local_processes"] is False
        assert payload["worker_types"] == ["Worker"]
        assert payload["positional_processes"] is False
        assert payload["positional_worker_types"] == ["Worker"]
        assert payload["network_shape"][1] == 3
        assert payload["network_shape"][0] > 0
    finally:
        await sandbox.stop(save=False)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_live_kernel_survives_figure_cell_shadowing_path(tmp_path):
    """A figure cell importing matplotlib's Path must not disarm the kernel.

    ``from matplotlib.path import Path`` is ordinary in code that draws
    connectors, and it rebinds the name that the savefig redirect and the I/O
    write guard resolve through. In production one such cell cost a session
    both its figure and its runtime generation: every subsequent kernel write
    was refused, including the trusted runtime-persistence cell.
    """
    sandbox = _sandbox(tmp_path)
    try:
        await sandbox.start()
        poisoned = await sandbox.execute(
            """
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import pathlib as _probe_pathlib

plt.figure()
plt.plot([0, 1], [0, 1])
plt.savefig('shadowed_path.png')
plt.close('all')
print(f'shadowed={Path is not _probe_pathlib.Path}')
""",
            records_history=False,
            skip_safety_check=True,
        )
        assert poisoned.success, poisoned.error
        assert "shadowed=True" in poisoned.output
        assert (sandbox.working_dir / "figures" / "shadowed_path.jpg").is_file()
        assert not (sandbox.working_dir / "shadowed_path.jpg").exists()

        assert await sandbox.save_session() is True
        assert (
            sandbox.notebook_state.state_dir
            / f"{sandbox.session_id}_session.pkl"
        ).is_file()
    finally:
        await sandbox.stop(save=False)
