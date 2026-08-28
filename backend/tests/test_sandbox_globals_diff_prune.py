"""Sandbox global-diff pruning.

When user code deletes a global (del), the runtime dependency tracker must
drop it from the required set so a declared-then-deleted transient object
(e.g. a loompy connection) does not permanently break snapshot capture.
"""

from unittest.mock import AsyncMock

import pytest

from backend.utils.sandbox import CodeExecutionResult, LocalJupyterSandbox
from backend.services.snapshot_v2.runtime_registry import RuntimeDependencyTracker


def _bare_sandbox() -> LocalJupyterSandbox:
    sb = object.__new__(LocalJupyterSandbox)
    sb._known_user_globals = set()
    sb._last_execution_diff = {"created": set(), "loaded": set(), "removed": set()}
    sb.runtime_dependency_tracker = RuntimeDependencyTracker()
    return sb


def test_removed_global_pruned_from_tracker():
    sb = _bare_sandbox()
    # Cell 1: loom_con is created and declared as a persistent output.
    sb._record_globals_diff(before=set(), after={"loom_con"}, declared={"loom_con"})
    assert sb.runtime_dependency_tracker.required_outputs == {"loom_con"}

    # Cell 2: loom_con is deleted (present before, absent after).
    sb._record_globals_diff(before={"loom_con"}, after=set(), declared=set())
    assert sb.runtime_dependency_tracker.required_outputs == set()


def test_only_names_actually_loaded_are_promoted():
    sb = _bare_sandbox()
    sb._record_globals_diff(before=set(), after={"used", "scratch"})
    sb._record_globals_diff(
        before={"used", "scratch"},
        after={"used", "scratch", "result"},
        loaded=sb._loaded_global_names_from_code("result = used + 1"),
    )
    assert sb.runtime_dependency_tracker.required_outputs == {"used"}


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "with loompy.connect(path) as ds:\n    print(ds.shape)",
            {"loompy", "path", "print"},
        ),
        (
            "for col in columns:\n    print(col)",
            {"columns", "print"},
        ),
        (
            "result = build_result()\nprint(result)",
            {"build_result", "print"},
        ),
        (
            "result = result + 1",
            {"result"},
        ),
        (
            "import loompy\nwith loompy.connect(path) as ds:\n    print(ds.shape)",
            {"path", "print"},
        ),
        (
            "from IPython.display import display\ndisplay(result)",
            {"result"},
        ),
        (
            "try:\n    with loompy.connect(path) as ds:\n        print(ds.shape)\n"
            "except OSError:\n    pass",
            {"loompy", "path", "print", "OSError"},
        ),
    ],
)
def test_loaded_globals_respect_same_cell_rebinding(code, expected):
    sb = _bare_sandbox()

    assert sb._loaded_global_names_from_code(code) == expected


def test_rebound_loom_handle_is_not_promoted_to_required_runtime():
    sb = _bare_sandbox()
    live = {"ds", "loompy", "path"}
    sb._record_globals_diff(before=set(), after=live)

    code = (
        "import loompy\n"
        "with loompy.connect(path) as ds:\n"
        "    print(ds.shape)"
    )
    sb._record_globals_diff(
        before=live,
        after=live,
        loaded=sb._loaded_global_names_from_code(code),
    )

    assert sb.runtime_dependency_tracker.required_outputs == {"path"}


def test_legacy_unobserved_declarations_are_pruned_but_created_dependencies_fail_closed():
    sb = _bare_sandbox()
    sb.runtime_dependency_tracker = RuntimeDependencyTracker.from_state_dict(
        {
            "created": ["adata_vel_myeloid"],
            "required": ["adata_vel_myeloid", "comm_comp"],
        }
    )

    stale = sb._prune_legacy_unobserved_required(live_names=set())

    assert stale == {"comm_comp"}
    assert sb.runtime_dependency_tracker.required_outputs == {"adata_vel_myeloid"}
    assert sb.runtime_dependency_tracker.needs_legacy_declared_cleanup is False


@pytest.mark.asyncio
async def test_failed_execution_still_prunes_deleted_required_global(tmp_path, monkeypatch):
    sb = LocalJupyterSandbox(
        session_id="failed-diff-prune",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    sb.runtime_dependency_tracker = RuntimeDependencyTracker()
    sb.runtime_dependency_tracker.merge_declared({"comm_comp"})
    monkeypatch.setattr(
        sb,
        "_snapshot_user_globals",
        AsyncMock(side_effect=[{"comm_comp"}, set()]),
    )
    monkeypatch.setattr(
        sb,
        "_execute_unlocked",
        AsyncMock(return_value=CodeExecutionResult(success=False, error="cell failed")),
    )

    result = await sb.execute("del comm_comp; raise RuntimeError('failed')", track_globals=True)

    assert result.success is False
    assert sb.runtime_dependency_tracker.required_outputs == set()


@pytest.mark.asyncio
async def test_missing_declared_output_fails_execution_without_poisoning_tracker(tmp_path, monkeypatch):
    sb = LocalJupyterSandbox(
        session_id="missing-declared-output",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(
        sb,
        "_snapshot_user_globals",
        AsyncMock(side_effect=[set(), set()]),
    )
    monkeypatch.setattr(
        sb,
        "_execute_unlocked",
        AsyncMock(return_value=CodeExecutionResult(success=True, output="error swallowed")),
    )

    result = await sb.execute(
        "try:\n    raise RuntimeError('failed')\nexcept Exception:\n    pass",
        declared_outputs=["interact_df"],
        track_globals=True,
    )

    assert result.success is False
    assert "Declared persistent outputs were not created: interact_df" in result.error
    assert sb.get_runtime_dependency_names() == set()


@pytest.mark.asyncio
async def test_present_declared_output_is_recorded(tmp_path, monkeypatch):
    sb = LocalJupyterSandbox(
        session_id="present-declared-output",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(
        sb,
        "_snapshot_user_globals",
        AsyncMock(side_effect=[set(), {"interact_df"}]),
    )
    monkeypatch.setattr(
        sb,
        "_execute_unlocked",
        AsyncMock(return_value=CodeExecutionResult(success=True)),
    )

    result = await sb.execute(
        "interact_df = {'rows': 1}",
        declared_outputs=["interact_df"],
        track_globals=True,
    )

    assert result.success is True
    assert sb.get_runtime_dependency_names() == {"interact_df"}
