"""Sandbox global-diff pruning.

When user code deletes a global (del), the runtime dependency tracker must
drop it from the required set so a declared-then-deleted transient object
(e.g. a loompy connection) does not permanently break snapshot capture.
"""

from backend.utils.sandbox import LocalJupyterSandbox
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
