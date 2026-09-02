"""Test that path boundary checking prevents prefix-matching bypass."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile


def test_is_write_allowed_boundary():
    from backend.utils.sandbox import _path_is_within

    # Should match: /data/sessions/abc/x is within /data/sessions/abc
    assert _path_is_within("/data/sessions/abc/x", "/data/sessions/abc")

    # Should NOT match: /data/sessions/abc-evil/x is NOT within /data/sessions/abc
    assert not _path_is_within("/data/sessions/abc-evil/x", "/data/sessions/abc")

    # Should match: exact root
    assert _path_is_within("/data/sessions/abc", "/data/sessions/abc")

    # Symlink resolution
    with tempfile.TemporaryDirectory() as tmp:
        real = os.path.join(tmp, "real")
        link = os.path.join(tmp, "link")
        os.mkdir(real)
        os.symlink(real, link)
        assert _path_is_within(os.path.join(link, "x"), real)


def test_allowed_roots_no_bare_tmp(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox
    sb = LocalJupyterSandbox(
        session_id="s-test-h8",
        state_dir=tmp_path / "states",
        working_dir=tmp_path,
    )
    roots = sb._allowed_write_roots()
    assert "/tmp" not in roots
    assert any("/tmp/scai-" in r for r in roots)


def test_numba_cache_is_inside_session_temp_root(tmp_path):
    from backend.utils.sandbox import LocalJupyterSandbox, _path_is_within

    sb = LocalJupyterSandbox(
        session_id="numba-boundary",
        state_dir=tmp_path / "states",
        working_dir=tmp_path,
    )
    cache_path = sb._session_tmp / "numba_cache"
    assert _path_is_within(str(cache_path), str(sb._session_tmp))


# The kernel write guard lives in the same ``__main__`` namespace as the
# analysis code it guards. Anything it looks up by name there can be rebound by
# a cell -- ``from matplotlib.path import Path`` in a figure cell is the real
# case that took a session down -- so the guard is exercised here against a
# poisoned namespace rather than a pristine one.
_WRITE_GUARD_PROBE = """\
import json
import sys
import types

# The generated source imports scanpy only to point its figdir at the
# workspace. A stub keeps this probe fast and independent of that install.
sys.modules.setdefault("scanpy", types.ModuleType("scanpy"))

source_path, scenario, workspace_probe, state_probe = sys.argv[1:5]
with open(source_path) as handle:
    source = handle.read()

namespace = {"__name__": "__main__"}
exec(compile(source, "<kernel-path-setup>", "exec"), namespace)

if scenario == "shadow_path":
    # Stands in for matplotlib.path.Path, which figure cells import as
    # ``from matplotlib.path import Path``. It reads its argument as vertex
    # coordinates, so a filesystem path raises ValueError.
    class ShadowPath:
        def __init__(self, vertices, codes=None):
            raise ValueError(
                "could not convert string to float: %r" % (vertices,)
            )

    namespace["Path"] = ShadowPath
elif scenario == "shadow_os":
    namespace["os"] = types.SimpleNamespace()

outcomes = {}
if scenario == "unresolvable":
    try:
        namespace["open"](object(), "w")
        outcomes["unresolvable"] = "ALLOWED"
    except BaseException as exc:
        outcomes["unresolvable"] = type(exc).__name__
else:
    targets = (("workspace", workspace_probe), ("state", state_probe))
    for label, target in targets:
        try:
            with namespace["open"](target, "w") as handle:
                handle.write("ok")
            outcomes[label] = "ALLOWED"
        except BaseException as exc:
            outcomes[label] = type(exc).__name__

print("__WRITE_GUARD__" + json.dumps(outcomes, sort_keys=True))
"""


def _run_write_guard_probe(tmp_path, scenario):
    # Execute the generated path-setup source in its own interpreter: it
    # patches builtins.open, os.open and several Path methods process-wide,
    # so it cannot be exec'd inside the test process.
    from backend.utils.sandbox import LocalJupyterSandbox
    from backend.utils.sandbox_parts.kernel_bootstrap import (
        build_path_setup_source,
    )

    sandbox = LocalJupyterSandbox(
        session_id="write-guard-" + scenario.replace("_", "-"),
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    source_file = tmp_path / "path_setup.py"
    source_file.write_text(build_path_setup_source(sandbox))
    probe_file = tmp_path / "write_guard_probe.py"
    probe_file.write_text(_WRITE_GUARD_PROBE)

    completed = subprocess.run(
        [
            sys.executable,
            str(probe_file),
            str(source_file),
            scenario,
            str(sandbox.working_dir / "probe.txt"),
            str(sandbox.notebook_state.state_dir / "probe_session.pkl.tmp"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    marker = "__WRITE_GUARD__"
    line = next(
        row for row in completed.stdout.splitlines() if row.startswith(marker)
    )
    return json.loads(line.removeprefix(marker)), completed.stdout


def test_write_guard_survives_user_code_shadowing_path(tmp_path):
    outcomes, _ = _run_write_guard_probe(tmp_path, "shadow_path")
    assert outcomes == {"workspace": "ALLOWED", "state": "ALLOWED"}


def test_write_guard_survives_user_code_shadowing_os(tmp_path):
    outcomes, _ = _run_write_guard_probe(tmp_path, "shadow_os")
    assert outcomes == {"workspace": "ALLOWED", "state": "ALLOWED"}


def test_unresolvable_write_target_names_the_resolution_failure(tmp_path):
    outcomes, stdout = _run_write_guard_probe(tmp_path, "unresolvable")
    assert outcomes == {"unresolvable": "PermissionError"}
    assert "[sandbox] Path resolution failed" in stdout
