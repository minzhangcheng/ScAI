"""Process-level write-isolation contracts for Jupyter kernels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import uuid
import sqlite3

import pytest

from backend.utils.sandbox import LocalJupyterSandbox


def _startup_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "utils" / "kernel_startup"


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_sitecustomize_blocks_python_and_native_writes(tmp_path):
    allowed = tmp_path / "session"
    blocked = tmp_path / "other-session"
    allowed.mkdir()
    blocked.mkdir()
    (blocked / "existing.txt").write_text("untouched")
    metadata_mode = (blocked / "existing.txt").stat().st_mode
    metadata_mtime = (blocked / "existing.txt").stat().st_mtime_ns
    connection = sqlite3.connect(blocked / "readonly.db")
    connection.execute("create table values_table (value integer)")
    connection.execute("insert into values_table values (7)")
    connection.commit()
    connection.close()
    import h5py

    with h5py.File(blocked / "readonly.h5", "w") as handle:
        handle.create_dataset("value", data=[7])

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(_startup_dir()),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SCAI_KERNEL_WRITE_ROOTS": json.dumps([str(allowed)]),
        "SCAI_KERNEL_REQUIRE_LANDLOCK": "1",
        "SCAI_KERNEL_LANDLOCK_ACTIVE": "999",
        "HOME": str(allowed / "home"),
        "TMPDIR": str(allowed / "tmp"),
        "TEMP": str(allowed / "tmp"),
        "TMP": str(allowed / "tmp"),
    })
    (allowed / "home").mkdir()
    (allowed / "tmp").mkdir()

    code = textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path
        import sqlite3
        import subprocess
        import h5py

        blocked = Path({str(blocked)!r})
        allowed = Path({str(allowed)!r})
        outcomes = {{}}

        def attempt(name, operation):
            try:
                operation()
            except Exception as exc:
                outcomes[name] = type(exc).__name__
            else:
                outcomes[name] = "ALLOWED"

        def fork_once():
            child = os.fork()
            if child == 0:
                os._exit(0)
            os.waitpid(child, 0)

        attempt("builtins", lambda: open(blocked / "existing.txt", "r+").write("bad"))
        attempt("bytes", lambda: open(os.fsencode(blocked / "existing.txt"), "r+"))
        attempt("sqlite", lambda: sqlite3.connect(blocked / "native.db"))
        attempt("h5py", lambda: h5py.File(blocked / "native.h5", "w"))
        attempt("chmod", lambda: os.chmod(blocked / "existing.txt", 0o600))
        attempt(
            "posix_spawn-chmod",
            lambda: os.posix_spawn(
                "/usr/bin/chmod",
                ["chmod", "000", str(blocked / "existing.txt")],
                os.environ.copy(),
            ),
        )
        attempt(
            "subprocess-chmod",
            lambda: subprocess.run(
                ["/usr/bin/chmod", "000", str(blocked / "existing.txt")],
                check=True,
            ),
        )
        attempt("os-system", lambda: os.system("/usr/bin/true"))
        if hasattr(os, "fork"):
            attempt("os-fork", fork_once)
        attempt("utime", lambda: os.utime(blocked / "existing.txt", (1, 1)))
        attempt("rename", lambda: os.rename(blocked / "existing.txt", blocked / "renamed.txt"))

        (allowed / "python.txt").write_text("ok")
        open(os.fsencode(allowed / "bytes.bin"), "wb").close()
        open(os.devnull, "w").close()
        connection = sqlite3.connect(allowed / "inside.db")
        connection.execute("create table ok (value integer)")
        connection.commit()
        connection.close()
        with h5py.File(allowed / "inside.h5", "w") as handle:
            handle.create_dataset("value", data=[1, 2, 3])
        readonly = sqlite3.connect(
            (blocked / "readonly.db").as_uri() + "?mode=ro",
            uri=True,
        )
        readonly_value = readonly.execute(
            "select value from values_table"
        ).fetchone()[0]
        readonly.close()
        with h5py.File(blocked / "readonly.h5", "r") as handle:
            h5_value = int(handle["value"][0])

        print(json.dumps({{
            "active": os.environ.get("SCAI_KERNEL_LANDLOCK_ACTIVE"),
            "outcomes": outcomes,
            "readonly_value": readonly_value,
            "h5_value": h5_value,
        }}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=allowed,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert int(payload["active"]) >= 3
    assert payload["active"] != "999"
    assert set(payload["outcomes"].values()).isdisjoint({"ALLOWED"})
    assert payload["readonly_value"] == 7
    assert payload["h5_value"] == 7
    assert (blocked / "existing.txt").read_text() == "untouched"
    assert (blocked / "existing.txt").stat().st_mode == metadata_mode
    assert (blocked / "existing.txt").stat().st_mtime_ns == metadata_mtime
    assert not (blocked / "renamed.txt").exists()
    assert not (blocked / "native.db").exists()
    assert not (blocked / "native.h5").exists()
    assert (allowed / "python.txt").read_text() == "ok"
    assert (allowed / "bytes.bin").is_file()
    assert (allowed / "inside.db").is_file()
    assert (allowed / "inside.h5").is_file()


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_trusted_launcher_fails_closed_when_write_root_is_invalid(tmp_path):
    env = os.environ.copy()
    env.update({
        "SCAI_KERNEL_WRITE_ROOTS": json.dumps([str(tmp_path / "missing")]),
        "SCAI_KERNEL_REQUIRE_LANDLOCK": "1",
    })
    launcher = _startup_dir() / "launcher.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(launcher),
            sys.executable,
            "-c",
            "print('kernel-started')",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 78
    assert "kernel-started" not in completed.stdout
    assert "startup failed" in completed.stderr


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
async def test_live_kernel_write_entrypoints_and_modes_are_session_scoped(tmp_path):
    current = tmp_path / "session-current"
    other = tmp_path / "session-other"
    workspace = current / "workspace"
    states = current / "states"
    workspace.mkdir(parents=True)
    states.mkdir()
    other.mkdir()
    existing = other / "existing.txt"
    existing.write_text("untouched")
    session_metadata = current / "session.json"
    session_metadata.write_text("host-owned")
    symlink = workspace / "outside-link.txt"
    symlink.symlink_to(existing)
    (workspace / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "Path('untrusted-sitecustomize-loaded').write_text('yes')\n"
    )
    (workspace / "local_workspace_module.py").write_text("VALUE = 42\n")

    sandbox = LocalJupyterSandbox(
        session_id=f"isolation-{uuid.uuid4().hex[:8]}",
        state_dir=str(states),
        working_dir=str(workspace),
        timeout=60,
    )
    await sandbox.start()
    try:
        probe = await sandbox.execute(
            textwrap.dedent(
                f"""
                import bz2
                import builtins
                import anndata as ad
                import h5py
                import io
                import json
                import numpy as np
                import os
                import pandas as pd
                from pathlib import Path
                import sqlite3
                import sys
                import tarfile
                import tempfile
                import zipfile
                import local_workspace_module

                outside = Path({str(other)!r})
                existing = outside / "existing.txt"
                session_metadata = Path({str(session_metadata)!r})
                outcomes = {{}}

                def attempt(name, operation):
                    try:
                        operation()
                    except Exception as exc:
                        outcomes[name] = type(exc).__name__
                    else:
                        outcomes[name] = "ALLOWED"

                for index, mode in enumerate((
                    "r+", "rt+", "r+t", "rb+", "r+b", "br+",
                    "w", "wb", "w+", "a", "ab", "a+",
                )):
                    target = existing if mode.startswith("r") else outside / f"mode-{{index}}"
                    attempt(f"mode:{{mode}}", lambda target=target, mode=mode: open(target, mode))
                for index, mode in enumerate(("x", "xb", "x+")):
                    target = outside / f"exclusive-{{index}}"
                    attempt(f"mode:{{mode}}", lambda target=target, mode=mode: open(target, mode))

                attempt("builtins.open", lambda: builtins.open(existing, "r+"))
                attempt("io.open", lambda: io.open(existing, "r+"))
                attempt("Path.open", lambda: existing.open("r+"))
                attempt("os.open-rdwr", lambda: os.open(existing, os.O_RDWR))
                attempt(
                    "os.open-create",
                    lambda: os.open(outside / "low-level", os.O_CREAT | os.O_TRUNC | os.O_WRONLY),
                )
                attempt("Path.touch", lambda: (outside / "touch").touch())
                attempt(
                    "tempfile",
                    lambda: tempfile.NamedTemporaryFile(dir=outside, delete=False),
                )
                attempt("bz2", lambda: bz2.open(outside / "archive.bz2", "w"))
                attempt("zip", lambda: zipfile.ZipFile(outside / "archive.zip", "w"))
                attempt("tar", lambda: tarfile.open(outside / "archive.tar", "w"))
                attempt("sqlite", lambda: sqlite3.connect(outside / "native.db"))
                attempt("h5py", lambda: h5py.File(outside / "native.h5", "w"))
                attempt("symlink", lambda: open({str(symlink)!r}, "w"))
                attempt("original-open", lambda: _ORIGINAL_OPEN(existing, "r+"))
                attempt("original-os-open", lambda: _ORIGINAL_OS_OPEN(existing, os.O_RDWR))
                attempt("session-metadata", lambda: open(session_metadata, "r+"))

                process_module = sys.modules.get("sub" + "process")
                outcomes["process-module-loaded"] = process_module is not None
                if process_module is not None:
                    attempt(
                        "dynamic-process-chmod",
                        lambda: process_module.run(
                            ["/usr/bin/chmod", "400", str(existing)],
                            check=True,
                        ),
                    )

                directory_fd = os.open(outside, os.O_RDONLY)
                try:
                    attempt(
                        "dir-fd",
                        lambda: os.open("existing.txt", os.O_RDWR, dir_fd=directory_fd),
                    )
                finally:
                    os.close(directory_fd)

                readable = existing.read_text()
                open = io.open
                attempt("shadowed-bare-open", lambda: open(existing, "r+"))

                inside = Path("isolation-compatible")
                inside.mkdir(exist_ok=True)
                (inside / "python.txt").write_text("ok")
                np.save(inside / "array.npy", np.array([1, 2, 3]))
                pd.DataFrame({{"value": [1, 2]}}).to_csv(inside / "frame.csv", index=False)
                ad.AnnData(np.arange(6).reshape(2, 3)).write_h5ad(inside / "adata.h5ad")
                with h5py.File(inside / "inside.h5", "w") as handle:
                    handle.create_dataset("value", data=[1, 2, 3])
                connection = sqlite3.connect(inside / "inside.db")
                connection.execute("create table ok (value integer)")
                connection.commit()
                connection.close()
                with tempfile.NamedTemporaryFile(dir=inside, delete=False) as handle:
                    handle.write(b"ok")

                print("__SCAI_ISOLATION__" + json.dumps({{
                    "active": os.environ.get("SCAI_KERNEL_LANDLOCK_ACTIVE"),
                    "bare_open_guarded": builtins.open is _sandboxed_open,
                    "readable": readable,
                    "local_import": local_workspace_module.VALUE,
                    "outcomes": outcomes,
                }}, sort_keys=True))
                """
            ),
            records_history=False,
        )
        assert probe.success, probe.error
        payload_line = next(
            line for line in probe.output.splitlines()
            if line.startswith("__SCAI_ISOLATION__")
        )
        payload = json.loads(payload_line.removeprefix("__SCAI_ISOLATION__"))
        assert int(payload["active"]) >= 3
        assert payload["bare_open_guarded"] is True
        assert payload["readable"] == "untouched"
        assert payload["local_import"] == 42
        assert payload["outcomes"].pop("process-module-loaded") is True
        assert set(payload["outcomes"].values()).isdisjoint({"ALLOWED"})

        assert existing.read_text() == "untouched"
        assert session_metadata.read_text() == "host-owned"
        assert sorted(path.name for path in other.iterdir()) == ["existing.txt"]
        compatible = workspace / "isolation-compatible"
        assert (compatible / "python.txt").read_text() == "ok"
        assert (compatible / "array.npy").is_file()
        assert (compatible / "frame.csv").is_file()
        assert (compatible / "adata.h5ad").is_file()
        assert (compatible / "inside.db").is_file()
        assert (compatible / "inside.h5").is_file()
        assert not (workspace / "untrusted-sitecustomize-loaded").exists()

        await sandbox._restart_kernel()
        after_restart = await sandbox.execute(
            textwrap.dedent(
                f"""
                import json
                import os
                from pathlib import Path

                target = Path({str(existing)!r})
                try:
                    open(target, "r+")
                except Exception as exc:
                    outcome = type(exc).__name__
                else:
                    outcome = "ALLOWED"
                print("__SCAI_RESTART_ISOLATION__" + json.dumps({{
                    "active": os.environ.get("SCAI_KERNEL_LANDLOCK_ACTIVE"),
                    "outcome": outcome,
                    "sitecustomize_loaded": Path(
                        "untrusted-sitecustomize-loaded"
                    ).exists(),
                }}))
                """
            ),
            records_history=False,
        )
        assert after_restart.success, after_restart.error
        restart_line = next(
            line for line in after_restart.output.splitlines()
            if line.startswith("__SCAI_RESTART_ISOLATION__")
        )
        restart_payload = json.loads(
            restart_line.removeprefix("__SCAI_RESTART_ISOLATION__")
        )
        assert int(restart_payload["active"]) >= 3
        assert restart_payload["outcome"] != "ALLOWED"
        assert restart_payload["sitecustomize_loaded"] is False
        assert existing.read_text() == "untouched"
    finally:
        await sandbox.stop(save=False)
