"""Subprocess contracts for direct startup dependency identity and fail-fast paths.

The harness deliberately copies only the small startup surface into a temporary
checkout.  It never uses the repository's dependency directory, network, or a
real service process.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StartupCase:
    system_python: Literal["absent", "valid", "broken", "wrong_minor"]
    uv: Literal[
        "absent",
        "existing_valid",
        "find_broken",
        "find_wrong_minor",
        "existing_outside_root",
        "implicit_download_trap",
        "install_valid",
        "install_fail",
        "post_install_find_fail",
        "post_install_broken",
        "post_install_wrong_minor",
        "install_outside_root",
    ] = "absent"
    existing_venv: Literal[
        "absent",
        "valid",
        "missing",
        "broken",
        "wrong_minor",
        "regular_file",
        "symlink_directory",
        "dangling_symlink",
    ] = "absent"
    fail_install: Literal["none", "pip_upgrade", "requirements", "nodeps", "npm_ci"] = "none"


_FAKE_UVICORN = r'''#!/bin/bash
printf '{"event":"service"}\n' >> "$SCAI_FAKE_EVENT_LOG"
for _i in $(seq 1 100); do
  if [ -f "$SCAI_FAKE_PID_FILE" ]; then
    printf '{"event":"pid"}\n' >> "$SCAI_FAKE_EVENT_LOG"
    exit 0
  fi
  sleep 0.01
done
exit 0
'''


_FAKE_PYTHON = r'''#!/usr/bin/python3
import json
import os
import shutil
import sys
import time
from pathlib import Path

MINOR = {minor!r}
BROKEN = {broken!r}
UVICORN_BODY = {uvicorn_body!r}
EVENTS = Path(os.environ["SCAI_FAKE_EVENT_LOG"])
VITE_LOCK_QUERY = 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["packages"]["node_modules/vite"]["version"])'


def event(name, **values):
    record = {{"event": name}}
    record.update(values)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


if BROKEN:
    raise SystemExit(1)

args = sys.argv[1:]
if len(args) == 3 and args[0] == "-c" and args[1] == VITE_LOCK_QUERY:
    lock = json.loads(Path(args[2]).read_text(encoding="utf-8"))
    print(lock["packages"]["node_modules/vite"]["version"])
    raise SystemExit(0)
if "-c" in args:
    code = args[args.index("-c") + 1]
    if "sys.version_info" in code:
        print(MINOR)
        raise SystemExit(0)
    if "urllib.request" in code:
        event("health")
        raise SystemExit(0)
    raise SystemExit(0)

if len(args) >= 2 and args[0] == "-m" and args[1] == "venv":
    target = Path(args[-1])
    target.mkdir(parents=True, exist_ok=False)
    (target / "bin").mkdir()
    shutil.copy2(Path(sys.argv[0]), target / "bin" / "python")
    (target / "bin" / "python").chmod(0o755)
    (target / "bin" / "activate").write_text(
        "VIRTUAL_ENV=%s\nexport VIRTUAL_ENV\n" % target,
        encoding="utf-8",
    )
    uvicorn = target / "bin" / "uvicorn"
    uvicorn.write_text(UVICORN_BODY, encoding="utf-8")
    uvicorn.chmod(0o755)
    event("venv", path=str(target))
    raise SystemExit(0)

if len(args) >= 2 and args[0] == "-m" and args[1] == "pip":
    pip_args = args[2:]
    event("pip", argv=pip_args, fail=os.environ.get("SCAI_FAIL_INSTALL", "none"))
    if pip_args[:2] == ["install", "--help"]:
        print("--build-constraint")
        raise SystemExit(0)
    if pip_args[:2] == ["install", "-q"] and "--upgrade" in pip_args:
        failed = os.environ.get("SCAI_FAIL_INSTALL") == "pip_upgrade"
    elif "--no-deps" in pip_args:
        failed = os.environ.get("SCAI_FAIL_INSTALL") == "nodeps"
    else:
        failed = os.environ.get("SCAI_FAIL_INSTALL") == "requirements"
    raise SystemExit(1 if failed else 0)

raise SystemExit(0)
'''


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _with_current_python_shebang(body: str) -> str:
    lines = body.splitlines(keepends=True)
    assert lines and lines[0].startswith("#!")
    lines[0] = f"#!{sys.executable}\n"
    result = "".join(lines)
    assert result.splitlines()[0] == f"#!{sys.executable}"
    return result


def _fake_interpreter(path: Path, *, minor: str = "3.11", broken: bool = False) -> None:
    body = _FAKE_PYTHON.format(minor=minor, broken=broken, uvicorn_body=_FAKE_UVICORN)
    _write_executable(path, _with_current_python_shebang(body))


def _log_script(path: Path, name: str, *, fail: str | None = None) -> None:
    if name == "node":
        body = "#!/bin/bash\nif [ \"$1\" = \"--version\" ]; then echo v20.19.0; exit 0; fi\nexit 0\n"
    else:
        body = r'''#!/bin/bash
set -eu
printf '{"event":"npm","argv":[' >> "$SCAI_FAKE_EVENT_LOG"
first=1
for arg in "$@"; do
  if [ "$first" -eq 0 ]; then printf ',' >> "$SCAI_FAKE_EVENT_LOG"; fi
  first=0
  printf '%s' "$arg" | __SCAI_TEST_PYTHON__ -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")), end="")' >> "$SCAI_FAKE_EVENT_LOG"
done
printf ']}\n' >> "$SCAI_FAKE_EVENT_LOG"
if [ "$1" = "ci" ]; then
  if [ "${SCAI_FAIL_INSTALL:-none}" = "npm_ci" ]; then exit 1; fi
  mkdir -p node_modules
fi
exit 0
'''
        body = body.replace("__SCAI_TEST_PYTHON__", shlex.quote(sys.executable))
    _write_executable(path, body)


_FAKE_HARNESS_ROOTS: set[Path] = set()


def _probe_executable_directory(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".scai-startup-exec-probe-{uuid.uuid4().hex}"
    try:
        probe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        probe.chmod(0o755)
        return subprocess.run([str(probe)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def _fake_harness_root(tmp_path: Path) -> Path:
    candidates = []
    output_root = Path("/results/output")
    if output_root.is_dir() and os.access(output_root, os.W_OK | os.X_OK):
        candidates.append(output_root / f".scai-startup-fake-{uuid.uuid4().hex}")
    candidates.append(tmp_path / f"scai-startup-fake-{uuid.uuid4().hex}")
    for candidate in candidates:
        if _probe_executable_directory(candidate):
            _FAKE_HARNESS_ROOTS.add(candidate)
            return candidate
        shutil.rmtree(candidate, ignore_errors=True)
    raise RuntimeError("no executable test-only directory is available for startup harness")


@pytest.fixture(autouse=True)
def _cleanup_fake_harness_roots():
    yield
    for root in list(_FAKE_HARNESS_ROOTS):
        shutil.rmtree(root, ignore_errors=True)
        _FAKE_HARNESS_ROOTS.discard(root)


def _prepare_checkout(tmp_path: Path, case: StartupCase) -> tuple[Path, Path]:
    harness_root = _fake_harness_root(tmp_path)
    checkout = harness_root / "scai"
    checkout.mkdir()
    for relative in (
        "start.sh",
        "Dockerfile",
        "requirements.txt",
        "requirements-nodeps.txt",
        "build-constraints.txt",
    ):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    helper = checkout / "scripts/link_frontend_node_modules.sh"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/link_frontend_node_modules.sh", helper)
    assert stat.S_IMODE(helper.stat().st_mode) == 0o644
    frontend = checkout / "frontend"
    frontend.mkdir(parents=True)
    package_json = {"name": "fake-frontend", "version": "1.0.0"}
    lock_json = {
        "name": "fake-frontend",
        "version": "1.0.0",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "fake-frontend", "version": "1.0.0"},
            "node_modules/vite": {"version": "8.0.0"},
        },
    }
    (frontend / "package.json").write_text(json.dumps(package_json), encoding="utf-8")
    (frontend / "package-lock.json").write_text(json.dumps(lock_json), encoding="utf-8")
    excluded_roots = {"dist", "node_modules", ".pytest_cache", ".benchmarks"}
    frontend_inputs = [
        path
        for path in frontend.rglob("*")
        if path.is_file()
        and path.relative_to(frontend).parts[0] not in excluded_roots
    ]
    latest_input_mtime_ns = max(path.stat().st_mtime_ns for path in frontend_inputs)
    dist_index = frontend / "dist/index.html"
    dist_index.parent.mkdir()
    dist_index.write_text("<!doctype html>\n", encoding="utf-8")
    fresh_dist_mtime_ns = latest_input_mtime_ns + 1_000_000_000
    os.utime(dist_index, ns=(fresh_dist_mtime_ns, fresh_dist_mtime_ns))

    fake_bin = harness_root / "bin"
    event_log = checkout / "events.jsonl"
    event_log.touch()
    pid_file = checkout / ".scai.pid"

    if case.system_python != "absent":
        _fake_interpreter(
            fake_bin / "python3.11",
            minor="3.10" if case.system_python == "wrong_minor" else "3.11",
            broken=case.system_python == "broken",
        )
    if case.uv != "absent":
        managed_root = harness_root / "dep/python/managed"
        managed_path = managed_root / "bin/python"
        outside_path = harness_root / "outside-python"
        if case.uv in {"existing_valid", "install_valid", "implicit_download_trap"}:
            _fake_interpreter(managed_path)
        elif case.uv in {"find_broken", "post_install_broken"}:
            _fake_interpreter(managed_path, broken=True)
        elif case.uv in {"find_wrong_minor", "post_install_wrong_minor"}:
            _fake_interpreter(managed_path, minor="3.10")
        elif case.uv in {"existing_outside_root", "install_outside_root"}:
            _fake_interpreter(outside_path)
        uv_body = r'''#!/usr/bin/python3
import json
import os
import sys
from pathlib import Path

mode = os.environ["SCAI_UV_MODE"]
state = Path(os.environ["SCAI_UV_STATE"])
try:
    finds = int(state.read_text(encoding="utf-8"))
except FileNotFoundError:
    finds = 0
args = sys.argv[1:]
record = {
    "event": "uv",
    "argv": args,
    "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR"),
    "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR"),
}
with Path(os.environ["SCAI_FAKE_EVENT_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
if args[:2] == ["python", "find"]:
    state.write_text(str(finds + 1), encoding="utf-8")
    if mode == "implicit_download_trap" or mode in {"install_fail", "post_install_find_fail"} and finds == 0:
        raise SystemExit(1)
    if mode == "existing_valid":
        print(str(Path(os.environ["UV_PYTHON_INSTALL_DIR"]) / "managed/bin/python"))
        raise SystemExit(0)
    if mode == "existing_outside_root":
        print(os.environ["SCAI_UV_OUTSIDE"])
        raise SystemExit(0)
    if finds == 0:
        raise SystemExit(1)
    if mode == "post_install_find_fail":
        raise SystemExit(1)
    if mode in {"post_install_broken", "post_install_wrong_minor", "install_outside_root"}:
        if mode == "install_outside_root":
            print(os.environ["SCAI_UV_OUTSIDE"])
        else:
            print(str(Path(os.environ["UV_PYTHON_INSTALL_DIR"]) / "managed/bin/python"))
        raise SystemExit(0)
    print(str(Path(os.environ["UV_PYTHON_INSTALL_DIR"]) / "managed/bin/python"))
    raise SystemExit(0)
if args[:2] == ["python", "install"]:
    if mode == "install_fail" or mode == "implicit_download_trap":
        raise SystemExit(1)
    raise SystemExit(0)
raise SystemExit(2)
'''
        _write_executable(fake_bin / "uv", _with_current_python_shebang(uv_body))

    _log_script(fake_bin / "node", "node")
    _log_script(fake_bin / "npm", "npm")

    dep_dir = harness_root / "dep"
    if case.existing_venv != "absent":
        venv = dep_dir / "venv-py311"
        if case.existing_venv == "regular_file":
            venv.parent.mkdir(parents=True, exist_ok=True)
            venv.write_bytes(b"existing regular file sentinel\n")
        elif case.existing_venv == "dangling_symlink":
            venv.parent.mkdir(parents=True, exist_ok=True)
            venv.symlink_to(harness_root / "missing-existing-venv", target_is_directory=True)
        elif case.existing_venv == "symlink_directory":
            target = harness_root / "outside-existing-venv"
            _fake_interpreter(target / "bin/python")
            (target / "bin/activate").write_text("VIRTUAL_ENV=%s\n" % target, encoding="utf-8")
            _write_executable(target / "bin/uvicorn", _FAKE_UVICORN)
            venv.parent.mkdir(parents=True, exist_ok=True)
            venv.symlink_to(target, target_is_directory=True)
        elif case.existing_venv == "missing":
            venv.mkdir(parents=True)
        else:
            _fake_interpreter(
                venv / "bin/python",
                minor="3.10" if case.existing_venv == "wrong_minor" else "3.11",
                broken=case.existing_venv == "broken",
            )
            (venv / "bin/activate").parent.mkdir(parents=True, exist_ok=True)
            (venv / "bin/activate").write_text("VIRTUAL_ENV=%s\n" % venv, encoding="utf-8")
            _write_executable(venv / "bin/uvicorn", _FAKE_UVICORN)

    (harness_root / "uv-state").write_text("0", encoding="utf-8")
    env_file = harness_root / "harness-env.json"
    env_file.write_text(
        json.dumps(
            {
                "event_log": str(event_log),
                "pid_file": str(pid_file),
                "uv_state": str(harness_root / "uv-state"),
                "outside": str(harness_root / "outside-python"),
            }
        ),
        encoding="utf-8",
    )
    return checkout, fake_bin


def _run_startup(checkout: Path, fake_bin: Path, case: StartupCase) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "SCAI_FAKE_EVENT_LOG": str(checkout / "events.jsonl"),
            "SCAI_FAKE_PID_FILE": str(checkout / ".scai.pid"),
            "SCAI_UV_MODE": case.uv,
            "SCAI_UV_STATE": str(checkout.parent / "uv-state"),
            "SCAI_UV_OUTSIDE": str(checkout.parent / "outside-python"),
            "SCAI_FAIL_INSTALL": case.fail_install,
        }
    )
    args = ["bash", "start.sh", "--dev" if case.fail_install == "npm_ci" else "--prod"]
    return subprocess.run(args, cwd=checkout, env=env, text=True, capture_output=True, timeout=15)


def _events(checkout: Path) -> list[dict[str, object]]:
    path = checkout / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _existing_venv_snapshot(path: Path) -> tuple[object, ...]:
    """Capture the exact pre-existing path without following it as the path itself."""

    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode):
        target = os.readlink(path)
        resolved_target = (path.parent / target).resolve(strict=False)
        contents: tuple[tuple[str, bytes], ...] = ()
        if resolved_target.is_dir():
            contents = tuple(
                (
                    str(item.relative_to(resolved_target)),
                    item.read_bytes(),
                )
                for item in sorted(resolved_target.rglob("*"))
                if item.is_file()
            )
        return ("symlink", target, resolved_target.exists(), contents)
    if stat.S_ISREG(path_stat.st_mode):
        return ("regular_file", path.read_bytes())
    return ("other", path_stat.st_mode)


def test_startup_harness_probes_executable_root_and_cleans_fake_roots():
    source = Path(__file__).read_text(encoding="utf-8")
    for marker in (
        "_probe_executable_directory",
        'Path(\"/results/output\")',
        "_FAKE_HARNESS_ROOTS",
        "@pytest.fixture(autouse=True)",
        "shutil.rmtree(root, ignore_errors=True)",
        "stat.S_IMODE(helper.stat().st_mode) == 0o644",
        "_with_current_python_shebang",
        "sys.executable",
        "_FAKE_UVICORN",
        "UVICORN_BODY = {uvicorn_body!r}",
        "_write_executable(venv / \"bin/uvicorn\", _FAKE_UVICORN)",
    ):
        assert marker in source


def test_fake_interpreters_use_the_test_process_shebang(tmp_path: Path):
    checkout, fake_bin = _prepare_checkout(tmp_path, StartupCase("valid", "existing_valid"))
    expected = f"#!{sys.executable}"
    for script in (
        fake_bin / "python3.11",
        fake_bin / "uv",
        checkout.parent / "dep/python/managed/bin/python",
    ):
        assert script.read_text(encoding="utf-8").splitlines()[0] == expected


def test_startup_fixture_marks_prebuilt_dist_strictly_newer_than_frontend_inputs(tmp_path: Path):
    checkout, _ = _prepare_checkout(tmp_path, StartupCase("valid"))
    frontend = checkout / "frontend"
    dist_index = frontend / "dist/index.html"
    excluded_roots = {"dist", "node_modules", ".pytest_cache", ".benchmarks"}
    frontend_inputs = [
        path
        for path in frontend.rglob("*")
        if path.is_file()
        and path.relative_to(frontend).parts[0] not in excluded_roots
    ]
    assert frontend_inputs
    assert dist_index.stat().st_mtime_ns > max(
        path.stat().st_mtime_ns for path in frontend_inputs
    )


def _case_key(case: StartupCase) -> str:
    return ":".join((case.system_python, case.uv, case.existing_venv, case.fail_install))


@pytest.mark.parametrize(
    "case, succeeds",
    [
        (StartupCase("valid"), True),
        (StartupCase("broken"), False),
        (StartupCase("wrong_minor"), False),
        (StartupCase("absent", "existing_valid"), True),
        (StartupCase("absent", "find_broken"), False),
        (StartupCase("absent", "find_wrong_minor"), False),
        (StartupCase("absent", "existing_outside_root"), False),
        (StartupCase("absent", "implicit_download_trap"), False),
        (StartupCase("absent", "install_valid"), True),
        (StartupCase("absent", "absent"), False),
        (StartupCase("absent", "install_fail"), False),
        (StartupCase("absent", "post_install_find_fail"), False),
        (StartupCase("absent", "post_install_broken"), False),
        (StartupCase("absent", "post_install_wrong_minor"), False),
        (StartupCase("absent", "install_outside_root"), False),
        (StartupCase("valid", existing_venv="valid"), True),
        (StartupCase("valid", existing_venv="missing"), False),
        (StartupCase("valid", existing_venv="broken"), False),
        (StartupCase("valid", existing_venv="wrong_minor"), False),
        (StartupCase("valid", fail_install="pip_upgrade"), False),
        (StartupCase("valid", fail_install="requirements"), False),
        (StartupCase("valid", fail_install="nodeps"), False),
        (StartupCase("valid", fail_install="npm_ci"), False),
    ],
    ids=lambda value: _case_key(value) if isinstance(value, StartupCase) else str(value),
)
def test_startup_provider_venv_and_install_failure_matrix(tmp_path: Path, case: StartupCase, succeeds: bool):
    checkout, fake_bin = _prepare_checkout(tmp_path, case)
    result = _run_startup(checkout, fake_bin, case)
    events = _events(checkout)
    event_names = {str(item.get("event")) for item in events}
    if succeeds:
        assert result.returncode == 0, result.stderr
        assert {"service", "pid"}.issubset(event_names)
    else:
        assert result.returncode != 0, result.stdout
        assert not ({"service", "pid"} & event_names)
    if case.fail_install == "npm_ci":
        npm_events = [event for event in events if event.get("event") == "npm"]
        assert len(npm_events) == 1
        assert npm_events[0]["argv"] == ["ci"]
        assert "locked frontend npm ci failed; services were not started" in (
            result.stdout + result.stderr
        )
        assert not (checkout.parent / "dep/frontend/.install-complete").exists()
        assert not ({"service", "pid"} & event_names)


@pytest.mark.parametrize("existing_venv", ["regular_file", "symlink_directory", "dangling_symlink"])
def test_startup_rejects_nonexact_existing_venv_paths_before_creation_install_or_service(
    tmp_path: Path, existing_venv: str
):
    case = StartupCase("valid", existing_venv=existing_venv)  # type: ignore[arg-type]
    checkout, fake_bin = _prepare_checkout(tmp_path, case)
    venv = checkout.parent / "dep/venv-py311"
    before = _existing_venv_snapshot(venv)

    result = _run_startup(checkout, fake_bin, case)
    event_names = {str(event.get("event")) for event in _events(checkout)}

    assert result.returncode != 0, result.stdout
    assert _existing_venv_snapshot(venv) == before
    assert "not a usable Python" in result.stderr
    assert "Move it aside and retry" in result.stderr
    assert not ({"venv", "pip", "npm", "service", "pid"} & event_names)


def test_uv_find_receives_cache_install_env_and_all_managed_flags(tmp_path: Path):
    case = StartupCase("absent", "existing_valid")
    checkout, fake_bin = _prepare_checkout(tmp_path, case)
    result = _run_startup(checkout, fake_bin, case)
    assert result.returncode == 0, result.stderr
    harness_root = checkout.parent
    uv = [event for event in _events(checkout) if event["event"] == "uv"]
    assert uv
    assert uv[0]["argv"] == [
        "python",
        "find",
        "--no-project",
        "--no-python-downloads",
        "--managed-python",
        "--no-config",
        "--resolve-links",
        "3.11",
    ]
    assert uv[0]["UV_CACHE_DIR"] == str(harness_root / "dep/uv-cache")
    assert uv[0]["UV_PYTHON_INSTALL_DIR"] == str(harness_root / "dep/python")


def test_uv_install_receives_explicit_install_dir_and_no_bin(tmp_path: Path):
    case = StartupCase("absent", "install_valid")
    checkout, fake_bin = _prepare_checkout(tmp_path, case)
    result = _run_startup(checkout, fake_bin, case)
    assert result.returncode == 0, result.stderr
    harness_root = checkout.parent
    uv = [event for event in _events(checkout) if event["event"] == "uv"]
    install = next(event for event in uv if event["argv"][:2] == ["python", "install"])
    assert install["argv"] == [
        "python",
        "install",
        "--no-bin",
        "--install-dir",
        str(harness_root / "dep/python"),
        "--no-config",
        "3.11",
    ]
    assert install["UV_CACHE_DIR"] == str(harness_root / "dep/uv-cache")
    assert install["UV_PYTHON_INSTALL_DIR"] == str(harness_root / "dep/python")


def test_uv_returned_realpath_must_remain_under_install_root(tmp_path: Path):
    case = StartupCase("absent", "existing_outside_root")
    checkout, fake_bin = _prepare_checkout(tmp_path, case)
    result = _run_startup(checkout, fake_bin, case)
    assert result.returncode != 0
    assert "outside" in result.stderr.lower()
    assert "service" not in {str(event.get("event")) for event in _events(checkout)}


def test_frontend_build_is_not_wrapped_by_f0_dependency_helper():
    script = (ROOT / "start.sh").read_text(encoding="utf-8")
    assert 'run_dependency_install "npm run build"' not in script
    assert "npm run build" in script
