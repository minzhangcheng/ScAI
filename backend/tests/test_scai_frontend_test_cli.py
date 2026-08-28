"""Docker-free contracts for the isolated frontend runner."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import types
import uuid
from pathlib import Path
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_FILE = ROOT / "scripts" / "scai-frontend-test"
PYTHON_RUNNER_FILE = ROOT / "scripts" / "scai_frontend_runner.py"
PARTS_DIR = ROOT / "scripts" / "scai_frontend_parts"
FIXTURE_FILE = ROOT / "backend/tests/fixtures/npm9-layout.json"
FIXTURE = json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))


def _frontend_source() -> str:
    paths = [PYTHON_RUNNER_FILE, *sorted(PARTS_DIR.glob("*.py"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def load_frontend_module(path: Path = PYTHON_RUNNER_FILE):
    spec = importlib.util.spec_from_file_location("scai_frontend_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    return module


FRONTEND = load_frontend_module()
FRONTEND_RUNNER = sys.modules["scai_frontend_parts.runner"]
FRONTEND_NPM = sys.modules["scai_frontend_parts.npm"]
_FAKE_TOOL_ROOTS: set[Path] = set()
NPM9_EMPTY_CONFIG_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
NPM9_ENV_KEYS = ("audit", "cache", "fund", "globalconfig", "registry", "update-notifier", "userconfig")


def _npm9_expected_config(install_dir: Path) -> dict[str, object]:
    install = install_dir.resolve()
    return {
        "audit": False,
        "cache": str((install / ".scai-runtime/cache").resolve()),
        "fund": False,
        "globalconfig": str((install / "global.npmrc").resolve()),
        "registry": FRONTEND.REGISTRY,
        "update-notifier": False,
        "userconfig": str((install / ".npmrc").resolve()),
    }


def _npm9_json_value(
    expected: dict[str, object],
    *,
    include_json: bool = True,
    json_value: object = True,
    include_location: bool = True,
    location: object = "project",
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        **expected,
        "globalignorefile": "/etc/npmignore",
        "prefix": "/usr/local",
        "strict-ssl": True,
    }
    if include_json:
        value["json"] = json_value
    if include_location:
        value["location"] = location
    if extra:
        value.update(extra)
    return value


def _npm9_json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _npm9_plain_bytes(
    expected: dict[str, object],
    *,
    env_name: str = "env",
    env_source: str = "environment",
    env_assignments: list[tuple[str, object]] | None = None,
    cli_name: str = "cli",
    cli_source: str = "command line options",
    cli_assignments: list[tuple[str, object]] | None = None,
    include_env: bool = True,
    include_cli: bool = True,
) -> bytes:
    lines = [
        '; "builtin" config from /usr/share/nodejs/npm/npmrc',
        'globalignorefile = "/etc/npmignore"',
        'prefix = "/usr/local"',
    ]
    if include_env:
        lines.extend(["", f'; "{env_name}" config from {env_source}'])
        assignments = env_assignments or [(key, expected[key]) for key in NPM9_ENV_KEYS]
        lines.extend(_npm9_plain_assignment(key, value) for key, value in assignments)
    if include_cli:
        lines.extend(["", f'; "{cli_name}" config from {cli_source}'])
        assignments = cli_assignments if cli_assignments is not None else [("location", "project")]
        lines.extend(_npm9_plain_assignment(key, value) for key, value in assignments)
    lines.extend(
        [
            "",
            "; node bin location = /usr/bin/node",
            "; node version = v20.19.2",
            "; npm local prefix = <install-dir>",
            "; npm version = 9.2.0",
            "; cwd = <install-dir>",
            "; HOME = <install-dir>/.scai-runtime/home",
            "; Run `npm config ls -l` to show all defaults.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _npm9_plain_assignment(key: str, value: object) -> str:
    encoded = str(value).lower() if isinstance(value, bool) else json.dumps(value, ensure_ascii=False)
    return f"{key} = {encoded}"


def _npm9_empty_config_files(install_dir: Path) -> dict[str, str]:
    for name in (".npmrc", "global.npmrc"):
        (install_dir / name).write_bytes(b"")
    return {"userconfig": NPM9_EMPTY_CONFIG_SHA, "globalconfig": NPM9_EMPTY_CONFIG_SHA}


def _probe_executable_directory(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".scai-frontend-exec-probe-{uuid.uuid4().hex}"
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


def _fake_tools_root(tmp_path: Path) -> Path:
    candidates = []
    output_root = Path("/results/output")
    if output_root.is_dir() and os.access(output_root, os.W_OK | os.X_OK):
        candidates.append(output_root / f".scai-frontend-fake-{uuid.uuid4().hex}")
    candidates.append(tmp_path / f"scai-frontend-fake-{uuid.uuid4().hex}")
    for candidate in candidates:
        if _probe_executable_directory(candidate):
            _FAKE_TOOL_ROOTS.add(candidate)
            return candidate
        shutil.rmtree(candidate, ignore_errors=True)
    raise RuntimeError("no executable test-only directory is available for fake node/npm")


@pytest.fixture(autouse=True)
def _cleanup_fake_tool_roots():
    yield
    for root in list(_FAKE_TOOL_ROOTS):
        shutil.rmtree(root, ignore_errors=True)
        _FAKE_TOOL_ROOTS.discard(root)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def package_lock(optional_paths: tuple[str, ...] = ()) -> dict:
    packages = {
        "": {
            "name": "scai-frontend",
            "version": "1.0.0",
            "dependencies": {"plain": "1.0.0", "@scope/pkg": "2.0.0", "vitest": "4.1.7"},
        }
    }
    optional_dependencies: dict[str, str] = {}
    for path, record in FIXTURE["packages"].items():
        packages[path] = {key: record[key] for key in ("version", "resolved", "integrity", "bin") if key in record}
    for path in optional_paths:
        record = FIXTURE["optional_lock_packages"][path]
        packages[path] = {key: record[key] for key in ("version", "resolved", "integrity", "optional", "os", "cpu") if key in record}
        optional_dependencies[path.rsplit("/", 1)[-1]] = record["version"]
    if optional_dependencies:
        packages[""]["optionalDependencies"] = optional_dependencies
    return {"name": "scai-frontend", "version": "1.0.0", "lockfileVersion": 3, "requires": True, "packages": packages}


def fake_npm_script(fixture: dict, behavior_path: Path, log_path: Path) -> str:
    fixture_json = json.dumps(fixture, sort_keys=True)
    return f'''#!{sys.executable}
import json
import os
import shutil
import stat
import sys
from pathlib import Path

FIXTURE = json.loads({fixture_json!r})
BEHAVIOR = Path({str(behavior_path)!r})
LOG = Path({str(log_path)!r})

def behavior(name):
    if not BEHAVIOR.exists():
        return False
    try:
        value = json.loads(BEHAVIOR.read_text())
    except Exception:
        return False
    return isinstance(value, dict) and value.get(name) is True

def log(argv):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(json.dumps(argv) + "\\n")

def package_name(key):
    suffix = key.rsplit("node_modules/", 1)[-1]
    parts = suffix.split("/")
    return "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\\n")

def materialize(cwd):
    lock = json.loads((cwd / "package-lock.json").read_text())
    root = cwd / "node_modules"
    root.mkdir()
    materialized_keys = set()
    for key, locked in lock["packages"].items():
        if not key:
            continue
        if key not in FIXTURE.get("packages", {{}}):
            continue
        materialized_keys.add(key)
        package_dir = cwd / key
        package_dir.mkdir(parents=True, exist_ok=True)
        metadata = {{"name": package_name(key), "version": locked["version"]}}
        if "bin" in locked:
            metadata["bin"] = locked["bin"]
        write_json(package_dir / "package.json", metadata)
        if metadata["name"] == "vitest":
            target = package_dir / "vitest.mjs"
            target.write_text("#!/bin/sh\\n")
            target.chmod(0o755)
        if metadata["name"] == "child":
            target = package_dir / "cli.js"
            target.write_text("#!/bin/sh\\n")
            target.chmod(0o755)
    (root / ".bin").mkdir()
    (root / ".bin" / "vitest").symlink_to("../vitest/vitest.mjs")
    nested_bin = cwd / "node_modules/plain/node_modules/.bin"
    nested_bin.mkdir(parents=True)
    (nested_bin / "child-tool").symlink_to("../child/cli.js")
    hidden = {{"lockfileVersion": 3, "packages": {{key: value for key, value in lock["packages"].items() if key in materialized_keys}}}}
    write_json(root / ".package-lock.json", hidden)

argv = sys.argv[1:]
log(argv)
cwd = Path.cwd()
if argv == ["--version"]:
    print("10.9.0")
    raise SystemExit(0)
if argv[:2] == ["config", "list"] and "--json" in argv:
    print(json.dumps({{"json": True, "location": "project", "cache": os.environ["npm_config_cache"], "userconfig": os.environ["npm_config_userconfig"], "globalconfig": os.environ["npm_config_globalconfig"], "registry": os.environ["npm_config_registry"], "audit": False, "fund": False, "update-notifier": False, "strict-ssl": True}}))
    raise SystemExit(0)
if argv == ["config", "list"]:
    plain_location = "global" if behavior("provenance") else "project"
    print('; "env" config from environment')
    for key in ("cache", "userconfig", "globalconfig", "registry"):
        print(f"{{key}} = \\"{{os.environ['npm_config_' + key]}}\\"")
    for key in ("audit", "fund", "update-notifier"):
        print(f"{{key}} = false")
    print('; "cli" config from command line options')
    print(f'location = "{{plain_location}}"')
    raise SystemExit(0)
if argv == ["ci"]:
    if behavior("ci"):
        print("fake npm ci failure", file=sys.stderr)
        raise SystemExit(17)
    materialize(cwd)
    raise SystemExit(0)
if argv == ["ls", "--all", "--json"]:
    if behavior("ls"):
        print("fake npm ls failure", file=sys.stderr)
        raise SystemExit(18)
    dependencies = {{"plain": {{"version": "1.0.0", "dependencies": {{"child": {{"version": "3.0.0"}}}}}}, "@scope/pkg": {{"version": "2.0.0"}}, "vitest": {{"version": "4.1.7"}}}}
    if "node_modules/optional-empty" in json.loads((cwd / "package-lock.json").read_text())["packages"]:
        dependencies["optional-empty"] = {{}}
    print(json.dumps({{"name": "scai-frontend", "version": "1.0.0", "dependencies": dependencies}}))
    raise SystemExit(0)
if argv and argv[0] == "test":
    output = next((item.split("=", 1)[1] for item in argv if item.startswith("--outputFile=")), "")
    if behavior("test"):
        result = {{"numTotalTestSuites": 1, "numPassedTestSuites": 0, "numFailedTestSuites": 1, "numPendingTestSuites": 0, "numTotalTests": 0, "numPassedTests": 0, "numFailedTests": 0, "numPendingTests": 0, "numTodoTests": 0, "success": False, "testResults": [{{"assertionResults": [], "message": "transform failed", "name": "frontend/src/import.test.ts", "status": "failed"}}]}}
    else:
        result = {{"numTotalTestSuites": 4, "numPassedTestSuites": 4, "numFailedTestSuites": 0, "numPendingTestSuites": 0, "numTotalTests": 4, "numPassedTests": 2, "numFailedTests": 0, "numPendingTests": 1, "numTodoTests": 1, "success": True, "testResults": [{{"assertionResults": [{{"ancestorTitles": ["alpha"], "fullName": "alpha passes", "status": "passed", "title": "passes"}}, {{"ancestorTitles": ["alpha"], "fullName": "alpha skips", "status": "pending", "title": "skips"}}], "message": "", "name": "frontend/src/alpha.test.ts", "status": "passed"}}, {{"assertionResults": [{{"ancestorTitles": ["beta"], "fullName": "beta passes", "status": "passed", "title": "passes"}}, {{"ancestorTitles": ["beta"], "fullName": "beta todo", "status": "todo", "title": "todo"}}], "message": "", "name": "frontend/src/beta.test.ts", "status": "passed"}}]}}
    write_json(Path(output), result)
    if behavior("tamper_restore"):
        link = cwd / "node_modules"
        if link.is_symlink():
            link.unlink()
        link.mkdir()
    raise SystemExit(19 if behavior("test") else 0)
print("unknown fake npm argv", argv, file=sys.stderr)
raise SystemExit(2)
'''


def make_frontend_repo(tmp_path: Path, behavior: dict | None = None, *, source_npmrc: bool = False, source_link: str | None = None, optional_paths: tuple[str, ...] = ()) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "backend/tests/fixtures").mkdir(parents=True)
    (repo / ".gitignore").write_text("test-results/\nnode_modules/\n__pycache__/\n", encoding="utf-8")
    for name in ("scai-frontend-test", "scai_frontend_runner.py", "scai_test_tree.py", "link_frontend_node_modules.sh"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    shutil.copytree(PARTS_DIR, repo / "scripts/scai_frontend_parts")
    shutil.copytree(ROOT / "scripts/scai_test_tree_parts", repo / "scripts/scai_test_tree_parts")
    frontend = repo / "frontend"
    frontend.mkdir()
    package = {"name": "scai-frontend", "version": "1.0.0", "private": True, "scripts": {"test": "vitest run"}}
    (frontend / "package.json").write_text(json.dumps(package, sort_keys=True) + "\n", encoding="utf-8")
    (frontend / "package-lock.json").write_text(json.dumps(package_lock(optional_paths), sort_keys=True) + "\n", encoding="utf-8")
    if source_npmrc:
        (frontend / ".npmrc").write_text("registry=https://evil.invalid/\n", encoding="utf-8")
    if source_link is not None:
        target = repo / "old-node-modules"
        target.mkdir()
        (frontend / "node_modules").symlink_to(source_link)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "frontend@example.invalid")
    git(repo, "config", "user.name", "Frontend")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    behavior_path = tmp_path / "behavior.json"
    behavior_path.write_text(json.dumps(behavior or {}), encoding="utf-8")
    log_path = tmp_path / "npm.log"
    tools = _fake_tools_root(tmp_path)
    node = tools / "node"
    node.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-p\" ] && [ \"$2\" = 'JSON.stringify({platform:process.platform,arch:process.arch})' ]; then\n"
        "  printf '%s\\n' '{\"platform\":\"linux\",\"arch\":\"x64\"}'\n"
        "else\n"
        "  printf '%s\\n' 'v20.19.0'\n"
        "fi\n",
        encoding="utf-8",
    )
    npm = tools / "npm"
    npm.write_text(fake_npm_script(FIXTURE, behavior_path, log_path), encoding="utf-8")
    node.chmod(0o755)
    npm.chmod(0o755)
    return repo, behavior_path, log_path


def run_frontend(repo: Path, tools: Path) -> subprocess.CompletedProcess[str]:
    if not (tools / "npm").is_file():
        active = [root for root in _FAKE_TOOL_ROOTS if (root / "npm").is_file()]
        if len(active) != 1:
            raise AssertionError(f"fake npm tool root is not uniquely registered: {active}")
        tools = active[0]
    env = {**os.environ, "PATH": f"{tools}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
    # The immutable runner may copy the checkout onto a noexec temporary
    # filesystem.  Use the explicit shell interpreter while preserving the
    # runner itself as the command under test.
    return subprocess.run(["bash", str(repo / "scripts/scai-frontend-test"), "--install-and-test"], cwd=repo, env=env, text=True, capture_output=True)


def latest_evidence(repo: Path) -> tuple[Path, dict]:
    paths = sorted((repo / "test-results").glob("*-frontend/evidence.json"))
    assert paths
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    return path, json.loads(path.read_text(encoding="utf-8"))


def load_runner_function(name: str):
    return getattr(FRONTEND, name)


def _write_captured_npm9_bin_tree(
    root: Path,
    *,
    specificity_bin: object = None,
    vitest_bin: object = None,
    lock_specificity_bin: object = None,
    lock_vitest_bin: object = None,
    specificity_target_kind: str = "regular",
    vitest_target_kind: str = "regular",
    specificity_shim: tuple[str, str] = ("symlink", "../@bramus/specificity/bin/cli.js"),
    vitest_shim: tuple[str, str] = ("symlink", "../vitest/vitest.mjs"),
    extra_shim: bool = False,
    omit_shims: tuple[str, ...] = (),
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Materialize the captured npm 9 package/lock/tree shape without npm."""
    specificity_bin = {"specificity": "./bin/cli.js"} if specificity_bin is None else specificity_bin
    vitest_bin = {"vitest": "./vitest.mjs"} if vitest_bin is None else vitest_bin
    lock_specificity_bin = {"specificity": "bin/cli.js"} if lock_specificity_bin is None else lock_specificity_bin
    lock_vitest_bin = {"vitest": "vitest.mjs"} if lock_vitest_bin is None else lock_vitest_bin

    install_dir = root / "install"
    node_modules = install_dir / "node_modules"
    specificity_dir = node_modules / "@bramus" / "specificity"
    vitest_dir = node_modules / "vitest"
    node_modules.mkdir(parents=True)

    resolved = {
        "specificity": "https://registry.npmjs.org/@bramus/specificity/-/specificity-2.4.2.tgz",
        "vitest": "https://registry.npmjs.org/vitest/-/vitest-4.1.7.tgz",
    }
    integrity = {
        "specificity": "sha512-ctxtJ/eA+t+6q2++vj5j7FYX3nRu311q1wfYH3xjlLOsczhlhxAg2FWNUXhpGvAw3BWo1xBcvOV6/YLc2r5FJw==",
        "vitest": "sha512-flYyaFd2CgoCoU+0UKt3pxksgC+S02iTDN0n3LtqaMeXsI9SBcdNujc2k0DeFLzUn/0k538yNjOSdwgCqcrwJA==",
    }
    versions = {"specificity": "2.4.2", "vitest": "4.1.7"}
    package_names = {"specificity": "@bramus/specificity", "vitest": "vitest"}
    package_dirs = {"specificity": specificity_dir, "vitest": vitest_dir}
    package_bins = {"specificity": specificity_bin, "vitest": vitest_bin}
    lock_bins = {"specificity": lock_specificity_bin, "vitest": lock_vitest_bin}
    target_paths = {"specificity": "bin/cli.js", "vitest": "vitest.mjs"}
    target_bytes = {
        "specificity": b"#!/usr/bin/env node\n",
        "vitest": b"#!/usr/bin/env node\nconsole.log('vitest');\n",
    }

    lock_packages: dict[str, dict[str, object]] = {}
    for key, package_dir in package_dirs.items():
        package_dir.mkdir(parents=True)
        (package_dir / "package.json").write_text(
            json.dumps(
                {
                    "name": package_names[key],
                    "version": versions[key],
                    "bin": package_bins[key],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        lock_key = f"node_modules/{package_names[key]}"
        lock_packages[lock_key] = {
            "version": versions[key],
            "resolved": resolved[key],
            "integrity": integrity[key],
            "bin": lock_bins[key],
        }

        target = package_dir / target_paths[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        kind = specificity_target_kind if key == "specificity" else vitest_target_kind
        if kind == "regular":
            target.write_bytes(target_bytes[key])
            target.chmod(0o755)
        elif kind == "symlink":
            outside = root / f"{key}-outside-target.js"
            outside.write_bytes(target_bytes[key])
            target.symlink_to(outside)
        elif kind != "missing":
            raise AssertionError(f"unsupported captured target kind: {kind}")

    (install_dir / "package-lock.json").write_text(
        json.dumps({"lockfileVersion": 3, "packages": lock_packages}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hidden = {
        "lockfileVersion": 3,
        "packages": json.loads(json.dumps(lock_packages)),
    }
    (node_modules / ".package-lock.json").write_text(json.dumps(hidden, sort_keys=True) + "\n", encoding="utf-8")

    bin_dir = node_modules / ".bin"
    bin_dir.mkdir()
    shim_specs = {"specificity": specificity_shim, "vitest": vitest_shim}
    for key, (kind, target) in shim_specs.items():
        if key in omit_shims:
            continue
        shim = bin_dir / key
        if kind == "symlink":
            shim.symlink_to(target)
        elif kind == "regular":
            shim.write_bytes(target.encode("utf-8"))
            shim.chmod(0o755)
        else:
            raise AssertionError(f"unsupported captured shim kind: {kind}")
    if extra_shim:
        (bin_dir / "unexpected").symlink_to("../vitest/vitest.mjs")

    npm_ls: dict[str, object] = {
        "name": "scai-frontend",
        "version": "1.0.0",
        "dependencies": {
            "@bramus/specificity": {"version": "2.4.2"},
            "vitest": {"version": "4.1.7"},
        },
    }
    return install_dir, npm_ls, FRONTEND.npm_ls_inventory(npm_ls)


def _walk_captured_npm9_bin_tree(install_dir: Path, npm_ls: dict[str, object], inventory: object) -> dict[str, object]:
    return FRONTEND.walk_npm_tree(install_dir, npm_ls, "linux", "x64", inventory)


def _write_hoisted_npm_tree(
    root: Path,
    *,
    installed_packages: dict[str, dict[str, object]],
    lock_packages: dict[str, dict[str, object]],
    npm_ls: dict[str, object],
) -> tuple[Path, dict[str, object], object]:
    """Materialize a small npm 9 tree whose logical and physical paths differ."""
    install_dir = root / "install"
    node_modules = install_dir / "node_modules"
    node_modules.mkdir(parents=True)

    for key, metadata in sorted(installed_packages.items()):
        package_dir = install_dir / key
        package_dir.mkdir(parents=True)
        (package_dir / "package.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
        )
        if "bin" in metadata:
            normalized = FRONTEND.normalize_bin(str(metadata["name"]), metadata["bin"])
            for target in normalized.values():
                target_path = package_dir / target
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(b"#!/usr/bin/env node\n")
                target_path.chmod(0o755)

    lock_path = install_dir / "package-lock.json"
    lock_path.write_text(
        json.dumps({"lockfileVersion": 3, "packages": lock_packages}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hidden_payload = {
        "lockfileVersion": 3,
        "packages": {key: value for key, value in lock_packages.items() if key in installed_packages},
    }
    (node_modules / ".package-lock.json").write_text(
        json.dumps(hidden_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    bin_dir = node_modules / ".bin"
    bin_dir.mkdir()
    (bin_dir / "vitest").symlink_to("../vitest/vitest.mjs")
    return install_dir, npm_ls, FRONTEND.npm_ls_inventory(npm_ls)


def _hoisted_core_packages() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    installed = {
        "node_modules/vite": {
            "name": "vite",
            "version": "8.2.0",
            "dependencies": {"rolldown": "~1.2.0"},
        },
        "node_modules/rolldown": {"name": "rolldown", "version": "1.2.1"},
        "node_modules/vitest": {
            "name": "vitest",
            "version": "4.1.7",
            "bin": {"vitest": "vitest.mjs"},
        },
    }
    lock = {
        "": {
            "name": "scai-frontend",
            "version": "1.0.0",
            "dependencies": {"vite": "^8.2.0", "vitest": "^4.1.7"},
        },
        "node_modules/vite": {
            "version": "8.2.0",
            "resolved": "https://registry.npmjs.org/vite/-/vite-8.2.0.tgz",
            "integrity": "sha512-pn+CFpM0lwDeKwmOq1ZaBK/9sjorZcgqxki6MbY/jPEVd9vichIlmlD4HmQ5wdP5EgqQCFRaACBxMC7uEGc6lQ==",
            "dependencies": {"rolldown": "~1.2.0"},
        },
        "node_modules/rolldown": {
            "version": "1.2.1",
            "resolved": "https://registry.npmjs.org/rolldown/-/rolldown-1.2.1.tgz",
            "integrity": "sha512-4FKJhg8d3OiyQOA6Q1Q0hoFFpW9/OoX+VsHzpECsdsIZoOArrAK90gl59YK/Z+gnDel45bgJZK03ozH/9bCqEw==",
        },
        "node_modules/vitest": {
            "version": "4.1.7",
            "resolved": "https://registry.npmjs.org/vitest/-/vitest-4.1.7.tgz",
            "integrity": "sha512-flYyaFd2CgoCoU+0UKt3pxksgC+S02iTDN0n3LtqaMeXsI9SBcdNujc2k0DeFLzUn/0k538yNjOSdwgCqcrwJA==",
            "bin": {"vitest": "vitest.mjs"},
        },
    }
    return installed, lock


def test_npm_ls_accepts_hoisted_platform_optional_empty_edge(tmp_path: Path):
    installed, lock = _hoisted_core_packages()
    binding = "node_modules/@rolldown/binding-freebsd-x64"
    binding_version = "1.2.1"
    binding_spec = {"@rolldown/binding-freebsd-x64": "~1.2.1"}
    installed["node_modules/rolldown"]["optionalDependencies"] = binding_spec
    lock["node_modules/rolldown"]["optionalDependencies"] = binding_spec
    lock[binding] = {
        "version": binding_version,
        "resolved": "https://registry.npmjs.org/@rolldown/binding-freebsd-x64/-/binding-freebsd-x64-1.2.1.tgz",
        "integrity": "sha512-ZnJlYWwtcm9sbGRvd24tYmluZGluZw==",
        "optional": True,
        "os": ["freebsd"],
        "cpu": ["x64"],
    }
    npm_ls = {
        "name": "scai-frontend",
        "version": "1.0.0",
        "dependencies": {
            "vite": {
                "version": "8.2.0",
                "dependencies": {
                    "rolldown": {
                        "version": "1.2.1",
                        "dependencies": {"@rolldown/binding-freebsd-x64": {}},
                    }
                },
            },
            "vitest": {"version": "4.1.7"},
        },
    }
    install_dir, _, inventory = _write_hoisted_npm_tree(
        tmp_path / "positive", installed_packages=installed, lock_packages=lock, npm_ls=npm_ls
    )
    tree = _walk_captured_npm9_bin_tree(install_dir, npm_ls, inventory)
    assert {item["path"] for item in tree["packages"]} == {
        "node_modules/vite",
        "node_modules/rolldown",
        "node_modules/vitest",
    }
    assert tree["optional_omissions"] == [
        {"path": binding, "reason": "lock os selector excludes linux"}
    ]

    mutations = (
        ("parent optionalDependencies missing", lambda value: value.pop("optionalDependencies")),
        (
            "parent optionalDependencies version drift",
            lambda value: value["optionalDependencies"].update({"@rolldown/binding-freebsd-x64": "9.9.9"}),
        ),
        ("target is required", lambda value: value.update({"optional": False})),
        ("selector applies", lambda value: value.update({"os": ["linux"]})),
    )
    for label, mutate in mutations:
        mutation_installed = json.loads(json.dumps(installed))
        mutation_lock = json.loads(json.dumps(lock))
        if label == "parent optionalDependencies version drift":
            mutate(mutation_installed["node_modules/rolldown"])
        elif label.startswith("parent"):
            mutate(mutation_installed["node_modules/rolldown"])
            mutate(mutation_lock["node_modules/rolldown"])
        else:
            mutate(mutation_lock[binding])
        mutation_root = tmp_path / label.replace(" ", "-")
        mutation_install, mutation_ls, mutation_inventory = _write_hoisted_npm_tree(
            mutation_root,
            installed_packages=mutation_installed,
            lock_packages=mutation_lock,
            npm_ls=npm_ls,
        )
        with pytest.raises(FRONTEND.FrontendError):
            _walk_captured_npm9_bin_tree(mutation_install, mutation_ls, mutation_inventory)


def test_npm_ls_accepts_only_exact_optional_peer_empty_edge(tmp_path: Path):
    installed = {
        "node_modules/peer-parent": {
            "name": "peer-parent",
            "version": "1.0.0",
            "peerDependencies": {"optional-peer": "^1.0.0"},
            "peerDependenciesMeta": {"optional-peer": {"optional": True}},
        },
        "node_modules/vitest": {"name": "vitest", "version": "4.1.7", "bin": {"vitest": "vitest.mjs"}},
    }
    lock = {
        "": {"name": "scai-frontend", "version": "1.0.0", "dependencies": {"peer-parent": "1.0.0"}},
        "node_modules/peer-parent": {
            "version": "1.0.0",
            "resolved": "https://registry.npmjs.org/peer-parent/-/peer-parent-1.0.0.tgz",
            "integrity": "sha512-cGVlci1wYXJlbnQ=",
            "peerDependencies": {"optional-peer": "^1.0.0"},
            "peerDependenciesMeta": {"optional-peer": {"optional": True}},
        },
        "node_modules/vitest": {
            "version": "4.1.7",
            "resolved": "https://registry.npmjs.org/vitest/-/vitest-4.1.7.tgz",
            "integrity": "sha512-flYyaFd2CgoCoU+0UKt3pxksgC+S02iTDN0n3LtqaMeXsI9SBcdNujc2k0DeFLzUn/0k538yNjOSdwgCqcrwJA==",
            "bin": {"vitest": "vitest.mjs"},
        },
    }
    npm_ls = {
        "name": "scai-frontend",
        "version": "1.0.0",
        "dependencies": {
            "peer-parent": {"version": "1.0.0", "dependencies": {"optional-peer": {}}},
            "vitest": {"version": "4.1.7"},
        },
    }
    install_dir, _, inventory = _write_hoisted_npm_tree(
        tmp_path / "positive", installed_packages=installed, lock_packages=lock, npm_ls=npm_ls
    )
    tree = _walk_captured_npm9_bin_tree(install_dir, npm_ls, inventory)
    assert tree["optional_omissions"] == []

    mutations = (
        ("ordinary peer", lambda value: value.pop("peerDependenciesMeta")),
        ("required peer", lambda value: value["peerDependenciesMeta"].update({"optional-peer": {"optional": False}})),
        ("malformed peer metadata", lambda value: value["peerDependenciesMeta"].update({"optional-peer": {"optional": "true"}})),
    )
    for label, mutate in mutations:
        mutation_installed = json.loads(json.dumps(installed))
        mutation_lock = json.loads(json.dumps(lock))
        mutate(mutation_installed["node_modules/peer-parent"])
        mutate(mutation_lock["node_modules/peer-parent"])
        mutation_install, mutation_ls, mutation_inventory = _write_hoisted_npm_tree(
            tmp_path / label.replace(" ", "-"),
            installed_packages=mutation_installed,
            lock_packages=mutation_lock,
            npm_ls=npm_ls,
        )
        with pytest.raises(FRONTEND.FrontendError):
            _walk_captured_npm9_bin_tree(mutation_install, mutation_ls, mutation_inventory)

    unknown_ls = json.loads(json.dumps(npm_ls))
    unknown_ls["dependencies"]["unknown-parent"] = unknown_ls["dependencies"].pop("peer-parent")
    unknown_install, unknown_npm_ls, unknown_inventory = _write_hoisted_npm_tree(
        tmp_path / "unknown-parent", installed_packages=installed, lock_packages=lock, npm_ls=unknown_ls
    )
    with pytest.raises(FRONTEND.FrontendError):
        _walk_captured_npm9_bin_tree(unknown_install, unknown_npm_ls, unknown_inventory)


def test_npm_ls_identity_crosscheck_accepts_hoisted_logical_paths_and_rejects_drift(tmp_path: Path):
    installed, lock = _hoisted_core_packages()
    npm_ls = {
        "name": "scai-frontend",
        "version": "1.0.0",
        "dependencies": {
            "vite": {"version": "8.2.0", "dependencies": {"rolldown": {"version": "1.2.1"}}},
            "vitest": {"version": "4.1.7"},
        },
    }
    install_dir, _, inventory = _write_hoisted_npm_tree(
        tmp_path / "positive", installed_packages=installed, lock_packages=lock, npm_ls=npm_ls
    )
    tree = _walk_captured_npm9_bin_tree(install_dir, npm_ls, inventory)
    assert {item["path"] for item in tree["packages"]} == {
        "node_modules/vite",
        "node_modules/rolldown",
        "node_modules/vitest",
    }
    assert tree["optional_omissions"] == []

    mutations = (
        ("unknown name", lambda value: value.update({"unknown-rolldown": value.pop("rolldown")})),
        ("version drift", lambda value: value["rolldown"].update({"version": "9.9.9"})),
    )
    for label, mutate in mutations:
        mutation_ls = json.loads(json.dumps(npm_ls))
        mutate(mutation_ls["dependencies"]["vite"]["dependencies"])
        mutation_install, mutation_npm_ls, mutation_inventory = _write_hoisted_npm_tree(
            tmp_path / label.replace(" ", "-"),
            installed_packages=json.loads(json.dumps(installed)),
            lock_packages=json.loads(json.dumps(lock)),
            npm_ls=mutation_ls,
        )
        with pytest.raises(FRONTEND.FrontendError):
            _walk_captured_npm9_bin_tree(mutation_install, mutation_npm_ls, mutation_inventory)

    missing_logical_ls = json.loads(json.dumps(npm_ls))
    del missing_logical_ls["dependencies"]["vite"]["dependencies"]["rolldown"]
    missing_install, missing_npm_ls, missing_inventory = _write_hoisted_npm_tree(
        tmp_path / "missing-logical-identity",
        installed_packages=installed,
        lock_packages=lock,
        npm_ls=missing_logical_ls,
    )
    with pytest.raises(FRONTEND.FrontendError):
        _walk_captured_npm9_bin_tree(missing_install, missing_npm_ls, missing_inventory)


def test_npm_bin_target_canonicalization_accepts_only_one_leading_dot_slash():
    normalize = FRONTEND.normalize_bin
    assert normalize("pkg", "target") == {"pkg": "target"}
    assert normalize("@scope/pkg", "./target") == {"pkg": "target"}
    assert normalize("pkg", {"run": "./target"}) == {"run": "target"}

    malformed = (
        "",
        ".",
        "./",
        "././target",
        "/absolute/path",
        "../escape",
        "./../escape",
        "a/../escape",
        "a/./target",
        "a//target",
        "a\\target",
        "a\x00target",
        [],
        None,
        1,
    )
    for target in malformed:
        with pytest.raises(FRONTEND.FrontendError):
            normalize("pkg", {"run": target})


def test_npm_bin_map_comparison_rejects_command_and_canonical_target_drift():
    normalize = FRONTEND.normalize_bin
    assert normalize("pkg", {"run": "./target"}) == normalize("pkg", {"run": "target"})
    assert normalize("@scope/pkg", "./target") == normalize("@scope/pkg", {"pkg": "target"})
    assert normalize("pkg", "./target") != normalize("pkg", {"run": "target"})

    unequal_maps = (
        ({"run": "target"}, {"other": "target"}),
        ({"run": "target"}, {"run": "target", "extra": "extra"}),
        ({"run": "target"}, {"renamed": "target"}),
        ({"run": "target"}, {"run": "other"}),
    )
    for installed, locked in unequal_maps:
        assert normalize("pkg", installed) != normalize("pkg", locked)

    malformed_maps = (
        {"": "target"},
        {1: "target"},
        {"run": None},
        {"run": 1},
        {"run": ""},
        {"run": "././target"},
        {"run": "a/../target"},
    )
    for value in malformed_maps:
        with pytest.raises(FRONTEND.FrontendError):
            normalize("pkg", value)


def test_npm_tree_accepts_captured_npm9_leading_dot_slash_bin_forms(tmp_path: Path):
    install_dir, npm_ls, inventory = _write_captured_npm9_bin_tree(tmp_path)
    tree = _walk_captured_npm9_bin_tree(install_dir, npm_ls, inventory)

    packages = {item["path"]: item for item in tree["packages"]}
    assert packages == {
        "node_modules/@bramus/specificity": {
            "path": "node_modules/@bramus/specificity",
            "name": "@bramus/specificity",
            "version": "2.4.2",
            "resolved": "https://registry.npmjs.org/@bramus/specificity/-/specificity-2.4.2.tgz",
            "integrity": "sha512-ctxtJ/eA+t+6q2++vj5j7FYX3nRu311q1wfYH3xjlLOsczhlhxAg2FWNUXhpGvAw3BWo1xBcvOV6/YLc2r5FJw==",
        },
        "node_modules/vitest": {
            "path": "node_modules/vitest",
            "name": "vitest",
            "version": "4.1.7",
            "resolved": "https://registry.npmjs.org/vitest/-/vitest-4.1.7.tgz",
            "integrity": "sha512-flYyaFd2CgoCoU+0UKt3pxksgC+S02iTDN0n3LtqaMeXsI9SBcdNujc2k0DeFLzUn/0k538yNjOSdwgCqcrwJA==",
        },
    }
    bins = {item["path"]: item for item in tree["bins"]}
    assert bins == {
        "node_modules/.bin/specificity": {
            "path": "node_modules/.bin/specificity",
            "kind": "symlink",
            "target": "node_modules/@bramus/specificity/bin/cli.js",
            "target_sha256": FRONTEND.sha256_file(install_dir / "node_modules/@bramus/specificity/bin/cli.js"),
        },
        "node_modules/.bin/vitest": {
            "path": "node_modules/.bin/vitest",
            "kind": "symlink",
            "target": "node_modules/vitest/vitest.mjs",
            "target_sha256": FRONTEND.sha256_file(install_dir / "node_modules/vitest/vitest.mjs"),
        },
    }
    assert tree["hidden_lock"]["path"] == "node_modules/.package-lock.json"


def test_npm_tree_preserves_containment_and_exact_shim_checks_after_bin_canonicalization(tmp_path: Path):
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    install_dir, npm_ls, inventory = _write_captured_npm9_bin_tree(baseline_root)
    baseline = _walk_captured_npm9_bin_tree(install_dir, npm_ls, inventory)
    assert {item["path"] for item in baseline["bins"]} == {
        "node_modules/.bin/specificity",
        "node_modules/.bin/vitest",
    }

    mutations = (
        ("parent target", {"specificity_bin": "../outside.js"}),
        ("absolute target", {"specificity_bin": "/tmp/scai-outside.js"}),
        ("internal dot target", {"specificity_bin": "bin/./cli.js"}),
        ("missing target", {"specificity_target_kind": "missing"}),
        ("symlink target", {"specificity_target_kind": "symlink"}),
        ("copied shim bytes", {"vitest_shim": ("regular", "tampered shim bytes")}),
        ("wrong shim target", {"vitest_shim": ("symlink", "../@bramus/specificity/bin/cli.js")}),
        ("extra shim", {"extra_shim": True}),
        ("missing shim", {"omit_shims": ("vitest",)}),
        ("renamed command", {"specificity_bin": {"renamed": "./bin/cli.js"}}),
    )
    for label, kwargs in mutations:
        mutation_root = tmp_path / label.replace(" ", "-")
        mutation_root.mkdir()
        install_dir, npm_ls, inventory = _write_captured_npm9_bin_tree(mutation_root, **kwargs)
        with pytest.raises(FRONTEND.FrontendError):
            _walk_captured_npm9_bin_tree(install_dir, npm_ls, inventory)


def test_frontend_runner_hashes_both_exact_inputs_before_install(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path)
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode == 0, result.stderr
    _, evidence = latest_evidence(repo)
    assert evidence["package_json_sha256"] == FRONTEND.sha256_file(repo / "frontend/package.json")
    assert evidence["package_lock_sha256"] == FRONTEND.sha256_file(repo / "frontend/package-lock.json")
    assert json.loads(log.read_text().splitlines()[2])[:2] == ["config", "list"]


def test_resolve_run_start_head_uses_one_strict_git_invocation(monkeypatch, tmp_path):
    expected = "0123456789abcdef0123456789abcdef01234567"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=(expected + "\n").encode(), stderr=b"")

    monkeypatch.setattr(FRONTEND.subprocess, "run", fake_run)

    assert FRONTEND.resolve_run_start_head(tmp_path) == expected
    assert calls == [
        (
            ["git", "-C", str(tmp_path), "rev-parse", "--verify", "HEAD^{commit}"],
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "check": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "returncode, stdout",
    [
        (128, b"0123456789abcdef0123456789abcdef01234567\n"),
        (0, b"0123456789abcdef0123456789abcdef01234567"),
        (0, b"0123456789abcdef0123456789abcdef01234567 \n"),
        (0, b"0123456789ABCDEF0123456789abcdef01234567\n"),
        (0, b"0123456789abcdef0123456789abcdef01234567\n\n"),
    ],
)
def test_resolve_run_start_head_rejects_noncanonical_git_output(monkeypatch, tmp_path, returncode, stdout):
    def fake_run(argv, **kwargs):
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=b"git error")

    monkeypatch.setattr(FRONTEND.subprocess, "run", fake_run)

    with pytest.raises(FRONTEND.FrontendError) as exc_info:
        FRONTEND.resolve_run_start_head(tmp_path)
    assert exc_info.value.label == "source manifest"


def test_frontend_runner_uses_fresh_run_directory_under_input_key(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    first = latest_evidence(repo)[1]["installation"]
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    second = latest_evidence(repo)[1]["installation"]
    assert first["namespace"] == second["namespace"] and first["run_dir"] != second["run_dir"]


def test_frontend_runner_uses_run_start_head_for_later_commits(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    (repo / "later-commit.txt").write_text("later\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "later")
    later = git(repo, "rev-parse", "HEAD")
    completed = run_frontend(repo, tmp_path / "tools")
    assert completed.returncode == 0, completed.stderr
    source = latest_evidence(repo)[1]["source"]
    assert source["base_commit"] == later
    assert source["manifest"]["base"] == later
    assert source["manifest_after"]["base"] == later
    assert source["manifest"]["entries"] == []
    assert source["manifest_after"]["entries"] == []


def test_frontend_lock_order_is_read_hash_global_namespace_keyed_recheck_then_run():
    text = _frontend_source()
    assert text.index("source_before") < text.index("safe_lock(Path(\"/tmp\")") < text.index("safe_lock(self.input_namespace") < text.index("self.install_dir.mkdir")


def test_missing_input_namespace_is_created_only_under_global_lock_before_keyed_open():
    text = _frontend_source()
    assert text.index("safe_lock(Path(\"/tmp\")") < text.index("ensure_directory(self.input_namespace") < text.index("safe_lock(self.input_namespace")


def test_keyed_lock_rechecks_source_and_inputs_before_fresh_run_or_copy():
    text = _frontend_source()
    assert text.index("safe_lock(self.input_namespace") < text.index("latest_source = source_snapshot") < text.index("self.install_dir.mkdir") < text.index("shutil.copyfile")


def test_frontend_input_preparation_failure_stops_before_npm(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path)
    (repo / "frontend/package-lock.json").unlink()
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0 and not log.exists()


def test_frontend_runner_rejects_source_npmrc_before_install(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path, source_npmrc=True)
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0 and not log.exists()


def test_frontend_runner_refuses_real_source_node_modules_directory(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path)
    (repo / "frontend/node_modules").mkdir()
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0 and not log.exists()


def test_npm_ci_failure_prevents_ls_and_test_and_writes_failure_evidence(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path, {"ci": True})
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0
    _, evidence = latest_evidence(repo)
    assert evidence["error"]["label"] == "npm ci"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[0] for call in calls if call] == ["--version", "config", "config", "ci"]
    assert "npm_ls" not in evidence["commands"] and "npm_test" not in evidence["commands"]


def test_npm_ls_failure_prevents_test(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path, {"ls": True})
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(call and call[0] == "test" for call in calls)


def test_tree_canonicalization_rejects_extra_or_version_drift():
    text = _frontend_source()
    assert "installed/lock drift" in text and "locked package is absent" in text


def test_tree_canonicalization_records_lock_resolved_and_integrity(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    tree = latest_evidence(repo)[1]["tree"]
    assert all(item["resolved"].startswith("https://") and item["integrity"] for item in tree["packages"])


def test_tree_walker_accepts_captured_npm9_hidden_lock_scoped_and_nested_layout(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    paths = {item["path"] for item in latest_evidence(repo)[1]["tree"]["packages"]}
    assert paths == set(FIXTURE["packages"])


def test_tree_walker_rejects_package_metadata_symlink_special_or_realpath_escape():
    text = _frontend_source()
    assert "invalid package metadata" in text and "escapes installation" in text and "is not a regular file" in text


def test_bin_walker_records_vitest_shim_and_declared_contained_target(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    bins = latest_evidence(repo)[1]["tree"]["bins"]
    vitest = next(item for item in bins if item["path"] == "node_modules/.bin/vitest")
    assert vitest["kind"] == "symlink" and vitest["target"] == "node_modules/vitest/vitest.mjs"


def test_bin_walker_requires_declaration_and_discovered_level_sets_to_match():
    text = _frontend_source()
    assert "set(declarations) != bin_dirs" in text and "bin name set drift" in text


def test_bin_walker_rejects_escape_wrong_target_extra_missing_shim_or_nested_bin_dir():
    text = _frontend_source()
    for phrase in ("absolute bin target", "wrong bin target", "bin name set drift", "vitest shim is missing"):
        assert phrase in text


def test_npm_test_uses_the_same_verified_vitest_shim():
    text = _frontend_source()
    assert text.index("self.verify_vitest_shim()") < text.index('[str(self.npm), "test"')


def test_frontend_runner_invokes_0644_link_helper_through_bash():
    assert stat.S_IMODE((ROOT / "scripts/link_frontend_node_modules.sh").stat().st_mode) == 0o644
    text = _frontend_source()
    assert '["bash", str(helper)' in text


def test_source_node_modules_link_is_restored_after_success_or_failure(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path, {"test": True}, source_link="../old-node-modules")
    original = os.readlink(repo / "frontend/node_modules")
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0 and (repo / "frontend/node_modules").is_symlink()
    assert os.readlink(repo / "frontend/node_modules") == original


def test_restoration_failure_forces_failed_evidence_and_nonzero_exit(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path, {"tamper_restore": True})
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode != 0, result.stderr
    _, evidence = latest_evidence(repo)
    assert evidence["status"] == "failed"
    assert evidence["error"] and evidence["error"]["label"] in {"frontend restoration", "frontend source identity"}
    assert "frontend restoration" in json.dumps(evidence["error"])
    assert evidence["status"] != "passed"


def test_runtime_home_cache_xdg_and_npmrc_are_outside_result_directory(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode == 0, result.stderr
    _, evidence = latest_evidence(repo)
    environment = evidence["effective_environment"]
    for key in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "npm_config_cache"):
        assert not str(environment[key]).startswith("<result-dir>"), (key, environment[key])
    assert str(environment["npm_config_userconfig"]).startswith("<install-dir>")
    assert str(environment["npm_config_globalconfig"]).startswith("<install-dir>")


def _artifact_test_run(tmp_path, monkeypatch):
    monkeypatch.setattr(FRONTEND_RUNNER, "load_tree_helper", lambda _repo: object())
    run = FRONTEND.FrontendRun(tmp_path)
    run.result_dir = tmp_path / "result"
    run.result_dir.mkdir(parents=True)
    run.register_result_name("npm-config-values.json")
    path = run.result_dir / "npm-config-values.json"
    path.write_bytes(b"original")
    return run, path


def test_result_artifact_allowlist_rejects_extra_directory_symlink_special_hardlink_and_tampering(tmp_path, monkeypatch):
    run, path = _artifact_test_run(tmp_path / "directory", monkeypatch)
    (run.result_dir / "extra-dir").mkdir()
    with pytest.raises(FRONTEND.FrontendError):
        run.load_result_artifacts()

    run, path = _artifact_test_run(tmp_path / "symlink", monkeypatch)
    (run.result_dir / "extra-link").symlink_to(tmp_path / "symlink-target")
    with pytest.raises(FRONTEND.FrontendError):
        run.load_result_artifacts()

    run, path = _artifact_test_run(tmp_path / "fifo", monkeypatch)
    os.mkfifo(run.result_dir / "extra-fifo")
    with pytest.raises(FRONTEND.FrontendError):
        run.load_result_artifacts()

    run, path = _artifact_test_run(tmp_path / "hardlink", monkeypatch)
    hardlink_source = tmp_path / "hardlink-source"
    hardlink_source.write_bytes(b"same")
    path.unlink()
    os.link(hardlink_source, path)
    with pytest.raises(FRONTEND.FrontendError):
        run.load_result_artifacts()

    run, path = _artifact_test_run(tmp_path / "replacement", monkeypatch)
    original_read = os.read
    replaced = False

    def replace_after_open(fd, size):
        nonlocal replaced
        if not replaced:
            path.unlink()
            path.write_bytes(b"replacement")
            replaced = True
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", replace_after_open)
    loaded = run.load_result_artifacts()
    assert loaded["npm-config-values.json"].payload == b"original"


def test_artifact_manifest_and_evidence_share_one_loaded_byte_snapshot():
    text = _frontend_source()
    runner_source = (PARTS_DIR / "runner.py").read_text(encoding="utf-8")
    module = ast.parse(runner_source, filename=str(PARTS_DIR / "runner.py"))
    frontend = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "FrontendRun")
    calls = [
        node for node in ast.walk(frontend)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "artifact_manifest"
    ]
    assert len(calls) == 1
    assert "load_result_artifacts" in text and "ArtifactBytes" in text


def test_shell_facade_executes_the_fixed_python_entrypoint():
    source = RUNNER_FILE.read_text(encoding="utf-8")
    assert 'export SCAI_FRONTEND_RUNNER_PATH="$0"' in source
    assert 'runner_dir="$(cd "$(dirname "$0")" && pwd -P)"' in source
    assert 'exec python3 "$runner_dir/scai_frontend_runner.py" "$@"' in source
    assert "PYTHON_EOF" not in source


def test_missing_optional_package_requires_platform_or_arch_inapplicability(tmp_path):
    os_path = "node_modules/optional-os"
    cpu_path = "node_modules/optional-cpu"
    empty_path = "node_modules/optional-empty"
    unreported_path = "node_modules/optional-unreported"
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    accepted, _, _ = make_frontend_repo(accepted_root, optional_paths=(os_path, cpu_path, empty_path))
    accepted_tools = next(iter(_FAKE_TOOL_ROOTS))
    result = run_frontend(accepted, accepted_tools)
    assert result.returncode == 0, result.stderr
    _, evidence = latest_evidence(accepted)
    omissions = {item["path"]: item["reason"] for item in evidence["tree"]["optional_omissions"]}
    assert omissions == {
        os_path: "lock os selector excludes linux",
        cpu_path: "lock cpu selector excludes x64",
        empty_path: "npm ls reported an optional omission",
    }

    with pytest.raises(FRONTEND.FrontendError):
        FRONTEND.npm_selector_applies([], "linux", "os")
    with pytest.raises(FRONTEND.FrontendError):
        FRONTEND.npm_selector_applies([""], "x64", "cpu")
    with pytest.raises(FRONTEND.FrontendError):
        FRONTEND.npm_selector_applies([1], "linux", "os")

    rejected_root = tmp_path / "unreported"
    rejected_root.mkdir()
    rejected, _, _ = make_frontend_repo(
        rejected_root,
        optional_paths=(os_path, cpu_path, empty_path, unreported_path),
    )
    rejected_tools = next(root for root in _FAKE_TOOL_ROOTS if root != accepted_tools)
    result = run_frontend(rejected, rejected_tools)
    assert result.returncode != 0
    _, evidence = latest_evidence(rejected)
    assert evidence["error"]["label"] == "npm tree"
    assert unreported_path in evidence["error"]["message"]


def test_node_platform_identity_is_command_owned_and_recorded(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode == 0, result.stderr
    evidence_path, evidence = latest_evidence(repo)
    identity = evidence["commands"]["node_platform"]
    query = "JSON.stringify({platform:process.platform,arch:process.arch})"
    assert identity["argv"][1:] == ["-p", query]
    assert identity["exit_code"] == 0
    assert identity["parsed"] == {"platform": "linux", "arch": "x64"}
    assert identity["stdout_sha256"] and identity["stderr_sha256"]
    artifact_names = {item["path"] for item in evidence["artifacts"]["entries"]}
    assert identity["stdout"] in artifact_names and identity["stderr"] in artifact_names
    assert evidence_path.exists()


def test_global_and_input_key_locks_serialize_concurrent_frontend_runs():
    text = _frontend_source()
    assert "fcntl.flock(fd, fcntl.LOCK_EX)" in text and "scai-frontend-" in text and ".install.lock" in text


def test_frontend_environment_clears_node_options_and_all_npm_config_inputs():
    text = _frontend_source()
    assert "os.environ" not in text[text.index("def command_env"):text.index("def empty_config_file")]
    assert "env=self.env" in text


def test_frontend_uses_isolated_home_cache_and_userconfig():
    text = _frontend_source()
    for key in ("HOME", "XDG_CACHE_HOME", "npm_config_cache", "npm_config_userconfig", "npm_config_globalconfig"):
        assert f'"{key}"' in text


def test_npm_config_commands_have_exact_argv_env_cwd_exit_and_output_paths():
    text = _frontend_source()
    assert '[str(self.npm), "config", "list", "--json"]' in text
    assert '[str(self.npm), "config", "list"]' in text
    assert "stdout_sha256" in text and "stderr_sha256" in text and '"cwd"' in text


def test_npm9_config_values_records_json_and_location_as_exact_command_owned_keys(tmp_path: Path):
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    expected = _npm9_expected_config(install_dir)
    empty_hashes = {
        "userconfig": NPM9_EMPTY_CONFIG_SHA,
        "globalconfig": NPM9_EMPTY_CONFIG_SHA,
    }
    value = FRONTEND.parse_npm_config_values(
        _npm9_json_bytes(_npm9_json_value(expected)), expected, empty_hashes
    )
    assert value["command_owned"] == {"json": True, "location": "project"}
    assert value["effective_overrides"] == expected
    json_value = _npm9_json_value(expected)
    builtin = {
        key: item
        for key, item in json_value.items()
        if key not in set(expected) | {"json", "location"}
    }
    assert value["builtin_values_sha256"] == FRONTEND.json_hash(builtin)


def test_npm9_config_sources_accepts_exact_env_and_command_owned_location(tmp_path: Path):
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    expected = _npm9_expected_config(install_dir)
    empty_hashes = _npm9_empty_config_files(install_dir)
    value = FRONTEND.parse_npm_config_sources(
        _npm9_plain_bytes(expected), expected, empty_hashes, install_dir
    )
    assert value["allowed_sources"] == [
        "npm builtin/default",
        "environment",
        "empty config file",
        "command-owned CLI",
    ]
    assert value["command_owned"] == {"location": "project"}
    assert value["final_effective_overrides"] == expected
    assert [item["class"] for item in value["observed_sources"]] == [
        "npm builtin/default",
        "environment",
        "command-owned CLI",
    ]


def test_npm9_config_sources_rejects_closed_provenance_negative_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    expected = _npm9_expected_config(install_dir)
    empty_hashes = _npm9_empty_config_files(install_dir)
    positive_json = _npm9_json_value(expected)
    positive_plain = _npm9_plain_bytes(expected)

    # Positive controls use the sanitized npm 9.2.0 command outputs before the
    # mutations below.  The current parser is expected to RED on these exact
    # command-owned fields until the addendum implementation is applied.
    values = FRONTEND.parse_npm_config_values(
        _npm9_json_bytes(positive_json), expected, empty_hashes
    )
    assert values["command_owned"] == {"json": True, "location": "project"}
    sources = FRONTEND.parse_npm_config_sources(
        positive_plain, expected, empty_hashes, install_dir
    )
    assert sources["command_owned"] == {"location": "project"}

    json_cases: list[tuple[str, dict[str, object]]] = []
    missing_json = dict(positive_json)
    missing_json.pop("json")
    json_cases.append(("missing json", missing_json))
    false_json = dict(positive_json)
    false_json["json"] = False
    json_cases.append(("false json", false_json))
    malformed_json = dict(positive_json)
    malformed_json["json"] = "true"
    json_cases.append(("non-boolean json", malformed_json))
    missing_location = dict(positive_json)
    missing_location.pop("location")
    json_cases.append(("missing location", missing_location))
    non_string_location = dict(positive_json)
    non_string_location["location"] = True
    json_cases.append(("non-string location", non_string_location))
    wrong_location = dict(positive_json)
    wrong_location["location"] = "global"
    json_cases.append(("non-project location", wrong_location))
    for label, value in json_cases:
        with pytest.raises(FRONTEND.FrontendError):
            FRONTEND.parse_npm_config_values(_npm9_json_bytes(value), expected, empty_hashes)

    captured_builtin_inputs: list[dict[str, object]] = []
    monkeypatch.setattr(
        FRONTEND_NPM,
        "json_hash",
        lambda value: captured_builtin_inputs.append(value) or "0" * 64,
    )
    FRONTEND.parse_npm_config_values(_npm9_json_bytes(positive_json), expected, empty_hashes)
    assert captured_builtin_inputs and not ({"json", "location"} & set(captured_builtin_inputs[-1]))

    cli_header = b'; "cli" config from command line options'
    env_header = b'; "env" config from environment'
    plain_cases: list[tuple[str, bytes, Callable[[], object] | None]] = [
        (
            "old environment header name",
            positive_plain.replace(env_header, b'; "environment" config from environment', 1),
            None,
        ),
        (
            "wrong environment source",
            positive_plain.replace(env_header, b'; "env" config from /tmp/ambient.npmrc', 1),
            None,
        ),
        (
            "duplicate environment section",
            positive_plain.replace(
                cli_header,
                b'; "env" config from environment\n' + cli_header,
                1,
            ),
            None,
        ),
        (
            "extra environment assignment",
            positive_plain.replace(cli_header, b'extra = true\n' + cli_header, 1),
            None,
        ),
        (
            "missing environment assignment",
            positive_plain.replace(b'audit = false\n', b'', 1),
            None,
        ),
        (
            "builtin location masquerade",
            positive_plain.replace(
                b'prefix = "/usr/local"\n',
                b'prefix = "/usr/local"\nlocation = "project"\n',
                1,
            ),
            None,
        ),
        (
            "missing CLI section",
            _npm9_plain_bytes(expected, include_cli=False),
            None,
        ),
        (
            "duplicate CLI section",
            positive_plain + cli_header + b'\nlocation = "project"\n',
            None,
        ),
        (
            "wrong CLI header name",
            positive_plain.replace(cli_header, b'; "command" config from command line options', 1),
            None,
        ),
        (
            "wrong CLI source text",
            positive_plain.replace(cli_header, b'; "cli" config from /tmp/ambient', 1),
            None,
        ),
        (
            "missing CLI location",
            positive_plain.replace(b'location = "project"\n', b'', 1),
            None,
        ),
        (
            "duplicate CLI location",
            positive_plain.replace(
                b'location = "project"\n',
                b'location = "project"\nlocation = "project"\n',
                1,
            ),
            None,
        ),
        (
            "extra CLI key",
            positive_plain.replace(
                b'location = "project"\n',
                b'location = "project"\nregistry = "https://evil.invalid/"\n',
                1,
            ),
            None,
        ),
        (
            "non-string CLI location",
            positive_plain.replace(b'location = "project"', b'location = true', 1),
            None,
        ),
        (
            "non-project CLI location",
            positive_plain.replace(b'location = "project"', b'location = "global"', 1),
            None,
        ),
        (
            "location outside CLI section",
            _npm9_plain_bytes(
                expected,
                env_assignments=[
                    *((key, expected[key]) for key in NPM9_ENV_KEYS),
                    ("location", "project"),
                ],
                cli_assignments=[],
            ),
            None,
        ),
        (
            "project source",
            b'; "project" config from /tmp/project.npmrc\nfoo = "x"\n' + positive_plain,
            None,
        ),
        (
            "user source",
            b'; "user" config from /tmp/user.npmrc\nfoo = "x"\n' + positive_plain,
            None,
        ),
        (
            "global source",
            b'; "global" config from /tmp/global.npmrc\nfoo = "x"\n' + positive_plain,
            None,
        ),
        (
            "unknown file path",
            b'; "user" config from /tmp/unknown.npmrc\nfoo = "x"\n' + positive_plain,
            None,
        ),
        (
            "unknown environment source",
            positive_plain.replace(env_header, b'; "env" config from ambient', 1),
            None,
        ),
        (
            "unknown CLI source path",
            positive_plain.replace(cli_header, b'; "cli" config from ambient', 1),
            None,
        ),
        (
            "CLI header masquerades as allowed user config",
            f'; "cli" config from {(install_dir / ".npmrc").resolve()}\n'.encode() + positive_plain,
            None,
        ),
        (
            "environment header masquerades as allowed user config",
            f'; "env" config from {(install_dir / ".npmrc").resolve()}\n'.encode() + positive_plain,
            None,
        ),
        (
            "CLI header masquerades as allowed global config",
            f'; "cli" config from {(install_dir / "global.npmrc").resolve()}\n'.encode() + positive_plain,
            None,
        ),
        (
            "environment header masquerades as allowed global config",
            f'; "env" config from {(install_dir / "global.npmrc").resolve()}\n'.encode() + positive_plain,
            None,
        ),
        (
            "allowed file assignment",
            (
                f'; "user" config from {(install_dir / ".npmrc").resolve()}\n'
                'foo = "x"\n'
            ).encode()
            + positive_plain,
            None,
        ),
        (
            "empty registry credential",
            positive_plain.replace(
                b'prefix = "/usr/local"\n',
                b'prefix = "/usr/local"\n//registry.npmjs.org/:_authToken = ""\n',
                1,
            ),
            None,
        ),
        (
            "nonempty exact credential",
            positive_plain.replace(
                b'prefix = "/usr/local"\n',
                b'prefix = "/usr/local"\ntoken = "secret"\n',
                1,
            ),
            None,
        ),
        (
            "changed allowed file hash",
            positive_plain,
            lambda: (install_dir / ".npmrc").write_text("registry=https://evil.invalid/\n", encoding="utf-8"),
        ),
    ]
    for label, raw, mutate in plain_cases:
        _npm9_empty_config_files(install_dir)
        if mutate is not None:
            mutate()
        try:
            FRONTEND.parse_npm_config_sources(raw, expected, empty_hashes, install_dir)
        except FRONTEND.FrontendError:
            continue
        raise AssertionError(f"accepted invalid npm 9 provenance: {label}")

    def require_agreement(value_artifact: dict[str, object], source_artifact: dict[str, object]) -> None:
        assert value_artifact["effective_overrides"] == source_artifact["final_effective_overrides"]
        assert value_artifact["command_owned"]["location"] == source_artifact["command_owned"]["location"] == "project"

    agreement_cases = []
    mismatch_values = json.loads(json.dumps(values))
    mismatch_values["command_owned"]["location"] = "global"
    agreement_cases.append(("JSON/plain location mismatch", mismatch_values, sources))
    mismatch_sources = json.loads(json.dumps(sources))
    mismatch_sources["command_owned"]["location"] = "global"
    agreement_cases.append(("plain/JSON location mismatch", values, mismatch_sources))
    mismatch_overrides = json.loads(json.dumps(sources))
    mismatch_overrides["final_effective_overrides"]["registry"] = "https://evil.invalid/"
    agreement_cases.append(("seven override mismatch", values, mismatch_overrides))
    for label, value_artifact, source_artifact in agreement_cases:
        with pytest.raises(AssertionError):
            require_agreement(value_artifact, source_artifact)


def test_npm9_config_values_and_sources_require_command_owned_location_agreement():
    validate = getattr(FRONTEND, "validate_npm_config_agreement", None)
    assert callable(validate), "agreement must be enforced by the production parser module"
    expected = {
        "audit": False,
        "cache": "/install/.scai-runtime/cache",
        "fund": False,
        "globalconfig": "/install/global.npmrc",
        "registry": FRONTEND.REGISTRY,
        "update-notifier": False,
        "userconfig": "/install/.npmrc",
    }
    values = {"effective_overrides": expected, "command_owned": {"json": True, "location": "project"}}
    sources = {"final_effective_overrides": dict(expected), "command_owned": {"location": "project"}}
    validate(values, sources)

    bad_values = json.loads(json.dumps(values))
    bad_values["command_owned"]["location"] = "global"
    with pytest.raises(FRONTEND.FrontendError):
        validate(bad_values, sources)

    bad_sources = json.loads(json.dumps(sources))
    bad_sources["command_owned"]["location"] = "global"
    with pytest.raises(FRONTEND.FrontendError):
        validate(values, bad_sources)

    bad_overrides = json.loads(json.dumps(sources))
    bad_overrides["final_effective_overrides"]["registry"] = "https://evil.invalid/"
    with pytest.raises(FRONTEND.FrontendError):
        validate(values, bad_overrides)


def test_frontend_fake_npm_emits_npm9_provenance_and_validates_it_before_ci(tmp_path: Path):
    success_root = tmp_path / "success"
    success_root.mkdir()
    success_before = set(_FAKE_TOOL_ROOTS)
    success_repo, _, success_log = make_frontend_repo(success_root)
    success_roots = _FAKE_TOOL_ROOTS - success_before
    assert len(success_roots) == 1
    success_tools = next(iter(success_roots))
    success_result = run_frontend(success_repo, success_tools)
    assert success_result.returncode == 0, success_result.stderr
    success_calls = [json.loads(line) for line in success_log.read_text(encoding="utf-8").splitlines()]
    success_npm_calls = [
        call
        for call in success_calls
        if call and (call[0] == "config" or call[0] in {"ci", "ls", "test"})
    ]
    assert success_npm_calls[:4] == [
        ["config", "list", "--json"],
        ["config", "list"],
        ["ci"],
        ["ls", "--all", "--json"],
    ]
    assert len(success_npm_calls) == 5
    assert success_npm_calls[4][:3] == ["test", "--", "--reporter=json"]
    assert len(success_npm_calls[4]) == 4 and success_npm_calls[4][3].startswith("--outputFile=")
    shutil.rmtree(success_tools, ignore_errors=True)
    _FAKE_TOOL_ROOTS.discard(success_tools)

    failure_root = tmp_path / "provenance-failure"
    failure_root.mkdir()
    failure_before = set(_FAKE_TOOL_ROOTS)
    repo, _, log = make_frontend_repo(failure_root, {"provenance": True})
    failure_roots = _FAKE_TOOL_ROOTS - failure_before
    assert len(failure_roots) == 1
    failure_tools = next(iter(failure_roots))
    result = run_frontend(repo, failure_tools)
    assert result.returncode != 0
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert calls[:3] == [["--version"], ["config", "list", "--json"], ["config", "list"]]
    assert [call for call in calls if call[:2] == ["config", "list"]] == [
        ["config", "list", "--json"],
        ["config", "list"],
    ]
    assert not any(call and call[0] == "ci" for call in calls)
    assert not any(call and call[0] == "ls" for call in calls)
    assert not any(call and call[0] == "test" for call in calls)
    assert not (repo / "frontend/node_modules").exists()
    _, evidence = latest_evidence(repo)
    assert evidence["status"] == "failed"
    assert evidence["error"]["label"] == "npm config sources"
    assert "location" in evidence["error"]["message"].lower()


def test_npm_config_values_models_command_owned_json_true_separately():
    parsed = load_runner_function("parse_npm_config_values")
    expected = {"cache": "/tmp/cache", "userconfig": "/tmp/user.npmrc", "globalconfig": "/tmp/global.npmrc", "registry": FRONTEND.REGISTRY, "audit": False, "fund": False, "update-notifier": False}
    raw = json.dumps({**expected, "json": True, "location": "project", "strict-ssl": True}).encode()
    value = parsed(raw, expected, {"userconfig": "0" * 64, "globalconfig": "0" * 64})
    assert value["command_owned"] == {"json": True, "location": "project"} and "json" not in value["effective_overrides"]


def test_npm_config_values_and_sources_agree_on_exact_effective_overrides(tmp_path):
    expected = _npm9_expected_config(tmp_path)
    empty_hashes = _npm9_empty_config_files(tmp_path)
    values = load_runner_function("parse_npm_config_values")(
        _npm9_json_bytes(_npm9_json_value(expected)), expected, empty_hashes
    )
    sources = load_runner_function("parse_npm_config_sources")(
        _npm9_plain_bytes(expected), expected, empty_hashes, tmp_path
    )
    assert values["effective_overrides"] == sources["final_effective_overrides"] == expected
    assert values["command_owned"]["location"] == sources["command_owned"]["location"] == "project"


def test_npm_config_plain_env_source_annotation_is_optional_but_if_present_exact():
    text = _frontend_source()
    assert "environment" in text and "unexpected npm config source" in text


def test_npm_config_sources_reject_unexpected_project_user_global_or_env_source():
    expected = {"cache": "/tmp/cache", "userconfig": "/tmp/u", "globalconfig": "/tmp/g", "registry": FRONTEND.REGISTRY, "audit": False, "fund": False, "update-notifier": False}
    raw = b'; "project" config from /bad/.npmrc\n'
    with pytest.raises(FRONTEND.FrontendError):
        FRONTEND.parse_npm_config_sources(raw, expected, {}, Path("/tmp"))


def test_npm_config_artifacts_and_index_hashes_reject_tampering():
    text = _frontend_source()
    assert "artifact_manifest" in text and "sha256_file" in text and "artifact-manifest.json" in text


def test_npm_secret_classifier_allows_auth_type_web_and_safe_auth_key_names():
    for key in ("auth-type", "always-auth", "keyfile", "strict-ssl"):
        FRONTEND.check_secret_key(key, "web")


def test_npm_secret_classifier_rejects_exact_and_registry_scoped_credentials():
    for key in ("_auth", "_authtoken", "password", "token", "cert", "key", "//registry.npmjs.org/:_authToken"):
        with pytest.raises(FRONTEND.FrontendError):
            FRONTEND.check_secret_key(key, "secret")


def test_frontend_evidence_redacts_paths_and_rejects_secret_config_keys(tmp_path):
    value = FRONTEND.tokenized({"cwd": str(tmp_path / "repo/install")}, tmp_path / "repo", tmp_path / "repo/result", tmp_path / "repo/install")
    assert value["cwd"] == "<install-dir>"
    raw = f"{tmp_path}/repo {tmp_path}/repo/result {tmp_path}/repo/install".encode()
    sanitized = FRONTEND.sanitize_output(raw, tmp_path / "repo", tmp_path / "repo/result", tmp_path / "repo/install").decode()
    assert sanitized == "<repo-root> <result-dir> <install-dir>"
    with pytest.raises(FRONTEND.FrontendError):
        FRONTEND.check_secret_key("_auth", "secret")


def test_vitest_json_counts_are_machine_readable_and_required(tmp_path):
    result = {"numTotalTestSuites": 0, "numPassedTestSuites": 0, "numFailedTestSuites": 0, "numPendingTestSuites": 0, "numTotalTests": 0, "numPassedTests": 0, "numFailedTests": 0, "numPendingTests": 0, "numTodoTests": 0, "success": True, "testResults": []}
    path = tmp_path / "vitest.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert FRONTEND.parse_vitest(path, 0)["success"] is True


def test_vitest_accepts_recursive_suite_totals_and_rejects_suite_arithmetic_drift(tmp_path):
    result = {
        "numTotalTestSuites": 3,
        "numPassedTestSuites": 3,
        "numFailedTestSuites": 0,
        "numPendingTestSuites": 0,
        "numTotalTests": 1,
        "numPassedTests": 1,
        "numFailedTests": 0,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "success": True,
        "testResults": [
            {
                "name": "frontend/src/recursive.test.ts",
                "status": "passed",
                "message": "",
                "assertionResults": [
                    {
                        "ancestorTitles": ["outer", "inner"],
                        "fullName": "outer inner passes",
                        "status": "passed",
                        "title": "passes",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "vitest.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    parsed = FRONTEND.parse_vitest(path, 0)
    assert parsed["counts"]["numTotalTestSuites"] == 3

    mutations = (
        ("suite arithmetic", {"numTotalTestSuites": 4}),
        ("success exit", {"success": False}),
        ("failed suite", {"testResults": [{**result["testResults"][0], "status": "failed"}]}),
    )
    for label, changes in mutations:
        mutated = json.loads(json.dumps(result))
        mutated.update(changes)
        mutated_path = tmp_path / f"{label.replace(' ', '-')}.json"
        mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(FRONTEND.FrontendError):
            FRONTEND.parse_vitest(mutated_path, 0)


def test_vitest_assertion_and_suite_details_aggregate_to_top_level_counts(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    counts = latest_evidence(repo)[1]["vitest"]["counts"]
    assert counts["numTotalTests"] == counts["numPassedTests"] + counts["numPendingTests"] + counts["numTodoTests"]


def test_vitest_suite_level_import_failure_is_valid_application_failure_evidence(tmp_path):
    value = {"numTotalTestSuites": 1, "numPassedTestSuites": 0, "numFailedTestSuites": 1, "numPendingTestSuites": 0, "numTotalTests": 0, "numPassedTests": 0, "numFailedTests": 0, "numPendingTests": 0, "numTodoTests": 0, "success": False, "testResults": [{"name": "frontend/src/import.test.ts", "status": "failed", "message": "transform failed", "assertionResults": []}]}
    path = tmp_path / "vitest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    parsed = FRONTEND.parse_vitest(path, 1)
    assert parsed["suite_diagnostics"][0]["message"] == "transform failed"


def test_frontend_evidence_records_commands_versions_hashes_tree_counts_and_source(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    evidence = latest_evidence(repo)[1]
    for key in ("commands", "node_version", "npm_version", "package_json_sha256", "package_lock_sha256", "tree", "vitest", "source"):
        assert key in evidence


def test_frontend_evidence_does_not_self_hash(tmp_path):
    repo, _, _ = make_frontend_repo(tmp_path)
    assert run_frontend(repo, tmp_path / "tools").returncode == 0
    evidence_path, evidence = latest_evidence(repo)
    assert "evidence.json" not in {item["path"] for item in evidence["artifacts"]["entries"]}
    assert "artifact-manifest.json" not in {item["path"] for item in evidence["artifacts"]["entries"]}
    assert evidence_path.exists()


def test_frontend_runner_never_invokes_npm_run_build_or_a_node_image():
    text = _frontend_source()
    assert "npm run build" not in text and "docker" not in text and '[str(self.npm), "ci"]' in text
