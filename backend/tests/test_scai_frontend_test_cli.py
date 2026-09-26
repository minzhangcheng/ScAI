"""Docker-free contracts for the isolated frontend runner."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_FILE = ROOT / "scripts" / "scai-frontend-test"
PYTHON_RUNNER_FILE = ROOT / "scripts" / "scai_frontend_runner.py"
PARTS_DIR = ROOT / "scripts" / "scai_frontend_parts"
FIXTURE_FILE = ROOT / "backend/tests/fixtures/npm9-layout.json"
FIXTURE = json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))


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


def test_frontend_runner_hashes_both_exact_inputs_before_install(tmp_path):
    repo, _, log = make_frontend_repo(tmp_path)
    result = run_frontend(repo, tmp_path / "tools")
    assert result.returncode == 0, result.stderr
    _, evidence = latest_evidence(repo)
    assert evidence["package_json_sha256"] == FRONTEND.sha256_file(repo / "frontend/package.json")
    assert evidence["package_lock_sha256"] == FRONTEND.sha256_file(repo / "frontend/package-lock.json")
    assert json.loads(log.read_text().splitlines()[2])[:2] == ["config", "list"]


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
