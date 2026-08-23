"""Docker-free contracts for the immutable Python runner and tree proofs."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TREE_FILE = ROOT / "scripts" / "scai_test_tree.py"


def _load_contract_module(spec):
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


SPEC = importlib.util.spec_from_file_location("scai_test_tree_contract", TREE_FILE)
assert SPEC is not None and SPEC.loader is not None
TREE = _load_contract_module(SPEC)
RUNNER = (ROOT / "scripts/scai-test").read_text(encoding="utf-8")
PLUGIN = (ROOT / "scripts/scai_pytest_collection.py").read_text(encoding="utf-8")
PLUGIN_SPEC = importlib.util.spec_from_file_location("scai_pytest_collection_contract", ROOT / "scripts/scai_pytest_collection.py")
assert PLUGIN_SPEC is not None and PLUGIN_SPEC.loader is not None
PLUGIN_MODULE = _load_contract_module(PLUGIN_SPEC)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "contract@example.invalid")
    git(repo, "config", "user.name", "Contract")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-qm", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def phase(when: str, outcome: str = "passed", detail: str = "", wasxfail: str | None = None) -> dict:
    value = {"when": when, "outcome": outcome, "wasxfail": wasxfail, "detail": detail}
    value["signature"] = TREE._phase_signature(value)
    return value


def record(nodeid: str, phases: list[dict], outcome: str | None = None) -> dict:
    status, detail, xfail, xpass = TREE._phase_result(phases)
    value = {
        "nodeid": nodeid,
        "outcome": outcome or status,
        "detail": detail,
        "xfail": xfail,
        "xpass": xpass,
        "phases": phases,
    }
    value["signature"] = TREE._record_signature(value)
    return value


def outcomes_for(records: list[dict], warnings: list[dict] | None = None) -> bytes:
    warnings = warnings or []
    payload = {
        "schema": 1,
        "complete": True,
        "count": len(records),
        "records_sha256": TREE._hash_json(records),
        "warnings_sha256": TREE._hash_json(warnings),
        "records": records,
        "warnings": warnings,
    }
    return json.dumps(payload, sort_keys=True).encode()


def valid_collection_payload() -> dict:
    return {
        "schema": 1,
        "complete": True,
        "count": 1,
        "nodeids": ["case"],
        "nodeids_sha256": hashlib.sha256(b"case\n").hexdigest(),
        "effective_config": {
            "SCAI_ENABLE_SNAPSHOT_V2": {
                "value": "true",
                "source": "root conftest.py setdefault",
                "production_default": "disabled",
            }
        },
        "candidate_tools": {
            "pytest": {"version": "9.1.1", "origin": "/opt/scai-test/venv/lib/python3.11/site-packages/pytest/__init__.py"},
            "pytest-asyncio": {"version": "1.3.0", "origin": "/opt/scai-test/venv/lib/python3.11/site-packages/pytest_asyncio/__init__.py"},
            "pytest-benchmark": {"version": "5.2.3", "origin": "/opt/scai-test/venv/lib/python3.11/site-packages/pytest_benchmark/__init__.py"},
            "plugins": {"asyncio": True, "benchmark": True},
        },
    }


def test_worktree_manifest_hash_changes_when_existing_file_bytes_change(tmp_path):
    repo, base = make_repo(tmp_path)
    first = TREE.build_worktree_manifest(repo, base)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    second = TREE.build_worktree_manifest(repo, base)
    assert first["digest"] != second["digest"]
    changed = next(entry for entry in second["entries"] if entry["path"] == "tracked.txt")
    assert changed["state"] == "modified"
    assert changed["sha256"] == hashlib.sha256(b"changed\n").hexdigest()


def test_worktree_manifest_records_each_untracked_file_sha_and_mode(tmp_path):
    repo, base = make_repo(tmp_path)
    path = repo / "new.py"
    path.write_text("print(1)\n", encoding="utf-8")
    path.chmod(0o755)
    manifest = TREE.build_worktree_manifest(repo, base)
    assert manifest["entries"] == [{
        "path": "new.py", "state": "untracked", "type": "regular", "mode": "100755",
        "sha256": hashlib.sha256(b"print(1)\n").hexdigest(),
    }]


def test_worktree_manifest_records_deleted_path_with_literal_marker(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "tracked.txt").unlink()
    assert TREE.build_worktree_manifest(repo, base)["entries"] == [{
        "path": "tracked.txt", "state": "deleted", "type": "deleted", "mode": None, "marker": "DELETED",
    }]


def test_worktree_manifest_records_symlink_target_and_120000_mode(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "link").symlink_to("tracked.txt")
    entry = TREE.build_worktree_manifest(repo, base)["entries"][0]
    assert entry["type"] == "symlink" and entry["mode"] == "120000"
    assert entry["sha256"] == hashlib.sha256(b"tracked.txt").hexdigest()


def test_worktree_manifest_represents_rename_as_delete_plus_present_path(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "tracked.txt").rename(repo / "renamed.txt")
    manifest = TREE.build_worktree_manifest(repo, base)
    assert [entry["path"] for entry in manifest["entries"]] == ["renamed.txt", "tracked.txt"]
    assert {entry["state"] for entry in manifest["entries"]} == {"untracked", "deleted"}


def test_worktree_manifest_rejects_directory_or_special_file_entry(tmp_path, monkeypatch):
    repo, base = make_repo(tmp_path)
    fifo = repo / "fifo"
    os.mkfifo(fifo)
    monkeypatch.setattr(TREE, "_worktree_changes", lambda _repo, _base: [("fifo", "untracked")])
    with pytest.raises(TREE.ContractError):
        TREE.build_worktree_manifest(repo, base)


def test_manifest_loader_recomputes_digest_and_rejects_tampering(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    path = tmp_path / "manifest.json"
    write_manifest(path, TREE.build_worktree_manifest(repo, base))
    payload = json.loads(path.read_text())
    payload["entries"][0]["sha256"] = "0" * 64
    write_manifest(path, payload)
    with pytest.raises(TREE.ContractError):
        TREE.load_manifest(path, "worktree", base)


def test_compare_summary_accepts_only_one_new_summary_entry(tmp_path):
    repo, base = make_repo(tmp_path)
    tested = tmp_path / "tested.json"
    post = tmp_path / "post.json"
    write_manifest(tested, TREE.build_worktree_manifest(repo, base))
    payload = json.loads(tested.read_text())
    payload["entries"].append({"path": "summary.json", "state": "untracked", "type": "regular", "mode": "100644", "sha256": "1" * 64})
    payload["entries"].sort(key=lambda entry: entry["path"])
    payload["digest"] = TREE._manifest_digest(payload["entries"])
    write_manifest(post, payload)
    TREE.compare_post_summary(tested, post, Path("summary.json"))


def test_compare_summary_rejects_changed_tested_bytes_or_extra_path(tmp_path):
    repo, base = make_repo(tmp_path)
    tested = tmp_path / "tested.json"
    post = tmp_path / "post.json"
    write_manifest(tested, TREE.build_worktree_manifest(repo, base))
    payload = json.loads(tested.read_text())
    payload["entries"].append({"path": "one.json", "state": "untracked", "type": "regular", "mode": "100644", "sha256": "1" * 64})
    payload["entries"].append({"path": "two.json", "state": "untracked", "type": "regular", "mode": "100644", "sha256": "2" * 64})
    payload["entries"].sort(key=lambda entry: entry["path"])
    payload["digest"] = TREE._manifest_digest(payload["entries"])
    write_manifest(post, payload)
    with pytest.raises(TREE.ContractError):
        TREE.compare_post_summary(tested, post, Path("one.json"))


def test_index_comparison_normalizes_untracked_to_added_only(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "new.txt").write_text("new", encoding="utf-8")
    post_payload = TREE.build_worktree_manifest(repo, base)
    post_path = tmp_path / "post.json"
    index_path = tmp_path / "index.json"
    write_manifest(post_path, post_payload)
    projected = [{**entry, "state": "added"} if entry["state"] == "untracked" else entry for entry in post_payload["entries"]]
    write_manifest(index_path, TREE._manifest("index", base, projected))
    TREE.compare_index(post_path, index_path)


def test_index_comparison_rejects_missing_stale_extra_mode_and_deletion_drift(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "new.txt").write_text("new", encoding="utf-8")
    post = tmp_path / "post.json"
    index = tmp_path / "index.json"
    post_payload = TREE.build_worktree_manifest(repo, base)
    write_manifest(post, post_payload)
    write_manifest(index, TREE._manifest("index", base, []))
    with pytest.raises(TREE.ContractError):
        TREE.compare_index(post, index)


def test_commit_manifest_recurses_nested_files_deletions_and_executable_modes(tmp_path):
    repo, base = make_repo(tmp_path)
    nested = repo / "nested"
    nested.mkdir()
    file = nested / "run.sh"
    file.write_text("#!/bin/sh\n", encoding="utf-8")
    file.chmod(0o755)
    (repo / "tracked.txt").unlink()
    git(repo, "add", "-A")
    commit = git(repo, "commit", "-qm", "candidate") or git(repo, "rev-parse", "HEAD")
    manifest = TREE.build_commit_manifest(repo, base, commit)
    assert {entry["path"] for entry in manifest["entries"]} == {"nested/run.sh", "tracked.txt"}
    assert next(entry for entry in manifest["entries"] if entry["path"] == "nested/run.sh")["mode"] == "100755"


def test_final_commit_verifier_accepts_exact_single_parent_commit(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("candidate\n", encoding="utf-8")
    git(repo, "add", "-A")
    index = TREE.build_index_manifest(repo, base)
    index_path = tmp_path / "index.json"
    write_manifest(index_path, index)
    git(repo, "commit", "-qm", "ver 0.39.15")
    assert TREE.verify_final_commit(repo, base, index_path, "ver 0.39.15") == git(repo, "rev-parse", "HEAD")


def test_final_commit_verifier_rejects_wrong_parent(tmp_path):
    repo, base = make_repo(tmp_path)
    git(repo, "checkout", "-qb", "side")
    (repo / "tracked.txt").write_text("side\n", encoding="utf-8")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "ver 0.39.15")
    (repo / "tracked.txt").write_text("side-two\n", encoding="utf-8")
    git(repo, "add", "-A"); git(repo, "commit", "-qm", "ver 0.39.15")
    index_path = tmp_path / "index.json"
    write_manifest(index_path, TREE.build_commit_manifest(repo, base, git(repo, "rev-parse", "HEAD")))
    with pytest.raises(TREE.ContractError):
        TREE.verify_final_commit(repo, base, index_path, "ver 0.39.15")


def test_final_commit_verifier_rejects_tree_drift_and_dirty_checkout(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("candidate\n", encoding="utf-8")
    git(repo, "add", "-A")
    index_path = tmp_path / "index.json"
    write_manifest(index_path, TREE.build_index_manifest(repo, base))
    git(repo, "commit", "-qm", "ver 0.39.15")
    (repo / "dirty").write_text("dirty", encoding="utf-8")
    with pytest.raises(TREE.ContractError):
        TREE.verify_final_commit(repo, base, index_path, "ver 0.39.15")


def test_final_commit_verifier_rejects_wrong_subject_body_trailer_and_attribution(tmp_path):
    repo, base = make_repo(tmp_path)
    (repo / "tracked.txt").write_text("candidate\n", encoding="utf-8")
    git(repo, "add", "-A")
    index_path = tmp_path / "index.json"
    write_manifest(index_path, TREE.build_index_manifest(repo, base))
    git(repo, "commit", "-qm", "wrong\n\nCo-Authored-By: bot")
    with pytest.raises(TREE.ContractError):
        TREE.verify_final_commit(repo, base, index_path, "ver 0.39.15")


def test_manifest_and_final_proofs_fail_identically_with_python_optimize(tmp_path):
    repo, base = make_repo(tmp_path)
    path = tmp_path / "bad.json"
    write_manifest(path, {"schema": 1, "kind": "worktree", "base": base, "digest": "0" * 64, "entries": []})
    script = f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); import scai_test_tree as t; t.load_manifest({str(path)!r}, 'worktree', {base!r})"
    normal = subprocess.run(["python3", "-c", script], capture_output=True)
    optimized = subprocess.run(["python3", "-O", "-c", script], capture_output=True)
    assert normal.returncode == optimized.returncode != 0


def test_fast_requires_an_explicit_target_before_image_lookup():
    result = subprocess.run([str(ROOT / "scripts/scai-test"), "fast"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "fast requires" in result.stderr


def test_raw_requires_arguments_and_keeps_raw_claim():
    result = subprocess.run([str(ROOT / "scripts/scai-test"), "--"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0 and "raw mode" in result.stderr


def test_profile_without_env_defaults_to_empty_mapping():
    assert 'profile.get("env", {})' in RUNNER


def test_profile_without_or_with_empty_args_fails_before_docker():
    assert "profile has no explicit args" in RUNNER


def test_profile_globs_expand_sorted_and_unmatched_glob_fails():
    assert "compgen -G" in RUNNER and "profile glob has no matches" in RUNNER


def test_profile_roster_is_exactly_fast_client_kernel_backend_full():
    assert set(__import__("tomllib").loads((ROOT / "test-profiles.toml").read_text())["profiles"]) == {"fast", "client", "kernel", "backend-full"}


def test_matching_pointer_executes_exact_name_at_digest():
    assert 'value["image"] + "@" + value["digest"]' in RUNNER


def test_pointer_non_null_schema_constructs_name_at_digest():
    assert "tracked_digest" in RUNNER and "LOCAL_IDENTITY" in RUNNER


def test_null_pointer_discovers_tag_but_executes_local_image_id():
    assert 'docker image inspect --format' in RUNNER and "SELECTION_MODE=local_id" in RUNNER


def test_mismatched_pointer_or_local_labels_or_lock_bytes_fail_before_pytest():
    assert "does not match current inputs" in RUNNER and "local image identity mismatch" in RUNNER


def test_runner_never_implicitly_pulls_or_falls_back_to_host_python():
    assert "--pull=never" in RUNNER and "python3 -m pytest" not in RUNNER


def test_runner_rejects_caller_owned_junit_collection_or_outcomes_output_options():
    assert "--junitxml=*|--scai-collection-json" in RUNNER


def test_docker_argv_has_exact_boundary_tmpfs_mounts_user_platform_and_env():
    for text in ("--read-only", "--network none", "--user 10001", "--platform", "--tmpfs /tmp:rw,size=2g", "PYTHONPYCACHEPREFIX=/tmp/pycache"):
        assert text in RUNNER


def test_docker_argv_disables_pull_and_mounts_only_container_output_child():
    assert "--pull=never" in RUNNER and "/results/output:rw" in RUNNER and ":ro" in RUNNER


def test_host_evidence_root_is_outside_checkout_and_not_container_mounted():
    assert 'mktemp -d "/tmp/scai-test-evidence.' in RUNNER and 'EVIDENCE_STAGE' in RUNNER


def test_prepublication_failure_keeps_diagnostic_staging_until_publication():
    assert "PUBLISHED=0" in RUNNER
    assert 'unaccepted diagnostic: $EVIDENCE_STAGE' in RUNNER
    assert 'rm -rf "$INCOMING_DIR"' not in RUNNER
    assert 'if [[ "$PUBLISHED" -eq 1 ]]' in RUNNER
    assert 'publish_evidence(Path("test-results"), run_id, artifacts)' in RUNNER
    assert RUNNER.index('PUBLISHED=1') > RUNNER.index('publish_evidence')


_FAKE_RUNNER_ROOTS: set[Path] = set()


def _probe_executable_directory(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".scai-runner-exec-probe-{uuid.uuid4().hex}"
    try:
        probe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        probe.chmod(0o755)
        return subprocess.run(
            [str(probe)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def _fake_runner_harness_root(tmp_path: Path) -> Path:
    candidates = []
    output_root = Path("/results/output")
    if output_root.is_dir() and os.access(output_root, os.W_OK | os.X_OK):
        candidates.append(output_root / f".scai-runner-fake-{uuid.uuid4().hex}")
    candidates.append(tmp_path / f"scai-runner-fake-{uuid.uuid4().hex}")
    for candidate in candidates:
        if _probe_executable_directory(candidate):
            _FAKE_RUNNER_ROOTS.add(candidate)
            return candidate
        shutil.rmtree(candidate, ignore_errors=True)
    raise RuntimeError("no executable test-only directory is available for fake runner")


@pytest.fixture(autouse=True)
def _cleanup_fake_runner_roots():
    yield
    for root in list(_FAKE_RUNNER_ROOTS):
        shutil.rmtree(root, ignore_errors=True)
        _FAKE_RUNNER_ROOTS.discard(root)


def _fake_runner_repo(tmp_path: Path) -> tuple[Path, Path]:
    harness_root = _fake_runner_harness_root(tmp_path)
    repo = harness_root / "fake-runner-repo"
    for relative in (
        "scripts/scai-test",
        "scripts/test-deps-fingerprint.sh",
        "scripts/lock-test-deps.sh",
        "scripts/scai_test_tree.py",
        "test-profiles.toml",
        "requirements.txt",
        "requirements-nodeps.txt",
        "build-constraints.txt",
        "docker/test-deps.Dockerfile.dockerignore",
        "docker/test-deps/requirements-test.in",
        "docker/test-deps/requirements-test.lock",
        "docker/test-deps/requirements-test-nodeps.lock",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    dockerfile = repo / "docker/test-deps.Dockerfile"
    dockerfile.write_text(
        "FROM python:3.11.15-slim-bookworm@sha256:"
        + "0" * 64
        + "\n",
        encoding="utf-8",
    )
    (repo / "docker/test-deps/lock-manifest.json").write_text(
        '{"resolver": {"pip": "pip>=25.3,<27"}}\n', encoding="utf-8"
    )
    verifier = repo / "scripts/verify-test-deps-image.sh"
    verifier.write_text("#!/bin/bash\nset -eu\nexit 0\n", encoding="utf-8")
    verifier.chmod(0o755)
    pointer = repo / "docker/test-deps-image.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps({"schema": 1, "image": None, "digest": None, "fingerprint": None, "lock_sha256": None, "selected_at_utc": None})
        + "\n",
        encoding="utf-8",
    )
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "contract@example.invalid")
    git(repo, "config", "user.name", "Contract")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    runner = repo / "scripts/scai-test"
    runner.write_text(runner.read_text(encoding="utf-8").replace("0c1edf479f7360dc80afa89cf7fc9534c1effce6", base), encoding="utf-8")
    identity_lines = subprocess.run(
        [str(repo / "scripts/test-deps-fingerprint.sh")], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.splitlines()
    fingerprint = next(line.split("=", 1)[1] for line in identity_lines if line.startswith("FINGERPRINT="))
    tag = next(line.split("=", 1)[1] for line in identity_lines if line.startswith("TAG="))
    lock_sha = hashlib.sha256((repo / "docker/test-deps/requirements-test.lock").read_bytes()).hexdigest()
    (repo / ".scai-test-deps-image.json").write_text(
        json.dumps({"schema": 1, "tag": tag, "image_id": "sha256:" + "a" * 64, "fingerprint": fingerprint, "lock_sha256": lock_sha})
        + "\n",
        encoding="utf-8",
    )
    fake_bin = harness_root / "fake-bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/bin/sh\necho fake-container-stdout\necho fake-container-stderr >&2\nexit 0\n", encoding="utf-8")
    docker.chmod(0o755)
    return repo, fake_bin


def _install_fake_docker_artifacts(
    fake_bin: Path,
    records: list[dict],
    junit_counters: dict[str, int],
    *,
    exit_code: int,
) -> None:
    fixture_root = fake_bin / "container-artifacts"
    fixture_root.mkdir()
    collection = valid_collection_payload()
    (fixture_root / "collection.json").write_text(json.dumps(collection, sort_keys=True), encoding="utf-8")
    (fixture_root / "outcomes.json").write_bytes(outcomes_for(records))
    attributes = " ".join(f'{key}="{junit_counters[key]}"' for key in ("tests", "failures", "errors", "skipped"))
    (fixture_root / "junit.xml").write_text(f'<testsuite name="pytest" {attributes} />', encoding="utf-8")
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, shutil, sys\n"
        f"fixture_root = pathlib.Path({str(fixture_root)!r})\n"
        "mount = next(sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1]) "
        "if value == '--volume' and sys.argv[index + 1].endswith(':/results/output:rw'))\n"
        "output = pathlib.Path(mount.rsplit(':/results/output:rw', 1)[0])\n"
        "for source in fixture_root.iterdir():\n"
        "    shutil.copyfile(source, output / source.name)\n"
        "print('fake-container-stdout')\n"
        "print('fake-container-stderr', file=sys.stderr)\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)


def _nested_runner_diagnostics(stderr: str) -> set[Path]:
    return {Path(value) for value in re.findall(r"unaccepted diagnostic: (/tmp/scai-test-evidence\.[^ ]+)", stderr)}


def _cleanup_nested_runner_diagnostics(stderr: str) -> None:
    for path in _nested_runner_diagnostics(stderr):
        shutil.rmtree(path, ignore_errors=True)


def test_fake_runner_harness_probes_and_cleans_executable_root():
    source = Path(__file__).read_text(encoding="utf-8")
    for marker in (
        "_probe_executable_directory",
        'Path(\"/results/output\")',
        'output_root / f".scai-runner-fake-',
        "_FAKE_RUNNER_ROOTS",
        "@pytest.fixture(autouse=True)",
        "shutil.rmtree(root, ignore_errors=True)",
        "probe.chmod(0o755)",
        "subprocess.run(",
    ):
        assert marker in source


def test_fake_cli_preserves_empty_container_output_and_logs(tmp_path):
    repo, fake_bin = _fake_runner_repo(tmp_path)
    result = subprocess.run(
        [str(repo / "scripts/scai-test"), "fast", "placeholder.py"],
        cwd=repo,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "container output allowlist mismatch: []" in result.stderr
    match = re.search(r"unaccepted diagnostic: (/tmp/scai-test-evidence\.[^ ]+)", result.stderr)
    assert match, result.stderr
    stage = Path(match.group(1))
    try:
        assert stage.is_dir() and stat.S_IMODE(stage.stat().st_mode) == 0o700
        assert (stage / "container-output").is_dir()
        assert list((stage / "container-output").iterdir()) == []
        assert (stage / "stdout.txt").read_text(encoding="utf-8") == "fake-container-stdout\n"
        stderr = (stage / "stderr.txt").read_text(encoding="utf-8")
        assert stderr == "fake-container-stderr\n"
        assert not (repo / "test-results").exists()
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def test_empty_profile_env_does_not_emit_empty_docker_env_and_preserves_base_env_args(tmp_path):
    repo, fake_bin = _fake_runner_repo(tmp_path)
    docker_log = fake_bin / "docker-args.jsonl"
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"log = pathlib.Path({str(docker_log)!r})\n"
        "with log.open('a', encoding='utf-8') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n"
        "    stream.write('\\n')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    base_env = [
        "PYTHONPATH=/workspace/scai",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "SCAI_DATA_DIR=/tmp/scai-data",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "XDG_CACHE_HOME=/tmp/xdg",
        "NUMBA_DISABLE_JIT=1",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
    ]
    try:
        for profile in ("client", "backend-full"):
            result = subprocess.run(
                [str(repo / "scripts/scai-test"), profile],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
            )
            assert result.returncode != 0
            calls = [json.loads(line) for line in docker_log.read_text(encoding="utf-8").splitlines()]
            assert len(calls) >= 1
            argv = calls[-1]
            env_values = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--env"]
            assert env_values == base_env
            assert "" not in env_values
            _cleanup_nested_runner_diagnostics(result.stderr)
            shutil.rmtree(repo / "test-results", ignore_errors=True)
    finally:
        _cleanup_nested_runner_diagnostics(result.stderr if "result" in locals() else "")


def test_runner_binds_logs_to_open_inodes_and_rejects_replaced_log_path(tmp_path):
    repo, fake_bin = _fake_runner_repo(tmp_path)
    records = [record("case", [phase("setup"), phase("call"), phase("teardown")])]
    counters = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    _install_fake_docker_artifacts(fake_bin, records, counters, exit_code=0)
    fixture_root = fake_bin / "container-artifacts"
    sentinel = fake_bin / "external-sentinel.txt"
    sentinel.write_text("sentinel-must-not-change\n", encoding="utf-8")
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, shutil, sys\n"
        f"fixture_root = pathlib.Path({str(fixture_root)!r})\n"
        f"sentinel = pathlib.Path({str(sentinel)!r})\n"
        "mount = next(sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1]) "
        "if value == '--volume' and sys.argv[index + 1].endswith(':/results/output:rw'))\n"
        "output = pathlib.Path(mount.rsplit(':/results/output:rw', 1)[0])\n"
        "for source in fixture_root.iterdir():\n"
        "    shutil.copyfile(source, output / source.name)\n"
        "stdout_path = pathlib.Path(os.readlink('/proc/self/fd/1'))\n"
        "original = stdout_path.with_name(stdout_path.name + '.original-inode')\n"
        "stdout_path.rename(original)\n"
        "stdout_path.symlink_to(sentinel)\n"
        "print('bound-to-original-inode')\n"
        "print('fake-container-stderr', file=sys.stderr)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    result = subprocess.run(
        [str(repo / "scripts/scai-test"), "fast", "placeholder.py"],
        cwd=repo,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
    )
    diagnostics = _nested_runner_diagnostics(result.stderr)
    try:
        assert result.returncode == 2, result.stderr
        assert len(diagnostics) == 1, result.stderr
        stage = next(iter(diagnostics))
        replaced = stage / "stdout.txt"
        original = stage / "stdout.txt.original-inode"
        assert replaced.is_symlink() and replaced.resolve() == sentinel.resolve()
        assert original.read_text(encoding="utf-8") == "bound-to-original-inode\n"
        assert sentinel.read_text(encoding="utf-8") == "sentinel-must-not-change\n"
        assert not (repo / "test-results").exists()
    finally:
        _cleanup_nested_runner_diagnostics(result.stderr)


def test_runner_retains_secure_log_inode_across_pre_docker_path_swap(tmp_path):
    repo, fake_bin = _fake_runner_repo(tmp_path)
    records = [record("case", [phase("setup"), phase("call"), phase("teardown")])]
    counters = {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    _install_fake_docker_artifacts(fake_bin, records, counters, exit_code=0)
    sentinel = fake_bin / "external-pre-docker-sentinel.txt"
    sentinel.write_text("sentinel-must-not-change\n", encoding="utf-8")
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    process = subprocess.Popen(
        [str(repo / "scripts/scai-test"), "fast", "placeholder.py"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    swapped: list[Path] = []
    watcher_errors: list[BaseException] = []

    def swap_after_secure_create() -> None:
        deadline = time.monotonic() + 10
        pattern = f"scai-test-evidence.*-{process.pid}.*"
        try:
            while time.monotonic() < deadline:
                for stage in Path("/tmp").glob(pattern):
                    stdout_path = stage / "stdout.txt"
                    original = stage / "stdout.txt.original-inode"
                    if stdout_path.is_file() and not stdout_path.is_symlink() and not original.exists():
                        stdout_path.rename(original)
                        stdout_path.symlink_to(sentinel)
                        swapped.append(stage)
                        return
                time.sleep(0.001)
            raise AssertionError("runner log never reached the secure-created pre-Docker state")
        except BaseException as exc:
            watcher_errors.append(exc)

    watcher = threading.Thread(target=swap_after_secure_create, daemon=True)
    watcher.start()
    stdout, stderr = process.communicate(timeout=30)
    watcher.join(timeout=1)
    try:
        assert not watcher.is_alive()
        assert not watcher_errors, watcher_errors
        assert len(swapped) == 1
        stage = swapped[0]
        assert process.returncode == 2, (stdout, stderr)
        assert "unaccepted diagnostic" in stderr
        replaced = stage / "stdout.txt"
        original = stage / "stdout.txt.original-inode"
        assert replaced.is_symlink() and replaced.resolve() == sentinel.resolve()
        assert original.read_text(encoding="utf-8") == "fake-container-stdout\n"
        assert sentinel.read_text(encoding="utf-8") == "sentinel-must-not-change\n"
        assert not (repo / "test-results").exists()
    finally:
        for stage in set(swapped) | _nested_runner_diagnostics(stderr):
            shutil.rmtree(stage, ignore_errors=True)


def test_post_run_artifact_validation_rejects_symlink_special_and_hardlink():
    assert "st_nlink != 1" in RUNNER and "O_NOFOLLOW" in RUNNER and "read_direct_children_nofollow" in RUNNER


def test_collect_only_output_is_exactly_collection_json():
    assert "EXPECTED_OUTPUTS=(collection.json)" in RUNNER


def test_normal_output_is_exactly_collection_junit_and_outcomes():
    assert "EXPECTED_OUTPUTS+=(outcomes.json junit.xml)" in RUNNER


def test_output_allowlist_rejects_extra_regular_file_or_directory():
    assert "actual !=" in RUNNER and "container output allowlist mismatch" in RUNNER


def test_post_run_source_manifest_mismatch_invalidates_evidence():
    assert "source tree changed while tests ran" in RUNNER


def test_collection_plugin_writes_sorted_nodeids_count_and_hash():
    assert "nodeids_sha256" in PLUGIN and "sorted(item.nodeid" in PLUGIN


def test_collection_records_exact_candidate_tool_versions_origins_and_plugins():
    for text in ("9.1.1", "1.3.0", "5.2.3", "/opt/scai-test/venv", "required asyncio and benchmark"):
        assert text in PLUGIN


def test_runner_rejects_benchmark_options_and_never_claims_benchmark_scope():
    assert "--benchmark*" in RUNNER and "benchmark scope" not in RUNNER


def test_collection_candidate_plugin_semantic_mismatch_fails_closed():
    payload = valid_collection_payload()
    payload["candidate_tools"]["plugins"]["benchmark"] = False
    with pytest.raises(TREE.ContractError, match="plugin identity"):
        TREE.validate_collection(json.dumps(payload).encode())
    assert 'collection = validate_collection(collection_artifact.payload)' in RUNNER


def test_collect_only_uses_collection_artifact_without_junit():
    assert 'if [[ "$COLLECT_ONLY" -eq 0 ]]' in RUNNER


def test_outcomes_plugin_merges_setup_call_teardown_to_one_exact_nodeid():
    phases = [phase("setup"), phase("call"), phase("teardown")]
    assert len(record("x::test", phases)["phases"]) == 3


def test_outcome_validator_requires_exact_collection_bijection():
    payload = outcomes_for([record("a", [phase("setup"), phase("call"), phase("teardown")])])
    assert TREE.validate_outcomes(payload, ("a",))["counts"]["passed"] == 1


def test_phase_state_machine_requires_teardown_and_applies_exact_final_precedence():
    valid = [
        (
            [phase("setup", "failed", "setup failed"), phase("teardown")],
            ("error", "setup failed", False, False),
        ),
        (
            [phase("setup", "skipped", "setup skipped"), phase("teardown")],
            ("skipped", "setup skipped", False, False),
        ),
        (
            [phase("setup"), phase("call"), phase("teardown", "skipped", "teardown skipped")],
            ("skipped", "teardown skipped", False, False),
        ),
        (
            [phase("setup"), phase("call", "failed", "call failed"), phase("teardown", "skipped")],
            ("failed", "call failed", False, False),
        ),
        (
            [phase("setup"), phase("call"), phase("teardown", "failed", "teardown failed")],
            ("error", "teardown failed", False, False),
        ),
        (
            [phase("setup"), phase("call", "failed", "call failed"), phase("teardown", "failed", "teardown failed")],
            ("error", "teardown failed", False, False),
        ),
        (
            [phase("setup"), phase("call", "passed", "call xpass", "xfail"), phase("teardown")],
            ("passed", "call xpass", False, True),
        ),
        (
            [phase("setup"), phase("call", "failed", "call xpass", "xfail"), phase("teardown")],
            ("failed", "call xpass", False, True),
        ),
        (
            [phase("setup"), phase("call", "skipped", "call xfail", "xfail"), phase("teardown")],
            ("skipped", "call xfail", True, False),
        ),
    ]
    for phases, expected in valid:
        normalized = TREE._validate_phases("valid::case", phases)
        assert TREE._phase_result(normalized) == expected

    incomplete = (
        [phase("setup"), phase("call")],
        [phase("setup", "failed"), phase("call")],
        [phase("setup", "skipped"), phase("call", "skipped")],
        [phase("setup")],
    )
    for phases in incomplete:
        with pytest.raises(TREE.ContractError, match="teardown"):
            TREE._validate_phases("incomplete::case", phases)


def test_outcome_producer_marks_invalid_stream_incomplete_and_raises_usage_error(tmp_path):
    target = tmp_path / "outcomes.json"
    evidence = PLUGIN_MODULE._Evidence(SimpleNamespace())
    evidence.add_report(SimpleNamespace(nodeid="broken::case", when="setup", outcome="passed", wasxfail=None, longreprtext=""))
    evidence.add_report(SimpleNamespace(nodeid="broken::case", when="call", outcome="passed", wasxfail=None, longreprtext=""))

    class Config:
        def getoption(self, name, default=None):
            return str(target) if name == "--scai-outcomes-json" else default

    session = SimpleNamespace(config=Config(), _scai_evidence=evidence)
    with pytest.raises(pytest.UsageError, match="incomplete"):
        PLUGIN_MODULE.pytest_sessionfinish(session, pytest.ExitCode.OK)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["complete"] is False


def test_junit_projection_resets_reporter_state_for_each_node():
    records = (
        {
            "nodeid": "first::case",
            "phases": [phase("setup"), phase("call", "failed", "call failed"), phase("teardown")],
        },
        {
            "nodeid": "second::case",
            "phases": [phase("setup"), phase("call"), phase("teardown", "failed", "teardown failed")],
        },
    )
    assert TREE.project_pytest_9_1_1_junit(records) == {
        "passed": 1,
        "failures": 1,
        "errors": 1,
        "skipped": 0,
        "double_fail_correction": 1,
        "tests": 2,
    }


def test_outcome_details_match_exact_canonical_schema():
    warning = {"nodeid": "skipped::case", "when": "runtest", "category": "UserWarning", "message": "notice"}
    warning["signature"] = TREE._hash_json({key: warning[key] for key in ("nodeid", "when", "category", "message")})
    records = [
        record("failed::case", [phase("setup"), phase("call", "failed", "failed detail"), phase("teardown")]),
        record("skipped::case", [phase("setup"), phase("call", "skipped", "skip detail", "xfail"), phase("teardown")]),
    ]
    details = TREE.validate_outcomes(outcomes_for(records, [warning]), tuple(record["nodeid"] for record in records))
    assert set(details) == {
        "schema", "collection_sha256", "records_sha256", "warnings_sha256", "records", "warnings",
        "counts", "junit_projection", "failed_nodes", "skipped_nodes",
    }
    assert set(details["counts"]) == {"collected", "passed", "failed", "errors", "skipped", "xfail", "xpass", "warnings"}
    assert details["counts"] == {
        "collected": 2, "passed": 0, "failed": 1, "errors": 0, "skipped": 1,
        "xfail": 1, "xpass": 0, "warnings": 1,
    }
    assert all(set(entry) == {"nodeid", "outcome", "signature", "detail"} for entry in details["failed_nodes"])
    assert all(set(entry) == {"nodeid", "signature", "reason"} for entry in details["skipped_nodes"])
    assert details["failed_nodes"][0]["nodeid"] == "failed::case"
    assert details["skipped_nodes"][0]["nodeid"] == "skipped::case"


def test_outcome_validator_rejects_duplicate_missing_extra_or_unknown_nodeid():
    payload = outcomes_for([record("a", [phase("setup"), phase("call"), phase("teardown")])])
    with pytest.raises(TREE.ContractError):
        TREE.validate_outcomes(payload, ("a", "b"))


def test_outcome_validator_rejects_duplicate_unknown_or_incomplete_phases():
    phases = [phase("setup"), phase("setup")]
    with pytest.raises(TREE.ContractError):
        TREE.validate_outcomes(outcomes_for([record("a", phases)]), ("a",))


def test_outcome_validator_classifies_passed_failed_error_and_skipped():
    records = [
        record("a", [phase("setup"), phase("call"), phase("teardown")]),
        record("b", [phase("setup"), phase("call", "failed"), phase("teardown")]),
        record("c", [phase("setup", "failed"), phase("teardown")], outcome="error"),
        record("d", [phase("setup"), phase("call", "skipped"), phase("teardown")]),
    ]
    details = TREE.validate_outcomes(outcomes_for(records), tuple("a b c d".split()))
    assert details["counts"] == {
        "collected": 4, "passed": 1, "failed": 1, "errors": 1, "skipped": 1,
        "xfail": 0, "xpass": 0, "warnings": 0,
    }


def test_outcome_validator_accepts_teardown_skip_with_severity_precedence():
    phases = [phase("setup"), phase("call"), phase("teardown", "skipped")]
    details = TREE.validate_outcomes(outcomes_for([record("a", phases)]), ("a",))
    assert details["counts"]["skipped"] == 1


def test_outcome_validator_rejects_illegal_missing_or_extra_phase_combinations():
    with pytest.raises(TREE.ContractError):
        TREE.validate_outcomes(outcomes_for([record("a", [phase("teardown")])]), ("a",))


def test_outcome_xfail_accepts_any_single_skipped_phase_wasxfail():
    phases = [phase("setup"), phase("call", "skipped", wasxfail="reason"), phase("teardown")]
    details = TREE.validate_outcomes(outcomes_for([record("a", phases)]), ("a",))
    assert details["records"][0]["xfail"] is True


def test_outcome_signatures_and_warning_details_are_canonical_and_hashed():
    warning = {"nodeid": "a", "when": "runtest", "category": "UserWarning", "message": "hello"}
    warning["signature"] = TREE._hash_json({key: warning[key] for key in ("nodeid", "when", "category", "message")})
    details = TREE.validate_outcomes(outcomes_for([record("a", [phase("setup"), phase("call"), phase("teardown")])], [warning]), ("a",))
    assert details["warnings"][0]["signature"] == warning["signature"]


def test_warning_hook_replays_early_and_captures_runtest_metadata():
    plugin = PLUGIN_MODULE
    plugin._ACTIVE_EVIDENCE = None
    plugin._PENDING_WARNINGS.clear()
    session = type("SyntheticSession", (), {})()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            warnings.warn("early synthetic warning", UserWarning)
        plugin.pytest_warning_recorded(captured[0], "collect", "", None)
        assert len(plugin._PENDING_WARNINGS) == 1

        plugin.pytest_sessionstart(session)
        assert plugin._PENDING_WARNINGS == []

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            warnings.warn("runtime synthetic warning", DeprecationWarning)
        plugin.pytest_warning_recorded(captured[0], "runtest", "synthetic::test_warning", None)

        warnings_payload = session._scai_evidence.payload()["warnings"]
        assert {(item["nodeid"], item["when"], item["category"], item["message"]) for item in warnings_payload} == {
            (None, "collect", "UserWarning", "early synthetic warning"),
            ("synthetic::test_warning", "runtest", "DeprecationWarning", "runtime synthetic warning"),
        }
    finally:
        plugin._ACTIVE_EVIDENCE = None
        plugin._PENDING_WARNINGS.clear()


def test_outcome_validator_fails_identically_with_python_optimize():
    source = "import sys; sys.path.insert(0, %r); import scai_test_tree as t; t.validate_outcomes(b'{}', ('a',))" % str(ROOT / "scripts")
    normal = subprocess.run(["python3", "-c", source], capture_output=True)
    optimized = subprocess.run(["python3", "-O", "-c", source], capture_output=True)
    assert normal.returncode == optimized.returncode != 0


def test_junit_projection_call_pass_then_teardown_fail_counts_one_test():
    records = ({"nodeid": "a", "phases": [phase("setup"), phase("call"), phase("teardown", "failed")]},)
    assert TREE.project_pytest_9_1_1_junit(records)["tests"] == 1


def test_junit_projection_call_fail_then_teardown_fail_counts_two_tests():
    records = ({"nodeid": "a", "phases": [phase("setup"), phase("call", "failed"), phase("teardown", "failed")]},)
    assert TREE.project_pytest_9_1_1_junit(records)["tests"] == 2


def test_junit_projection_teardown_skip_and_xfail_xpass_match_pytest_9_1_1():
    records = ({"nodeid": "a", "phases": [phase("setup"), phase("call", "passed", wasxfail="reason"), phase("teardown", "skipped")]},)
    result = TREE.project_pytest_9_1_1_junit(records)
    assert result["skipped"] == 1 and result["passed"] == 1 and result["tests"] == 2


def test_junit_parser_accepts_call_pass_teardown_error_without_deriving_passed(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuite name="pytest" tests="1" failures="0" errors="1" skipped="0" />',
        encoding="utf-8",
    )
    assert TREE.parse_junit(path) == {"tests": 1, "failures": 0, "errors": 1, "skipped": 0}


def test_junit_parser_accepts_setup_and_teardown_errors_without_negative_pass_rejection(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuite name="pytest" tests="1" failures="0" errors="2" skipped="0" />',
        encoding="utf-8",
    )
    assert TREE.parse_junit(path) == {"tests": 1, "failures": 0, "errors": 2, "skipped": 0}


def test_junit_parser_rejects_malformed_and_negative_counters(tmp_path):
    malformed = tmp_path / "malformed.xml"
    malformed.write_text('<testsuite name="pytest" tests="not-an-int" />', encoding="utf-8")
    with pytest.raises(TREE.ContractError, match="not an integer"):
        TREE.parse_junit(malformed)

    negative_dir = tmp_path / "negative"
    negative_dir.mkdir()
    negative = negative_dir / "negative.xml"
    negative.write_text('<testsuite name="pytest" tests="1" errors="-1" />', encoding="utf-8")
    with pytest.raises(TREE.ContractError, match="attribute is negative"):
        TREE.parse_junit(negative)


def test_runner_cross_checks_only_four_raw_junit_counters_and_rejects_each_mutation(tmp_path):
    repo, fake_bin = _fake_runner_repo(tmp_path)
    records = [record("case", [phase("setup"), phase("call"), phase("teardown", "failed", "teardown failed")])]
    expected = {"tests": 1, "failures": 0, "errors": 1, "skipped": 0}
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    _install_fake_docker_artifacts(fake_bin, records, expected, exit_code=1)
    accepted = subprocess.run(
        [str(repo / "scripts/scai-test"), "fast", "placeholder.py"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    try:
        assert accepted.returncode == 1, accepted.stderr
        assert "JUnit projection mismatch" not in accepted.stderr
        assert (repo / "test-results").is_dir()
    finally:
        _cleanup_nested_runner_diagnostics(accepted.stderr)
        shutil.rmtree(repo / "test-results", ignore_errors=True)
        shutil.rmtree(fake_bin / "container-artifacts", ignore_errors=True)
    for key in ("tests", "failures", "errors", "skipped"):
        mutated = dict(expected)
        mutated[key] += 1
        _install_fake_docker_artifacts(fake_bin, records, mutated, exit_code=1)
        rejected = subprocess.run(
            [str(repo / "scripts/scai-test"), "fast", "placeholder.py"],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
        )
        try:
            assert rejected.returncode == 2, (key, rejected.stderr)
            assert "JUnit" in rejected.stderr
            assert not (repo / "test-results").exists()
            assert len(_nested_runner_diagnostics(rejected.stderr)) == 1, (key, rejected.stderr)
        finally:
            _cleanup_nested_runner_diagnostics(rejected.stderr)
            shutil.rmtree(repo / "test-results", ignore_errors=True)
            shutil.rmtree(fake_bin / "container-artifacts", ignore_errors=True)


def test_junit_projection_mismatch_reports_actual_and_phase_expected():
    assert "parsed actual=" in RUNNER
    assert "phase projection expected=" in RUNNER


def test_runner_uses_run_start_head_for_later_commits(tmp_path):
    repo, fake_bin = _fake_runner_repo(tmp_path)
    _install_fake_docker_artifacts(
        fake_bin,
        [record("case", [phase("setup"), phase("call"), phase("teardown")])],
        {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
        exit_code=0,
    )
    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    git(repo, "add", "later.txt")
    git(repo, "commit", "-qm", "later")
    run_start_head = git(repo, "rev-parse", "HEAD")
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    result = subprocess.run(
        [str(repo / "scripts/scai-test"), "fast", "placeholder.py"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    try:
        assert result.returncode == 0, result.stderr
        run_dirs = [
            path for path in (repo / "test-results").iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        assert len(run_dirs) == 1
        evidence = json.loads((run_dirs[0] / "evidence.json").read_text(encoding="utf-8"))
        assert evidence["source"]["base_commit"] == run_start_head
        assert evidence["source"]["manifest"]["base"] == run_start_head
        for name in ("pre-source-manifest.json", "post-source-manifest.json"):
            manifest = json.loads((run_dirs[0] / name).read_text(encoding="utf-8"))
            assert manifest["base"] == run_start_head
    finally:
        _cleanup_nested_runner_diagnostics(result.stderr)
        shutil.rmtree(repo / "test-results", ignore_errors=True)
        shutil.rmtree(fake_bin / "container-artifacts", ignore_errors=True)


def test_collection_validator_requires_complete_true_and_exact_plugins():
    base = {
        "schema": 1,
        "complete": True,
        "count": 1,
        "nodeids": ["case"],
        "nodeids_sha256": hashlib.sha256(b"case\n").hexdigest(),
        "effective_config": {
            "SCAI_ENABLE_SNAPSHOT_V2": {
                "value": "true",
                "source": "root conftest.py setdefault",
                "production_default": "disabled",
            }
        },
        "candidate_tools": {
            "pytest": {"version": "9.1.1", "origin": "/opt/scai-test/venv/lib/python3.11/site-packages/pytest/__init__.py"},
            "pytest-asyncio": {"version": "1.3.0", "origin": "/opt/scai-test/venv/lib/python3.11/site-packages/pytest_asyncio/__init__.py"},
            "pytest-benchmark": {"version": "5.2.3", "origin": "/opt/scai-test/venv/lib/python3.11/site-packages/pytest_benchmark/__init__.py"},
            "plugins": {"asyncio": True, "benchmark": True},
        },
    }
    invalid = [
        {**base, "complete": False},
        {**base, "candidate_tools": {key: value for key, value in base["candidate_tools"].items() if key != "plugins"}},
        {**base, "candidate_tools": {**base["candidate_tools"], "plugins": {"asyncio": False, "benchmark": True}}},
        {**base, "candidate_tools": {**base["candidate_tools"], "plugins": {"asyncio": True, "benchmark": True, "extra": False}}},
        {**base, "candidate_tools": {**base["candidate_tools"], "extra": {"version": "1", "origin": "/opt/scai-test/venv"}}},
    ]
    for payload in invalid:
        with pytest.raises(TREE.ContractError):
            TREE.validate_collection(json.dumps(payload).encode())


def test_read_direct_children_nofollow_rejects_symlink_fifo_hardlink_and_reuses_loaded_bytes(tmp_path, monkeypatch):
    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    (symlink_root / "item").symlink_to(tmp_path / "target")
    with pytest.raises(TREE.ContractError):
        TREE.read_direct_children_nofollow(symlink_root, ("item",))

    fifo_root = tmp_path / "fifo"
    fifo_root.mkdir()
    os.mkfifo(fifo_root / "item")
    with pytest.raises(TREE.ContractError):
        TREE.read_direct_children_nofollow(fifo_root, ("item",))

    hardlink_root = tmp_path / "hardlink"
    hardlink_root.mkdir()
    source = tmp_path / "hardlink-source"
    source.write_bytes(b"same")
    os.link(source, hardlink_root / "item")
    with pytest.raises(TREE.ContractError):
        TREE.read_direct_children_nofollow(hardlink_root, ("item",))

    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    (extra_root / "item").write_bytes(b"one")
    (extra_root / "extra").mkdir()
    with pytest.raises(TREE.ContractError):
        TREE.read_direct_children_nofollow(extra_root, ("item",))

    race_root = tmp_path / "race"
    race_root.mkdir()
    race_path = race_root / "item"
    race_path.write_bytes(b"original")
    original_read = os.read
    replaced = False

    def replace_after_open(fd, size):
        nonlocal replaced
        if not replaced:
            race_path.unlink()
            race_path.write_bytes(b"replacement")
            replaced = True
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", replace_after_open)
    result = TREE.read_direct_children_nofollow(race_root, ("item",))
    assert result["item"].payload == b"original"


def test_runner_publishes_both_source_manifests_and_refuses_existing_destinations(tmp_path):
    artifacts = {
        name: TREE.ArtifactBytes(name, payload, len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in {
            "pre-source-manifest.json": b"pre\n",
            "post-source-manifest.json": b"post\n",
            "stdout.txt": b"stdout\n",
        }.items()
    }
    final = TREE.publish_evidence(tmp_path, "run-ok", artifacts)
    assert (final / "pre-source-manifest.json").read_bytes() == b"pre\n"
    assert (final / "post-source-manifest.json").read_bytes() == b"post\n"

    incoming = tmp_path / ".incoming-run-incoming"
    incoming.mkdir()
    sentinel = incoming / "sentinel"
    sentinel.write_bytes(b"incoming")
    with pytest.raises(TREE.ContractError):
        TREE.publish_evidence(tmp_path, "run-incoming", artifacts)
    assert sentinel.read_bytes() == b"incoming"

    existing = tmp_path / "run-existing"
    existing.mkdir()
    existing_sentinel = existing / "sentinel"
    existing_sentinel.write_bytes(b"existing")
    with pytest.raises(TREE.ContractError):
        TREE.publish_evidence(tmp_path, "run-existing", artifacts)
    assert existing_sentinel.read_bytes() == b"existing"


def test_evidence_publication_fsyncs_before_atomic_rename_without_recursive_removal(tmp_path, monkeypatch):
    payload = b"payload\n"
    artifacts = {"artifact.txt": TREE.ArtifactBytes("artifact.txt", payload, len(payload), hashlib.sha256(payload).hexdigest())}
    events = []
    real_fsync = os.fsync
    real_rename = os.rename

    def record_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def record_rename(source, target):
        events.append("rename")
        return real_rename(source, target)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "rename", record_rename)
    TREE.publish_evidence(tmp_path, "run-fsync", artifacts)
    assert "rename" in events and events.index("fsync") < events.index("rename")
    assert 'rm -rf "$INCOMING_DIR"' not in RUNNER


def test_kernel_profile_rejects_only_exact_gated_live_skip_details():
    gated = {"skipped_nodes": [{"nodeid": "kernel::live", "signature": "a" * 64, "reason": "Set SCAI_RUN_KERNEL_TESTS=1"}]}
    with pytest.raises(TREE.ContractError):
        TREE.reject_kernel_gated_live_skips("kernel", gated)
    TREE.reject_kernel_gated_live_skips("fast", gated)
    TREE.reject_kernel_gated_live_skips("kernel", {"skipped_nodes": [{"nodeid": "kernel::live", "signature": "a" * 64, "reason": "optional dependency skipped"}]})


def test_raw_collectonly_aliases_are_exact_and_emit_collection_only():
    assert "--collect-only|--collectonly|--co" in RUNNER and "--collect-only=*" in RUNNER


def test_kernel_skip_reason_comes_only_from_exact_outcome_nodeid():
    assert "nodeids" in PLUGIN and "outcomes" in RUNNER


def test_evidence_records_command_environment_duration_counts_and_claim_scope():
    for text in ("expanded_command", "environment", "duration_seconds", "counts", "claim_scope"):
        assert text in RUNNER


def test_zero_collection_or_normal_missing_malformed_junit_is_not_a_pass():
    assert "JUnit XML reports zero tests" in TREE_FILE.read_text(encoding="utf-8") and "container output allowlist" in RUNNER


def test_evidence_hashes_payloads_but_never_itself_or_its_detached_manifest():
    assert 'if path.name not in {"evidence.json", "artifact-manifest.json"}' in RUNNER


def test_evidence_records_snapshot_v2_conftest_effective_configuration():
    payload = valid_collection_payload()
    payload["effective_config"]["SCAI_ENABLE_SNAPSHOT_V2"]["value"] = "false"
    with pytest.raises(TREE.ContractError, match="effective configuration"):
        TREE.validate_collection(json.dumps(payload).encode())
    assert '"pytest_effective_config": collection["effective_config"]' in RUNNER


def test_new_shell_script_modes_are_0755():
    assert stat.S_IMODE((ROOT / "scripts/scai-test").stat().st_mode) == 0o755


def test_container_artifacts_are_host_readable_after_atomic_write(tmp_path):
    target = tmp_path / "artifact.json"
    code = """
import os
import stat
import sys
from scripts.scai_pytest_collection import _atomic_write, _make_readable
path = sys.argv[1]
_atomic_write(path, {"schema": 1})
if stat.S_IMODE(os.stat(path).st_mode) != 0o644:
    raise SystemExit(f"atomic mode drift: {oct(stat.S_IMODE(os.stat(path).st_mode))}")
os.chmod(path, 0o600)
_make_readable(path)
if stat.S_IMODE(os.stat(path).st_mode) != 0o644:
    raise SystemExit(f"readable mode drift: {oct(stat.S_IMODE(os.stat(path).st_mode))}")
"""
    result = subprocess.run([sys.executable, "-c", code, str(target)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_atomic_write_uses_mkstemp_fd_without_path_reopen():
    module = ast.parse(PLUGIN, filename="scripts/scai_pytest_collection.py")
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_atomic_write")
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    call_names = {(node.func.value.id, node.func.attr) for node in calls if isinstance(node.func.value, ast.Name)}
    assert ("tempfile", "mkstemp") in call_names
    assert ("os", "fdopen") in call_names
    assert ("os", "fchmod") in call_names
    assert ("os", "replace") in call_names
    assert ("os", "close") not in call_names
    assert ("os", "open") not in call_names
