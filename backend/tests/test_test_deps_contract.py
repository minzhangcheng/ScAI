"""Docker-free contracts for the F0 immutable dependency environment."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_DIGEST = "sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15"
PYTHON_PATCH = "3.11.15"
PLATFORM = "linux/amd64"
EXPECTED_PIP_CHECK = "decoupler 1.8.0 has requirement numba<0.61.0,>=0.60.0, but you have numba 0.61.2."
FIXED_HEAD = "0c1edf479f7360dc80afa89cf7fc9534c1effce6"
EXPECTED_IMAGE_LABELS = {
    "org.opencontainers.image.source": "https://github.com/minzhangcheng/scai",
    "org.opencontainers.image.revision": FIXED_HEAD,
    "org.opencontainers.image.version": "0.39.15",
    "io.scai.test-deps.fingerprint": "4a7cbb66ad06298e8b6ccc36ebcd0fe98e5a1f245a51eb1aa2a1bab166eaeec4",
    "io.scai.test-deps.lock-sha256": "d4df0f49351c6854bbcb53130959f53aca557e8420bc64470a2698da4900aaf4",
    "io.scai.test-deps.python": PYTHON_PATCH,
    "io.scai.test-deps.platform": PLATFORM,
    "io.scai.test-deps.base-digest": BASE_DIGEST,
}
FAKE_IMAGE_ID = "sha256:" + "1" * 64
FAKE_LOCAL_DIGEST = "sha256:" + "2" * 64
FAKE_REGISTRY_DIGEST = "sha256:" + "3" * 64
UNVERIFIED_EXTRA_LABEL = "com.example.unverified"


def _frontend_source() -> str:
    source = (ROOT / "scripts/scai-frontend-test").read_text(encoding="utf-8")
    return source.split("<<'PYTHON_EOF'\n", 1)[1].rsplit("\nPYTHON_EOF\n", 1)[0]


def _markdown_section(path: Path, heading: str, next_heading_prefix: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(heading)
    end = text.find(f"\n{next_heading_prefix}", start + len(heading))
    if end < 0:
        end = len(text)
    return text[start:end].rstrip()


def _tiered_test_strategy(path: Path) -> str:
    return _markdown_section(path, "#### Tiered Test Strategy", "### ")


def _handover_development_guide() -> str:
    return _markdown_section(ROOT / "docs/HANDOVER.md", "## 11. Development Guide", "## ")


def _lock_entries(path: Path) -> list[tuple[str, str, str]]:
    pattern = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s]+) --hash=sha256:([0-9a-f]{64})$")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        assert match, f"malformed lock line: {line}"
        result.append(match.groups())
    return result


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _manifest() -> dict:
    return json.loads((ROOT / "docker/test-deps/lock-manifest.json").read_text(encoding="utf-8"))


def _probe_executable_directory(directory: Path) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".scai-image-contract-probe-{uuid.uuid4().hex}"
    try:
        probe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        probe.chmod(0o755)
        return subprocess.run(
            [str(probe)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode == 0
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def _copy_image_contract_repo(repo: Path) -> None:
    paths = (
        "scripts/build-test-deps.sh",
        "scripts/verify-test-deps-image.sh",
        "scripts/test-deps-fingerprint.sh",
        "scripts/lock-test-deps.sh",
        "docker/test-deps.Dockerfile",
        "docker/test-deps.Dockerfile.dockerignore",
        "docker/test-deps/requirements-test.in",
        "docker/test-deps/requirements-test.lock",
        "docker/test-deps/requirements-test-nodeps.lock",
        "docker/test-deps/lock-manifest.json",
        "requirements.txt",
        "requirements-nodeps.txt",
        "build-constraints.txt",
    )
    for relative in paths:
        source = ROOT / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_fake_verifier(repo: Path) -> None:
    path = repo / "scripts/verify-test-deps-image.sh"
    path.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "printf '%s\\n' \"$*\" >> \"${FAKE_VERIFY_LOG}\"\n"
        "if [ \"${FAKE_VERIFY_FAIL:-0}\" = 1 ]; then exit 23; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fake_docker_script() -> str:
    return """#!__PYTHON__
import hashlib
import json
import os
import re
import sys
from pathlib import Path

state_path = Path(os.environ["FAKE_DOCKER_STATE"])
log_path = Path(os.environ["FAKE_DOCKER_LOG"])
state = json.loads(state_path.read_text(encoding="utf-8"))
argv = sys.argv[1:]
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\\n")

def save():
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

if argv[:2] == ["image", "inspect"]:
    if "--format" not in argv:
        raise SystemExit(0)
    template = argv[argv.index("--format") + 1]
    reference = argv[-1]
    if "{{.Id}}" in template:
        print(state["image_id"])
    elif "{{json .Config.Labels}}" in template:
        print(json.dumps(state["labels"], sort_keys=True))
    elif ".RepoDigests" in template:
        image_name = reference.rsplit(":", 1)[0] if reference.startswith("ghcr.io/") else state["repo_name"]
        value = f"{image_name}@{state['local_digest']}"
        if "json" in template:
            print(json.dumps([value]))
        else:
            print(value)
    elif "index .Config.Labels" in template:
        match = re.search(r'Labels\\s+"([^"]+)"', template)
        print(state["labels"].get(match.group(1), "") if match else "")
    raise SystemExit(0)

if argv[:2] == ["buildx", "build"]:
    state["build_argv"] = argv
    save()
    raise SystemExit(0)

if argv and argv[0] == "push":
    state["push_argv"] = argv
    save()
    print("pushed")
    raise SystemExit(0)

if argv[:3] == ["buildx", "imagetools", "inspect"]:
    print(state["registry_digest"])
    raise SystemExit(0)

if argv and argv[0] == "run":
    command = argv[-1] if argv else ""
    if command == "id -u":
        print("10001")
    elif command == "cat /opt/scai-test/manifest/runtime-user.txt":
        print("10001 scai-test")
    elif command == "command -v python":
        print("/opt/scai-test/venv/bin/python")
    elif command == "python --version":
        print("Python 3.11.15")
    elif command.startswith("touch "):
        raise SystemExit(1)
    elif command.startswith("sha256sum "):
        match = re.search(r"sha256sum (/opt/scai-test/[^\\s|]+)", command)
        if match:
            relative = match.group(1).removeprefix("/opt/scai-test/")
            source = Path(os.environ["FAKE_REPO"]) / relative
            if relative in {"requirements-test.lock", "requirements-test-nodeps.lock", "lock-manifest.json"}:
                source = Path(os.environ["FAKE_REPO"]) / "docker/test-deps" / relative
            payload = source.read_bytes()
            print(hashlib.sha256(payload).hexdigest())
    elif command.startswith("cd "):
        pass
    elif command.startswith("python "):
        pass
    raise SystemExit(0)

raise SystemExit(0)
""".replace("__PYTHON__", sys.executable)


@contextlib.contextmanager
def _image_contract_harness(tmp_path: Path, *, labels: dict[str, str] | None = None):
    candidates: list[Path] = []
    output_root = Path("/results/output")
    if output_root.is_dir() and os.access(output_root, os.W_OK | os.X_OK):
        candidates.append(output_root / f".scai-image-contract-{uuid.uuid4().hex}")
    candidates.append(tmp_path / f"scai-image-contract-{uuid.uuid4().hex}")
    harness_root = next((candidate for candidate in candidates if _probe_executable_directory(candidate)), None)
    if harness_root is None:
        raise RuntimeError("no executable test-only directory is available for fake Docker")
    repo = harness_root / "repo"
    repo.mkdir()
    _copy_image_contract_repo(repo)
    state_path = harness_root / "docker-state.json"
    log_path = harness_root / "docker.log"
    verify_log = harness_root / "verify.log"
    state_path.write_text(
        json.dumps(
            {
                "image_id": FAKE_IMAGE_ID,
                "labels": labels or {**EXPECTED_IMAGE_LABELS, UNVERIFIED_EXTRA_LABEL: "do-not-publish"},
                "repo_name": "ghcr.io/minzhangcheng/scai-test-deps",
                "local_digest": FAKE_LOCAL_DIGEST,
                "registry_digest": FAKE_REGISTRY_DIGEST,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log_path.write_text("", encoding="utf-8")
    verify_log.write_text("", encoding="utf-8")
    docker = harness_root / "docker"
    docker.write_text(_fake_docker_script(), encoding="utf-8")
    docker.chmod(0o755)
    git = harness_root / "git"
    git.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = rev-parse ] && [ \"$2\" = HEAD ]; then\n"
        "  printf '%s\\n' \"${FAKE_GIT_HEAD}\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{harness_root}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "FAKE_DOCKER_STATE": str(state_path),
        "FAKE_DOCKER_LOG": str(log_path),
        "FAKE_VERIFY_LOG": str(verify_log),
        "FAKE_REPO": str(repo),
        "FAKE_GIT_HEAD": FIXED_HEAD,
    }
    try:
        yield repo, env, state_path, log_path, verify_log, harness_root
    finally:
        shutil.rmtree(harness_root, ignore_errors=True)


def _run_copied_verifier(repo: Path, env: dict[str, str], image_id: str = FAKE_IMAGE_ID) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "scripts/verify-test-deps-image.sh"), image_id],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )


def _run_copied_builder(repo: Path, env: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "scripts/build-test-deps.sh"), *arguments],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )


def _docker_events(path: Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _docker_copy_commands(text: str) -> list[str]:
    commands = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("COPY "):
            index += 1
            continue
        parts = [stripped]
        while parts[-1].endswith("\\"):
            index += 1
            assert index < len(lines), "unterminated Dockerfile COPY"
            parts.append(lines[index].strip())
        commands.append(" ".join(part.rstrip("\\").strip() for part in parts))
        index += 1
    return commands


def _target_test_files() -> list[Path]:
    roots = [ROOT / "backend/tests", ROOT / "clients/python/tests"]
    return sorted(path for root in roots if root.is_dir() for path in root.rglob("*.py"))


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def test_dockerfile_uses_literal_base_in_every_stage():
    text = (ROOT / "docker/test-deps.Dockerfile").read_text(encoding="utf-8")
    from_lines = [line.strip() for line in text.splitlines() if line.strip().startswith("FROM ")]
    expected = f"python:3.11.15-slim-bookworm@{BASE_DIGEST}"
    assert len(from_lines) >= 2
    assert all(expected in line for line in from_lines[:2])


def test_dockerfile_cannot_copy_source_or_define_a_production_command():
    text = (ROOT / "docker/test-deps.Dockerfile").read_text(encoding="utf-8")
    source_allowlist = {
        "build-constraints.txt",
        "docker/test-deps/requirements-test.lock",
        "docker/test-deps/requirements-test-nodeps.lock",
        "docker/test-deps/lock-manifest.json",
    }
    source_paths = []
    for command in _docker_copy_commands(text):
        tokens = command.split()
        assert len(tokens) >= 3
        assert tokens[0] == "COPY"
        if "--from=builder" not in tokens:
            source_paths.extend(tokens[1:-1])
            assert set(tokens[1:-1]) <= source_allowlist
            assert tokens[-1].startswith("/opt/scai-test/")
        assert "frontend/" not in command and "backend/" not in command
    assert set(source_paths) == source_allowlist
    assert not re.search(r"^(ENTRYPOINT|CMD|HEALTHCHECK|EXPOSE)\b", text, re.MULTILINE)
    for forbidden in ("/dep", "/extra", "/data", "/workspace", "/app"):
        assert forbidden not in text


def test_dockerignore_admits_only_test_dependency_inputs():
    lines = [line.strip() for line in (ROOT / "docker/test-deps.Dockerfile.dockerignore").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    assert lines == [
        "*",
        "**",
        "!build-constraints.txt",
        "!docker/",
        "!docker/test-deps/",
        "!docker/test-deps/requirements-test.in",
        "!docker/test-deps/requirements-test.lock",
        "!docker/test-deps/requirements-test-nodeps.lock",
        "!docker/test-deps/lock-manifest.json",
    ]


def test_test_input_has_only_candidate_pytest_pins():
    values = [
        line.strip()
        for line in (ROOT / "docker/test-deps/requirements-test.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert values == ["pytest==9.1.1", "pytest-asyncio==1.3.0", "pytest-benchmark==5.2.3"]


def test_target_tests_have_no_pytest_benchmark_consumer():
    for path in _target_test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = [alias.name for alias in node.names]
                assert "pytest_benchmark" not in imported
                assert all(not name.startswith("pytest_benchmark.") for name in imported)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                argument_names = {argument.arg for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
                if node.args.vararg:
                    argument_names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    argument_names.add(node.args.kwarg.arg)
                assert "benchmark" not in argument_names, f"benchmark fixture consumer: {path}:{node.lineno}"
                for decorator in node.decorator_list:
                    assert _dotted_name(decorator) != "pytest.mark.benchmark", f"benchmark decorator consumer: {path}:{node.lineno}"


def test_each_lock_line_is_exact_and_hashed():
    for path in (ROOT / "docker/test-deps/requirements-test.lock", ROOT / "docker/test-deps/requirements-test-nodeps.lock"):
        entries = _lock_entries(path)
        assert entries == sorted(entries, key=lambda item: (_normalized(item[0]), item[1]))
        assert len({_normalized(item[0]) for item in entries}) == len(entries)
        if path.name == "requirements-test.lock":
            versions = {_normalized(name): version for name, version, _ in entries}
            assert {name: versions.get(name) for name in ("pytest", "pytest-asyncio", "pytest-benchmark")} == {
                "pytest": "9.1.1",
                "pytest-asyncio": "1.3.0",
                "pytest-benchmark": "5.2.3",
            }


def test_lock_manifest_inputs_tool_installer_inventory_and_allowlist_agree():
    manifest = _manifest()
    assert manifest["schema"] == 1
    assert manifest["platform"] == {"python": PYTHON_PATCH, "os": PLATFORM}
    assert manifest["base_image_digest"] == BASE_DIGEST
    expected_inputs = {
        "requirements.txt",
        "requirements-nodeps.txt",
        "build-constraints.txt",
        "docker/test-deps/requirements-test.in",
        "scripts/lock-test-deps.sh",
    }
    assert set(manifest["inputs"]) == expected_inputs
    for relative, digest in manifest["inputs"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert manifest["lock_tool"]["path"] == "scripts/lock-test-deps.sh"
    assert manifest["lock_tool"]["sha256"] == hashlib.sha256((ROOT / "scripts/lock-test-deps.sh").read_bytes()).hexdigest()
    assert manifest["resolver_bootstrap_specs"] == ["pip>=25.3,<27", "setuptools<82", "wheel"]
    assert set(manifest["resolver"]) == {"pip", "setuptools", "wheel"}
    assert all(isinstance(manifest["resolver"][name], str) and manifest["resolver"][name] for name in manifest["resolver"])
    assert manifest["installer"]["argv"] == ["/opt/scai-test/venv/bin/python", "-m", "pip"]
    assert manifest["allowed_pip_check"] == [EXPECTED_PIP_CHECK]


def test_lock_allowlist_is_exact_and_cannot_be_manually_widened():
    manifest = _manifest()
    assert manifest["allowed_pip_check"] == [EXPECTED_PIP_CHECK]
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert "EXPECTED_PIP_CHECK" not in source
    assert "decoupler 1.8.0 has requirement numba<0.61.0,>=0.60.0, but you have numba 0.61.2." in source
    assert not re.search(r"^\s*assert\s+", source, re.MULTILINE)


def test_lock_resolver_applies_build_constraints_to_pep517_isolation():
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert "--build-constraint" in source
    assert "PIP_CONSTRAINT" in source
    assert "setuptools<82" in (ROOT / "build-constraints.txt").read_text(encoding="utf-8")


def test_resolver_clean_installer_and_image_builder_share_tracked_build_constraint():
    lock_source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    docker_source = (ROOT / "docker/test-deps.Dockerfile").read_text(encoding="utf-8")
    build_source = (ROOT / "scripts/build-test-deps.sh").read_text(encoding="utf-8")
    assert (ROOT / "build-constraints.txt").read_text(encoding="utf-8").splitlines()[-1] == "setuptools<82"
    assert "constraints = repo / 'build-constraints.txt'" in lock_source
    assert lock_source.count("constraints = repo / 'build-constraints.txt'\n") == 1
    assert "resolver_python = Path(sys.executable)" in lock_source
    assert "installer_env['PIP_CONSTRAINT'] = str(constraints)" in lock_source
    assert "installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes'" in lock_source
    assert "--build-constraint', str(constraints)" in lock_source
    assert "COPY build-constraints.txt /opt/scai-test/build-constraints.txt" in docker_source
    assert "COPY --from=builder /opt/scai-test/build-constraints.txt /opt/scai-test/build-constraints.txt" in docker_source
    assert docker_source.count("PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt") == 2
    assert "--file docker/test-deps.Dockerfile" in build_source
    assert '"${build_args[@]}" .' in build_source


def test_image_builder_verifies_constraint_hash_before_lock_install():
    source = (ROOT / "docker/test-deps.Dockerfile").read_text(encoding="utf-8")
    start = source.index("RUN mkdir -p /opt/scai-test/manifest")
    end = source.index("\n\nRUN python -m pip freeze", start)
    install_block = source[start:end]
    assert '["inputs"]["build-constraints.txt"]' in install_block
    assert 'actual_hash="$(sha256sum /opt/scai-test/build-constraints.txt' in install_block
    assert '[ "$actual_hash" = "$expected_hash" ]' in install_block
    assert install_block.index('[ "$actual_hash" = "$expected_hash" ]') < install_block.index("PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt")
    assert install_block.count("PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt") == 2
    assert install_block.count("--require-hashes") == 2
    assert "/opt/scai-test/manifest/build-constraints.sha256" in source


def test_prelock_check_proves_three_constraint_consumers_before_missing_lock():
    result = subprocess.run(
        ["scripts/lock-test-deps.sh", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert "constraint source contract: PASS" in output
    if result.returncode:
        assert "missing" in output


def test_lock_resolver_identity_is_separate_from_clean_installer_identity():
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert "resolver_python = Path(sys.executable)" in source
    assert "resolver_pip_spec = 'pip>=25.3,<27'" in source
    assert "resolver_bootstrap_specs = [resolver_pip_spec, 'setuptools<82', 'wheel']" in source
    assert "str(resolver_python), '-m', 'pip', 'install'" in source
    assert "str(python), '-m', 'pip', 'install', '--require-hashes'" in source
    installer_start = source.index("installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes'")
    installer_end = source.index("final_freeze = output", installer_start)
    assert "'--upgrade'" not in source[installer_start:installer_end]
    assert "'resolver': resolver" in source
    assert "'installer': {'argv': ['/opt/scai-test/venv/bin/python', '-m', 'pip']" in source


def test_lock_resolver_verifies_report_hashes_against_downloaded_bytes():
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert "pip', 'download'" in source
    assert "downloaded_hashes" in source
    assert "reported artifact bytes were not downloaded" in source


def test_lock_resolver_reproduces_builder_sdist_tool_surface():
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert '"apt-get", "install", "-y", "--no-install-recommends", "build-essential", "curl", "gosu", "pkg-config"' in source
    assert '"apt-get", "update"' in source


def test_clean_installer_keeps_identity_and_receives_build_constraints():
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert "installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes'" in source
    assert "installer_env['PIP_CONSTRAINT'] = str(constraints)" in source
    installer_start = source.index("installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes'")
    installer_end = source.index("final_freeze = output", installer_start)
    assert "'--upgrade'" not in source[installer_start:installer_end]


def test_lock_resolver_cleanup_is_safe_after_function_return():
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    assert 'trap \'rm -rf "$output_dir"\' EXIT' not in source
    assert "LOCK_OUTPUT_DIR" in source
    assert "trap cleanup_lock_output EXIT" in source
    assert "prepare_lock_output()" in source
    assert 'chmod 0700 -- "$LOCK_OUTPUT_DIR"' in source
    assert 'mkdir -- "$LOCK_OUTPUT_DIR/$download_name"' in source
    assert 'chmod 0777 -- "$LOCK_OUTPUT_DIR/$download_name"' in source
    assert 'rm -rf -- "$LOCK_OUTPUT_DIR"' in source


def test_freeze_records_initial_and_final_actual_inventories():
    manifest = _manifest()
    for key in ("initial_actual_bootstrap_installers", "expected_final_freeze", "final_actual_freeze"):
        assert isinstance(manifest[key], list)
        assert manifest[key] == sorted(manifest[key], key=lambda item: item["name"])
        assert all(set(item) == {"name", "version"} for item in manifest[key])
    assert manifest["expected_final_freeze"] == manifest["final_actual_freeze"]


def test_freeze_uses_locked_version_for_replaced_bootstrap_names():
    manifest = _manifest()
    locked = {_normalized(item["name"]): item["version"] for item in manifest["requirements"] + manifest["nodeps"]}
    initial = {item["name"]: item["version"] for item in manifest["initial_actual_bootstrap_installers"]}
    expected = sorted(
        [{"name": name, "initial_version": initial[name], "locked_version": locked[name]} for name in initial.keys() & locked.keys()],
        key=lambda item: item["name"],
    )
    assert manifest["replaced_bootstrap_installers"] == expected


def test_freeze_retains_only_unlocked_bootstrap_extras():
    manifest = _manifest()
    locked = {_normalized(item["name"]) for item in manifest["requirements"] + manifest["nodeps"]}
    expected = sorted(
        [item for item in manifest["initial_actual_bootstrap_installers"] if item["name"] not in locked],
        key=lambda item: item["name"],
    )
    assert manifest["final_unlocked_bootstrap_extras"] == expected


def test_fingerprint_covers_every_declared_input():
    source = (ROOT / "scripts/test-deps-fingerprint.sh").read_text(encoding="utf-8")
    for required in ("requirements.txt", "requirements-nodeps.txt", "build-constraints.txt", "requirements-test.in", "requirements-test.lock", "requirements-test-nodeps.lock", "lock-manifest.json", "scripts/lock-test-deps.sh"):
        assert required in source
    assert "FINGERPRINT=" in source and "TAG=" in source


def test_build_is_local_by_default_and_never_edits_pointer():
    source = (ROOT / "scripts/build-test-deps.sh").read_text(encoding="utf-8")
    assert "--load" in source
    assert "--push" in source
    assert "docker/test-deps-image.json" not in source
    assert ".scai-test-deps-image.json" in source


def test_verifier_requires_exact_required_oci_label_values(tmp_path):
    with _image_contract_harness(tmp_path) as (repo, env, _state_path, _log, _verify_log, _root):
        baseline = _run_copied_verifier(repo, env)
        assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    for key in EXPECTED_IMAGE_LABELS:
        with _image_contract_harness(tmp_path) as (repo, env, state_path, _log, _verify_log, _root):
            labels = dict(EXPECTED_IMAGE_LABELS)
            labels[key] = "mutated-label"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["labels"] = labels
            state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
            result = _run_copied_verifier(repo, env)
            assert result.returncode != 0, (
                f"verifier accepted mutated OCI label {key}:\nstdout={result.stdout}\nstderr={result.stderr}"
            )


def test_build_requires_fixed_head_and_verifies_immutable_id_before_identity_write(tmp_path):
    with _image_contract_harness(tmp_path) as (repo, env, _state_path, log_path, verify_log, _root):
        wrong_head_env = {**env, "FAKE_GIT_HEAD": "deadbeef"}
        result = _run_copied_builder(repo, wrong_head_env, "--force")
        assert result.returncode != 0
        assert _docker_events(log_path) == [], "fixed-HEAD rejection must precede every Docker call"
        assert not (repo / ".scai-test-deps-image.json").exists()

    with _image_contract_harness(tmp_path) as (repo, env, _state_path, _log, verify_log, _root):
        _write_fake_verifier(repo)
        failing_verify_env = {**env, "FAKE_VERIFY_FAIL": "1"}
        result = _run_copied_builder(repo, failing_verify_env, "--force")
        assert result.returncode != 0, result.stdout + result.stderr
        assert verify_log.read_text(encoding="utf-8").strip()
        assert not (repo / ".scai-test-deps-image.json").exists(), "identity must follow immutable verification"


def test_default_build_never_pushes_and_records_only_verified_labels(tmp_path):
    source = (ROOT / "scripts/build-test-deps.sh").read_text(encoding="utf-8")
    verify_call = source.index("verify-test-deps-image.sh")
    identity_write = source.index('python3 - "$LOCAL_IDENTITY"')
    assert verify_call < identity_write
    with _image_contract_harness(tmp_path) as (repo, env, _state_path, log_path, verify_log, _root):
        _write_fake_verifier(repo)
        result = _run_copied_builder(repo, env, "--force")
        assert result.returncode == 0, result.stdout + result.stderr
        events = _docker_events(log_path)
        build = next(event for event in events if event[:3] == ["buildx", "build", "--file"])
        assert "--load" in build and "--push" not in build
        assert not any(event and event[0] == "push" for event in events)
        assert verify_log.read_text(encoding="utf-8").strip()
        identity = json.loads((repo / ".scai-test-deps-image.json").read_text(encoding="utf-8"))
        assert identity["labels"] == EXPECTED_IMAGE_LABELS
        assert UNVERIFIED_EXTRA_LABEL not in identity["labels"]


def test_authorized_push_uses_verified_local_content_and_cross_checks_registry_digest(tmp_path):
    with _image_contract_harness(tmp_path) as (repo, env, state_path, log_path, verify_log, _root):
        _write_fake_verifier(repo)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["local_digest"] = FAKE_REGISTRY_DIGEST
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        result = _run_copied_builder(repo, env, "--force", "--push")
        assert result.returncode == 0, result.stdout + result.stderr
        events = _docker_events(log_path)
        build_index = next(index for index, event in enumerate(events) if event[:3] == ["buildx", "build", "--file"])
        push_index = next(index for index, event in enumerate(events) if event and event[0] == "push")
        inspect_indexes = [index for index, event in enumerate(events) if event[:3] == ["buildx", "imagetools", "inspect"]]
        repo_digest_inspects = [
            event for event in events
            if event[:2] == ["image", "inspect"] and any(".RepoDigests" in argument for argument in event)
        ]
        assert "--load" in events[build_index] and "--push" not in events[build_index]
        assert build_index < push_index < inspect_indexes[-1]
        assert len(repo_digest_inspects) == 1 and repo_digest_inspects[0][-1] == FAKE_IMAGE_ID
        assert verify_log.read_text(encoding="utf-8").strip()
        assert f"@{FAKE_REGISTRY_DIGEST}" in (result.stdout + result.stderr)

        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["registry_digest"] = "sha256:" + "4" * 64
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        mismatch = _run_copied_builder(repo, env, "--force", "--push")
        assert mismatch.returncode != 0
        assert "Registry digest (proposal only)" not in (mismatch.stdout + mismatch.stderr)
        assert f"@{state['registry_digest']}" not in (mismatch.stdout + mismatch.stderr)


def test_verifier_requires_immutable_ref_and_scientific_imports():
    source = (ROOT / "scripts/verify-test-deps-image.sh").read_text(encoding="utf-8")
    assert "^sha256:[0-9a-f]{64}$" in source
    assert "@sha256:[0-9a-f]{64}" in source
    for name in ("anndata", "zarr", "dask_image", "pytest_benchmark"):
        assert name in source
    for version in ("pytest", "pytest-asyncio", "pytest-benchmark", "9.1.1", "1.3.0", "5.2.3"):
        assert version in source
    assert "build-constraints.txt" in source
    assert "expected_final_freeze" in source
    assert "sha256sum -c manifest/build-constraints.sha256" in source
    assert "final-os-packages.txt" in source
    assert "--pull=never" in source and "--network none" in source and "--user 10001" in source


def test_verifier_proves_namespace_search_locations_and_distribution_files():
    source = (ROOT / "scripts/verify-test-deps-image.sh").read_text(encoding="utf-8")
    assert "importlib.util.find_spec" in source
    assert "spec.submodule_search_locations" in source
    assert "distribution.files" in source
    assert "distribution.locate_file" in source
    assert "distribution file is missing" in source
    assert "relative_to(venv_root)" in source
    assert '"langgraph": "langgraph"' in source


def test_runner_consumes_the_build_identity_tag_field():
    runner = (ROOT / "scripts/scai-test").read_text(encoding="utf-8")
    builder = (ROOT / "scripts/build-test-deps.sh").read_text(encoding="utf-8")
    assert 'value.get("tag")' in runner
    assert '"tag": tag' in builder
    assert 'value.get("discovery_tag")' not in runner


def test_final_stage_records_and_verifies_post_setup_os_inventory():
    text = (ROOT / "docker/test-deps.Dockerfile").read_text(encoding="utf-8")
    assert text.index("useradd") < text.index("final-os-packages.txt") < text.index("hashes.txt")
    assert "dpkg-query" in text and "runtime-user.txt" in text


def test_final_runtime_has_git_and_exact_requested_apt_packages():
    text = (ROOT / "docker/test-deps.Dockerfile").read_text(encoding="utf-8")
    installs = re.findall(r"apt-get install -y --no-install-recommends ([^\\\n]+)", text)
    assert {tuple(item.split()) for item in installs} == {
        ("build-essential", "curl", "gosu", "pkg-config"),
        ("build-essential", "curl", "git", "gosu", "pkg-config"),
    }
    assert "git --version" in (ROOT / "scripts/verify-test-deps-image.sh").read_text(encoding="utf-8") or "git" in text


def test_pointer_is_null_or_a_complete_matching_registry_identity():
    pointer = json.loads((ROOT / "docker/test-deps-image.json").read_text(encoding="utf-8"))
    assert set(pointer) == {"schema", "image", "digest", "fingerprint", "lock_sha256", "selected_at_utc"}
    assert pointer["schema"] == 1
    if pointer["image"] is None:
        assert all(pointer[key] is None for key in ("digest", "fingerprint", "lock_sha256", "selected_at_utc"))
    else:
        assert "@" not in pointer["image"]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", pointer["digest"])
        assert re.fullmatch(r"[0-9a-f]{64}", pointer["fingerprint"])
        assert re.fullmatch(r"[0-9a-f]{64}", pointer["lock_sha256"])


def test_non_null_pointer_keeps_name_and_digest_in_separate_fields():
    source = (ROOT / "scripts/verify-test-deps-image.sh").read_text(encoding="utf-8")
    assert "@sha256:" in source
    assert 'value["image"] + "@" + value["digest"]' in (ROOT / "scripts/scai-test").read_text(encoding="utf-8") if (ROOT / "scripts/scai-test").exists() else True


def test_new_direct_shell_scripts_are_0755_and_link_helper_remains_0644():
    for relative in ("scripts/lock-test-deps.sh", "scripts/test-deps-fingerprint.sh", "scripts/build-test-deps.sh", "scripts/verify-test-deps-image.sh", "scripts/scai-frontend-test"):
        assert stat.S_IMODE((ROOT / relative).stat().st_mode) == 0o755
    assert stat.S_IMODE((ROOT / "scripts/link_frontend_node_modules.sh").stat().st_mode) == 0o644


def test_production_contracts_survive_python_optimize():
    result = subprocess.run(["scripts/lock-test-deps.sh", "--check"], cwd=ROOT, env={**os.environ, "PYTHONOPTIMIZE": "1"}, text=True, capture_output=True)
    output = f"{result.stdout}\n{result.stderr}"
    assert "constraint source contract: PASS" in output
    if result.returncode:
        assert "missing" in output


def test_production_files_do_not_import_test_image_logic():
    for relative in ("Dockerfile", "docker-entrypoint.sh", "start.sh", "build.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "scai-test-deps" not in text
        assert "lock-test-deps.sh" not in text


def test_runner_and_collection_plugin_are_source_owned_contracts():
    runner = (ROOT / "scripts/scai-test").read_text(encoding="utf-8")
    plugin = (ROOT / "scripts/scai_pytest_collection.py").read_text(encoding="utf-8")
    assert "--pull=never" in runner
    assert "--network none" in runner
    assert "--read-only" in runner
    assert "--user 10001" in runner
    assert "nodeids_sha256" in plugin
    assert "--scai-collection-json" in plugin
    assert "--scai-outcomes-json" in plugin
    assert "os.fchmod" in plugin
    assert "junit_target" in plugin


def test_runner_preserves_unaccepted_diagnostics_until_atomic_publication():
    runner = (ROOT / "scripts/scai-test").read_text(encoding="utf-8")
    tree = (ROOT / "scripts/scai_test_tree.py").read_text(encoding="utf-8")
    assert "PUBLISHED=0" in runner
    assert "DIAGNOSTIC_REASON" in runner
    assert 'unaccepted diagnostic: $EVIDENCE_STAGE' in runner
    assert 'rm -rf "$INCOMING_DIR"' not in runner
    assert 'rm -rf "$EVIDENCE_STAGE"' in runner
    assert 'if [[ "$PUBLISHED" -eq 1 ]]' in runner
    assert "publish_evidence" in runner
    assert "fcntl.flock(lock_fd, fcntl.LOCK_EX)" in tree
    assert "os.O_EXCL" in tree and "os.rename(incoming, final)" in tree
    assert "_lexists(incoming) or _lexists(final)" in tree
    assert "shutil.rmtree" not in tree
    assert runner.index("publish_evidence") < runner.index("PUBLISHED=1")


def test_runner_does_not_use_host_pytest_or_unbounded_profile_roster():
    runner = (ROOT / "scripts/scai-test").read_text(encoding="utf-8")
    assert "python3 -m pytest" not in runner
    assert "fast|client|kernel|backend-full" not in runner
    profiles = __import__("tomllib").loads((ROOT / "test-profiles.toml").read_text(encoding="utf-8"))
    assert set(profiles["profiles"]) == {"fast", "client", "kernel", "backend-full"}


def test_frontend_fixture_is_the_exact_npm9_layout_contract():
    fixture = json.loads((ROOT / "backend/tests/fixtures/npm9-layout.json").read_text(encoding="utf-8"))
    assert set(fixture) == {"schema", "directories", "bin_links", "packages", "optional_lock_packages", "root_hidden_lock", "npm_major"}
    assert fixture["schema"] == 1 and fixture["npm_major"] == 9
    assert fixture["root_hidden_lock"] == "node_modules/.package-lock.json"
    assert fixture["directories"] == sorted(fixture["directories"])
    assert fixture["bin_links"] == {
        "node_modules/.bin/vitest": "../vitest/vitest.mjs",
        "node_modules/plain/node_modules/.bin/child-tool": "../child/cli.js",
    }
    optional = fixture["optional_lock_packages"]
    assert set(optional) == {
        "node_modules/optional-os",
        "node_modules/optional-cpu",
        "node_modules/optional-empty",
        "node_modules/optional-unreported",
    }
    assert all(item.get("optional") is True for item in optional.values())
    assert optional["node_modules/optional-os"]["os"] == ["darwin"]
    assert optional["node_modules/optional-cpu"]["cpu"] == ["arm64"]
    assert "os" not in optional["node_modules/optional-empty"] and "cpu" not in optional["node_modules/optional-empty"]
    assert "os" not in optional["node_modules/optional-unreported"] and "cpu" not in optional["node_modules/optional-unreported"]
    for path, package in optional.items():
        assert set(package) >= {"optional", "version", "resolved", "integrity"}, path
        assert package["resolved"].startswith("https://registry.npmjs.org/")
        assert package["integrity"].startswith("sha512-")
    assert set(fixture["packages"]) == {
        "node_modules/@scope/pkg",
        "node_modules/plain",
        "node_modules/plain/node_modules/child",
        "node_modules/vitest",
    }
    for path, package in fixture["packages"].items():
        assert set(package) >= {"name", "version", "resolved", "integrity"}
        assert package["name"] == path.rsplit("node_modules/", 1)[-1]
        assert package["resolved"].startswith("https://registry.npmjs.org/")
        assert package["integrity"].startswith("sha512-")


def test_frontend_runner_has_the_separate_authorized_install_boundary():
    source = (ROOT / "scripts/scai-frontend-test").read_text(encoding="utf-8")
    assert 'argv != ["--install-and-test"]' in source
    assert "package_json_sha256" in source and "package_lock_sha256" in source and "input_key" in source
    assert "frontend-installations" in source and 'Path("/tmp") / f"scai-frontend-' in source
    assert '[str(self.npm), "ci"]' in source
    assert '[str(self.npm), "ls", "--all", "--json"]' in source
    assert '[str(self.npm), "test", "--", "--reporter=json"' in source
    assert "npm run build" not in source and "docker" not in source


def test_frontend_runner_owns_environment_config_and_cleanup_contract():
    source = (ROOT / "scripts/scai-frontend-test").read_text(encoding="utf-8")
    tree = ast.parse(_frontend_source(), filename="scripts/scai-frontend-test")
    command_env = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "command_env")
    returned = next(node for node in ast.walk(command_env) if isinstance(node, ast.Return))
    assert isinstance(returned.value, ast.Dict)
    keys = {item.value for item in returned.value.keys if isinstance(item, ast.Constant)}
    assert keys == {"PATH", "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "LANG", "LC_ALL", "npm_config_cache", "npm_config_userconfig", "npm_config_globalconfig", "npm_config_registry", "npm_config_audit", "npm_config_fund", "npm_config_update_notifier"}
    assert "NODE_OPTIONS" not in source and "NPM_CONFIG_" not in source
    for marker in ("empty_config_file", "verify_config_files", "signal.SIGHUP", "signal.SIGINT", "signal.SIGTERM", "restore_link", "os.replace"):
        assert marker in source


def test_frontend_runner_hashes_and_tokenizes_all_published_artifacts():
    source = (ROOT / "scripts/scai-frontend-test").read_text(encoding="utf-8")
    for marker in ("artifact_manifest", "sha256_file", "json_hash", "sanitize_output", "source_after", "npm-config-values.json", "npm-config-sources.json", "npm-tree.json", "vitest.json"):
        assert marker in source
    evidence_start = source.index("def evidence")
    evidence_end = source.index("def finish", evidence_start)
    evidence = source[evidence_start:evidence_end]
    assert "tokenized(self.config_values" in evidence
    assert "tokenized(self.config_sources" in evidence
    assert "evidence.json" not in evidence


def test_frontend_runner_tree_and_vitest_fail_closed_contract():
    source = (ROOT / "scripts/scai-frontend-test").read_text(encoding="utf-8")
    for marker in ("lockfileVersion"):
        assert marker in source
    for marker in ("installed/lock drift", "locked package is absent", "integrity", "resolved", "hidden lock package subset differs", "set(declarations) != bin_dirs", "vitest shim is missing", "success does not agree with details/process exit", "suite-level import/setup/transform application failure"):
        assert marker in source


def test_frontend_cli_contract_tests_cover_the_exact_plan_test_names():
    source = (ROOT / "backend/tests/test_scai_frontend_test_cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="backend/tests/test_scai_frontend_test_cli.py")
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")}
    required = {
        "test_frontend_runner_hashes_both_exact_inputs_before_install",
        "test_frontend_runner_uses_fresh_run_directory_under_input_key",
        "test_frontend_lock_order_is_read_hash_global_namespace_keyed_recheck_then_run",
        "test_missing_input_namespace_is_created_only_under_global_lock_before_keyed_open",
        "test_keyed_lock_rechecks_source_and_inputs_before_fresh_run_or_copy",
        "test_frontend_input_preparation_failure_stops_before_npm",
        "test_frontend_runner_rejects_source_npmrc_before_install",
        "test_frontend_runner_refuses_real_source_node_modules_directory",
        "test_npm_ci_failure_prevents_ls_and_test_and_writes_failure_evidence",
        "test_npm_ls_failure_prevents_test",
        "test_tree_canonicalization_rejects_extra_or_version_drift",
        "test_tree_canonicalization_records_lock_resolved_and_integrity",
        "test_tree_walker_accepts_captured_npm9_hidden_lock_scoped_and_nested_layout",
        "test_tree_walker_rejects_package_metadata_symlink_special_or_realpath_escape",
        "test_bin_walker_records_vitest_shim_and_declared_contained_target",
        "test_bin_walker_requires_declaration_and_discovered_level_sets_to_match",
        "test_bin_walker_rejects_escape_wrong_target_extra_missing_shim_or_nested_bin_dir",
        "test_npm_test_uses_the_same_verified_vitest_shim",
        "test_frontend_runner_invokes_0644_link_helper_through_bash",
        "test_source_node_modules_link_is_restored_after_success_or_failure",
        "test_global_and_input_key_locks_serialize_concurrent_frontend_runs",
        "test_frontend_environment_clears_node_options_and_all_npm_config_inputs",
        "test_frontend_uses_isolated_home_cache_and_userconfig",
        "test_npm_config_commands_have_exact_argv_env_cwd_exit_and_output_paths",
        "test_npm_config_values_models_command_owned_json_true_separately",
        "test_npm_config_values_and_sources_agree_on_exact_effective_overrides",
        "test_npm_config_plain_env_source_annotation_is_optional_but_if_present_exact",
        "test_npm_config_sources_reject_unexpected_project_user_global_or_env_source",
        "test_npm_config_artifacts_and_index_hashes_reject_tampering",
        "test_npm_secret_classifier_allows_auth_type_web_and_safe_auth_key_names",
        "test_npm_secret_classifier_rejects_exact_and_registry_scoped_credentials",
        "test_frontend_evidence_redacts_paths_and_rejects_secret_config_keys",
        "test_vitest_json_counts_are_machine_readable_and_required",
        "test_vitest_assertion_and_suite_details_aggregate_to_top_level_counts",
        "test_vitest_suite_level_import_failure_is_valid_application_failure_evidence",
        "test_frontend_evidence_records_commands_versions_hashes_tree_counts_and_source",
        "test_frontend_evidence_does_not_self_hash",
        "test_frontend_runner_never_invokes_npm_run_build_or_a_node_image",
    }
    assert required <= names


def test_frontend_fake_harness_probes_exec_mount_and_finalizes_tool_roots():
    source = (ROOT / "backend/tests/test_scai_frontend_test_cli.py").read_text(encoding="utf-8")
    for marker in ("_probe_executable_directory", 'Path("/results/output")', "_FAKE_TOOL_ROOTS", "@pytest.fixture(autouse=True)", "shutil.rmtree(root, ignore_errors=True)"):
        assert marker in source
    assert 'subprocess.run([str(probe)]' in source


def test_active_test_surfaces_share_the_immutable_runner_contract():
    agents = _tiered_test_strategy(ROOT / "AGENTS.md")
    claude = _tiered_test_strategy(ROOT / "CLAUDE.md")
    handover = _handover_development_guide()
    assert agents == claude
    required = (
        "scripts/scai-test fast",
        "scripts/scai-test client",
        "scripts/scai-test kernel",
        "scripts/scai-test backend-full",
        "scripts/scai-test --",
        "scripts/scai-frontend-test --install-and-test",
    )
    for label, section in (("AGENTS.md", agents), ("CLAUDE.md", claude), ("docs/HANDOVER.md", handover)):
        for command in required:
            assert command in section, f"{label} is missing active test command: {command}"
        command_lines = [line.strip() for line in section.splitlines() if line.strip()]
        assert not any(line.startswith(("pytest ", "python -m pytest")) for line in command_lines), label
    assert "focused edit-loop evidence only" in agents
    assert "not backend-full or release evidence" in agents


def test_test_strategy_documents_the_active_evidence_contract():
    text = (ROOT / "docs/test-strategy.md").read_text(encoding="utf-8")
    required = (
        "test-profiles.toml",
        "--pull=never",
        "--read-only",
        "--network none",
        "--user 10001",
        "--platform linux/amd64",
        "--tmpfs /tmp:rw,size=2g",
        "NUMBA_DISABLE_JIT=1",
        "SCAI_RUN_KERNEL_TESTS=1",
        "SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB=1",
        "pytest==9.1.1",
        "pytest-asyncio==1.3.0",
        "pytest-benchmark==5.2.3",
        "infercnvpy",
        "scripts/lock-test-deps.sh --check",
        "build-constraints.txt",
        "dependency context",
        "hash-before-install",
        "scripts/scai-frontend-test --install-and-test",
        "npm ci",
        "npm ls --all --json",
        "node-ID bijection",
        "pytest 9.1.1\nphase-to-JUnit",
        "SCAI_ENABLE_SNAPSHOT_V2=true",
        "production default, which remains\ndisabled",
        "host `pytest`",
        "baseline",
    )
    for marker in required:
        assert marker in text, f"test strategy is missing required contract: {marker}"
