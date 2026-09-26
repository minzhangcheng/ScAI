"""Docker-free contracts for the F0 immutable dependency environment."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
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
    "io.scai.test-deps.fingerprint": "",
    "io.scai.test-deps.lock-sha256": "",
    "io.scai.test-deps.python": PYTHON_PATCH,
    "io.scai.test-deps.platform": PLATFORM,
    "io.scai.test-deps.base-digest": BASE_DIGEST,
}
FAKE_IMAGE_ID = "sha256:" + "1" * 64


def _expected_image_labels(repo: Path) -> dict[str, str]:
    """Bind fake image bytes to this checkout, not a historical dependency lock."""
    output = subprocess.check_output(
        ["bash", str(repo / "scripts/test-deps-fingerprint.sh")], cwd=repo, text=True,
    )
    fingerprint = dict(line.split("=", 1) for line in output.splitlines())["FINGERPRINT"]
    return {
        **EXPECTED_IMAGE_LABELS,
        "io.scai.test-deps.fingerprint": fingerprint,
        "io.scai.test-deps.lock-sha256": hashlib.sha256(
            (repo / "docker/test-deps/requirements-test.lock").read_bytes()
        ).hexdigest(),
    }


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

if argv[:2] == ["image", "inspect"]:
    if "--format" in argv:
        template = argv[argv.index("--format") + 1]
        match = re.search(r'Labels\\s+"([^"]+)"', template)
        print(state["labels"].get(match.group(1), "") if match else "")
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
def _image_contract_harness(tmp_path: Path):
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
    state_path.write_text(
        json.dumps(
            {
                "labels": _expected_image_labels(repo),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log_path.write_text("", encoding="utf-8")
    docker = harness_root / "docker"
    docker.write_text(_fake_docker_script(), encoding="utf-8")
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{harness_root}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "FAKE_DOCKER_STATE": str(state_path),
        "FAKE_DOCKER_LOG": str(log_path),
        "FAKE_REPO": str(repo),
    }
    try:
        yield repo, env, state_path
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


def test_verifier_requires_exact_required_oci_label_values(tmp_path):
    with _image_contract_harness(tmp_path) as (repo, env, _state_path):
        baseline = _run_copied_verifier(repo, env)
        assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    for key in EXPECTED_IMAGE_LABELS:
        with _image_contract_harness(tmp_path) as (repo, env, state_path):
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["labels"][key] = "mutated-label"
            state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
            result = _run_copied_verifier(repo, env)
            assert result.returncode != 0, (
                f"verifier accepted mutated OCI label {key}:\nstdout={result.stdout}\nstderr={result.stderr}"
            )


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


def test_preserving_lock_update_keeps_all_existing_versions(tmp_path):
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    block = source.split("preserve_existing = os.environ.get", 1)[1].split("\n\ndef run(", 1)[0]
    block = "preserve_existing = os.environ.get" + block
    repo = tmp_path / "repo"
    out = tmp_path / "output"
    (repo / "docker/test-deps").mkdir(parents=True)
    out.mkdir()
    main = repo / "docker/test-deps/requirements-test.lock"
    nodeps = repo / "docker/test-deps/requirements-test-nodeps.lock"
    main.write_text("# lock\nnumpy==1.26.4 --hash=sha256:" + "a" * 64 + "\n")
    nodeps.write_text("decoupler==1.8.0 --hash=sha256:" + "b" * 64 + "\n")
    namespace = {
        "os": type("Environment", (), {"environ": {"SCAI_LOCK_PRESERVE_EXISTING": "1"}}),
        "repo": repo, "out": out, "hashlib": hashlib, "re": re,
    }
    exec(compile(block, "lock-preservation", "exec"), namespace)
    assert (out / "existing-versions.txt").read_text().splitlines() == [
        "numpy==1.26.4", "decoupler==1.8.0",
    ]
    assert namespace["preserved_lock_hashes"] == {
        "docker/test-deps/requirements-test.lock": hashlib.sha256(main.read_bytes()).hexdigest(),
        "docker/test-deps/requirements-test-nodeps.lock": hashlib.sha256(nodeps.read_bytes()).hexdigest(),
    }
    assert "args += ['-c', str(existing_constraints)]" in source
    assert "download_args += ['-c', str(existing_constraints)]" in source


def test_default_lock_update_does_not_constrain_existing_versions(tmp_path):
    source = (ROOT / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    block = source.split("preserve_existing = os.environ.get", 1)[1].split("\n\ndef run(", 1)[0]
    namespace = {
        "os": type("Environment", (), {"environ": {}}),
        "repo": tmp_path / "missing-repo", "out": tmp_path,
    }
    exec(compile("preserve_existing = os.environ.get" + block, "lock-preservation", "exec"), namespace)
    assert namespace["preserve_existing"] is False
    assert not (tmp_path / "existing-versions.txt").exists()
