#!/bin/bash
set -euo pipefail

# Offline lock checker and separately authorized resolver for the immutable
# Python test image.  --check never invokes Docker or a package index.
# PIP_CONSTRAINT=$REPO_ROOT/build-constraints.txt remains the single tracked
# constraint source for resolver and clean-installer build isolation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="$REPO_ROOT/docker/test-deps"
MAIN_LOCK="$LOCK_DIR/requirements-test.lock"
NODEPS_LOCK="$LOCK_DIR/requirements-test-nodeps.lock"
MANIFEST="$LOCK_DIR/lock-manifest.json"
DOCKERFILE="$REPO_ROOT/docker/test-deps.Dockerfile"
DOCKERIGNORE="$REPO_ROOT/docker/test-deps.Dockerfile.dockerignore"
BUILD_SCRIPT="$REPO_ROOT/scripts/build-test-deps.sh"
BASE_DIGEST="sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15"
PYTHON_PATCH="3.11.15"
PLATFORM="linux/amd64"

die() { echo "error: $*" >&2; exit 1; }

# Keep the disposable resolver output outside the function scope.  A trap that
# expands a local variable after run_update returns is unsafe under
# `set -u`: the failed resolver previously left a temporary directory behind
# and then replaced the useful resolver error with an unbound-variable error.
LOCK_OUTPUT_DIR=""
cleanup_lock_output() {
    if [ -z "$LOCK_OUTPUT_DIR" ]; then
        return 0
    fi
    case "$LOCK_OUTPUT_DIR" in
        /tmp/scai-test-deps-lock.*) rm -rf -- "$LOCK_OUTPUT_DIR" ;;
        *) echo "error: refusing to clean an unexpected lock output path: $LOCK_OUTPUT_DIR" >&2; return 1 ;;
    esac
}
trap cleanup_lock_output EXIT

prepare_lock_output() {
    LOCK_OUTPUT_DIR="$(mktemp -d /tmp/scai-test-deps-lock.XXXXXX)"
    chmod 0700 -- "$LOCK_OUTPUT_DIR"
    local download_name
    for download_name in main-download nodeps-download; do
        # Keep the directory host-owned while allowing the remapped container
        # writer to create artifacts.  The 0700 parent keeps these 0777
        # directories private to this host user.
        mkdir -- "$LOCK_OUTPUT_DIR/$download_name"
        chmod 0777 -- "$LOCK_OUTPUT_DIR/$download_name"
    done
}

check_inputs_exist() {
    local path
    for path in \
        "$DOCKERFILE" \
        "$DOCKERIGNORE" \
        "$BUILD_SCRIPT" \
        "$REPO_ROOT/requirements.txt" \
        "$REPO_ROOT/requirements-nodeps.txt" \
        "$REPO_ROOT/build-constraints.txt" \
        "$LOCK_DIR/requirements-test.in"; do
        [ -f "$path" ] || die "missing dependency input: $path"
    done
}

run_check() {
    check_inputs_exist
    python3 - "$REPO_ROOT" "$DOCKERFILE" "$DOCKERIGNORE" "$BUILD_SCRIPT" "$MAIN_LOCK" "$NODEPS_LOCK" "$MANIFEST" \
        "$BASE_DIGEST" "$PYTHON_PATCH" "$PLATFORM" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


class ContractError(Exception):
    pass


repo = Path(sys.argv[1])
dockerfile = Path(sys.argv[2])
dockerignore = Path(sys.argv[3])
build_script = Path(sys.argv[4])
main_lock = Path(sys.argv[5])
nodeps_lock = Path(sys.argv[6])
manifest_path = Path(sys.argv[7])
base_digest, python_patch, platform = sys.argv[8:11]
errors: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def compact_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def copy_commands(text: str) -> list[str]:
    commands: list[str] = []
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
            require(index < len(lines), "Dockerfile COPY continuation is unterminated")
            parts.append(lines[index].strip())
        commands.append(" ".join(part.rstrip("\\").strip() for part in parts))
        index += 1
    return commands


def require_image_builder_constraint_contract(*, expected_hash: str | None = None) -> None:
    constraint_text = compact_lines(repo / "build-constraints.txt")
    require(constraint_text == ["setuptools<82"], "build constraint source must contain only setuptools<82")

    ignore_lines = compact_lines(dockerignore)
    require(
        ignore_lines
        == [
            "*",
            "**",
            "!build-constraints.txt",
            "!docker/",
            "!docker/test-deps/",
            "!docker/test-deps/requirements-test.in",
            "!docker/test-deps/requirements-test.lock",
            "!docker/test-deps/requirements-test-nodeps.lock",
            "!docker/test-deps/lock-manifest.json",
        ],
        "Dockerfile-specific dependency context allowlist drift",
    )

    lock_source = (repo / "scripts/lock-test-deps.sh").read_text(encoding="utf-8")
    require(
        lock_source.count("constraints = repo / 'build-constraints.txt'\n") == 1,
        "lock resolver must have one tracked build-constraints source",
    )
    require("resolver_python = Path(sys.executable)" in lock_source, "resolver interpreter identity missing")
    require(
        "resolver_bootstrap_specs = [resolver_pip_spec, 'setuptools<82', 'wheel']" in lock_source,
        "resolver bootstrap constraint missing",
    )
    require("def run_with_build_constraint" in lock_source, "resolver build-constraint helper missing")
    require("PIP_CONSTRAINT" in lock_source, "resolver PEP 517 constraint fallback missing")
    require("'--build-constraint', str(constraints)" in lock_source, "resolver build constraint option missing")
    require("installer_env['PIP_CONSTRAINT'] = str(constraints)" in lock_source, "clean installer constraint missing")
    require(
        "installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes'" in lock_source,
        "clean installer hash install missing",
    )
    installer_start = lock_source.index("installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes'")
    installer_end = lock_source.index("final_freeze = output", installer_start)
    require("'--upgrade'" not in lock_source[installer_start:installer_end], "clean installer may not upgrade")
    require("Path('/tmp/scai-test-lock-venv')" in lock_source, "clean installer venv identity missing")

    docker_text = dockerfile.read_text(encoding="utf-8")
    expected_from = f"python:3.11.15-slim-bookworm@{base_digest}"
    from_lines = [line.strip() for line in docker_text.splitlines() if line.strip().startswith("FROM ")]
    require(len(from_lines) >= 2 and all(expected_from in line for line in from_lines[:2]), "Dockerfile base identity drift")
    source_commands = [command for command in copy_commands(docker_text) if "--from=builder" not in command]
    require(
        len(source_commands) == 2,
        "Dockerfile must have exactly two source COPY commands",
    )
    source_paths = set()
    for command in source_commands:
        tokens = command.split()
        require(len(tokens) >= 3 and tokens[0] == "COPY", f"malformed source COPY: {command}")
        source_paths.update(tokens[1:-1])
    require(
        source_paths
        == {
            "build-constraints.txt",
            "docker/test-deps/requirements-test.lock",
            "docker/test-deps/requirements-test-nodeps.lock",
            "docker/test-deps/lock-manifest.json",
        },
        "Dockerfile source COPY allowlist drift",
    )
    require("COPY build-constraints.txt /opt/scai-test/build-constraints.txt" in docker_text, "tracked constraint COPY missing")
    require('["inputs"]["build-constraints.txt"]' in docker_text, "builder manifest constraint hash lookup missing")
    require("actual_hash=\"$(sha256sum /opt/scai-test/build-constraints.txt" in docker_text, "builder constraint hash computation missing")
    require('[ "$actual_hash" = "$expected_hash" ]' in docker_text, "builder constraint hash comparison missing")
    require("/opt/scai-test/manifest/build-constraints.sha256" in docker_text, "builder constraint hash evidence missing")
    install_start = docker_text.index("RUN mkdir -p /opt/scai-test/manifest")
    install_end = docker_text.index("\n\nRUN python -m pip freeze", install_start)
    install_block = docker_text[install_start:install_end]
    require(install_block.count("PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt") == 2, "builder installs do not share PIP_CONSTRAINT")
    require(install_block.count("--require-hashes") == 2, "builder does not hash-install both lock files")
    require(
        install_block.index("[ \"$actual_hash\" = \"$expected_hash\" ]")
        < install_block.index("PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt"),
        "builder constraint hash is not checked before install",
    )
    require("COPY --from=builder /opt/scai-test/build-constraints.txt /opt/scai-test/build-constraints.txt" in docker_text, "final constraint retention missing")

    build_source = build_script.read_text(encoding="utf-8")
    require("--file docker/test-deps.Dockerfile" in build_source, "image builder Dockerfile path drift")
    require('"${build_args[@]}" .' in build_source, "image builder must use repository root context")
    if expected_hash is not None:
        require(
            hashlib.sha256((repo / "build-constraints.txt").read_bytes()).hexdigest() == expected_hash,
            "tracked build constraint hash does not equal manifest input hash",
        )


try:
    require_image_builder_constraint_contract()
    print("constraint source contract: PASS")
    require(main_lock.is_file(), "main lock is missing after source contract PASS")
    require(nodeps_lock.is_file(), "no-deps lock is missing after source contract PASS")
    require(manifest_path.is_file(), "lock manifest is missing after source contract PASS")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "lock manifest must be an object")
    require(manifest.get("schema") == 1, "lock manifest schema mismatch")
    require(manifest.get("platform") == {"python": python_patch, "os": platform}, "lock platform mismatch")
    require(manifest.get("base_image_digest") == base_digest, "base digest mismatch")

    expected_inputs = {
        "requirements.txt",
        "requirements-nodeps.txt",
        "build-constraints.txt",
        "docker/test-deps/requirements-test.in",
        "scripts/lock-test-deps.sh",
    }
    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict) and set(inputs) == expected_inputs, "lock input set mismatch")
    for relative, expected in inputs.items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"invalid input hash: {relative}")
        actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        require(actual == expected, f"input hash mismatch: {relative}")

    from_lines = [line.strip() for line in dockerfile.read_text(encoding="utf-8").splitlines() if line.strip().startswith("FROM ")]
    require(len(from_lines) >= 2, "Dockerfile must contain builder and runtime stages")
    exact_from = f"python:3.11.15-slim-bookworm@{base_digest}"
    require(all(exact_from in line for line in from_lines[:2]), "Dockerfile base digest/tag mismatch")

    line_re = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s]+) --hash=sha256:([0-9a-f]{64})$")

    def normalize(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    def parse_lock(path: Path) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        seen: set[str] = set()
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = line_re.fullmatch(line)
            require(match is not None, f"{path.name}:{number}: malformed or unhashed lock line")
            if match is None:
                raise ContractError(f"{path.name}:{number}: malformed lock line")
            name, version, digest = match.groups()
            normalized = normalize(name)
            require(normalized not in seen, f"{path.name}:{number}: duplicate normalized name {normalized}")
            seen.add(normalized)
            records.append({"name": name, "version": version, "sha256": digest})
        require(records == sorted(records, key=lambda item: (normalize(item["name"]), item["version"])), f"{path.name}: lock sort drift")
        return records

    main = parse_lock(main_lock)
    nodeps = parse_lock(nodeps_lock)

    def inventory(value: object, key: str) -> list[dict[str, str]]:
        require(isinstance(value, list), f"manifest.{key} must be a list")
        result: list[dict[str, str]] = []
        for record in value:
            require(isinstance(record, dict), f"manifest.{key} contains a non-object")
            require(set(record) == {"name", "version", "sha256"}, f"manifest.{key} record shape mismatch")
            require(isinstance(record["name"], str) and isinstance(record["version"], str), f"manifest.{key} name/version type mismatch")
            require(isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, f"manifest.{key} hash mismatch")
            result.append({"name": record["name"], "version": record["version"], "sha256": record["sha256"]})
        return result

    require(main == inventory(manifest.get("requirements"), "requirements"), "main lock inventory mismatch")
    require(nodeps == inventory(manifest.get("nodeps"), "nodeps"), "no-deps lock inventory mismatch")
    test_input = [
        line.strip()
        for line in (repo / "docker/test-deps/requirements-test.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(
        test_input == ["pytest==9.1.1", "pytest-asyncio==1.3.0", "pytest-benchmark==5.2.3"],
        "test-tool input mismatch",
    )

    expected_pip_check = "decoupler 1.8.0 has requirement numba<0.61.0,>=0.60.0, but you have numba 0.61.2."
    require(manifest.get("allowed_pip_check") == [expected_pip_check], "pip-check allowlist is not the code-owned singleton")
    require(manifest.get("resolver_bootstrap_specs") == ["pip>=25.3,<27", "setuptools<82", "wheel"], "resolver bootstrap contract drift")
    resolver = manifest.get("resolver")
    require(isinstance(resolver, dict) and set(resolver) == {"pip", "setuptools", "wheel"}, "resolver identity shape mismatch")
    require(all(isinstance(resolver[name], str) and resolver[name] for name in ("pip", "setuptools", "wheel")), "resolver identity is incomplete")
    lock_tool = manifest.get("lock_tool")
    require(isinstance(lock_tool, dict), "lock_tool identity missing")
    require(lock_tool.get("path") == "scripts/lock-test-deps.sh", "lock_tool path mismatch")
    require(lock_tool.get("contract_version") == 1, "lock_tool contract version mismatch")
    require(lock_tool.get("sha256") == hashlib.sha256((repo / "scripts/lock-test-deps.sh").read_bytes()).hexdigest(), "lock_tool hash mismatch")

    def by_name(records: list[dict[str, str]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for record in records:
            name = normalize(record["name"])
            require(name not in result, f"duplicate lock name: {name}")
            result[name] = record["version"]
        return result

    locked_main = by_name(main)
    locked_nodeps = by_name(nodeps)
    for name, version in {
        "pytest": "9.1.1",
        "pytest-asyncio": "1.3.0",
        "pytest-benchmark": "5.2.3",
    }.items():
        require(locked_main.get(name) == version, f"main lock candidate version drift: {name}")
    require(set(locked_main).isdisjoint(set(locked_nodeps)) or all(locked_main[name] == locked_nodeps[name] for name in set(locked_main) & set(locked_nodeps)), "same package has conflicting lock versions")
    locked = {**locked_main, **locked_nodeps}

    def typed_inventory(key: str) -> list[dict[str, str]]:
        value = manifest.get(key)
        require(isinstance(value, list), f"manifest.{key} missing")
        result = []
        for record in value:
            require(isinstance(record, dict) and set(record) == {"name", "version"}, f"manifest.{key} shape mismatch")
            require(isinstance(record["name"], str) and isinstance(record["version"], str), f"manifest.{key} type mismatch")
            result.append({"name": normalize(record["name"]), "version": record["version"]})
        return result

    initial = {record["name"]: record["version"] for record in typed_inventory("initial_actual_bootstrap_installers")}
    expected_replaced = sorted(
        [{"name": name, "initial_version": initial[name], "locked_version": locked[name]} for name in initial.keys() & locked.keys()],
        key=lambda item: item["name"],
    )
    expected_unlocked = sorted(
        [{"name": name, "version": version} for name, version in initial.items() if name not in locked],
        key=lambda item: item["name"],
    )
    expected_final = sorted(
        [{"name": name, "version": version} for name, version in locked.items()] + expected_unlocked,
        key=lambda item: item["name"],
    )
    require(manifest.get("replaced_bootstrap_installers") == expected_replaced, "bootstrap replacement transition drift")
    require(manifest.get("final_unlocked_bootstrap_extras") == expected_unlocked, "unlocked bootstrap extras drift")
    require(manifest.get("expected_final_freeze") == expected_final, "expected final freeze drift")
    require(manifest.get("final_actual_freeze") == expected_final, "actual final freeze drift")
    require_image_builder_constraint_contract(expected_hash=inputs["build-constraints.txt"])
    installer = manifest.get("installer")
    require(isinstance(installer, dict), "installer identity missing")
    require(installer.get("argv") == ["/opt/scai-test/venv/bin/python", "-m", "pip"], "installer argv mismatch")
    require(installer.get("distribution") == "pip" and installer.get("version") == initial.get("pip"), "installer identity drift")
    require("pip" in initial, "initial bootstrap pip identity is missing")
except (ContractError, OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
    errors.append(str(exc))

if errors:
    for error in errors:
        print(f"FAIL: {error}")
    raise SystemExit(1)
print("OK: immutable dependency lock is internally consistent")
PY
}

run_update() {
    check_inputs_exist
    command -v docker >/dev/null 2>&1 || die "docker is unavailable; --update requires a daemon"
    docker version >/dev/null 2>&1 || die "docker daemon is unavailable; --update requires a daemon"
    local base_ref
    prepare_lock_output
    base_ref="python:3.11.15-slim-bookworm@$BASE_DIGEST"
    docker run --rm -i --network default --platform "$PLATFORM" \
        --volume "$REPO_ROOT:/workspace/scai:ro" \
        --volume "$LOCK_OUTPUT_DIR:/out:rw" \
        --entrypoint python3 "$base_ref" - <<'PY'
import hashlib
import json
import os
import re
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path

repo = Path('/workspace/scai')
out = Path('/out')
out.mkdir(parents=True, exist_ok=True)
test_in = repo / 'docker/test-deps/requirements-test.in'
constraints = repo / 'build-constraints.txt'
resolver_python = Path(sys.executable)
resolver_pip_spec = 'pip>=25.3,<27'
resolver_bootstrap_specs = [resolver_pip_spec, 'setuptools<82', 'wheel']


def run(args: list[str], **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def output(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


# Match the builder's native build surface while resolving/building sdists in
# this disposable container.  These packages never enter the lock or runtime
# image; they only make the authorized resolver able to reproduce its build
# metadata and hash-enforced installation checks.
run(["apt-get", "update"])
run(["apt-get", "install", "-y", "--no-install-recommends", "build-essential", "curl", "gosu", "pkg-config"])
run(["apt-get", "clean"])


def report(requirements: list[str], output_path: Path, download_dir: Path, *, no_deps: bool = False) -> list[dict[str, str]]:
    args = [str(resolver_python), '-m', 'pip', 'install', '--dry-run', '--ignore-installed', '--report', str(output_path)]
    if no_deps:
        args.append('--no-deps')
    for path in requirements:
        args += ['-r', str(path)]
    if not no_deps:
        args += ['-c', str(constraints)]
    run_with_build_constraint(args, enabled=not no_deps)

    download_args = [
        str(resolver_python), '-m', 'pip', 'download', '--disable-pip-version-check',
        '--no-input', '--no-cache-dir', '--dest', str(download_dir),
    ]
    if no_deps:
        download_args.append('--no-deps')
    for path in requirements:
        download_args += ['-r', str(path)]
    if not no_deps:
        download_args += ['-c', str(constraints)]
    download_dir.mkdir(parents=True, exist_ok=True)
    run_with_build_constraint(download_args, enabled=not no_deps)

    payload = json.loads(output_path.read_text(encoding='utf-8'))
    entries = []
    for item in payload.get('install', []):
        metadata = item.get('metadata') or {}
        download = item.get('download_info') or {}
        archive = download.get('archive_info') or {}
        digest = archive.get('hash') or ''
        if digest.startswith('sha256='):
            digest = digest[7:]
        if not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise SystemExit(f'missing sha256 artifact hash for {metadata.get("name")}')
        entries.append({'name': metadata['name'], 'version': metadata['version'], 'sha256': digest})
    downloaded_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in download_dir.iterdir()
        if path.is_file()
    }
    missing_hashes = sorted({entry['sha256'] for entry in entries} - downloaded_hashes)
    if missing_hashes:
        raise SystemExit(f'reported artifact bytes were not downloaded: {missing_hashes!r}')
    entries.sort(key=lambda item: (re.sub(r'[-_.]+', '-', item['name']).lower(), item['version']))
    return entries


# Keep the clean install venv's ensurepip-provided installer untouched.  The
# base interpreter is a disposable, explicitly upgraded resolver environment;
# separating these identities is what lets newer pip resolve this graph while
# the manifest still records the actual installer used for hash-enforced
# installation.
run([
    str(resolver_python), '-m', 'pip', 'install', '--disable-pip-version-check',
    '--no-input', '--no-cache-dir', '--quiet', '--upgrade', *resolver_bootstrap_specs,
])
resolver = {
    'pip': output([str(resolver_python), '-m', 'pip', '--version']).split()[1],
    'setuptools': output([str(resolver_python), '-c', 'import setuptools; print(setuptools.__version__)']),
    'wheel': output([str(resolver_python), '-c', 'import wheel; print(wheel.__version__)']),
}

resolver_help = subprocess.run(
    [str(resolver_python), '-m', 'pip', 'install', '--help'],
    check=True,
    capture_output=True,
    text=True,
)
supports_build_constraint = '--build-constraint' in resolver_help.stdout


def run_with_build_constraint(args: list[str], *, enabled: bool) -> None:
    if not enabled:
        run(args)
        return
    if supports_build_constraint:
        run([*args, '--build-constraint', str(constraints)])
        return
    # Older pip releases do not expose --build-constraint.  Pip propagates
    # PIP_CONSTRAINT into PEP 517 build isolation, preserving setuptools<82.
    build_env = os.environ.copy()
    build_env['PIP_CONSTRAINT'] = str(constraints)
    run(args, env=build_env)

main = report([repo / 'requirements.txt', test_in], out / 'main-report.json', out / 'main-download')
nodeps = report([repo / 'requirements-nodeps.txt'], out / 'nodeps-report.json', out / 'nodeps-download', no_deps=True)


def write_lock(path: Path, entries: list[dict[str, str]]) -> None:
    lines = [
        '# ScAI test-dependency environment lock. Generated by scripts/lock-test-deps.sh.',
        '# Do not edit; run scripts/lock-test-deps.sh --update to regenerate.',
    ]
    lines.extend(f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}" for entry in entries)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


write_lock(out / 'requirements-test.lock', main)
write_lock(out / 'requirements-test-nodeps.lock', nodeps)

venv_dir = Path('/tmp/scai-test-lock-venv')
venv.EnvBuilder(with_pip=True).create(venv_dir)
python = venv_dir / 'bin/python'
initial_freeze = output([str(python), '-m', 'pip', 'freeze', '--all'])
installer_help = subprocess.run(
    [str(python), '-m', 'pip', 'install', '--help'],
    check=True,
    capture_output=True,
    text=True,
)
installer_install = [str(python), '-m', 'pip', 'install', '--require-hashes', '-r', str(out / 'requirements-test.lock')]
if '--build-constraint' in installer_help.stdout:
    run([*installer_install, '--build-constraint', str(constraints)])
else:
    installer_env = os.environ.copy()
    installer_env['PIP_CONSTRAINT'] = str(constraints)
    run(installer_install, env=installer_env)
run([str(python), '-m', 'pip', 'install', '--no-deps', '--require-hashes', '-r', str(out / 'requirements-test-nodeps.lock')])
final_freeze = output([str(python), '-m', 'pip', 'freeze', '--all'])


def freeze_entries(text: str) -> list[dict[str, str]]:
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or '==' not in line:
            continue
        name, version = line.split('==', 1)
        entries.append({'name': re.sub(r'[-_.]+', '-', name).lower(), 'version': version})
    return sorted(entries, key=lambda item: item['name'])


initial = freeze_entries(initial_freeze)
final = freeze_entries(final_freeze)
locked = {re.sub(r'[-_.]+', '-', item['name']).lower(): item['version'] for item in main + nodeps}
initial_map = {item['name']: item['version'] for item in initial}
unlocked = [{'name': name, 'version': version} for name, version in initial_map.items() if name not in locked]
unlocked.sort(key=lambda item: item['name'])
expected = [{'name': name, 'version': version} for name, version in locked.items()]
expected = [item for item in expected if item['name'] not in {extra['name'] for extra in unlocked}] + unlocked
expected.sort(key=lambda item: item['name'])
replaced = [{'name': name, 'initial_version': initial_map[name], 'locked_version': locked[name]} for name in sorted(initial_map.keys() & locked.keys())]

pip_check_process = subprocess.run([str(python), '-m', 'pip', 'check'], capture_output=True, text=True, check=False)
pip_check = [line.strip() for line in pip_check_process.stdout.splitlines() if line.strip()]
expected_pip_check = 'decoupler 1.8.0 has requirement numba<0.61.0,>=0.60.0, but you have numba 0.61.2.'
if pip_check != [expected_pip_check]:
    raise SystemExit(f'unexpected pip check output: {pip_check!r}')

manifest = {
    'schema': 1,
    'platform': {'python': '3.11.15', 'os': 'linux/amd64'},
    'base_image_digest': 'sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15',
    'inputs': {},
    'requirements': main,
    'nodeps': nodeps,
    'resolver': resolver,
    'resolver_bootstrap_specs': resolver_bootstrap_specs,
    'installer': {'argv': ['/opt/scai-test/venv/bin/python', '-m', 'pip'], 'distribution': 'pip', 'version': initial_map.get('pip', resolver['pip'])},
    'initial_actual_bootstrap_installers': initial,
    'replaced_bootstrap_installers': replaced,
    'final_unlocked_bootstrap_extras': unlocked,
    'expected_final_freeze': expected,
    'final_actual_freeze': final,
    'allowed_pip_check': [expected_pip_check],
    'generation_command': 'scripts/lock-test-deps.sh --update',
    'generated_at_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
    'lock_tool': {'path': 'scripts/lock-test-deps.sh', 'contract_version': 1, 'sha256': hashlib.sha256((repo / 'scripts/lock-test-deps.sh').read_bytes()).hexdigest()},
}
for relative in ['requirements.txt', 'requirements-nodeps.txt', 'build-constraints.txt', 'docker/test-deps/requirements-test.in', 'scripts/lock-test-deps.sh']:
    manifest['inputs'][relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
(out / 'lock-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
    cp "$LOCK_OUTPUT_DIR/requirements-test.lock" "$MAIN_LOCK"
    cp "$LOCK_OUTPUT_DIR/requirements-test-nodeps.lock" "$NODEPS_LOCK"
    cp "$LOCK_OUTPUT_DIR/lock-manifest.json" "$MANIFEST"
    run_check
}

case "${1:-}" in
    --check) run_check ;;
    --update) run_update ;;
    *) die "usage: scripts/lock-test-deps.sh --check|--update" ;;
esac
