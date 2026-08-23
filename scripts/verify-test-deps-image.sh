#!/bin/bash
set -euo pipefail

# Immutable local/registry test-image verifier. Mutable tags are rejected.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
LOCK_DIR="docker/test-deps"
MANIFEST="$LOCK_DIR/lock-manifest.json"
BASE_DIGEST="sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15"
EXPECTED_SOURCE="https://github.com/minzhangcheng/scai"
EXPECTED_REVISION="0c1edf479f7360dc80afa89cf7fc9534c1effce6"
EXPECTED_VERSION="0.39.15"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "OK: $*"; }

REF="${1:-}"
[ -n "$REF" ] || fail "usage: scripts/verify-test-deps-image.sh sha256:<image-id>|name@sha256:<digest>"
if [[ ! "$REF" =~ ^sha256:[0-9a-f]{64}$ && ! "$REF" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]]; then
    fail "ref must be an immutable image ID or name@sha256:digest"
fi
[ -f "$MANIFEST" ] || fail "missing lock manifest; resolve the lock first"
command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
docker image inspect "$REF" >/dev/null 2>&1 || fail "image is not available locally: $REF"

expected_fingerprint="$(scripts/test-deps-fingerprint.sh | sed -n 's/^FINGERPRINT=//p')"
expected_lock="$(sha256sum "$LOCK_DIR/requirements-test.lock" | cut -d' ' -f1)"
expected_nodeps_lock="$(sha256sum "$LOCK_DIR/requirements-test-nodeps.lock" | cut -d' ' -f1)"
expected_constraints="$(sha256sum build-constraints.txt | cut -d' ' -f1)"
expected_manifest="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"

label() {
    docker image inspect --format "{{index .Config.Labels \"$1\"}}" "$REF" 2>/dev/null || true
}

[ "$(label io.scai.test-deps.fingerprint)" = "$expected_fingerprint" ] || fail "fingerprint label mismatch"
[ "$(label io.scai.test-deps.lock-sha256)" = "$expected_lock" ] || fail "lock label mismatch"
[ "$(label io.scai.test-deps.python)" = "3.11.15" ] || fail "Python label mismatch"
[ "$(label io.scai.test-deps.platform)" = "linux/amd64" ] || fail "platform label mismatch"
[ "$(label io.scai.test-deps.base-digest)" = "$BASE_DIGEST" ] || fail "base digest label mismatch"
[ "$(label org.opencontainers.image.source)" = "$EXPECTED_SOURCE" ] || fail "source label mismatch"
[ "$(label org.opencontainers.image.revision)" = "$EXPECTED_REVISION" ] || fail "revision label mismatch"
[ "$(label org.opencontainers.image.version)" = "$EXPECTED_VERSION" ] || fail "version label mismatch"
ok "OCI labels match current inputs"

run_image() {
    docker run --rm --pull=never --read-only --network none --user 10001 \
        --platform linux/amd64 --tmpfs /tmp:rw,size=512m \
        --entrypoint /bin/sh "$REF" -c "$1"
}

[ "$(run_image 'id -u')" = "10001" ] || fail "runtime UID is not 10001"
[ "$(run_image 'cat /opt/scai-test/manifest/runtime-user.txt')" = "10001 scai-test" ] || fail "runtime user evidence mismatch"
[ "$(run_image 'command -v python')" = "/opt/scai-test/venv/bin/python" ] || fail "python is not from the immutable venv"
[ "$(run_image 'python --version')" = "Python 3.11.15" ] || fail "Python patch mismatch"
run_image 'git --version' >/dev/null || fail "git is unavailable to the runtime user"
if run_image 'touch /opt/scai-test/venv/.write-probe' >/dev/null 2>&1; then
    fail "runtime user can write the immutable venv"
fi
ok "runtime UID, Python identity, and read-only venv verified"

container_lock="$(run_image 'sha256sum /opt/scai-test/requirements-test.lock | cut -d" " -f1')"
[ "$container_lock" = "$expected_lock" ] || fail "embedded lock bytes differ from tracked lock"
container_nodeps_lock="$(run_image 'sha256sum /opt/scai-test/requirements-test-nodeps.lock | cut -d" " -f1')"
[ "$container_nodeps_lock" = "$expected_nodeps_lock" ] || fail "embedded no-deps lock bytes differ from tracked lock"
container_constraints="$(run_image 'sha256sum /opt/scai-test/build-constraints.txt | cut -d" " -f1')"
[ "$container_constraints" = "$expected_constraints" ] || fail "embedded build-constraints bytes differ from tracked input"
container_manifest="$(run_image 'sha256sum /opt/scai-test/lock-manifest.json | cut -d" " -f1')"
[ "$container_manifest" = "$expected_manifest" ] || fail "embedded lock manifest bytes differ from tracked manifest"
run_image 'cd /opt/scai-test/manifest && sha256sum -c hashes.txt' >/dev/null || fail "detached image evidence hashes fail"
run_image 'cd /opt/scai-test && sha256sum -c manifest/build-constraints.sha256' >/dev/null || fail "builder constraint hash evidence fails"

run_image 'python - <<"PY"
import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path

EXPECTED_PIP_CHECK = "decoupler 1.8.0 has requirement numba<0.61.0,>=0.60.0, but you have numba 0.61.2."
root = Path("/opt/scai-test")
manifest = json.loads((root / "lock-manifest.json").read_text(encoding="utf-8"))
if manifest.get("allowed_pip_check") != [EXPECTED_PIP_CHECK]:
    raise SystemExit("manifest pip-check allowlist drift")
if hashlib.sha256((root / "build-constraints.txt").read_bytes()).hexdigest() != manifest["inputs"]["build-constraints.txt"]:
    raise SystemExit("in-image build constraint does not match manifest input")

line_re = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s]+) --hash=sha256:([0-9a-f]{64})$")

def parse_lock(path):
    records = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = line_re.fullmatch(line)
        if match is None:
            raise SystemExit(f"malformed lock line: {path}: {line}")
        records.append(match.groups())
    return records

main = parse_lock(root / "requirements-test.lock")
nodeps = parse_lock(root / "requirements-test-nodeps.lock")
main_versions = {re.sub(r"[-_.]+", "-", name).lower(): version for name, version, _ in main}
for name, version in {
    "pytest": "9.1.1",
    "pytest-asyncio": "1.3.0",
    "pytest-benchmark": "5.2.3",
}.items():
    if main_versions.get(name) != version:
        raise SystemExit(f"main lock version drift: {name}")
locked = {
    re.sub(r"[-_.]+", "-", name).lower(): version
    for name, version, _ in [*main, *nodeps]
}
if locked.get("setuptools") != "81.0.0":
    raise SystemExit(f"locked setuptools identity drift: {locked.get('setuptools')!r}")

def freeze(text):
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "==" not in line:
            continue
        name, version = line.split("==", 1)
        entries.append({"name": re.sub(r"[-_.]+", "-", name).lower(), "version": version})
    return sorted(entries, key=lambda item: item["name"])

actual_freeze = freeze(subprocess.run(["python", "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True).stdout)
unlocked = manifest.get("final_unlocked_bootstrap_extras")
if not isinstance(unlocked, list):
    raise SystemExit("manifest final unlocked bootstrap extras missing")
expected_freeze = sorted(
    [{"name": name, "version": version} for name, version in locked.items()] + unlocked,
    key=lambda item: item["name"],
)
if manifest.get("expected_final_freeze") != expected_freeze or actual_freeze != expected_freeze:
    raise SystemExit("final pip freeze differs from lock-owned versions and extras")
for distribution, version in {
    "pytest": "9.1.1",
    "pytest-asyncio": "1.3.0",
    "pytest-benchmark": "5.2.3",
}.items():
    if metadata.version(distribution) != version:
        raise SystemExit(f"installed distribution version drift: {distribution}")

actual_check = [line.strip() for line in subprocess.run(["python", "-m", "pip", "check"], capture_output=True, text=True, check=False).stdout.splitlines() if line.strip()]
if actual_check != [EXPECTED_PIP_CHECK]:
    raise SystemExit(f"pip check drift: {actual_check!r}")
PY' >/dev/null || fail "lock, freeze, exact tool identities, or pip check failed"

run_image 'python - <<"PY"
import importlib
import importlib.util
from importlib import metadata
from pathlib import Path
from packaging.version import Version
venv_root = Path("/opt/scai-test/venv").resolve()

def require_inside(path, description):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(venv_root)
    except ValueError:
        raise SystemExit(f"{description} resolved outside the immutable venv: {resolved}")

def require_module(module_name, distribution_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise SystemExit(f"{module_name} did not resolve to a module spec")
    locations = spec.submodule_search_locations
    if locations is not None:
        locations = list(locations)
        if not locations:
            raise SystemExit(f"{module_name} has no namespace search locations")
        for location in locations:
            require_inside(location, f"{module_name} namespace search location")
        if spec.origin not in {None, "namespace"}:
            require_inside(spec.origin, f"{module_name} package module file")
    else:
        if not spec.origin or spec.origin in {"built-in", "frozen"}:
            raise SystemExit(f"{module_name} did not resolve to a file module")
        require_inside(spec.origin, f"{module_name} module file")
    distribution = metadata.distribution(distribution_name)
    files = distribution.files
    if not files:
        raise SystemExit(f"{distribution_name} has no distribution file inventory")
    for relative in files:
        located = Path(distribution.locate_file(relative))
        if not located.is_file():
            raise SystemExit(f"{distribution_name} distribution file is missing: {relative}")
        require_inside(located, f"{distribution_name} distribution file {relative}")
    return importlib.import_module(module_name)

zarr = require_module("zarr", "zarr")
if not Version("2.18.7") <= Version(zarr.__version__) < Version("3.0"):
    raise SystemExit(f"unsupported zarr version: {zarr.__version__}")
for module_name, distribution_name in {
    "anndata": "anndata",
    "dask_image": "dask-image",
    "fastapi": "fastapi",
    "langgraph": "langgraph",
    "scanpy": "scanpy",
    "h5py": "h5py",
    "jupyter_client": "jupyter_client",
    "pytest": "pytest",
    "pytest_asyncio": "pytest-asyncio",
    "pytest_benchmark": "pytest-benchmark",
}.items():
    require_module(module_name, distribution_name)
PY' >/dev/null || fail "scientific/test imports or Zarr range failed"

run_image 'tmp_inventory="$(mktemp)"; dpkg-query -W -f="${Package} ${Version}\n" | sort >"$tmp_inventory"; cmp -s "$tmp_inventory" /opt/scai-test/manifest/final-os-packages.txt; rm -f "$tmp_inventory"' \
    || fail "final OS inventory differs from detached evidence"

for forbidden in /dep /extra /data /workspace /app /scai; do
    if run_image "test ! -e '$forbidden'" >/dev/null 2>&1; then :; else fail "forbidden path present: $forbidden"; fi
done
if run_image 'find /opt -maxdepth 3 \( -name backend -o -name frontend \) -print -quit' | grep -q .; then
    fail "application source found in image"
fi
ok "source and forbidden-volume absence verified"
echo "ALL CHECKS PASSED: $REF"
