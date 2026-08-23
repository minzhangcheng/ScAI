#!/bin/bash
set -euo pipefail

# Dependency-intent fingerprint.  This is not an image content digest and the
# discovery tag emitted below is never an execution reference.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DOCKERFILE="docker/test-deps.Dockerfile"
DOCKERIGNORE="docker/test-deps.Dockerfile.dockerignore"
LOCK_DIR="docker/test-deps"
MANIFEST="$LOCK_DIR/lock-manifest.json"

for path in "$DOCKERFILE" "$DOCKERIGNORE" "$LOCK_DIR/requirements-test.in" \
    "$LOCK_DIR/requirements-test.lock" "$LOCK_DIR/requirements-test-nodeps.lock" \
    "$MANIFEST" requirements.txt requirements-nodeps.txt build-constraints.txt \
    scripts/lock-test-deps.sh; do
    [ -f "$path" ] || {
        echo "error: missing fingerprint input $path; run the prerequisite lock step" >&2
        exit 1
    }
done

BASE_DIGEST="$(grep -oE '@sha256:[0-9a-f]{64}' "$DOCKERFILE" | head -1 | tr -d '@')"
[ -n "$BASE_DIGEST" ] || { echo "error: missing digest-pinned base" >&2; exit 1; }

RESOLVER_PIP="$(python3 - "$MANIFEST" <<'PY'
import json
import sys
try:
    print(json.loads(open(sys.argv[1], encoding="utf-8").read())["resolver"]["pip"])
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
PY
)" || { echo "error: malformed lock manifest" >&2; exit 1; }

records="$({
    printf 'base-image %s\n' "$BASE_DIGEST"
    printf 'platform linux/amd64\n'
    printf 'python 3.11.15\n'
    printf 'resolver-pip %s\n' "$RESOLVER_PIP"
    for path in \
        "$DOCKERFILE" "$DOCKERIGNORE" \
        requirements.txt requirements-nodeps.txt build-constraints.txt \
        "$LOCK_DIR/requirements-test.in" \
        "$LOCK_DIR/requirements-test.lock" \
        "$LOCK_DIR/requirements-test-nodeps.lock" \
        "$MANIFEST" scripts/lock-test-deps.sh; do
        printf 'file %s %s\n' "$path" "$(sha256sum "$path" | cut -d' ' -f1)"
    done
} | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"

printf 'FINGERPRINT=%s\n' "$records"
printf 'TAG=ghcr.io/minzhangcheng/scai-test-deps:py311-amd64-%s\n' "${records:0:16}"
