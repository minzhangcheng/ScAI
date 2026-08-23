#!/bin/bash
set -euo pipefail

# Local immutable dependency-image build. Registry writes require --push and
# the tracked pointer is never changed by this script.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
FIXED_HEAD="0c1edf479f7360dc80afa89cf7fc9534c1effce6"
PUSH=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=1 ;;
        --force) FORCE=1 ;;
        *) echo "error: unknown option $arg" >&2; exit 2 ;;
    esac
done

scripts/lock-test-deps.sh --check
mapfile -t IDENTITY < <(scripts/test-deps-fingerprint.sh)
FINGERPRINT="${IDENTITY[0]#FINGERPRINT=}"
TAG="${IDENTITY[1]#TAG=}"
LOCK_SHA256="$(sha256sum docker/test-deps/requirements-test.lock | cut -d' ' -f1)"
REVISION="$(git rev-parse HEAD)"
LOCAL_IDENTITY=".scai-test-deps-image.json"
[ "$REVISION" = "$FIXED_HEAD" ] || {
    echo "error: build source HEAD must be $FIXED_HEAD (got $REVISION)" >&2
    exit 1
}

if [ "$FORCE" -eq 0 ]; then
    existing_fp="$(docker image inspect --format '{{index .Config.Labels "io.scai.test-deps.fingerprint"}}' "$TAG" 2>/dev/null || true)"
    existing_lock="$(docker image inspect --format '{{index .Config.Labels "io.scai.test-deps.lock-sha256"}}' "$TAG" 2>/dev/null || true)"
    if [ -n "$existing_fp" ] && { [ "$existing_fp" != "$FINGERPRINT" ] || [ "$existing_lock" != "$LOCK_SHA256" ]; }; then
        echo "error: existing discovery tag has a conflicting fingerprint/lock label; use --force only for a reviewed local replacement" >&2
        exit 1
    fi
fi

build_args=(
    docker buildx build
    --file docker/test-deps.Dockerfile
    --platform linux/amd64
    --tag "$TAG"
    --build-arg "SCAI_TEST_FINGERPRINT=$FINGERPRINT"
    --build-arg "SCAI_TEST_LOCK_SHA256=$LOCK_SHA256"
    --build-arg "SCAI_SOURCE_REVISION=$REVISION"
    --provenance=false
    --load
)
"${build_args[@]}" .

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$TAG")"
scripts/verify-test-deps-image.sh "$IMAGE_ID"
LABELS="$(docker image inspect --format '{{json .Config.Labels}}' "$IMAGE_ID")"
python3 - "$LOCAL_IDENTITY" "$TAG" "$IMAGE_ID" "$FINGERPRINT" "$LOCK_SHA256" "$REVISION" "$LABELS" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, tag, image_id, fingerprint, lock_sha, revision, labels = sys.argv[1:]
observed_labels = json.loads(labels)
required_label_keys = (
    "org.opencontainers.image.source",
    "org.opencontainers.image.revision",
    "org.opencontainers.image.version",
    "io.scai.test-deps.fingerprint",
    "io.scai.test-deps.lock-sha256",
    "io.scai.test-deps.python",
    "io.scai.test-deps.platform",
    "io.scai.test-deps.base-digest",
)
if not isinstance(observed_labels, dict) or any(key not in observed_labels for key in required_label_keys):
    raise SystemExit("immutable image is missing a required verified label")
verified_labels = {key: observed_labels[key] for key in required_label_keys}
payload = {
    "schema": 1,
    "tag": tag,
    "image_id": image_id,
    "fingerprint": fingerprint,
    "lock_sha256": lock_sha,
    "revision": revision,
    "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "labels": verified_labels,
    "manifest_hashes": {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (
            "docker/test-deps/requirements-test.lock",
            "docker/test-deps/requirements-test-nodeps.lock",
            "docker/test-deps/lock-manifest.json",
        )
    },
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Built local immutable image $IMAGE_ID for $TAG"
echo "Identity written only to ignored $LOCAL_IDENTITY"
if [ "$PUSH" -eq 1 ]; then
    docker push "$TAG"
    IMAGE_NAME="${TAG%:*}"
    LOCAL_REPO_DIGEST="$(docker image inspect --format '{{json .RepoDigests}}' "$IMAGE_ID")"
    LOCAL_REPO_DIGEST="$(python3 - "$IMAGE_NAME" "$LOCAL_REPO_DIGEST" <<'PY'
import json
import re
import sys

image_name, raw = sys.argv[1:]
try:
    values = json.loads(raw)
except (TypeError, ValueError) as exc:
    raise SystemExit(f"malformed local RepoDigests: {exc}")
if not isinstance(values, list):
    raise SystemExit("local RepoDigests is not a list")
pattern = re.compile(re.escape(image_name) + r"@sha256:[0-9a-f]{64}")
matches = [value for value in values if isinstance(value, str) and pattern.fullmatch(value)]
if len(matches) != 1:
    raise SystemExit(f"local RepoDigests does not contain one exact digest for {image_name}")
print(matches[0])
PY
)"
REGISTRY_DIGEST="$(docker buildx imagetools inspect --format '{{.Manifest.Digest}}' "$TAG")"
LOCAL_DIGEST="${LOCAL_REPO_DIGEST##*@}"
[[ "$REGISTRY_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo "error: registry manifest digest is not an exact sha256 identity" >&2
    exit 1
}
[ "$LOCAL_DIGEST" = "$REGISTRY_DIGEST" ] || {
    echo "error: pushed registry digest does not match verified local RepoDigest" >&2
    exit 1
}
echo "Registry digest (proposal only): $LOCAL_REPO_DIGEST"
echo "Tracked pointer update remains a separate reviewed action."
fi
