# syntax=docker/dockerfile:1

# Source-independent immutable Python test environment for the 0.39.1 target.
# The digest is part of the lock contract and must be identical in every stage.
FROM python:3.11.15-slim-bookworm@sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl gosu pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/scai-test/venv
ENV PATH=/opt/scai-test/venv/bin:$PATH

COPY docker/test-deps/requirements-test.lock \
     docker/test-deps/requirements-test-nodeps.lock \
     docker/test-deps/lock-manifest.json /opt/scai-test/
COPY build-constraints.txt /opt/scai-test/build-constraints.txt

RUN mkdir -p /opt/scai-test/manifest \
    && expected_hash="$(python -c 'import json; print(json.load(open("/opt/scai-test/lock-manifest.json", encoding="utf-8"))["inputs"]["build-constraints.txt"])')" \
    && actual_hash="$(sha256sum /opt/scai-test/build-constraints.txt | cut -d' ' -f1)" \
    && [ "$actual_hash" = "$expected_hash" ] \
    && printf '%s  build-constraints.txt\n' "$actual_hash" > /opt/scai-test/manifest/build-constraints.sha256 \
    && PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt \
       python -m pip install --require-hashes --no-cache-dir \
       -r /opt/scai-test/requirements-test.lock \
    && PIP_CONSTRAINT=/opt/scai-test/build-constraints.txt \
       python -m pip install --no-deps --require-hashes --no-cache-dir \
       -r /opt/scai-test/requirements-test-nodeps.lock

RUN python -m pip freeze --all | sort > /opt/scai-test/manifest/pip-freeze.txt \
    && (python -m pip check > /opt/scai-test/manifest/pip-check.txt || true) \
    && sort -o /opt/scai-test/manifest/pip-check.txt /opt/scai-test/manifest/pip-check.txt \
    && dpkg-query -W -f='${Package} ${Version}\n' | sort > /opt/scai-test/manifest/builder-os-packages.txt \
    && python - <<'PY'
import json
import re
from pathlib import Path

manifest = json.loads(Path('/opt/scai-test/lock-manifest.json').read_text())
actual = [line.strip() for line in Path('/opt/scai-test/manifest/pip-check.txt').read_text().splitlines() if line.strip()]
allowed = manifest.get('allowed_pip_check', [])
if sorted(actual) != sorted(allowed):
    raise SystemExit(f'pip check differs from manifest allowlist: {actual!r} != {allowed!r}')

def freeze(text):
    entries = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or '==' not in line:
            continue
        name, version = line.split('==', 1)
        entries.append({'name': re.sub(r'[-_.]+', '-', name).lower(), 'version': version})
    return sorted(entries, key=lambda item: item['name'])

actual_freeze = freeze(Path('/opt/scai-test/manifest/pip-freeze.txt').read_text())
expected_freeze = manifest.get('expected_final_freeze')
if actual_freeze != expected_freeze:
    raise SystemExit(f'pip freeze differs from manifest: {actual_freeze!r} != {expected_freeze!r}')
PY

ARG SCAI_TEST_FINGERPRINT=UNSET
ARG SCAI_TEST_LOCK_SHA256=UNSET
ARG SCAI_SOURCE_REVISION=UNSET

FROM python:3.11.15-slim-bookworm@sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15

ARG SCAI_TEST_FINGERPRINT=UNSET
ARG SCAI_TEST_LOCK_SHA256=UNSET
ARG SCAI_SOURCE_REVISION=UNSET

LABEL org.opencontainers.image.source="https://github.com/minzhangcheng/scai" \
      org.opencontainers.image.revision="$SCAI_SOURCE_REVISION" \
      org.opencontainers.image.version="0.39.15" \
      io.scai.test-deps.fingerprint="$SCAI_TEST_FINGERPRINT" \
      io.scai.test-deps.lock-sha256="$SCAI_TEST_LOCK_SHA256" \
      io.scai.test-deps.python="3.11.15" \
      io.scai.test-deps.platform="linux/amd64" \
      io.scai.test-deps.base-digest="sha256:77923445c077d8eb971b14b2b114a1d9cd4a87edb4c75654820ca4832ee8cb15"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git gosu pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/scai-test/venv /opt/scai-test/venv
COPY --from=builder /opt/scai-test/requirements-test.lock /opt/scai-test/requirements-test-nodeps.lock /opt/scai-test/
COPY --from=builder /opt/scai-test/build-constraints.txt /opt/scai-test/build-constraints.txt
COPY --from=builder /opt/scai-test/lock-manifest.json /opt/scai-test/lock-manifest.json
COPY --from=builder /opt/scai-test/manifest /opt/scai-test/manifest

RUN useradd --uid 10001 --home-dir /tmp --no-create-home scai-test \
    && chown -R root:root /opt/scai-test \
    && chmod -R a-w /opt/scai-test \
    && chmod 0755 /opt/scai-test /opt/scai-test/venv /opt/scai-test/venv/bin \
    && chmod a+r /opt/scai-test/manifest/*

# This final evidence-producing layer runs after all package/user/permission
# mutations. It is detached from itself and is checked by the verifier.
RUN dpkg-query -W -f='${Package} ${Version}\n' | sort > /opt/scai-test/manifest/final-os-packages.txt \
    && printf '10001 scai-test\n' > /opt/scai-test/manifest/runtime-user.txt \
    && rm -f /opt/scai-test/manifest/hashes.txt \
    && find /opt/scai-test/manifest -maxdepth 1 -type f ! -name hashes.txt \
        -exec sha256sum {} + | sort > /opt/scai-test/manifest/hashes.txt \
    && chmod a+r /opt/scai-test/manifest/*

ENV PATH=/opt/scai-test/venv/bin:$PATH \
    HOME=/tmp \
    TMPDIR=/tmp

# No application source, command, entrypoint, healthcheck, port, or startup
# operation belongs in this dependency-only image.
