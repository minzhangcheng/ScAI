# Immutable Python test dependencies

This directory describes the source-independent Python 3.11.15/linux-amd64
test image used by `scripts/scai-test`. The image contains the complete
hash-locked dependency graph but no ScAI source, tests, frontend, Git metadata,
production dependency volume, or startup entrypoint.

`requirements-test.in` contains exactly the three target-tool pins:
`pytest==9.1.1`, `pytest-asyncio==1.3.0`, and `pytest-benchmark==5.2.3`.
The benchmark distribution is an unavoidable unconditional dependency of the
retained `infercnvpy` resolution; it does not authorize benchmark-specific
tests, and the target backend/client tests contain no benchmark fixture,
decorator, or import consumer. The generated lock files contain exact
`name==version --hash=sha256:...` records. The `lock-manifest.json` records
input hashes, resolver and installer identity, bootstrap transitions, the
exact code-owned `pip check` allowlist, and final freeze inventory.

The final image builder receives the tracked root `build-constraints.txt`
through the Dockerfile-specific root-context allowlist. It verifies that
file's SHA against the lock manifest before either hash-lock installation and
uses that same verified path through `PIP_CONSTRAINT` for both installs. The
constraint bytes are retained in the image and covered by detached evidence.

## Authority boundaries

- `scripts/lock-test-deps.sh --check` is offline and Docker-free.
- `scripts/lock-test-deps.sh --update` is the only test-lock resolution path;
  it requires explicit package-index authorization and writes only the two
  lock files plus `lock-manifest.json`.
- `scripts/build-test-deps.sh` builds locally by default and writes only the
  ignored local image identity. `--push` is a separate registry authorization;
  the tracked pointer is never edited automatically.
- `scripts/verify-test-deps-image.sh` accepts only an immutable image ID or
  `name@sha256:<digest>` and runs probes with UID 10001, read-only root,
  network disabled, and the declared platform.

A null `docker/test-deps-image.json` pointer is accepted when a matching local
immutable image ID has been independently verified. A non-null pointer keeps
the repository name and digest in separate fields; the runner constructs
`name@digest` only at execution time. The fingerprint discovery tag is never
an execution reference.

The image is not a production application image and is not pushed or deployed
by F0.
