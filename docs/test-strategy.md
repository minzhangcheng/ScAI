# Test Strategy

This document defines the active test boundary for the repository. It is a
test and evidence contract, not a release report. The supported Python entry
point is `scripts/scai-test`; the supported frontend entry point is
`scripts/scai-frontend-test`. `test-profiles.toml` is the single source of
truth for the Python profile roster and profile arguments.

## Immutable Python test boundary

`scripts/scai-test` selects a locally available, immutable
`scai-test-deps` image by a verified image ID and dependency fingerprint. The
runner refuses a missing or mismatched identity; it does not pull an image or
fall back to a host environment. Each container run uses the checkout
read-only and a separate writable evidence directory with these boundary
properties:

- `--pull=never`, `--read-only`, `--network none`, `--user 10001`, and
  `--platform linux/amd64`;
- a `--tmpfs /tmp:rw,size=2g` temporary filesystem and no `/dep`, `/extra`,
  production `/data`, or host virtual environment;
- `NUMBA_DISABLE_JIT=1` for the test environment;
- `SCAI_RUN_KERNEL_TESTS=1` for the `kernel` profile;
- `SCAI_ENABLE_SNAPSHOT_V2=true` as a test-only default from the test
  conftest, while the production default remains disabled.

The image contains the candidate testing tools in its immutable virtual
environment. The required identities are `pytest==9.1.1`,
`pytest-asyncio==1.3.0`, and `pytest-benchmark==5.2.3`. The benchmark pin is
retained because it is an unavoidable unconditional dependency of the
installed infercnvpy dependency graph; no target test consumes the benchmark
fixture or marker, and the dependency contract keeps that absence explicit.
Complete collection remains a pytest-plugin/asyncio/benchmark semantic gate,
not a benchmark feature or a performance claim.

## Supported profiles and claim scope

The profile roster is exactly the following five profiles:

| Profile | Invocation | Scope of a successful claim |
| --- | --- | --- |
| `fast` | `scripts/scai-test fast <explicit node or file>` | focused edit-loop evidence only |
| `client` | `scripts/scai-test client` | complete retained Python client suite |
| `kernel` | `scripts/scai-test kernel` | complete known real-kernel/integration evidence |
| `backend-full` | `scripts/scai-test backend-full` | complete backend suite |
| `backend-full-gate-off` | `scripts/scai-test backend-full-gate-off` | complete backend suite with the snapshot subsystem disabled, matching the production default |

The five claim scopes are deliberately distinct: `fast` is focused edit-loop
scope, `client` is the affected client-surface scope, `kernel` is integration
scope, `backend-full` is full-suite/release-gate scope, and
`backend-full-gate-off` is the same suite in the configuration production
actually runs. A complete result within one scope does not imply any of the
other scopes; in particular a `backend-full` result says nothing about the
subsystem-disabled configuration, because the repository conftest forces
`SCAI_ENABLE_SNAPSHOT_V2=true` while production defaults it off.

The raw form accepts only explicit pytest arguments and does not create another
profile:

```bash
scripts/scai-test fast backend/tests/test_x.py::test_y
scripts/scai-test client
scripts/scai-test kernel
scripts/scai-test backend-full
scripts/scai-test backend-full-gate-off
scripts/scai-test -- <raw pytest args>
```

Every run records the expanded command, image and dependency identity, source
identity, collection, outcome counts, duration, and artifact hashes below
`test-results/`. Focused edit-loop evidence supports only the selected nodes;
it is not backend-full or release evidence. A profile that was not run must
not be reported as passed, and broad profiles are not repeated when source,
image identity, configuration, and test inputs are unchanged. Known baseline
failures are reported separately; an F0 defect is not relabeled as baseline.

## Dependency locks and image provenance

`scripts/lock-test-deps.sh --check` is the offline pre-lock contract.
`scripts/lock-test-deps.sh --update` is a separate, explicitly authorized
network operation and publishes the main lock, the no-dependencies lock, and
the manifest atomically only after resolver reports, downloaded artifact bytes,
hashes, clean-installer identity, inventories, freeze, and the allowed
`pip check` observation agree.
The resolver bootstrap identity is deliberately separate from the clean
verification-venv installer identity; the clean installer is not silently
upgraded.

`build-constraints.txt` is one tracked input shared by the resolver's PEP 517
build constraints, the clean hash-lock installer, and the final test image
builder. The image builder receives only the dependency context, performs the
manifest hash-before-install check for that file before either hash-lock
install, retains the constraint and lock byte proofs, and records the final
inventories. Source files, application commands, and production data are not
part of the image context. A local image build (`scripts/build-test-deps.sh`,
with its local
`--load` path) and immutable verifier run are separate from selecting the
tracked image pointer. Registry push and any tracked pointer update are
separate reviewed actions.

## Frontend boundary

Frontend testing uses the same repository source but an isolated installation
tree. The only supported invocation is:

```bash
scripts/scai-frontend-test --install-and-test
```

This command is an independent authorization checkpoint because `npm ci`
requires network access. It hashes the package inputs before installation,
serializes the global and input-key locks, creates a fresh run directory,
performs `npm ci`, then runs `npm ls --all --json` and Vitest from that same
verified tree. The dual npm-config value and source-provenance artifacts and
the lock order are rechecked before execution;
the source `node_modules` tree is restored after success or failure. Client
and frontend package versions are independent surfaces and are not changed by
the Python dependency image workflow. Frontend evidence includes the source
and lock digests, effective npm configuration without secrets, the canonical
tree and bin-link inventory, and aggregated Vitest counts/details.

## Host-owned evidence and fail-closed validation

The host owns the final evidence directory. Files emitted by a test container
are untrusted until the runner checks an exact allowlist, regular-file type,
single-link count, safe resolved path, readable mode, and JSON/schema
contracts. Collection records a complete sorted node-id list and its hash.
Outcome records must form an exact node-ID bijection with that collection, and
host-owned hashed outcome-detail records carry the per-test status, phase,
warning, xfail/xpass, and traceback facts. JUnit is a pytest 9.1.1
phase-to-JUnit aggregate projection cross-check, not the authoritative outcome
source; its root-suite counters must agree deterministically with the outcome
projection. No result may describe a failure as a pass.

The runner captures warnings through pytest's warning hook, including historic
warnings emitted before session start, with category, message, node id, and
phase. It does not parse terminal output to manufacture warning evidence.
Source manifests are reloaded and measured before and after the container run
and their digests must be identical. The manifest excludes its own output
from its hash domain, and published artifact hashes are checked without
self-hashing the manifest. A kernel profile that intentionally skips an
unavailable kernel records an outcome-derived skip and its reason; it is not
silently counted as a successful kernel execution. Vitest details are
machine-readable and aggregate to the published top-level counts.

## Explicit exclusions and interpretation rules

Do not invoke host `pytest`, `python -m pytest`, `/dep`, `/extra`, xdist, the
timeout plugin or timeout options, benchmark-specific options or claims, a
Node image or candidate application image from this workflow. This strategy
does not authorize network access, Docker builds, registry pushes, deployment,
or production `node_modules` changes; each such boundary needs its own
explicit authorization.

The root `conftest.py` makes `SCAI_ENABLE_SNAPSHOT_V2=true` effective inside
pytest. Every Python evidence record and summary discloses that test-only
configuration; it makes no claim about the production default, which remains
disabled.

A passing focused profile is not full-suite proof, a healthy existing
application container is not proof that the current source was rebuilt, and
test evidence is not scientific approval. Do not copy later profiles, suite
timings, or failure counts into a current handoff. The evidence records what
ran and what the host could verify; scientific interpretation, baseline
disposition, and release acceptance remain separate decisions.
