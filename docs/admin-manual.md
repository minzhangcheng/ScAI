# ScAI Administrator Manual

## 1. Purpose and audience

This manual is for administrators who deploy and operate one ScAI instance, manage accounts and model credentials, protect persistent data, monitor service health, and support programmatic use. Analysis users should instead read the [user guide](user-guide.md). Python package details are also available in the [client README](../clients/python/README.md), but current client source is authoritative where that document differs.

Examples use visibly marked values such as `<API_KEY>`, `<SESSION_ID>`, and `<ARCHIVE_PATH>`. Run commands from the repository root unless stated otherwise.

## 2. Architecture and requirements

ScAI production serves the built Vite/React application, REST API, and WebSockets from one FastAPI/Uvicorn process on port `3000`. Development keeps a Vite server on `3000` and a proxied FastAPI/Uvicorn backend on `8000`. Both modes use local analysis kernels, external OpenAI-compatible model services, and disk-backed application/session storage.

The supported deployment choices are:

- Docker: a Node.js 20.19 build stage compiles the frontend; the Python 3.11 runtime contains no Node.js/npm and exposes the single application port `3000`.
- Direct startup: a Linux environment with GNU-compatible `find`, `grep`, `head`, `cut`, `tr`, `cmp`, and `cp`; `python3.12` for virtual-environment creation; `python3`; and `pgrep`, `pkill`, `lsof`, and `kill` is required. Node.js 20.19+ or 22.12+ and npm/npx are required for development and whenever production must rebuild stale frontend assets. The scripts are not designed as portable POSIX-only tooling.

Size storage for uploaded datasets, generated artifacts, logs, temporary uploads, and dependencies. The host must be able to reach configured model endpoints and the Python package registry during startup. Docker startup installs Python requirements, so the first start can be substantially slower; frontend dependencies are resolved only while the image is built.

### GPU acceleration for analysis methods

GPU access is optional. The shipped stack has a CPU fallback, and adding a GPU
does not change the scientific method or make every package GPU-accelerated.
The following classification describes the packages pinned in
`requirements.txt` and the default analysis workflow:

| Package or workload | GPU guidance |
|---|---|
| `scvi-tools==1.3.3` and its PyTorch backend (`SCVI`, `SOLO`, `SCAR`, `totalVI`, and similar models when selected) | CPU is supported; GPU is recommended for large single-cell/single-nucleus model fitting and repeated training epochs. The default `accelerator="auto"` selects a visible GPU and otherwise uses CPU. |
| Scanpy/AnnData preprocessing, QC, normalization, HVG selection, PCA, UMAP, Leiden/igraph, NumPy/SciPy/Numba/scikit-learn | CPU path in the shipped image; no GPU is required. |
| `scvelo`, `cellrank`, `harmonypy`, `bbknn`, `scanorama`, `scrublet`, `doubletdetection`, `celltypist` | CPU path in the shipped image; GPU is not required. |
| `gseapy`, `decoupler`, `pyscenic`, `infercnvpy`, `squidpy`, `liana`, `omnipath`, `pydeseq2`, and `muon` | CPU path in the shipped image; GPU is not required. |
| HDF5/Zarr I/O, Dask-image, Pillow, Matplotlib, Seaborn, and Plotly | CPU/I/O or rendering work; GPU is not required. |

The image does not install RAPIDS, cuML, cuGraph, or CuPy as alternate
backends. Therefore GPU mapping alone does not accelerate Scanpy PCA/UMAP,
Leiden, RNA velocity, integration, or the other CPU rows above. Those methods
would need a separately validated GPU backend and dependency image; do not add
one ad hoc to a production `/dep` environment. A GPU also does not replace the
RAM and disk capacity required by large AnnData objects.

### Creating a GPU-enabled Docker container

Use this option when the host has a compatible NVIDIA driver, NVIDIA Container
Toolkit, and at least one GPU available to Docker. The `--gpus` flag must be
present when the container is created; `docker restart` cannot add GPU devices
to an existing container.

```bash
docker pull ghcr.io/minzhangcheng/scai:latest
docker run -d --name scai \
  --gpus all \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  --env-file <PROTECTED_ENV_FILE> \
  --restart unless-stopped \
  ghcr.io/minzhangcheng/scai:latest
```

To assign only GPU 0, replace `--gpus all` with
`--gpus '"device=0"'`. Preserve the existing `/dep` and `/data` volumes and
the same `SCAI_JWT_SECRET` and `SCAI_CONFIG_SECRET` values when recreating an
existing instance. Apply a suitable `--memory` limit as described in the
memory-sizing section.

After startup and dependency installation, verify that PyTorch can see the
mapped device:

```bash
docker exec scai /dep/venv/bin/python -c \
  'import torch; print({"cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})'
```

`cuda_available` should be `True` and `device_count` should be at least `1`.
`torch.version.cuda` only reports the CUDA version supported by the installed
PyTorch build; it does not prove that the container has access to a GPU. If
the check is false, inspect the host's `nvidia-smi`, NVIDIA Container Toolkit,
Docker runtime configuration, and the container's device mapping before
starting an analysis.

### Memory sizing

Almost all of an instance's memory is consumed by the analysis kernels, not by
the backend. **Each session with a live analysis holds its own kernel process**,
so a host running N concurrent analyses needs N times the per-analysis figure
below plus the fixed platform overhead.

Single-cell workflows are memory-hungry for a structural reason: a stage-1 run
simultaneously holds the per-sample matrices, their QC-filtered copies, and a
merged object carrying a counts layer, a lognorm layer and `.raw`. That is six
to ten copies of the same expression data before any dense transformation.

**Reference measurement.** On the reference deployment, a session with 2 samples,
19,128 cells and 22,804 genes (392 MB on disk as `.h5ad`) reached a resident set
of **6.4 GiB** in its kernel, alongside 0.96 GiB for the backend process. The
sizing model below is fitted to that point:

```
kernel peak (GiB) ≈ 0.4 + (cells × 330 KB) + (samples × 50 MB)
platform overhead ≈ 1.6 GiB   (backend, frontend, OS headroom)
```

The recommended figures include **30% safety margin** over the modelled peak.

| Samples | Total cells | Kernel peak | **Recommended** |
|--------:|------------:|------------:|----------------:|
| 2  | 10,000  | 3.6 GiB   | **7 GiB**    |
| 2  | 20,000  | 6.8 GiB   | **11 GiB**   |
| 4  | 40,000  | 13.2 GiB  | **20 GiB**   |
| 6  | 60,000  | 19.6 GiB  | **28 GiB**   |
| 8  | 100,000 | 32.3 GiB  | **44 GiB**   |
| 12 | 200,000 | 63.9 GiB  | **86 GiB**   |
| 20 | 500,000 | 158.7 GiB | **209 GiB**  |
| 100 | 1,000,000 | 320.0 GiB | **418 GiB** |

The recommended column is **per concurrent analysis**. Earlier releases carried
a second, higher column for instances running snapshots; snapshots are gone
(section 8) and that margin is no longer needed. Meanwhile,
use the middle column.

**Assumptions and limits of this table.**

- Whole-transcriptome data with 15,000–35,000 genes. Panel data with far fewer
  genes uses proportionally less; targeted assays with very deep per-cell
  detection use more.
- Cell count dominates. Memory scales with the number of non-zero entries, which
  grows with cells, not with the size of the gene annotation.
- The model is fitted to **one** measured deployment. Treat the 200,000-,
  500,000-, and 1,000,000-cell rows as extrapolation, not measurement, and
  verify before committing hardware at that scale.
- Peaks are transient. A dense transformation of a full matrix costs
  `cells × genes × 4 bytes` on top of the table — 1.6 GiB for the reference
  dataset, and it scales with the gene count rather than with detected genes.
  Analyses that restrict such work to highly variable genes stay far below this.

The 100-sample row assumes approximately 10,000 retained cells per sample. A
different number of samples with the same total cell count has a similar matrix
cost; sample count adds smaller per-sample QC and metadata overhead. Treat this
row as capacity-planning guidance, not a claim that every analysis method is
validated at one million cells.

#### When the runtime is saved

The kernel's runtime is persisted on a schedule, not continuously:

| Boundary | Behaviour |
|---|---|
| **Round boundary** | Always. The graph reaches it once per round, after the round's work is complete and before the next one starts. |
| **Step boundary** (an executor turn inside a round) | Only once `SCAI_RUNTIME_SAVE_FLOOR_MINUTES` (default **30**) have passed since the last **successful** save. The boundary is reached many times a round, so the floor is what keeps the cost down. |
| **Closing** (the analysis ends) | Exactly one attempt. A failure keeps the previous committed generation, is recorded, and does not hold the shutdown. |

So the work at risk from a crash is bounded by **one round, or the floor,
whichever comes first** — and that bound **holds only while saves succeed**.

Measured cost of one save, for sizing the floor: **13 s** for a 190 MiB dump and
**56 s** for a 972 MiB one, at roughly 54 MiB/s. Against a 30-minute floor that
is under 1% and about 3% of the interval respectively, so the default is
conservative rather than expensive; lowering it trades wall clock for a smaller
exposure window.
After a failed save the window runs from the last successful commit, not from
the failed attempt, and it keeps growing until one succeeds. The analysis page
stays quiet when a live kernel is merely ahead of its last successful commit,
because that is the expected state between save points. It still shows a
non-blocking warning for a failed save, an inconsistent generation, or a
non-live session whose committed generation is behind the session; those
warnings show the time of the last successful save rather than the last attempt.

The 30-minute default is not a measurement. It is a starting value pending
before/after save timings on real sessions; expect it to be confirmed or
adjusted in a later release. Lower it on an instance where the kernel is small
and crashes are the concern; raise it where saves are slow and the round
boundary is protection enough.

#### Runtime persistence budgets

Runtime persistence has its own resource gate. By default, its memory
ceilings are **automatic**, derived from the Docker cgroup memory limit when one
is present, otherwise from host memory:

| Budget | Automatic ceiling |
|---|---:|
| Total live AnnData size | 45% |
| Analysis-kernel RSS | 60% |

These are per-session persistence ceilings, not a global scheduler. On a shared
instance, size the container and any explicit per-session caps for the maximum
number of concurrently live analysis kernels. An unbounded container lets every
session see host capacity, so two sessions can each receive an individually
valid budget whose sum exceeds the host. Prefer a finite container memory limit,
or install explicit `SCAI_RUNTIME_MAX_*` ceilings from the environment-variable
table in section 17.

Unset capacity variables and the explicit value `auto` both select the automatic
ceiling. A positive numeric GiB value is an administrator ceiling; the effective
limit is the lower of that value and the automatic ceiling. For example:

```bash
docker run -d --name scai --memory=128g \
  -e SCAI_RUNTIME_MAX_RSS_GB=64 \
  -e SCAI_RUNTIME_MAX_ANNDATA_TOTAL_GB=48 \
  ... ghcr.io/minzhangcheng/scai:latest
```

Illustrative explicit profiles are:

| Total cells | RSS | AnnData total |
|---:|---:|---:|
| up to 50,000 | 32 GiB | 24 GiB |
| 50,000–200,000 | 64 GiB | 48 GiB |
| 200,000–500,000 | 256 GiB | 192 GiB |
| around 1,000,000 | 512 GiB | 384 GiB |

The last two profiles are extrapolations and assume one large analysis per
allocated memory share. Use the automatic defaults unless an operator needs a
stricter concurrency boundary. ScAI does not impose namespace-count,
single-object, regular-runtime-PKL-size, or save-duration ceilings; operators
are responsible for monitoring data-volume free
space and inode capacity. Disk capacity does not evict live Python objects or
reduce kernel RSS. Backed AnnData may also need materialisation during regular
persistence, so disk-backed input is not itself an out-of-core execution
guarantee.

**Do not run an instance without a memory ceiling.** An unbounded kernel that
exhausts host memory invites the kernel OOM killer to terminate an arbitrary
process, potentially another service on the same host. Set a container limit
sized from the table:

```bash
docker run -d --name scai --memory=32g ... ghcr.io/minzhangcheng/scai:latest
```

## 3. Docker deployment

### Pull and run

```bash
docker pull ghcr.io/minzhangcheng/scai:latest
docker run -d --name scai \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  --env-file <PROTECTED_ENV_FILE> \
  --restart unless-stopped \
  ghcr.io/minzhangcheng/scai:latest
```

**Warning:** reuse the same protected `SCAI_JWT_SECRET` and `SCAI_CONFIG_SECRET` values used by the previous container. The example uses an environment file for brevity; protect that file from other users. Recreating the container with different secrets invalidates existing JWTs and can make encrypted model credentials unreadable.

Open `http://localhost:3000`. `/dep` stores the Python environment; `/data` stores application data. Frontend assets are compiled into the image and served by FastAPI. The entrypoint creates both mounted directories, recursively assigns them to UID `10001` (`scai`), then drops root privileges. Session directories are normally restricted to mode `0700`. Ensure the host directories can safely have their ownership changed, and use elevated host privileges when backup tooling cannot read them. Do not change ownership on live data merely to make a backup command succeed.

The image exposes only `3000`; the UI, API, and WebSockets all use that port. There is no separate runtime service on `8000` to publish.

Useful container environment variables are:

```bash
docker run -d --name scai \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  -e SCAI_MODE=prod \
  -e SCAI_JWT_SECRET='<STABLE_RANDOM_SECRET>' \
  -e SCAI_CONFIG_SECRET='<STABLE_RANDOM_SECRET>' \
  -e PIP_INDEX_URL='<PYTHON_PACKAGE_INDEX_URL>' \
  ghcr.io/minzhangcheng/scai:latest
```

`SCAI_MODE` defaults to `prod`; `--prod` is accepted after the image name. The production runtime deliberately rejects `--dev` because it contains no Node.js/npm; use a source checkout with `./start.sh --dev` for Vite hot reload. Docker always exports `SCAI_DATA_DIR=/data`. `PIP_INDEX_URL` is passed to dependency installation when set.

### Health, logs, lifecycle, and removal

The built-in health check calls `http://127.0.0.1:3000/api/health` every 30 seconds, with a 5-second timeout and 60-second startup grace period.

```bash
docker ps --filter name=scai
docker inspect --format '{{.State.Health.Status}}' scai
curl -f http://localhost:3000/api/health
docker logs -f scai
docker stop scai
docker start scai
docker restart scai
```

Container output is also appended to `/data/logs/scai.log`.

### Update and backup

Back up the data volume before updating, then replace the container while retaining both host-mounted directories:

```bash
docker pull ghcr.io/minzhangcheng/scai:latest
docker stop scai
docker rm scai
docker run -d --name scai \
  -p 3000:3000 \
  -v "$(pwd)/scai-dep:/dep" \
  -v "$(pwd)/scai-data:/data" \
  --restart unless-stopped \
  ghcr.io/minzhangcheng/scai:latest
```

To remove the service but preserve data, stop and remove only the container. **Warning:** deleting `scai-data` destroys accounts, configuration, sessions, reports, and uploads.

## 4. Direct startup and shutdown

Install Python 3.12, clone the repository, and run. Install Node.js 20.19+ or 22.12+ with npm when the production frontend must be built or when using development mode:

```bash
./start.sh
```

Defaults are production mode, host `127.0.0.1`, and a single application port `3000`. In development, `--backport` selects the proxied backend port and defaults to `8000`. Supported options are exactly:

```text
./start.sh [--dev | --prod] [--host HOST] [--port PORT] [--backport PORT]
./start.sh -h
./start.sh --help
```

Example for network access and non-default ports:

```bash
./start.sh --prod --host 0.0.0.0 --port 3000
```

Relative to the repository directory `<REPO>`, startup stores dependencies in `<REPO>/../dep`, data in `<REPO>/../data`, and process metadata in `<REPO>/.scai.pid`. It creates the environment with `python3.12 -m venv`, updates pip, installs backend requirements, and starts Uvicorn on the application port. If any frontend build input is newer than `dist/index.html`, it installs the exact `package-lock.json` graph with `npm ci` outside the checkout and rebuilds first. Development instead records both Uvicorn and Vite PIDs.

Direct-run environment controls include `SCAI_DATA_DIR` (backend data-root override), `SCAI_JWT_SECRET`, `SCAI_CONFIG_SECRET`, `SCAI_ENABLE_REPRODUCIBILITY`, `SCAI_AUTO_FETCH_MARKERS=1`, and `SCAI_AUTO_FETCH_CELLTYPIST=1`. The two fetch options download optional resources if missing and do not block startup on failure: the marker databases for annotation priors, and the CellTypist model repertoire (each model is a few MB) into `$SCAI_DATA_DIR/celltypist`, which every analysis kernel sees through its `CELLTYPIST_FOLDER`. Without the cache a kernel downloads a model on demand and needs network access to do so. For the container, run `docker exec <container> /dep/venv/bin/python /app/scripts/fetch_celltypist_models.py` once; the cache persists on the data volume. When `SCAI_DATA_DIR` is set, `start.sh` still creates and prints `<REPO>/../data`, but the backend actually reads and writes the override path; monitor and back up the override, not the printed sibling directory.

Stop the instance with:

```bash
./stop.sh
```

**Warning:** `stop.sh` still has broad fallback process-name matching after it reads `.scai.pid`. It releases the application/backend ports recorded by `start.sh`, including custom values, but may terminate an unrelated process occupying one of those ports. On shared hosts or multi-instance installations, inspect the recorded PIDs and process command lines first and prefer stopping only the intended PIDs:

```bash
lsof -i:<FRONTEND_PORT>
lsof -i:<BACKEND_PORT>
kill <PID>
```

Do not delete a valid `.scai.pid` while services run. If startup reports a stale PID file, first verify that its PIDs no longer exist, then remove only that stale file.

## 5. Building and publishing images

Build directly:

```bash
docker build -t ghcr.io/minzhangcheng/scai:latest .
```

The release helper always builds both the selected tag and `latest`:

```text
./build.sh [--version TAG] [--push] [-h | --help]
```

```bash
./build.sh --version 0.57.10
./build.sh --version 0.57.10 --push
```

Without `--version`, the helper uses the newest Git tag returned by `git describe --tags --abbrev=0`, or `latest` if none exists. `--push` pushes both tags to `ghcr.io/minzhangcheng/scai`; authenticate Docker to that registry before using it. The helper does not accept registry, image-name, platform, or build-argument flags.

## 6. First-run and user administration

On first access, create the initial administrator through the setup screen. The backend permits this operation only until initialization is recorded. Usernames and optional email addresses must be unique.

Administrators can create active ordinary users, list users, reset passwords, change roles, change account status, and delete users. Self-service registration creates a `pending` account that cannot log in until an administrator activates it. Administrator-selectable status values are `pending`, `readonly`, and `active`; `deleting` is a server-owned, non-approvable state that means account cleanup must finish or be retried. Roles are `admin` and `user`. Administrators cannot alter their own role/status or delete themselves, cannot change the status, role, or password of a `deleting` account, and ScAI prevents removal or demotion of the last administrator.

**Warning:** before deleting a user, stop every analysis owned by that user and back up required data. User deletion evicts cached sessions and removes persisted session directories, legacy state files, API keys, encrypted user configuration, and database records, but it does not itself stop active workflow/task objects. After deletion, verify that no workflow or worker remains for the user's session IDs and that the expected files and database/session listings are gone.

**Upgrade note for installations predating 0.38.11:** an interrupted deletion created by older code may still be stored as `pending`, which is indistinguishable in the database from a genuine account awaiting approval. ScAI therefore does not migrate these rows automatically. Before approving historical `pending` accounts, reconcile them against deletion/cleanup logs and retained session or import-job records. For an account confirmed to be an interrupted deletion, retry Delete so the server records `deleting` and resumes cleanup; do not approve it or bulk-convert all `pending` rows.

Configure the global model API key, model base URL, and optional proxy in the administration UI. Saving configuration persists it but does **not** validate the credentials. Run the explicit validation action separately. Analysis startup checks that required configuration is present; provider authentication, selected-model availability, quota, and network failures may not become visible until the first model request. Active users may store per-user model configuration, which takes precedence according to server configuration resolution.

Each active user can create API keys for automation. A key begins with `scai_`, belongs to its creator, and is displayed in full only once. Give it a descriptive label, store it in a secret manager, and revoke unused keys. JWT login tokens expire after 72 hours; API keys remain until revoked.

## 7. Persistent data and directory layout

Docker uses `/data`; direct startup normally uses the sibling `data` directory unless `SCAI_DATA_DIR` overrides it.

| Path under the data root | Purpose |
|---|---|
| `app.db` | SQLite users, API-key hashes, global settings, and encrypted user model settings |
| `.jwt_secret` | Auto-generated JWT signing secret when no environment secret is supplied |
| `.config_secret` | Local encryption material when no configuration secret is supplied |
| `sessions/<SESSION_ID>/` | Authoritative per-session metadata and files |
| `sessions/<SESSION_ID>/session.json` | Session metadata |
| `sessions/<SESSION_ID>/uploads/` | Session inputs |
| `sessions/<SESSION_ID>/workspace/` | Analysis code and figures |
| `sessions/<SESSION_ID>/states/` | Workflow/checkpoint files |
| `sessions/<SESSION_ID>/reports/` | Generated reports |
| `sessions/<SESSION_ID>/logs/` | Session logs |
| `uploads/`, `states/`, `workspace/`, `reports/` | Legacy top-level storage and compatibility paths |
| `logs/` | Service logs; Docker writes `logs/scai.log` |
| `upload_temp/` | In-progress upload data |

Docker only needs `/dep/venv` persisted; the frontend assets are compiled into the image. Direct startup uses `<REPO>/../dep/venv` and `<REPO>/../dep/frontend/node_modules`. Dependencies are reproducible and may be rebuilt, but persistence reduces restart time.

## 8. Decision snapshots

> **Status: Experimental.** Decision snapshots may be unstable. Capture, restore, the
> on-disk format, and the user interface may change between versions. Do not
> treat snapshots as a backup; keep independent backups of required data.

A decision snapshot is a restore point taken immediately before one of the
four decisions an analysis asks about itself. Restoring one **creates a new
session**; the session it was taken from is never modified. The feature is off
until you give it somewhere to write.

This is not the snapshot subsystem earlier builds had. That one — its HTTP API,
its object store, its capture gates and the `SCAI_ENABLE_SNAPSHOT_V2` switch —
was removed, and this replaces it with something narrower on purpose: no
branch tree, no history browser, no way to make one by hand.

### Turning it on

Set `SCAI_SNAPSHOT_DIR` to an absolute path. Unset means off; there is no
separate switch, so the directory and the feature cannot disagree. Point it at
storage **separate from `SCAI_DATA_DIR`**: a snapshot is meant to survive the
loss of the volume a session lives on, and a root under a session tree is
reachable by analysis code. A relative path is refused and logged, and a root
inside the data directory is warned about but accepted, for the single-volume
case.

Sizing it is the decision that matters. A snapshot stores the session's kernel
dump whole, and finished sessions on a working deployment hold 3–8 GB dumps.
Two snapshots that hold the same bytes share one copy — the filesystem counts
the references — so a session that reaches several decisions without executing
anything in between costs one dump, not several. A session that works between
every decision costs one per decision.

### What is captured, and when

Four points, all of them immediately before a decision is made, and nowhere
else:

| Decision | When |
|---|---|
| Confirm the cell annotation | Round 1, Stage 1, before the confirmation dialog |
| Choose the research direction | End of round 1, after the exploration report |
| Choose the direction again | End of every integration round |
| Accept the round's result | End of every other round |

An error or incident dialog — an LLM failure, a disabled agent, a missing
marker database — is not one of these and is not snapshotted. A run with the
skip time set to zero, where the PI decides instead of the user, is: at this
layer that is the same decision made by a different party, and a user who comes
back to an unattended run is exactly who a rollback is for.

The capture holds the analysis for the time it takes to copy the session's
**workspace** and nothing else. The kernel dump is held by a hardlink, which
costs nothing regardless of size, and copied afterwards; the uploaded inputs
are copied afterwards too. Two bounds stop a slow or hung target from turning
those copies into an outage: `SCAI_SNAPSHOT_STALL_SECONDS` (default 120)
abandons a copy that has written no bytes for that long, and
`SCAI_SNAPSHOT_MAX_SECONDS` (default 1800) is the backstop under it. A capture
that is abandoned records the decision as having no snapshot and the analysis
carries on unaffected.

Both bounds govern the copies and nothing else, and each copy phase gets its
own clock. They do **not** bound the runtime save a capture asks for when the
kernel has moved since the last one, which is usually most of a capture's
elapsed time: measured on ordinary array storage, the copies took 6-11 s while
the saves took 46-73 s. So size these two from what the *copies* cost on your
storage, not from the `capture_seconds` the listing reports -- that figure
includes a save these knobs cannot stop. The save is the same one the ordinary
persistence cadence takes; see `SCAI_RUNTIME_SAVE_FLOOR_MINUTES` for the knob
that governs it.

### Space and snapshot importance

**Nothing is deleted automatically.** That is the default and it is deliberate:
a snapshot is the only copy of the decision it precedes. When space runs short,
delete them one at a time, starting with the lowest importance. The analysis
panel labels every snapshot with its importance rank (`1` is highest):

| Importance rank | Decision | Why |
|---|---|---|
| 1 (highest) | Round 1 cell annotation | The earliest decision; every later label depends on it |
| 2 | Round 1 research direction | The choice the rest of the analysis follows |
| 3 | Integration-round direction | The choice after each cross-direction analysis |
| 4 (lowest) | Round acceptance | Everything else |

Within one importance rank the **later** snapshot is given up first. The API and
`scai snapshots list` report `deletion_order` on every row; the optional prune
uses that exact order.

If you would rather that happened without you, set
`SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB`. While free space on the snapshot volume
is below it, a capture reclaims the next snapshot in that order — never the one
it just wrote, and never a round-1 annotation snapshot of a session that is
still running.

### What travels, and what does not

Deleting a session deletes its snapshots with it. If the feature is off at that
moment the tree is left where it is, under
`<SCAI_SNAPSHOT_DIR>/<session id>/`, and you can remove it by hand.

Snapshots **do not travel in an export**. There is no `include_snapshots`
parameter and no importer for one: they are local restore points, and an
archive nothing can read back would be dead weight. To move a session between
installations, use portable export/import (sections 9 and 13).

### Upgrading from a build that had the old subsystem

- **Existing snapshot data is inert, not migrated.** `sessions/<id>/v2/` and
  `sessions/<id>/snapshots/` are no longer read or written. They still occupy
  disk; deleting a session deletes them with it. You may delete those two
  directories to reclaim space once you are satisfied you do not need the old
  build. Nothing in this version can restore from them.

- **Existing snapshot data is inert, not migrated.** `sessions/<id>/v2/` and
  `sessions/<id>/snapshots/` are no longer read or written. They still occupy
  disk; deleting a session deletes them with it. You may delete those two
  directories to reclaim space once you are satisfied you do not need the old
  build. Nothing in this version can restore from them.
- **A session interrupted mid-restore will refuse to start.** If the previous
  build crashed while applying a restore, that session's files describe
  different points in the same run, and no code here can finish or roll it
  back. Startup logs the session IDs. The repair is to run the previous version
  once and let its recovery complete, then upgrade again. Every other session
  that was merely parked on a snapshot failure starts normally — the blocker is
  cleared on load.
- **Archives from an older build still import.** Their snapshot members are
  skipped, and the session imports with its content. An archive that carries
  *only* snapshot data and no session state is refused with an explanatory
  error.
- **Memory sizing is unaffected by this feature.** The "snapshots on" column
  in [Memory sizing](#memory-sizing) belonged to the old subsystem, which
  captured through the kernel; use the other one. A decision snapshot allocates
  no kernel memory at all — it copies files an ordinary save has already
  written.

**What protects a running analysis.** The kernel's runtime is saved at each persistence
boundary to `sessions/<id>/states/<id>_session.pkl` under a crash-recoverable
transaction, signed with this installation's integrity key, and reloaded when
the session's kernel next starts. That is a resume mechanism, not history: it
holds the current generation only, and there is no way to return a session to
an earlier point. For history and for moving a session between installations,
use portable export/import (sections 9 and 13); for protecting the service, use
a data-root backup (section 9).

## 9. Backup, upgrade, recovery, and rollback

**Stop long enough for the closing save.** A session with a live kernel writes
its runtime on shutdown, and that write is a full dump: measured at 57 s for a
1.28 GB kernel, and dumps of 3-8 GB are ordinary on real data. Docker's default
stop grace is 10 s, which kills the process partway through. What survives is
the last routine save, so an analysis that has run since then comes back unable
to continue: the kernel state on disk is older than the work the session
records, and the only remedies are a decision snapshot or re-running from the
data. Pausing first does **not** help -- a pause holds the executor between
cells and does not commit the runtime.

```bash
docker stop -t 300 scai      # not `docker stop`, and not `docker kill`
```

Give the timeout room for **every live session added together**, not for the
largest one. The closing saves run one after another, so an instance holding
two sessions needs both. Measured: a 400 s grace was not enough for two
sessions of a few gigabytes each, and the second came back with its kernel
behind its record while the first was clean.

The shutdown log ends with `Session saved successfully` followed by `Sandbox
stopped` for each live session; if it does not, the stop was too short. A
session whose closing save was cut off is recoverable -- it rolls back to its
last committed generation the next time the analysis is started -- but it gives
up the work since that save, which is the cost this whole procedure exists to
avoid.

Prefer a quiescent file-level backup so SQLite and session files agree:

1. Stop ScAI, with a grace period long enough for the closing save (above).
2. Copy the entire data root, including hidden files, preserving ownership, modes, timestamps, and links.
3. Record the application/image version and retain the stable secret values separately in a protected secret store.
4. Restart ScAI and verify health.

Example for Docker bind mounts:

```bash
docker stop -t 300 scai
sudo cp -a ./scai-data "./scai-data-backup-$(date +%Y%m%d-%H%M%S)"
sudo test -f "$(ls -dt ./scai-data-backup-* | head -1)/app.db"
docker start scai
```

**Warning:** confirm available disk space and backup destination before running recursive copies. `sudo cp -a` preserves UID `10001`, modes such as `0700`, timestamps, and links; do not `chown` or `chmod` the live volume for backup access. Verify the copied tree, file counts, sizes, hidden secret files, and a test recovery rather than relying only on the example's basic `app.db` check. Do not treat a copy taken during active writes as transactionally consistent.

For upgrades, back up, deploy the new code/image against the existing data root, then test login, configuration visibility, session listing, and a health request. Retain the old image tag/code revision until acceptance. For rollback, stop the new version and restore both the matching full data backup and previous application version. Never merge selected database and session files from different backup times unless performing a documented repair.

Recovery procedure: stop the service, move the damaged data root aside, restore the complete backup to the original path, restore required ownership, supply the same secrets, start the matching application version, and validate before allowing users back in.

Portable session export/import moves individual sessions between installations. Exported with `include_runtime=true` and imported with `X-Trust: true` it also carries the live kernel state and workflow checkpoint, so an imported session can continue analysing; the default export carries content only and cannot resume. Either way it is not a substitute for a complete service backup — see section 13 for the trade-off.

## 10. Monitoring and logs

Set `SCAI_CONTEXT_WINDOW_DEFAULT=200000` before backend startup (or pass it with
Docker `-e`) to set the default context budget for new analyses. The default is
200000 tokens; the minimum is 8192. An explicit environment value overrides
persisted global defaults, while manual Agent settings take precedence.
Explicit provider context-overflow errors reduce the affected Agent/model/endpoint
budget by 20% and rebuild the prompt within the existing retry allowance (normally
five total attempts). The reduced budget persists for that analysis across reloads.
Setting or resetting the Agent budget, or changing its model, clears the reduction.
The Agents page shows the effective budget. This setting is not the API output
`max_tokens` parameter. Original execution records remain available.

Production and Docker use the single application port for liveness. Only direct development mode exposes the backend separately on default port `8000`:

```bash
curl -f http://localhost:3000/api/health
curl -f http://127.0.0.1:8000/api/health  # development mode only
```

For Docker, combine health state and logs:

```bash
docker inspect --format '{{.State.Status}} {{.State.Health.Status}}' scai
docker logs --since 30m scai
docker stats scai
```

For direct operation, inspect ports and processes:

```bash
lsof -i:3000
lsof -i:8000
ps -fp "$(awk -F= '/_PID=/{printf "%s,", $2}' .scai.pid | sed 's/,$//')"
```

Monitor free disk and inode capacity on the data, dependency, and temporary-upload filesystems; process memory/CPU; container restart count; health latency; repeated authentication failures; model-provider errors; and growth of `logs/`, uploads, workspaces, and reports. Alert before disk exhaustion because uploads and report generation can be large.

## 11. Python client installation and authentication

The `scai-client` version follows the ScAI application release; it requires Python `>=3.10` and `httpx>=0.26.0`. Install from the repository checkout:

```bash
python3 -m pip install ./clients/python
```

Create an API key in the user interface and initialize the client:

```python
from scai_client import Client

client = Client(
    "http://localhost:3000",
    api_key="<API_KEY>",
    timeout=3600,
)
```

Constructor: `Client(base_url: str, *, api_key: str | None = None, jwt: str | None = None, timeout: int = 3600)`. At least one credential is required; when both are supplied, `jwt` is used. `base_url` has trailing slashes removed. `timeout` configures the underlying synchronous HTTP client.

Use a context manager to close connections automatically:

```python
with Client("http://localhost:3000", api_key="<API_KEY>") as client:
    sessions = client.list_sessions()
```

Public ordinary lifecycle methods are `close() -> None`, `__enter__() -> Client`, and `__exit__(*args) -> None`. Missing credentials or HTTP 401 responses raise `AuthError`.

## 12. Python client analysis jobs and artifacts

### Client job methods

`submit(config_path: str | Path) -> Job` reads a ConfigV1 JSON document, creates an analysis, uploads each existing `data.files[].local_path` (relative paths resolve from the config directory), and starts the job. Invalid configuration raises `ScaiError`; authentication raises `AuthError`. **A missing referenced local file is silently skipped, and the client still sends the start request.** Preflight every configured path before submission; do not assume the server will reject a job that started without the intended file. Uploads use 10 MiB chunks, three retries after the initial attempt per chunk, and server-side SHA-256 verification.

`get_job(job_id: str) -> Job` returns a handle without first checking existence.

```python
from scai_client import Client

with Client("http://localhost:3000", api_key="<API_KEY>") as client:
    job = client.submit("analysis-config.json")
    final = job.wait(interval=60, timeout=24 * 3600)
    print(final["status"])
    print(job.artifacts())
    job.download_artifact("report.html", "downloads/report.html")
    job.download_export("downloads/job-export.zip")
```

### Job constructor and methods

`Job(client: httpx.Client, job_id: str)` is normally created by `Client`; `job_id` is available as `job.job_id`.

| Method | Behavior and return value |
|---|---|
| `status() -> dict` | Fetch current job metadata. |
| `wait(interval: int = 60, timeout: int | None = None) -> dict` | Poll until `completed`; raises `JobFailed` on `failed` and `TimeoutError` after the limit. It does not treat `stopped`, `paused`, or any `waiting_*` status as terminal, so `timeout=None` can poll forever. Use a finite timeout and inspect `status()` for operator/user intervention states. |
| `artifacts() -> list` | Return the current status response's `artifacts` list, or `[]`. |
| `download_artifact(name: str, dest: str | Path) -> Path` | Download one named artifact, create parent directories, and return the destination. |
| `download_export(dest: str | Path) -> Path` | Download the first `*.zip` found at the top of the session workspace when one exists; otherwise build and return a ZIP of the session directory. Its content is not a stable portable-session contract. Use `Client.export_session` for portable session transfer. |
| `delete() -> dict` | Delete only when no active or initializing workflow owns the job. The server can return HTTP 409; stop the analysis and wait for initialization/cleanup to finish before retrying. |

All Job HTTP calls raise `AuthError` for HTTP 401 and may raise `httpx.HTTPStatusError` for other unsuccessful responses.

Client exceptions are `ScaiError` (base), `AuthError`, `FileIntegrityError`, `UploadError`, `JobFailed`, and `ImportFailed`. File access may also raise standard `OSError`/`FileNotFoundError`; JSON parsing may raise `json.JSONDecodeError`; polling may raise `TimeoutError`; and network failures may surface as `httpx` exceptions.

## 13. Python client portable session management

`list_sessions() -> list[dict]` returns sessions owned by the authenticated user, including identifiers, names, status, round, and timestamps.

Ordinary import usage:

```python
from scai_client import Client

def progress(info: dict) -> None:
    print(info.get("phase"), info.get("bytes_processed"), info.get("bytes_total"))

with Client("http://localhost:3000", api_key="<API_KEY>", timeout=86400) as client:
    session_id = client.import_session(
        "portable-session.tar.gz",
        wait=True,
        poll_interval=2.0,
        timeout=None,
        on_progress=progress,
    )
```

Ordinary signature: `import_session(archive_path: str | Path, *, wait: bool = True, poll_interval: float = 2.0, timeout: int | None = None, on_progress: Callable[[dict], None] | None = None) -> str`. Accepted server formats are `.tar.gz`, `.tgz`, `.zip`, and `.json`. Tar archives can be asynchronous: with `wait=False`, the returned string is an import job ID. Progress callback exceptions are ignored. Extraction progress compares uncompressed member bytes with compressed archive bytes, so its ratio is heuristic and can exceed 100%.

**Warning:** ordinary import reuses the session ID recorded in the archive only when that ID is free on the destination. When a session already owns it, the archive is materialized under a freshly allocated ID with every embedded path rebound to it, and the existing session's files are never written over — import always creates a session and never overwrites one. A collision therefore costs a new ID and exact workflow continuation (see the runtime paragraph below), not data, so no full backup or isolated instance is needed to guard against one. Tar import has no configured limit on compressed size, expanded size, or member count. Check archive inventory out of band and reserve sufficient data/temp disk and inodes to withstand extreme expansion or an archive bomb. Inline ZIP and JSON imports are limited to 100 MiB request bodies, but that limit does not make archive contents safe. Errors include `FileNotFoundError`, `AuthError`, `ScaiError`, `ImportFailed`, `TimeoutError`, and HTTP/network errors.

`cancel_import(import_id: str) -> dict` requests cooperative cancellation while upload processing advances and between archive members; cancellation is not an immediate transaction rollback. Both the status and cancel lookups are scoped to the user who started the job, and an unknown ID and another user's ID are answered identically with HTTP 404, so the endpoint cannot be used to probe for live imports and an ID alone is not a capability. The method returns the server response and raises `AuthError` or `ScaiError` for authentication, unknown IDs, or other API failures.

Ordinary export usage:

```python
with Client("http://localhost:3000", api_key="<API_KEY>") as client:
    path = client.export_session(
        "<SESSION_ID>",
        "exports/portable-session.tar.gz",
        timeout=None,
        chunk_size=1024 * 1024,
    )
```

Ordinary signature: `export_session(session_id: str, output_path: str | Path | None = None, *, timeout: float | None = None, chunk_size: int = 1048576) -> Path`. The default destination is `<FIRST_8_ID_CHARS>_export.tar.gz`; passing a directory places that name inside it. Data first goes to a `.partial` file, which is removed on failure and atomically renamed on success. HTTP 401 raises `AuthError`; ownership/not-found errors raise `ScaiError`; other HTTP failures raise `httpx.HTTPStatusError`.

The server fully builds a temporary compressed tar file before sending the first response byte. Ensure `upload_temp/` has space for that additional archive and allow enough first-byte time for large sessions; `timeout=None` disables read/write/pool timeouts for this request while retaining the client's connect timeout.

Portable export includes selected session metadata, uploads, workspace files (including figures/artifacts), reports, portable notebooks/state sidecars, externalized content, per-session LLM interaction logs (`sessions/<SESSION_ID>/logs/`), and an optional reproducibility bundle. The logs travel in every case: they are the only per-call record of model usage, and a report regenerated without them claims every agent made zero calls. Plaintext credentials and endpoints are stripped from every exported state JSON; with `include_runtime=true` the LangGraph checkpoint can still carry credentials sealed under the source installation's key, which the destination cannot read and drops on import.

Whether the imported session can *continue* analysing depends on one choice at export time — and, for the runtime option, on the import trusting the archive:

- **Default (`include_runtime=false`)** — no kernel pkl, no LangGraph checkpoint, and no h5ad runtime sidecars. Data, code, figures, reports and conversation history all survive and are viewable, but the session cannot resume: import marks its runtime `unavailable`, deletes the per-round bioinformatics checkpoint that an ordinary export does still carry, and resume tells the user to re-run from the imported data — which is what the session then does, restarting the current round instead of restoring a conversation whose kernel is gone. Use this to hand a session to someone else, or when only the results matter.
- **With runtime (`include_runtime=true`)** — adds the live kernel pkl (in-memory values such as `adata`), the LangGraph checkpoint, and any h5ad runtime sidecars a pre-upgrade build left in the pkl's place, so the analysis picks up where it left off. Import must pass `X-Trust: true` to accept the pkl and the h5ad sidecars, because loading a pkl executes code from the archive. Without that header the archive still imports, but no live kernel lands and both checkpoint families — the LangGraph checkpoint and the per-round bioinformatics checkpoint — are dropped, leaving exactly the default outcome above. The archive can be gigabytes larger.

Exact continuation needs three things together: `include_runtime=true` at export, `X-Trust: true` at import, and the archive's session ID still free on the destination. A runtime export resumes on a *different* installation as well as the original one, but that last condition is the one that quietly decides how precisely it resumes. Import never overwrites an existing session: when that ID is already taken it allocates a fresh one, and the LangGraph checkpoint — keyed by the archive's session ID and impossible to rename into place — is dropped along with it. The kernel pkl still arrives, so the imported session keeps its in-memory values, but Resume re-enters the graph at the start of a new round instead of continuing at the node the session was paused on. Re-importing a runtime archive onto the installation that produced it therefore gives exact continuation only if the original session was deleted first. Two installation-scoped mechanisms are handled at import: the pkl's HMAC sidecar is regenerated under the destination's key (it is never shipped in the archive), and credentials sealed with the source installation's key are dropped rather than treated as corruption — reconfigure the API key after import.

**A resumable export flushes the kernel first, and is refused rather than
approximated.** `include_runtime=true` writes the live kernel to disk, verifies
that generation, and only then builds the archive — so the runtime, the session
state and the workflow checkpoint inside it describe one moment. If the flush
fails, or the analysis advances while the archive is being built, the export
answers **409** with a reason instead of producing an archive that imports
cleanly and cannot continue. A results-only export is unaffected and never
interrupts the analysis.

The archive says what it is. Its `manifest.json` records `resumable`, the
committed generation's revision and digest, its coherence state, and which
variable names that generation holds and which it could not — so an archive
read months later is self-describing.

**Importing a resumable archive onto a session ID that already exists is
refused.** The workflow checkpoint is keyed by session ID and cannot be renamed
into place, so importing under a fresh ID would drop it: the runtime would
arrive, the import would report success, and Resume would restart the round.
Delete or rename the existing session first, or re-export without runtime.
A results-only archive still lands under a fresh ID as before — it has no
checkpoint to lose, and reallocating is what keeps a re-import from overwriting
the live session.


Resuming needs a session that was exported while paused or waiting for input. A session exported after it completed has a checkpoint sitting at the end of the graph, so there is nothing to continue; re-analysis starts a fresh round instead.

It remains a migration format, not a complete service backup.

## 14. Session Manager CLI

Install the local client package before using the repository helper:

```bash
python3 -m pip install ./clients/python
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' list
```

Global options are `--url URL` (default `http://localhost:3000`) and required `--api-key API_KEY`. The helper creates `Client(..., timeout=86400)`.

Ordinary commands are:

```bash
python3 session_manager.py --api-key '<API_KEY>' list
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' \
  import '<ARCHIVE_PATH>'
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' \
  export '<SESSION_ID>'
python3 session_manager.py --url http://localhost:3000 --api-key '<API_KEY>' \
  export '<SESSION_ID>' '<OUTPUT_PATH>'
```

`list` displays ID, name, status, current round, and creation time. `import` shows heuristic extraction progress (which can exceed 100%) and waits without a polling deadline; the HTTP client timeout is 24 hours. It reuses the archived session ID only when that ID is free, allocating a new one on collision rather than overwriting the existing session. `export` writes a compressed archive, using the client default name when output is omitted; the server needs temporary disk to finish building it before download starts. Quote secrets and paths, and avoid storing keys or asynchronous import job IDs in shell history or logs.

## 15. Security practices

- Set strong, stable `SCAI_JWT_SECRET` and `SCAI_CONFIG_SECRET` values through protected environment injection. Shared instances intended to accept the same tokens must use the same JWT secret.
- Never casually change `SCAI_CONFIG_SECRET`: previously encrypted stored model credentials may become unreadable. Back it up separately from data, with equivalent protection.
- Protect `.jwt_secret`, `.config_secret`, `app.db`, archives, backups, and all data directories with least-privilege ownership and permissions.
- Run application processes as an unprivileged account. The container drops to UID `10001` after volume preparation.
- Treat API keys, JWTs, model-provider keys, and proxy credentials as secrets. Do not place them in source control, command transcripts, tickets, or logs; rotate and revoke exposed values.
- Restrict network access to production port `3000` and, in development only, backend port `8000` according to user need. Do not bind direct startup to `0.0.0.0` without an explicit network-control plan.
- Assess the privacy and regulatory impact before sending prompts, images, or derived research content to external model services. Verify provider retention and regional policies.
- Validate archive origin and content, scan incoming files according to institutional policy, and import only into an appropriately isolated service account.
- Encrypt backups at rest and in transit, maintain offline/immutable copies where required, and test recovery regularly.
- Keep host OS, Docker, Python/Node dependencies, and ScAI images updated after staging and backup validation.

## 16. Troubleshooting

| Symptom | Checks and corrective action |
|---|---|
| Container remains unhealthy | Run `docker logs scai`; inspect `/data/logs/scai.log`; verify dependency installation, disk space, DNS/provider access, and `curl` to backend health inside the container. |
| Frontend opens but API calls fail | In production, check the single process and `curl http://127.0.0.1:3000/api/health`; in development, check the backend port and Vite proxy variables. Review browser/server logs. |
| Direct startup says instance is running | Read `.scai.pid` and verify each PID. On a dedicated host, run `./stop.sh`; on a shared host, stop only verified PIDs. Remove the PID file only if all recorded services are gone. |
| Address already in use | Use `lsof -i:<PORT>`, identify ownership, stop only the intended process, or choose `--port`/development `--backport`. `stop.sh` cleans the recorded custom ports but can kill an unrelated occupant of them. |
| `python3.12` not found | Install Python 3.12 and its venv support; `start.sh` checks generic `python3` first but explicitly invokes `python3.12` for initial environment creation. |
| Node/npm or package installation fails | Node is not used by the packaged runtime. For a source build or development, verify Node.js 20.19+ or 22.12+, npm, registry access, proxy/index settings, free space, and ownership of the dependency directory. Preserve logs before retrying. |
| Model configuration saves but analysis fails | Saving does not validate credentials. Use the validation action, confirm base URL/proxy/key, model availability, quotas, and outbound connectivity. |
| Stored model key disappears or cannot decrypt | Restore the original `SCAI_CONFIG_SECRET` or `.config_secret` from protected backup, then restart. Do not overwrite the database while diagnosing. |
| Existing login tokens stop working after restart | Restore the stable `SCAI_JWT_SECRET` or `.jwt_secret`. Users can log in again to receive new JWTs; API keys are database-backed. |
| Upload/import fails | Check supported format, source readability, ID collision, data/temp free space and inodes, client/server logs, HTTP timeouts, ownership, member count/expansion, and archive integrity. |
| Client gets 401 | Confirm key prefix/value, ownership, revocation state, account status, server URL, and that the Authorization header reaches the backend. |
| Client job wait never finishes | `Job.wait` recognizes only `completed` and `failed`; inspect `job.status()` for `stopped`, `paused`, and `waiting_*`, review server/provider logs, set a finite timeout, and handle `TimeoutError`. |
| Delete fails | `Job.delete` returns HTTP 409 while a workflow is active or initializing. Stop it and wait for cleanup. User deletion and role changes also enforce self/last-admin protections. |
| Disk usage grows | Review uploads, workspaces, reports, logs, temporary uploads, abandoned `.partial` files, and retained sessions. Back up before deleting supported data through the application. |

## 17. Parameter and default-value reference

### Ports, paths, and limits

| Item | Default/value | Scope |
|---|---|---|
| Production application port | `3000` | Docker and direct-start UI/API/WebSocket |
| Development backend port | `8000` | Direct development mode only |
| Docker mapping | `3000:3000` | Recommended host mapping |
| Docker data/dependencies | `/data`, `/dep` | Persistent volumes |
| Direct dependencies | `<REPO>/../dep` | `start.sh` |
| Direct data | `<REPO>/../data` | Unless environment overrides backend root |
| PID file | `<REPO>/.scai.pid` | Direct startup |
| Upload chunk | `10 MiB` | Backend/client upload |
| Maximum upload | `10 GiB` | Backend |
| JWT lifetime | `72 hours` | Login tokens |
| Client HTTP timeout | `3600 seconds` | `Client` default |
| Job poll interval | `60 seconds` | `Job.wait` default |
| Import poll interval | `2.0 seconds` | `Client.import_session` default |
| Export stream chunk | `1 MiB` | `Client.export_session` default |

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SCAI_DATA_DIR` | Resolved sibling `data` directory | Backend data root; Docker entrypoint forces `/data`. Direct `start.sh` still creates/prints sibling data when overridden. |
| `SCAI_JWT_SECRET` | Generated and stored as `.jwt_secret` | HS256 JWT signing |
| `SCAI_CONFIG_SECRET` | Generated local encryption material | Encrypts stored user/model secrets |
| `SCAI_ENABLE_REPRODUCIBILITY` | Disabled | Truthy `1`, `true`, or `yes` enables reproducibility bundles |
| `SCAI_RUNTIME_SAVE_FLOOR_MINUTES` | `30` | Minutes that must pass since the last **successful** runtime save before a step boundary saves again. Round and closing boundaries ignore it. Unset, non-integer or non-positive values log and fall back |
| `SCAI_RUNTIME_MAX_RSS_GB` | `auto` | Per-session kernel RSS ceiling for regular runtime persistence; numeric values are GiB administrator caps |
| `SCAI_RUNTIME_MAX_ANNDATA_TOTAL_GB` | `auto` | Total estimated size of live AnnData objects; numeric values are GiB administrator caps |
| `SCAI_SNAPSHOT_DIR` | unset | Root for decision snapshots. **Unset means snapshots are off; setting it turns them on** — there is no separate switch. Must be absolute, and should be on separate storage from `SCAI_DATA_DIR` |
| `SCAI_SNAPSHOT_STALL_SECONDS` | `120` | Abandon a snapshot **copy** after this long with no byte written. A hung mount makes no progress; a slow one does. Does not bound the runtime save a capture may ask for |
| `SCAI_SNAPSHOT_MAX_SECONDS` | `1800` | Backstop under the stall bound, for a target that dribbles bytes without ever stalling |
| `SCAI_SNAPSHOT_FREE_SPACE_RESERVE_GB` | unset | Free space below which the least valuable snapshots are reclaimed. Unset means **nothing is ever deleted automatically**, which is the default |
| `SCAI_MODE` | `prod` | Docker entrypoint mode; the production image rejects `dev` |
| `PIP_INDEX_URL` | pip environment/default | Docker Python package index |
| `SCAI_AUTO_FETCH_MARKERS` | `0` | Direct startup; `1` attempts marker-database download |
| `SCAI_AUTO_FETCH_CELLTYPIST` | `0` | Direct startup; `1` caches CellTypist models under `$SCAI_DATA_DIR/celltypist` |
| `SCAI_HOST` | Set internally | Vite proxy host; scripts set it from startup context |
| `SCAI_BACKPORT` | Set internally | Vite proxy backend port; scripts set it from startup context |


### Session storage capacity

ScAI imposes no per-session disk quota and no free-space reserve. Operators are
responsible for sizing and monitoring the data volume, including free space and
inode capacity. Archive member-count and metadata-size checks remain because
they protect import parsing and memory use rather than limiting persistent disk
consumption.

### Shell options

| Program | Option | Default/effect |
|---|---|---|
| `start.sh` | `--dev` | Development mode |
| `start.sh` | `--prod` | Production mode; default |
| `start.sh` | `--host HOST` | `127.0.0.1` |
| `start.sh` | `--port PORT` | Production application / development frontend `3000` |
| `start.sh` | `--backport PORT` | Development backend `8000`; ignored in production |
| `start.sh` | `-h`, `--help` | Help |
| `stop.sh` | none | Recorded PID/port cleanup plus broad host process-name fallback |
| `build.sh` | `--version TAG` | Newest Git tag, else `latest` |
| `build.sh` | `--push` | Push selected tag and `latest`; disabled by default |
| `build.sh` | `-h`, `--help` | Help |
| `session_manager.py` | `--url URL` | `http://localhost:3000` |
| `session_manager.py` | `--api-key KEY` | Required |
| `session_manager.py list` | none | List owned sessions |
| `session_manager.py import` | `archive` | Required archive path |
| `session_manager.py export` | `session_id`, optional `output` | Required ID and optional destination |

### Python public ordinary API

| Type | Signature |
|---|---|
| `Client` | `Client(base_url, *, api_key=None, jwt=None, timeout=3600)` |
| `Client` | `close()` |
| `Client` | `submit(config_path)` |
| `Client` | `get_job(job_id)` |
| `Client` | `list_sessions()` |
| `Client` | `import_session(archive_path, *, wait=True, poll_interval=2.0, timeout=None, on_progress=None)` |
| `Client` | `export_session(session_id, output_path=None, *, timeout=None, chunk_size=1048576)` |
| `Client` | `cancel_import(import_id)` |
| `Job` | `Job(client, job_id)` |
| `Job` | `status()` |
| `Job` | `wait(interval=60, timeout=None)` |
| `Job` | `artifacts()` |
| `Job` | `download_artifact(name, dest)` |
| `Job` | `download_export(dest)` |
| `Job` | `delete()` |
