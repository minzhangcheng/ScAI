# ScAI Administrator Manual

## 1. Purpose and audience

This manual is for administrators who deploy and operate one ScAI instance, manage accounts and model credentials, protect persistent data, monitor service health, and support programmatic use. Analysis users should instead read the [user guide](user-guide.md). Python package details are also available in the [client README](../clients/python/README.md), but current client source is authoritative where that document differs.

Examples use visibly marked values such as `<API_KEY>`, `<SESSION_ID>`, and `<ARCHIVE_PATH>`. Run commands from the repository root unless stated otherwise.

## 2. Architecture and requirements

ScAI consists of a Vite/React frontend on port `3000`, a FastAPI/Uvicorn backend on port `8000`, local analysis kernels, external OpenAI-compatible model services, and disk-backed application/session storage.

The supported deployment choices are:

- Docker: the image is based on Python 3.11, installs Node.js/npm, exposes frontend port `3000`, and runs the backend internally on `8000`.
- Direct startup: a Linux environment with GNU-compatible `find`, `stat`, `sort`, `grep`, `head`, `cut`, and `tr`; `python3.12` for virtual-environment creation; `python3`; Node.js/npm/npx; and `pgrep`, `pkill`, `lsof`, and `kill` is required. The scripts are not designed as portable POSIX-only tooling.

Size storage for uploaded datasets, generated artifacts, logs, temporary uploads, and dependencies. The host must be able to reach configured model endpoints and package registries during startup. Docker startup installs Python requirements and may install Node dependencies, so the first start can be substantially slower.

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

Open `http://localhost:3000`. `/dep` stores the Python environment and Node dependencies; `/data` stores application data. The entrypoint creates both, recursively assigns them to UID `10001` (`scai`), then drops root privileges. Session directories are normally restricted to mode `0700`. Ensure the host directories can safely have their ownership changed, and use elevated host privileges when backup tooling cannot read them. Do not change ownership on live data merely to make a backup command succeed.

The image exposes only `3000`; do not publish `8000` unless direct API access is operationally required. If it is required, add `-p 8000:8000` and restrict network access appropriately.

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

`SCAI_MODE` defaults to `prod`; the entrypoint also accepts `--dev` or `--prod` after the image name. Docker always exports `SCAI_DATA_DIR=/data`. `PIP_INDEX_URL` is passed to dependency installation when set.

### Health, logs, lifecycle, and removal

The built-in health check calls `http://127.0.0.1:8000/api/health` every 30 seconds, with a 5-second timeout and 60-second startup grace period.

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

Install Python 3.12 and Node.js/npm, clone the repository, and run:

```bash
./start.sh
```

Defaults are production mode, host `127.0.0.1`, frontend port `3000`, and backend port `8000`. Supported options are exactly:

```text
./start.sh [--dev | --prod] [--host HOST] [--port PORT] [--backport PORT]
./start.sh -h
./start.sh --help
```

Example for network access and non-default ports:

```bash
./start.sh --prod --host 0.0.0.0 --port 3000 --backport 8000
```

Relative to the repository directory `<REPO>`, startup stores dependencies in `<REPO>/../dep`, data in `<REPO>/../data`, and process metadata in `<REPO>/.scai.pid`. It creates the environment with `python3.12 -m venv`, updates pip, installs backend requirements, links frontend `node_modules` to the dependency directory, builds changed frontend sources in production mode, then records backend and frontend PIDs.

Direct-run environment controls include `SCAI_DATA_DIR` (backend data-root override), `SCAI_JWT_SECRET`, `SCAI_CONFIG_SECRET`, `SCAI_ENABLE_REPRODUCIBILITY`, and `SCAI_AUTO_FETCH_MARKERS=1`. The last option fetches optional marker databases if missing and does not block startup on failure. When `SCAI_DATA_DIR` is set, `start.sh` still creates and prints `<REPO>/../data`, but the backend actually reads and writes the override path; monitor and back up the override, not the printed sibling directory.

Stop the instance with:

```bash
./stop.sh
```

**Warning:** `stop.sh` is intentionally broad. After reading `.scai.pid`, it searches the whole host for command lines matching ScAI Uvicorn, Vite, and Node patterns. It then terminates **any process occupying default port `8000` or `3000`**, even if that process belongs to another application or another ScAI instance. On shared hosts or multi-instance installations, inspect the recorded PIDs and process command lines first; prefer stopping only the intended PIDs. For instances using custom ports, identify and stop those PIDs manually because the fallback remains hard-coded to the defaults:

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
./build.sh --version 0.36.11
./build.sh --version 0.36.11 --push
```

Without `--version`, the helper uses the newest Git tag returned by `git describe --tags --abbrev=0`, or `latest` if none exists. `--push` pushes both tags to `ghcr.io/minzhangcheng/scai`; authenticate Docker to that registry before using it. The helper does not accept registry, image-name, platform, or build-argument flags.

## 6. First-run and user administration

On first access, create the initial administrator through the setup screen. The backend permits this operation only until initialization is recorded. Usernames and optional email addresses must be unique.

Administrators can create active ordinary users, list users, reset passwords, change roles, change account status, and delete users. Self-service registration creates a `pending` account that cannot log in until an administrator activates it. Status values are `pending`, `readonly`, and `active`; roles are `admin` and `user`. Administrators cannot alter their own role/status or delete themselves, and ScAI prevents removal or demotion of the last administrator.

**Warning:** before deleting a user, stop every analysis owned by that user and back up required data. User deletion evicts cached sessions and removes persisted session directories, legacy state files, API keys, encrypted user configuration, and database records, but it does not itself stop active workflow/task objects. After deletion, verify that no workflow or worker remains for the user's session IDs and that the expected files and database/session listings are gone.

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

Docker `/dep` contains `/dep/venv` and `/dep/node_modules`. Direct startup uses `<REPO>/../dep/venv` and `<REPO>/../dep/node_modules`. Dependencies are reproducible and may be rebuilt, but persistence reduces restart time.

## 8. Backup, upgrade, recovery, and rollback

Prefer a quiescent file-level backup so SQLite and session files agree:

1. Stop ScAI.
2. Copy the entire data root, including hidden files, preserving ownership, modes, timestamps, and links.
3. Record the application/image version and retain the stable secret values separately in a protected secret store.
4. Restart ScAI and verify health.

Example for Docker bind mounts:

```bash
docker stop scai
sudo cp -a ./scai-data "./scai-data-backup-$(date +%Y%m%d-%H%M%S)"
sudo test -f "$(ls -dt ./scai-data-backup-* | head -1)/app.db"
docker start scai
```

**Warning:** confirm available disk space and backup destination before running recursive copies. `sudo cp -a` preserves UID `10001`, modes such as `0700`, timestamps, and links; do not `chown` or `chmod` the live volume for backup access. Verify the copied tree, file counts, sizes, hidden secret files, and a test recovery rather than relying only on the example's basic `app.db` check. Do not treat a copy taken during active writes as transactionally consistent.

For upgrades, back up, deploy the new code/image against the existing data root, then test login, configuration visibility, session listing, and a health request. Retain the old image tag/code revision until acceptance. For rollback, stop the new version and restore both the matching full data backup and previous application version. Never merge selected database and session files from different backup times unless performing a documented repair.

Recovery procedure: stop the service, move the damaged data root aside, restore the complete backup to the original path, restore required ownership, supply the same secrets, start the matching application version, and validate before allowing users back in.

Portable session export/import moves individual sessions between installations. Exported with `include_runtime=true` it also carries the live kernel state and workflow checkpoint, so an imported session can continue analysing; the default export carries content only and cannot resume. Either way it is not a substitute for a complete service backup — see section 12 for the trade-off.

## 9. Monitoring and logs

Use the backend health endpoint as the basic liveness check:

```bash
curl -f http://127.0.0.1:8000/api/health
curl -f http://localhost:3000/api/health
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

## 10. Python client installation and authentication

`scai-client` version `0.2.0` requires Python `>=3.10` and `httpx>=0.26.0`. Install from the repository checkout:

```bash
python3 -m pip install ./clients/python
```

Create an API key in the user interface and initialize the client:

```python
from scai_client import Client

client = Client(
    "http://localhost:8000",
    api_key="<API_KEY>",
    timeout=3600,
)
```

Constructor: `Client(base_url: str, *, api_key: str | None = None, jwt: str | None = None, timeout: int = 3600)`. At least one credential is required; when both are supplied, `jwt` is used. `base_url` has trailing slashes removed. `timeout` configures the underlying synchronous HTTP client.

Use a context manager to close connections automatically:

```python
with Client("http://localhost:8000", api_key="<API_KEY>") as client:
    sessions = client.list_sessions()
```

Public ordinary lifecycle methods are `close() -> None`, `__enter__() -> Client`, and `__exit__(*args) -> None`. Missing credentials or HTTP 401 responses raise `AuthError`.

## 11. Python client analysis jobs and artifacts

### Client job methods

`submit(config_path: str | Path) -> Job` reads a ConfigV1 JSON document, creates an analysis, uploads each existing `data.files[].local_path` (relative paths resolve from the config directory), and starts the job. Invalid configuration raises `ScaiError`; authentication raises `AuthError`. **A missing referenced local file is silently skipped, and the client still sends the start request.** Preflight every configured path before submission; do not assume the server will reject a job that started without the intended file. Uploads use 10 MiB chunks, three retries after the initial attempt per chunk, and server-side SHA-256 verification.

`get_job(job_id: str) -> Job` returns a handle without first checking existence.

```python
from scai_client import Client

with Client("http://localhost:8000", api_key="<API_KEY>") as client:
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

## 12. Python client portable session management

`list_sessions() -> list[dict]` returns sessions owned by the authenticated user, including identifiers, names, status, round, and timestamps.

Ordinary import usage:

```python
from scai_client import Client

def progress(info: dict) -> None:
    print(info.get("phase"), info.get("bytes_processed"), info.get("bytes_total"))

with Client("http://localhost:8000", api_key="<API_KEY>", timeout=86400) as client:
    session_id = client.import_session(
        "portable-session.tar.gz",
        wait=True,
        poll_interval=2.0,
        timeout=None,
        on_progress=progress,
    )
```

Ordinary signature: `import_session(archive_path: str | Path, *, wait: bool = True, poll_interval: float = 2.0, timeout: int | None = None, on_progress: Callable[[dict], None] | None = None) -> str`. Accepted server formats are `.tar.gz`, `.tgz`, `.zip`, and `.json`. Tar archives can be asynchronous: with `wait=False`, the returned string is an import job ID. Progress callback exceptions are ignored. Extraction progress compares uncompressed member bytes with compressed archive bytes, so its ratio is heuristic and can exceed 100%.

**Warning:** ordinary import retains the session ID recorded in the archive and can write over files for an existing session with that ID. Before import, list existing sessions, take a complete backup, and use an isolated instance when collision consequences are uncertain. Tar import has no configured limit on compressed size, expanded size, or member count. Check archive inventory out of band and reserve sufficient data/temp disk and inodes to withstand extreme expansion or an archive bomb. Inline ZIP and JSON imports are limited to 100 MiB request bodies, but that limit does not make archive contents safe. Errors include `FileNotFoundError`, `AuthError`, `ScaiError`, `ImportFailed`, `TimeoutError`, and HTTP/network errors.

`cancel_import(import_id: str) -> dict` requests cooperative cancellation while upload processing advances and between archive members; cancellation is not an immediate transaction rollback. An import job ID is not scoped to its creator by the current status/cancel lookup, so treat it as a sensitive capability: do not log, publish, or disclose it. The method returns the server response and raises `AuthError` or `ScaiError` for authentication, unknown IDs, or other API failures.

Ordinary export usage:

```python
with Client("http://localhost:8000", api_key="<API_KEY>") as client:
    path = client.export_session(
        "<SESSION_ID>",
        "exports/portable-session.tar.gz",
        timeout=None,
        chunk_size=1024 * 1024,
    )
```

Ordinary signature: `export_session(session_id: str, output_path: str | Path | None = None, *, timeout: float | None = None, chunk_size: int = 1048576) -> Path`. The default destination is `<FIRST_8_ID_CHARS>_export.tar.gz`; passing a directory places that name inside it. Data first goes to a `.partial` file, which is removed on failure and atomically renamed on success. HTTP 401 raises `AuthError`; ownership/not-found errors raise `ScaiError`; other HTTP failures raise `httpx.HTTPStatusError`.

The server fully builds a temporary compressed tar file before sending the first response byte. Ensure `upload_temp/` has space for that additional archive and allow enough first-byte time for large sessions; `timeout=None` disables read/write/pool timeouts for this request while retaining the client's connect timeout.

Portable export includes selected session metadata, uploads, workspace files (including figures/artifacts), reports, portable notebooks/state sidecars, externalized content, and an optional reproducibility bundle. It excludes session logs and credentials in every case.

Whether the imported session can *continue* analysing depends on one choice at export time:

- **Default (`include_runtime=false`)** — no kernel pkl and no workflow checkpoint. Data, code, figures, reports and conversation history all survive and are viewable, but the session cannot resume: import marks its runtime `unavailable` and resume tells the user to re-run from the imported data. Use this to hand a session to someone else, or when only the results matter.
- **With runtime (`include_runtime=true`)** — adds the live kernel pkl (in-memory values such as `adata`) and the LangGraph checkpoint, so the analysis picks up where it left off. Import must pass `X-Trust: true` to accept it, because loading a pkl executes code from the archive. The archive can be gigabytes larger.

A runtime export resumes on a *different* installation as well as the original one. Two installation-scoped mechanisms are handled at import: the pkl's HMAC sidecar is regenerated under the destination's key (it is never shipped in the archive), and credentials sealed with the source installation's key are dropped rather than treated as corruption — reconfigure the API key after import. Snapshot V2 does not need to be enabled for any of this.

Resuming needs a session that was exported while paused or waiting for input. A session exported after it completed has a checkpoint sitting at the end of the graph, so there is nothing to continue; re-analysis starts a fresh round instead.

It remains a migration format, not a complete service backup.

## 13. Session Manager CLI

Install the local client package before using the repository helper:

```bash
python3 -m pip install ./clients/python
python3 session_manager.py --url http://localhost:8000 --api-key '<API_KEY>' list
```

Global options are `--url URL` (default `http://localhost:8000`) and required `--api-key API_KEY`. The helper creates `Client(..., timeout=86400)`.

Ordinary commands are:

```bash
python3 session_manager.py --api-key '<API_KEY>' list
python3 session_manager.py --url http://localhost:8000 --api-key '<API_KEY>' \
  import '<ARCHIVE_PATH>'
python3 session_manager.py --url http://localhost:8000 --api-key '<API_KEY>' \
  export '<SESSION_ID>'
python3 session_manager.py --url http://localhost:8000 --api-key '<API_KEY>' \
  export '<SESSION_ID>' '<OUTPUT_PATH>'
```

`list` displays ID, name, status, current round, and creation time. `import` shows heuristic extraction progress (which can exceed 100%) and waits without a polling deadline; the HTTP client timeout is 24 hours. It retains the archived session ID, so list and back up existing sessions before use. `export` writes a compressed archive, using the client default name when output is omitted; the server needs temporary disk to finish building it before download starts. Quote secrets and paths, and avoid storing keys or asynchronous import job IDs in shell history or logs.

## 14. Security practices

- Set strong, stable `SCAI_JWT_SECRET` and `SCAI_CONFIG_SECRET` values through protected environment injection. Shared instances intended to accept the same tokens must use the same JWT secret.
- Never casually change `SCAI_CONFIG_SECRET`: previously encrypted stored model credentials may become unreadable. Back it up separately from data, with equivalent protection.
- Protect `.jwt_secret`, `.config_secret`, `app.db`, archives, backups, and all data directories with least-privilege ownership and permissions.
- Run application processes as an unprivileged account. The container drops to UID `10001` after volume preparation.
- Treat API keys, JWTs, model-provider keys, and proxy credentials as secrets. Do not place them in source control, command transcripts, tickets, or logs; rotate and revoke exposed values.
- Restrict network access to ports `3000` and `8000` according to user need. Do not bind direct startup to `0.0.0.0` without an explicit network-control plan.
- Assess the privacy and regulatory impact before sending prompts, images, or derived research content to external model services. Verify provider retention and regional policies.
- Validate archive origin and content, scan incoming files according to institutional policy, and import only into an appropriately isolated service account.
- Encrypt backups at rest and in transit, maintain offline/immutable copies where required, and test recovery regularly.
- Keep host OS, Docker, Python/Node dependencies, and ScAI images updated after staging and backup validation.

## 15. Troubleshooting

| Symptom | Checks and corrective action |
|---|---|
| Container remains unhealthy | Run `docker logs scai`; inspect `/data/logs/scai.log`; verify dependency installation, disk space, DNS/provider access, and `curl` to backend health inside the container. |
| Frontend opens but API calls fail | Check backend port/process, `curl http://127.0.0.1:8000/api/health`, frontend proxy variables set by startup, and browser/server logs. |
| Direct startup says instance is running | Read `.scai.pid` and verify each PID. On a dedicated host, run `./stop.sh`; on a shared host, stop only verified PIDs. Remove the PID file only if all recorded services are gone. |
| Address already in use | Use `lsof -i:<PORT>`, identify ownership, stop only the intended process, or choose `--port`/`--backport`. `stop.sh` can kill any occupant of 3000/8000 and does not clean custom ports. |
| `python3.12` not found | Install Python 3.12 and its venv support; `start.sh` checks generic `python3` first but explicitly invokes `python3.12` for initial environment creation. |
| Node/npm or package installation fails | Verify Node.js/npm, registry access, proxy/index settings, free space, and ownership of the dependency directory. Preserve logs before retrying. |
| Model configuration saves but analysis fails | Saving does not validate credentials. Use the validation action, confirm base URL/proxy/key, model availability, quotas, and outbound connectivity. |
| Stored model key disappears or cannot decrypt | Restore the original `SCAI_CONFIG_SECRET` or `.config_secret` from protected backup, then restart. Do not overwrite the database while diagnosing. |
| Existing login tokens stop working after restart | Restore the stable `SCAI_JWT_SECRET` or `.jwt_secret`. Users can log in again to receive new JWTs; API keys are database-backed. |
| Upload/import fails | Check supported format, source readability, ID collision, data/temp free space and inodes, client/server logs, HTTP timeouts, ownership, member count/expansion, and archive integrity. |
| Client gets 401 | Confirm key prefix/value, ownership, revocation state, account status, server URL, and that the Authorization header reaches the backend. |
| Client job wait never finishes | `Job.wait` recognizes only `completed` and `failed`; inspect `job.status()` for `stopped`, `paused`, and `waiting_*`, review server/provider logs, set a finite timeout, and handle `TimeoutError`. |
| Delete fails | `Job.delete` returns HTTP 409 while a workflow is active or initializing. Stop it and wait for cleanup. User deletion and role changes also enforce self/last-admin protections. |
| Disk usage grows | Review uploads, workspaces, reports, logs, temporary uploads, abandoned `.partial` files, and retained sessions. Back up before deleting supported data through the application. |

## 16. Parameter and default-value reference

### Ports, paths, and limits

| Item | Default/value | Scope |
|---|---|---|
| Frontend port | `3000` | Docker and direct startup |
| Backend port | `8000` | Internal Docker backend and direct startup |
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
| `SCAI_MODE` | `prod` | Docker entrypoint mode (`prod` or `dev`) |
| `PIP_INDEX_URL` | pip environment/default | Docker Python package index |
| `SCAI_AUTO_FETCH_MARKERS` | `0` | Direct startup; `1` attempts marker-database download |
| `SCAI_HOST` | Set internally | Vite proxy host; scripts set it from startup context |
| `SCAI_BACKPORT` | Set internally | Vite proxy backend port; scripts set it from startup context |

### Shell options

| Program | Option | Default/effect |
|---|---|---|
| `start.sh` | `--dev` | Development mode |
| `start.sh` | `--prod` | Production mode; default |
| `start.sh` | `--host HOST` | `127.0.0.1` |
| `start.sh` | `--port PORT` | Frontend `3000` |
| `start.sh` | `--backport PORT` | Backend `8000` |
| `start.sh` | `-h`, `--help` | Help |
| `stop.sh` | none | Broad host process matching plus unconditional cleanup of occupants on default ports 3000/8000 |
| `build.sh` | `--version TAG` | Newest Git tag, else `latest` |
| `build.sh` | `--push` | Push selected tag and `latest`; disabled by default |
| `build.sh` | `-h`, `--help` | Help |
| `session_manager.py` | `--url URL` | `http://localhost:8000` |
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
