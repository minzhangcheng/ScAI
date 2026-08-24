# scai-client

`scai-client` is the synchronous Python SDK for submitting and managing ScAI batch analyses. The client package version follows the ScAI application release.

## Requirements and Installation

- Python 3.10 or newer
- `httpx>=0.26.0`
- A running ScAI server
- A ScAI API key or JWT

Install the package from the repository checkout:

```bash
python3 -m pip install ./clients/python
```

For editable development installation:

```bash
python3 -m pip install -e './clients/python[dev]'
pytest clients/python/tests
```

## Create an API Key

Sign in to ScAI, open the user menu, and select **API Keys**. Create a labeled key and store it securely; the full value is shown only when it is created. ScAI API keys begin with `scai_`.

Docker and direct production mode serve the UI, API, and WebSockets from the single application URL `http://localhost:3000`. Only native `./start.sh --dev` runs a separate backend on `http://localhost:8000`; the Vite development server on port `3000` proxies API requests to it.

## Create a Client

```python
from scai_client import Client

client = Client(
    base_url="http://localhost:3000",
    api_key="scai_your_key_here",
    timeout=3600,
)
```

Constructor:

```python
Client(
    base_url: str,
    *,
    api_key: str | None = None,
    jwt: str | None = None,
    timeout: int = 3600,
)
```

At least one credential is required. If both `jwt` and `api_key` are supplied, the JWT is used. Prefer a context manager so HTTP connections close automatically:

```python
with Client("http://localhost:3000", api_key="scai_your_key_here") as client:
    sessions = client.list_sessions()
```

## Submit an Analysis

`Client.submit()` accepts a ConfigV1 JSON file, creates an analysis, uploads referenced local files, and starts the job:

```python
from scai_client import Client

with Client("http://localhost:3000", api_key="scai_your_key_here") as client:
    job = client.submit("analysis-config.json")
    print(job.job_id)

    final_status = job.wait(interval=60, timeout=24 * 3600)
    print(final_status["status"])

    print(job.artifacts())
    job.download_artifact("report.html", "downloads/report.html")
```

Local paths in `data.files[].local_path` are resolved relative to the config file. The current client silently skips a referenced file that does not exist, while the server may still start the analysis. Validate every local path before submission to avoid an incomplete or later-failing analysis.

Uploads use 10 MiB chunks, retry failed chunks up to three times after the initial attempt, and complete with server-side SHA-256 verification.

## ConfigV1 Format

The top-level schema version is `1.0`. The following minimal example uses PubMed-assisted literature analysis and two single-cell files:

```json
{
  "schema_version": "1.0",
  "analysis": {
    "name": "Treatment versus control study",
    "initial_instruction": "Characterize treatment-associated cell states while accounting for biological replicates.",
    "round_wait_seconds": -1
  },
  "api_config": {
    "default_llm": {
      "llm_base_url": "https://example-provider.invalid/v1",
      "api_key": "provider-key",
      "model": "provider-model",
      "proxy_url": null
    },
    "literature": {
      "mode": "pubmed_llm",
      "pubmed_email": "researcher@example.org",
      "max_search_rounds": 20
    }
  },
  "data": {
    "groups": [
      {"key": "control", "name": "Control", "sample_count": 1, "order": 0},
      {"key": "treated", "name": "Treated", "sample_count": 1, "order": 1}
    ],
    "files": [
      {
        "local_path": "data/control.h5ad",
        "section": "sequencing",
        "data_type": "scRNA-seq",
        "group": "control",
        "sample_number": 1
      },
      {
        "local_path": "data/treated.h5ad",
        "section": "sequencing",
        "data_type": "scRNA-seq",
        "group": "treated",
        "sample_number": 1
      }
    ],
    "indicators": []
  },
  "agents": {}
}
```

Important constraints include:

- `analysis.round_wait_seconds`: `-1` for manual wait, `0` for no wait, or up to `3600` seconds.
- Group `sample_count`: 1-100.
- `section`: `pathology`, `sequencing`, or `biochemistry`.
- `sample_number`: positive and consistent with the intended group design.
- Agent names: `pi`, `pathology`, `bioinformatics`, `biochemistry`, `literature`, `painter`, `supervisor`, and `technique`.
- Literature mode: `pubmed_llm`, `deep_research`, `gemini_deep_research`, or `disabled`.

The config contains model-provider credentials. Do not commit real keys to source control or include them in shared examples.

## Work with Existing Jobs

```python
with Client("http://localhost:3000", api_key="scai_your_key_here") as client:
    job = client.get_job("job-or-session-id")
    status = job.status()
```

Public `Job` methods:

| Method | Behavior |
|---|---|
| `status() -> dict` | Fetch current job status and metadata. |
| `wait(interval=60, timeout=None) -> dict` | Poll until `completed`; raise `JobFailed` only for `failed`. |
| `artifacts() -> list` | Return the current artifact list. |
| `download_artifact(name, dest) -> Path` | Download one artifact. |
| `download_export(dest) -> Path` | Download the ZIP returned by the V1 job export endpoint. |
| `delete() -> dict` | Delete when the server has no active or initializing workflow for the job. |

`Job.wait()` does not treat `paused`, `stopped`, or any `waiting_*` state as terminal. With `timeout=None`, it may poll indefinitely while the workflow waits for human input. Automation should use a finite timeout or call `status()` and handle all workflow states explicitly.

`Job.download_export()` is intended for the V1 job output. If the workspace already contains a top-level ZIP, the endpoint may return that file; otherwise it builds a ZIP from the session directory. Use `Client.export_session()` when you specifically need an ordinary portable session archive.

## List and Transfer Sessions

List sessions owned by the authenticated user:

```python
sessions = client.list_sessions()
for session in sessions:
    print(session["id"], session["name"], session["status"])
```

Export a portable session archive:

```python
path = client.export_session(
    "session-id",
    "exports/session.tar.gz",
    timeout=None,
    chunk_size=1024 * 1024,
)
```

The server constructs the archive before sending the first response byte. Large exports require sufficient server temporary space and may require a long gateway first-byte timeout. The client writes to a `.partial` file and atomically renames it on success.

Import a portable archive and report progress:

```python
def show_progress(info: dict) -> None:
    print(info.get("phase"), info.get("files_processed"), info.get("current_file"))

session_id = client.import_session(
    "exports/session.tar.gz",
    wait=True,
    poll_interval=2.0,
    timeout=None,
    on_progress=show_progress,
)
```

With an asynchronous tar import, call `import_session(..., wait=False)` to obtain the import job ID, then request cooperative cancellation if necessary:

```python
import_id = client.import_session("exports/session.tar.gz", wait=False)
client.cancel_import(import_id)
```

Operational cautions:

- Ordinary import preserves the session ID stored in the archive. Check `list_sessions()` first; importing a duplicate ID may overwrite files in an existing session.
- `.zip` and `.json` imports are limited to 100 MiB. Tar imports use a disk-backed asynchronous path but do not enforce compressed-size, expanded-size, member-size, or member-count limits. Inspect untrusted archives offline and monitor disk and inode capacity.
- Import-job IDs are sensitive operational identifiers because status and cancellation are not currently scoped to the creating user.
- Portable archives contain selected metadata and artifacts; they are not complete service backups and do not preserve active in-memory execution.
- During extraction, `bytes_processed` can represent uncompressed member bytes while `bytes_total` is the compressed archive size, so the ratio may exceed 100%.

## Error Handling

SDK exceptions derive from `ScaiError`:

| Exception | Meaning |
|---|---|
| `AuthError` | Missing or rejected API key/JWT, including HTTP 401 responses. |
| `FileIntegrityError` | Server-side SHA-256 verification failed. |
| `UploadError` | Upload initialization, chunk transfer, or completion failed. |
| `JobFailed` | A polled analysis reached `failed`. |
| `ImportFailed` | An import reached `failed` or `cancelled`. |

Standard exceptions may also surface, including `FileNotFoundError`, `json.JSONDecodeError`, `TimeoutError`, `OSError`, and `httpx` network or HTTP errors.

```python
from scai_client import AuthError, JobFailed, ScaiError

try:
    with Client("http://localhost:3000", api_key="scai_your_key_here") as client:
        job = client.submit("analysis-config.json")
        job.wait(timeout=24 * 3600)
except AuthError:
    print("Check the ScAI API key and base URL")
except JobFailed as exc:
    print(exc.job_id, exc.error)
except ScaiError as exc:
    print(f"ScAI request failed: {exc}")
```

## Related Documentation

- [Biomedical User Guide](../../docs/user-guide.md)
- [Administrator Manual](../../docs/admin-manual.md)
- [Technical Handover](../../docs/HANDOVER.md)

## License

MIT
