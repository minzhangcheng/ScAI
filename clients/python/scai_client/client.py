"""ScAI Client — main entry point for the Python SDK.

Usage:
    from scai_client import Client

    client = Client(base_url="http://localhost:8000", api_key="scai_xxx")
    job = client.submit("my-config.json")
    job.wait(timeout=3600)
    job.download_export("results.zip")
"""
import json
import time
from pathlib import Path
from typing import Optional

import httpx

from .job import Job
from .upload import upload_file
from .errors import ScaiError, AuthError, ImportFailed


class Client:
    """ScAI API client.

    Args:
        base_url: Server URL (e.g., "http://localhost:8000")
        api_key: API key for authentication (starts with "scai_")
        jwt: JWT token for authentication
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        jwt: Optional[str] = None,
        timeout: int = 300,
    ):
        if not api_key and not jwt:
            raise AuthError("Either api_key or jwt must be provided")

        self.base_url = base_url.rstrip("/")
        if jwt:
            self._auth = {"Authorization": f"Bearer {jwt}"}
        else:
            self._auth = {"Authorization": f"Bearer {api_key}"}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=self._auth,
            timeout=timeout,
        )

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def submit(self, config_path: str | Path) -> Job:
        """Submit a complete analysis: create job -> upload files -> start.

        Args:
            config_path: Path to a ConfigV1 JSON file

        Returns:
            A Job object for tracking progress
        """
        config_path = Path(config_path)
        cfg = json.loads(config_path.read_text())

        # 1. Create the job
        r = self._http.post("/api/v1/analyses", json=cfg)
        if r.status_code == 422:
            raise ScaiError(f"Invalid config: {r.json()}")
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        r.raise_for_status()
        job_id = r.json()["job_id"]

        # 2. Upload each referenced file
        config_dir = config_path.parent
        for entry in cfg.get("data", {}).get("files", []):
            local = Path(entry["local_path"])
            if not local.is_absolute():
                local = config_dir / local
            if local.exists():
                upload_file(self._http, job_id, local)

        # 3. Start the analysis
        r = self._http.post(f"/api/v1/analyses/{job_id}/start")
        if r.status_code == 400:
            missing = r.json().get("detail", "")
            raise ScaiError(f"Cannot start: {missing}")
        r.raise_for_status()

        return Job(self._http, job_id)

    def get_job(self, job_id: str) -> Job:
        """Get a Job object for an existing analysis."""
        return Job(self._http, job_id)

    def list_sessions(self) -> list[dict]:
        """List all analysis sessions belonging to the authenticated user.

        Returns:
            A list of session dicts. Each dict contains:

            * ``id`` (str) — unique session identifier
            * ``name`` (str) — session title / display name
            * ``status`` (str) — TaskStatus value: ``"pending"``, ``"running"``,
              ``"waiting_input"``, ``"waiting_annotation"``,
              ``"generating_report"``, ``"completed"``, ``"failed"``, ``"paused"``
            * ``current_round`` (int) — current analysis round
            * ``created_at`` (str) — ISO 8601 creation timestamp
            * ``updated_at`` (str) — ISO 8601 last-modification timestamp

        Note:
            The session model has no separate ``description`` field;
            ``name`` serves as the title.
        """
        r = self._http.get("/api/sessions")
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        r.raise_for_status()
        return r.json().get("sessions", [])

    def import_session(
        self,
        archive_path: str | Path,
        *,
        wait: bool = True,
        poll_interval: float = 2.0,
        timeout: Optional[int] = None,
    ) -> str:
        """Import a session from a local archive (.tar.gz / .tgz / .zip / .json).

        The server processes ``.tar.gz``/``.tgz`` archives asynchronously; this
        method polls the import-status endpoint until completion when
        ``wait=True`` (default). ``.zip`` and plain JSON files are processed
        synchronously by the server and return the new ``session_id``
        immediately.

        Args:
            archive_path: Local path to the archive file.
            wait: If True (default), block until the import completes. Only
                relevant for tar.gz archives. When False on a tar.gz upload,
                returns the ``import_id`` instead of a ``session_id``.
            poll_interval: Seconds between import-status polls.
            timeout: Maximum seconds to wait. ``None`` means no limit.

        Returns:
            The newly created ``session_id`` (or ``import_id`` if
            ``wait=False`` for an async tar.gz upload).

        Raises:
            FileNotFoundError: If ``archive_path`` does not exist.
            AuthError: If authentication fails.
            ImportFailed: If the server reports a failed import.
            TimeoutError: If ``timeout`` is exceeded while polling.
        """
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise FileNotFoundError(str(archive_path))

        with open(archive_path, "rb") as fh:
            files = {"file": (archive_path.name, fh, "application/octet-stream")}
            r = self._http.post("/api/session/import", files=files)

        if r.status_code == 401:
            raise AuthError("Authentication failed")
        if r.status_code >= 400:
            raise ScaiError(f"Import failed: HTTP {r.status_code}: {r.text}")
        body = r.json()

        if body.get("session_id"):
            return body["session_id"]

        import_id = body.get("data", {}).get("import_id")
        if not import_id:
            raise ScaiError(f"Unexpected import response: {body}")

        if not wait:
            return import_id

        return self._wait_for_import(import_id, poll_interval, timeout)

    def export_session(
        self,
        session_id: str,
        output_path: str | Path | None = None,
        *,
        timeout: float | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Export a session as a tar.gz archive (streaming).

        Calls ``GET /api/session/{session_id}/export``. The server builds the
        archive synchronously before streaming the first byte, so large
        sessions may need ``timeout`` raised above the client's default.
        Bytes are written to ``{output_path}.partial`` first and atomically
        renamed on success — a network failure leaves no half-written file
        at ``output_path``.

        Args:
            session_id: The session ID to export.
            output_path: Destination file. If ``None``, writes
                ``{session_id[:8]}_export.tar.gz`` in the current directory.
                If a directory is given, the file is saved inside it with
                that default name.
            timeout: Per-request timeout in seconds. ``None`` disables the
                read timeout for this call (recommended for large archives).
            chunk_size: Bytes per ``iter_bytes`` chunk.

        Returns:
            Resolved ``Path`` of the downloaded archive.

        Raises:
            AuthError: HTTP 401.
            ScaiError: HTTP 403 (session not owned by caller) or HTTP 404
                (session not found).
            httpx.HTTPStatusError: Other HTTP errors.
        """
        default_name = f"{session_id[:8]}_export.tar.gz"
        if output_path is None:
            output_path = Path.cwd() / default_name
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / default_name

        partial = output_path.with_name(output_path.name + ".partial")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if timeout is None:
            # Disable read/write/pool timeouts for large streaming downloads,
            # but preserve the client-level connect timeout.
            base = self._http.timeout
            req_timeout = httpx.Timeout(
                connect=base.connect,
                read=None,
                write=None,
                pool=None,
            )
        else:
            req_timeout = httpx.Timeout(timeout)

        try:
            with self._http.stream(
                "GET",
                f"/api/session/{session_id}/export",
                timeout=req_timeout,
            ) as r:
                if r.status_code == 401:
                    raise AuthError("Authentication failed")
                if r.status_code == 403:
                    raise ScaiError(
                        f"Forbidden: session not owned by caller: {session_id}"
                    )
                if r.status_code == 404:
                    raise ScaiError(f"Session not found: {session_id}")
                r.raise_for_status()
                with open(partial, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
            partial.replace(output_path)
        except BaseException:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return output_path

    def _wait_for_import(
        self,
        import_id: str,
        poll_interval: float,
        timeout: Optional[int],
    ) -> str:
        start = time.time()
        while True:
            r = self._http.get(f"/api/session/import/{import_id}")
            if r.status_code == 401:
                raise AuthError("Authentication failed")
            r.raise_for_status()
            info = r.json()
            st = info.get("status")
            if st == "completed":
                session_id = info.get("session_id")
                if not session_id:
                    raise ScaiError(f"Import completed but no session_id: {info}")
                return session_id
            if st == "failed":
                raise ImportFailed(import_id, info.get("error", "unknown"))
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Import {import_id} did not complete within {timeout}s"
                )
            time.sleep(poll_interval)