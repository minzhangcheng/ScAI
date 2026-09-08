"""ScAI Client — main entry point for the Python SDK.

Usage:
    from scai_client import Client

    client = Client(base_url="http://localhost:3000", api_key="scai_xxx")
    job = client.submit("my-config.json")
    job.wait(timeout=3600)
    job.download_export("results.zip")
"""
import json
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Optional, Sequence

import httpx

from . import endpoints
from .config import load_config, resolve_data_files
from .job import Job
from .upload import upload_file
from .errors import ScaiError, AuthError, ImportFailed


def login(base_url: str, username: str, password: str, *, timeout: int = 60) -> Dict[str, Any]:
    """Exchange credentials for a short-lived JWT."""
    response = httpx.post(
        base_url.rstrip("/") + endpoints.login(),
        json={"username": username, "password": password},
        timeout=timeout,
    )
    if response.status_code == 401:
        raise AuthError("login failed: invalid credentials")
    if response.status_code == 429:
        raise AuthError("login failed: too many attempts; wait and try again")
    if response.status_code >= 400:
        raise ScaiError(
            f"login failed: HTTP {response.status_code}: {response.text[:400]}"
        )
    return response.json()


def verify_exported_bindings(
    config: Dict[str, Any],
    exported: Dict[str, Any],
    local_files: Optional[Sequence[Path]] = None,
) -> None:
    """Verify every exported V1 file binding against its declaration.

    The server's export is authoritative for the durable destination.  A
    successful checksum upload is not enough: a placeholder can still be
    attached to the wrong section, group, sample, or filename.  Keep this
    verifier shared by the SDK and CLI so both paths refuse the same unsafe
    start.
    """
    declared = list(config.get("data", {}).get("files", []))
    stored = exported.get("data", {}).get("files")
    if not isinstance(stored, list) or len(stored) != len(declared):
        raise ScaiError(
            "exported binding manifest length differs from the configuration"
        )
    if local_files is not None and len(local_files) != len(declared):
        raise ScaiError("resolved local files do not match the configuration order")

    declared_group_names = {
        str(group.get("key")): group.get("name")
        for group in config.get("data", {}).get("groups", [])
    }
    stored_group_names = {
        str(group.get("key")): group.get("name")
        for group in exported.get("data", {}).get("groups", [])
    }

    for index, entry in enumerate(declared):
        record = stored[index]
        if not isinstance(record, dict):
            raise ScaiError(f"exported binding at data.files[{index}] is invalid")
        local = local_files[index] if local_files is not None else None
        expected_filename = entry.get("filename") or (
            local.name if local is not None else Path(entry["local_path"]).name
        )
        exported_path = PurePosixPath(str(record.get("local_path") or ""))
        if exported_path.name != expected_filename:
            raise ScaiError(f"exported binding filename mismatch at data.files[{index}]")
        if record.get("section") != entry.get("section"):
            raise ScaiError(f"exported binding section mismatch at data.files[{index}]")
        if record.get("sample_number") != entry.get("sample_number"):
            raise ScaiError(
                f"exported binding sample_number mismatch at data.files[{index}]"
            )

        declared_group = entry.get("group")
        stored_group = record.get("group")
        if declared_group is None:
            if stored_group is not None:
                raise ScaiError(f"exported binding group mismatch at data.files[{index}]")
            minimum_parts = 4  # uploads/section/_ungrouped/filename
        else:
            if stored_group is None or (
                declared_group_names.get(str(declared_group))
                != stored_group_names.get(str(stored_group))
            ):
                raise ScaiError(f"exported binding group mismatch at data.files[{index}]")
            minimum_parts = 5  # uploads/section/group/sample/filename

        if (
            len(exported_path.parts) < minimum_parts
            or exported_path.parts[0] != "uploads"
        ):
            raise ScaiError(f"data.files[{index}] is unbound in the exported session")


class Client:
    """ScAI API client.

    Args:
        base_url: Server URL (e.g., "http://localhost:3000")
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
        timeout: int = 3600,
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

    def new(self, config_path: str | Path) -> Job:
        """Create, bind, upload, and start one complete ConfigV1 analysis."""
        config, resolved_path = load_config(config_path)
        local_files = resolve_data_files(config, resolved_path.parent)
        entries = config.get("data", {}).get("files", [])
        if len(local_files) != len(entries):
            raise ScaiError("resolved data files do not match the configuration order")

        job_id, manifest = self.create_analysis(config)
        file_ids = self._validate_placeholder_manifest(config, local_files, manifest)
        for local, file_id in zip(local_files, file_ids):
            self.upload_data_file(job_id, local, placeholder_id=file_id)
        verify_exported_bindings(config, self.export_config(job_id), local_files)
        self.start_analysis(job_id)
        return Job(self._http, job_id)

    @staticmethod
    def _validate_placeholder_manifest(
        config: Dict[str, Any], local_files: list[Path], manifest: Any
    ) -> list[str]:
        """Require a complete, ordered server manifest before any upload."""
        entries = config.get("data", {}).get("files", [])
        if not isinstance(manifest, list) or len(manifest) != len(entries):
            raise ScaiError(
                "Create response does not contain a complete placeholder manifest"
            )
        file_ids: list[str] = []
        seen_ids: set[str] = set()
        group_ids_by_key: dict[str, str] = {}
        group_keys_by_id: dict[str, str] = {}
        required = ("id", "filename", "section", "group_id", "sample_number")
        for index, (entry, local, item) in enumerate(zip(entries, local_files, manifest)):
            if not isinstance(item, dict):
                raise ScaiError(f"invalid placeholder entry at data.files[{index}]")
            missing = [field for field in required if field not in item]
            if missing:
                raise ScaiError(
                    f"Create response placeholder data.files[{index}] is missing "
                    + ", ".join(missing)
                )
            file_id = item["id"]
            if not isinstance(file_id, str) or not file_id.strip() or file_id != file_id.strip():
                raise ScaiError(f"placeholder id is invalid at data.files[{index}]")
            if file_id in seen_ids:
                raise ScaiError(f"placeholder id is repeated at data.files[{index}]")
            expected_filename = entry.get("filename") or local.name
            if item["filename"] != expected_filename:
                raise ScaiError(f"filename does not match data.files[{index}]")
            if item["section"] != entry.get("section"):
                raise ScaiError(f"section does not match data.files[{index}]")
            if item["sample_number"] != entry.get("sample_number"):
                raise ScaiError(f"sample_number does not match data.files[{index}]")
            configured_group = entry.get("group")
            manifest_group = item["group_id"]
            if configured_group is None:
                if manifest_group is not None:
                    raise ScaiError(f"group_id does not match data.files[{index}]")
            elif not isinstance(manifest_group, str) or not manifest_group.strip():
                raise ScaiError(f"group_id is missing at data.files[{index}]")
            else:
                if group_ids_by_key.setdefault(configured_group, manifest_group) != manifest_group:
                    raise ScaiError(f"group_id changes for {configured_group!r}")
                prior_key = group_keys_by_id.setdefault(manifest_group, configured_group)
                if prior_key != configured_group:
                    raise ScaiError("group_id is reused across configured groups")
            seen_ids.add(file_id)
            file_ids.append(file_id)
        return file_ids

    def submit(self, config_path: str | Path) -> Job:
        """Backward-compatible alias for :meth:`new`."""
        return self.new(config_path)

    def create_analysis(self, config: Dict[str, Any]) -> tuple[str, list[dict]]:
        r = self._http.post(endpoints.create_analysis(), json=config)
        if r.status_code == 422:
            raise ScaiError(f"Invalid config: {r.json()}")
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        r.raise_for_status()
        created = r.json()
        return created["job_id"], list(created.get("files") or [])

    def upload_data_file(
        self, job_id: str, path: str | Path, *, placeholder_id: Optional[str] = None
    ) -> None:
        if not isinstance(placeholder_id, str) or not placeholder_id.strip():
            raise ScaiError("upload requires the exact server placeholder id")
        upload_file(self._http, job_id, Path(path), file_id=placeholder_id)

    def start_analysis(self, job_id: str) -> dict:
        r = self._http.post(endpoints.start_analysis(job_id))
        if r.status_code == 400:
            raise ScaiError(f"Cannot start: {r.json().get('detail', '')}")
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        r.raise_for_status()
        return r.json()

    def export_config(self, session_id: str) -> Dict[str, Any]:
        r = self._http.get(endpoints.analysis_config(session_id))
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        r.raise_for_status()
        return r.json()

    def stop_session(self, session_id: str, action: str = "immediate") -> dict:
        r = self._http.post(endpoints.stop_session(session_id), json={"action": action})
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        r.raise_for_status()
        return r.json()

    def create_api_key(self, label: str) -> Dict[str, Any]:
        r = self._http.post(endpoints.api_keys(), json={"label": label})
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        if r.status_code >= 400:
            raise ScaiError(f"could not create an API key: HTTP {r.status_code}: {r.text[:400]}")
        return r.json()

    def health(self) -> Dict[str, Any]:
        r = self._http.get(endpoints.health())
        r.raise_for_status()
        return r.json()

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
        r = self._http.get(endpoints.sessions())
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
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        trust: bool = False,
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
            timeout: Maximum seconds to wait. ``None`` means no limit
                (recommended for multi-GB archives — extraction can take >1h).
            on_progress: Optional callback invoked once per poll with the raw
                status dict from the server. Useful for rendering a progress
                bar. Fields include ``phase`` (uploading/extracting/completed/
                failed/cancelled), ``bytes_total``, ``bytes_processed``,
                ``files_processed``, ``current_file``. Exceptions from the
                callback are swallowed so they cannot abort the import.
            trust: When True, instruct the server to restore sandbox session
                pickles from the archive. ``dill.load`` runs arbitrary code,
                so only pass True for archives whose origin you control.
                Default False matches the historical S4 policy (pickles are
                silently dropped).

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

        # Raw octet-stream upload — server reads request.stream() and writes
        # directly to disk, avoiding Starlette's SpooledTemporaryFile (which
        # added a redundant copy of the entire body, doubling disk I/O on
        # multi-GB archives).
        size = archive_path.stat().st_size
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            "X-Filename": archive_path.name,
        }
        if trust:
            headers["X-Trust"] = "true"
        with open(archive_path, "rb") as fh:
            r = self._http.post(
                endpoints.session_import(),
                content=fh,
                headers=headers,
            )

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

        return self._wait_for_import(import_id, poll_interval, timeout, on_progress)

    def export_session(
        self,
        session_id: str,
        output_path: str | Path | None = None,
        *,
        timeout: float | None = None,
        chunk_size: int = 1024 * 1024,
        include_runtime: bool = False,
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
            include_runtime: When True, the server includes the sandbox
                session pickle -- the live kernel's variable dump. It can be
                multi-GB and is only loadable on re-import with ``trust=True``
                on a trusted instance. Default False produces smaller,
                portable archives.

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
                endpoints.session_export(session_id),
                params={"include_runtime": "true"} if include_runtime else None,
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
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> str:
        start = time.time()
        while True:
            r = self._http.get(endpoints.import_status(import_id))
            if r.status_code == 401:
                raise AuthError("Authentication failed")
            r.raise_for_status()
            info = r.json()
            if on_progress is not None:
                try:
                    on_progress(info)
                except Exception:
                    # Caller's progress hook must never break the import loop.
                    pass
            st = info.get("status")
            if st == "completed":
                session_id = info.get("session_id")
                if not session_id:
                    raise ScaiError(f"Import completed but no session_id: {info}")
                return session_id
            if st == "failed":
                raise ImportFailed(import_id, info.get("error", "unknown"))
            if st == "cancelled":
                raise ImportFailed(import_id, info.get("error", "cancelled"))
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Import {import_id} did not complete within {timeout}s"
                )
            time.sleep(poll_interval)

    def cancel_import(self, import_id: str) -> Dict[str, Any]:
        """Request cancellation of an in-progress tar.gz import.

        Cooperative cancel — the server stops between archive members. Already-
        completed or failed jobs are returned unchanged with ``cancelled=False``.

        Args:
            import_id: The import job id returned from :meth:`import_session`
                (when called with ``wait=False``).

        Returns:
            The server's cancel response dict.

        Raises:
            AuthError: If authentication fails.
            ScaiError: If the import id is not found or another HTTP error occurs.
        """
        r = self._http.post(endpoints.import_cancel(import_id))
        if r.status_code == 401:
            raise AuthError("Authentication failed")
        if r.status_code == 404:
            raise ScaiError(f"Import job not found: {import_id}")
        if r.status_code >= 400:
            raise ScaiError(f"Cancel failed: HTTP {r.status_code}: {r.text}")
        return r.json()
