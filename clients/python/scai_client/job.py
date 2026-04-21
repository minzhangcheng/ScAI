"""Job object for polling analysis status and downloading results."""
import time
from pathlib import Path
from typing import Optional

import httpx

from .errors import JobFailed, AuthError


class Job:
    """Represents an analysis job on the ScAI server.

    Usage:
        job = client.submit("config.json")
        job.wait(timeout=3600)
        job.download_export("results.zip")
    """

    def __init__(self, client: httpx.Client, job_id: str):
        self._client = client
        self.job_id = job_id

    def _check_auth(self, response: httpx.Response) -> None:
        """Raise AuthError on 401 responses."""
        if response.status_code == 401:
            raise AuthError("Authentication failed (401)")

    def status(self) -> dict:
        """Get current job status and metadata."""
        r = self._client.get(f"/api/v1/analyses/{self.job_id}")
        self._check_auth(r)
        r.raise_for_status()
        return r.json()

    def wait(self, interval: int = 60, timeout: Optional[int] = None) -> dict:
        """Poll until the job is completed.

        Args:
            interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = unlimited)

        Returns:
            Final status dict

        Raises:
            JobFailed: If the job ends in 'failed' status
            AuthError: If authentication fails
            TimeoutError: If timeout is exceeded
        """
        start = time.time()
        while True:
            s = self.status()
            st = s.get("status", "unknown")
            if st == "completed":
                return s
            if st == "failed":
                raise JobFailed(self.job_id, s.get("error", "unknown"))
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Job {self.job_id} did not complete within {timeout}s"
                )
            time.sleep(interval)

    def artifacts(self) -> list:
        """List available artifacts."""
        return self.status().get("artifacts", [])

    def download_artifact(self, name: str, dest: str | Path) -> Path:
        """Download a single artifact by name."""
        dest = Path(dest)
        r = self._client.get(
            f"/api/v1/analyses/{self.job_id}/artifacts/{name}",
            follow_redirects=True,
        )
        self._check_auth(r)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def download_export(self, dest: str | Path) -> Path:
        """Download the full session export as a zip."""
        dest = Path(dest)
        with self._client.stream(
            "GET", f"/api/v1/analyses/{self.job_id}/export", follow_redirects=True
        ) as r:
            self._check_auth(r)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        return dest

    def delete(self) -> dict:
        """Delete the job (only allowed for non-running jobs)."""
        r = self._client.delete(f"/api/v1/analyses/{self.job_id}")
        self._check_auth(r)
        r.raise_for_status()
        return r.json()