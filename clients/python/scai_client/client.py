"""ScAI Client — main entry point for the Python SDK.

Usage:
    from scai_client import Client

    client = Client(base_url="http://localhost:8000", api_key="scai_xxx")
    job = client.submit("my-config.json")
    job.wait(timeout=3600)
    job.download_export("results.zip")
"""
import json
from pathlib import Path
from typing import Optional

import httpx

from .job import Job
from .upload import upload_file
from .errors import ScaiError, AuthError


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