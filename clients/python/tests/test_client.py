"""Unit tests for scai_client — uses httpx mock transport."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scai_client import Client, Job
from scai_client.errors import FileIntegrityError, JobFailed


def test_client_init():
    """Client should accept base_url and api_key."""
    c = Client(base_url="http://localhost:8000", api_key="scai_test")
    assert c.base_url == "http://localhost:8000"
    c.close()


def test_client_context_manager():
    """Client should work as a context manager."""
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        assert c.base_url == "http://localhost:8000"


def test_job_status():
    """Job.status() should return the parsed JSON."""
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        job = Job(c._http, "test-job-id")
        # Mock the HTTP call
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"job_id": "test-job-id", "status": "pending"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(c._http, "get", return_value=mock_resp):
            s = job.status()
            assert s["status"] == "pending"


def test_job_wait_completed():
    """Job.wait() should return when status is completed."""
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        job = Job(c._http, "test-job-id")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        # First call returns running, second returns completed
        mock_resp.json.side_effect = [
            {"status": "running"},
            {"status": "completed"},
        ]

        with patch.object(c._http, "get", return_value=mock_resp):
            with patch("time.sleep"):  # Don't actually sleep
                s = job.wait(interval=1)
                assert s["status"] == "completed"


def test_job_wait_failed():
    """Job.wait() should raise JobFailed when status is failed."""
    with Client(base_url="http://localhost:8000", api_key="scai_test") as c:
        job = Job(c._http, "test-job-id")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "failed", "error": "OOM"}

        with patch.object(c._http, "get", return_value=mock_resp):
            with pytest.raises(JobFailed):
                job.wait(interval=1)
