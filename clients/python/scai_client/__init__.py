"""ScAI Python SDK — submit batch analyses via the v1 API.

Usage:
    from scai_client import Client

    client = Client(base_url="http://localhost:3000", api_key="scai_xxx")

    # Submit and wait:
    job = client.submit("config.json")
    job.wait(timeout=3600)
    job.download_export("results.tar.gz")

    # List existing sessions and their status:
    for s in client.list_sessions():
        print(s["id"], s["status"], s.get("current_round"))

    # Import a previously exported archive:
    session_id = client.import_session("session-export.tar.gz")

    # Export an existing session by id:
    path = client.export_session("<session_id>")
    job = client.get_job(session_id)
"""
from .client import Client, login
from .job import Job
from .errors import (
    AuthError,
    FileIntegrityError,
    ImportFailed,
    JobFailed,
    ScaiError,
    UploadError,
)

__all__ = [
    "Client",
    "Job",
    "login",
    "AuthError",
    "FileIntegrityError",
    "ImportFailed",
    "JobFailed",
    "ScaiError",
    "UploadError",
]
# The client follows the application release version rather than maintaining
# an independent release line.
__version__ = "0.50.1"
