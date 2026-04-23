"""ScAI Python SDK — submit batch analyses via the v1 API."""
from .client import Client
from .job import Job
from .errors import AuthError, FileIntegrityError, JobFailed, ScaiError, UploadError

__all__ = [
    "Client",
    "Job",
    "AuthError",
    "FileIntegrityError",
    "JobFailed",
    "ScaiError",
    "UploadError",
]
__version__ = "0.1.0"
