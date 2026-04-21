"""ScAI Python SDK — submit batch analyses via the v1 API."""
from .client import Client
from .job import Job
from .errors import FileIntegrityError, UploadError, JobFailed

__all__ = ["Client", "Job", "FileIntegrityError", "UploadError", "JobFailed"]
__version__ = "0.1.0"
