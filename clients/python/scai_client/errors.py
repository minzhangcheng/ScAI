"""Custom exception types for the ScAI SDK."""


class ScaiError(Exception):
    """Base error for all ScAI SDK exceptions."""


class AuthError(ScaiError):
    """Raised when authentication fails (invalid JWT or API key)."""


class FileIntegrityError(ScaiError):
    """Raised when a file's SHA256 checksum doesn't match the expected value."""

    def __init__(self, filename: str, expected: str = "", actual: str = ""):
        self.filename = filename
        self.expected = expected
        self.actual = actual
        msg = f"File integrity check failed for {filename}"
        if expected:
            msg += f": expected {expected[:16]}..., got {actual[:16]}..."
        super().__init__(msg)


class UploadError(ScaiError):
    """Raised when a file upload fails."""
    pass


class JobFailed(ScaiError):
    """Raised when a job ends in 'failed' status."""

    def __init__(self, job_id: str, error: str = "unknown"):
        self.job_id = job_id
        self.error = error
        super().__init__(f"Job {job_id} failed: {error}")


class ImportFailed(ScaiError):
    """Raised when a session import job ends in 'failed' state."""

    def __init__(self, import_id: str, error: str = "unknown"):
        self.import_id = import_id
        self.error = error
        super().__init__(f"Import {import_id} failed: {error}")