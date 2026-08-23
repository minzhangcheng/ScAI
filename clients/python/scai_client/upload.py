"""Chunked file upload with SHA256 verification and retry support."""
import hashlib
from pathlib import Path

import httpx

from .errors import FileIntegrityError, UploadError, AuthError

DEFAULT_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def upload_file(
    client: httpx.Client,
    job_id: str,
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = 3,
    file_id: str | None = None,
) -> None:
    """Upload a file to a job using chunked upload with SHA256 verification.

    Args:
        client: Authenticated httpx client
        job_id: The analysis job ID
        file_path: Path to the local file
        chunk_size: Size of each chunk in bytes
        max_retries: Maximum retries per chunk
        file_id: Exact placeholder ID returned by the create endpoint
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    expected_sha256 = sha256_file(file_path)

    # 1. Initialize upload. The server requires file_id for every new client
    # upload; an omitted value is sent explicitly so the server can reject it.
    init_payload = {
        "file_id": file_id,
        "filename": file_path.name,
        "expected_sha256": expected_sha256,
        "total_size": file_size,
        "total_chunks": total_chunks,
        "chunk_size": chunk_size,
    }
    init_resp = client.post(
        f"/api/v1/analyses/{job_id}/files/init",
        json=init_payload,
    )
    if init_resp.status_code != 200:
        if init_resp.status_code == 401:
            raise AuthError("Authentication failed (401)")
        raise UploadError(f"Upload init failed: {init_resp.text}")

    upload_id = init_resp.json()["upload_id"]

    # 2. Upload each chunk
    for chunk_index in range(total_chunks):
        offset = chunk_index * chunk_size
        data = _read_chunk(file_path, offset, chunk_size)

        for attempt in range(max_retries + 1):
            try:
                resp = client.post(
                    f"/api/v1/analyses/{job_id}/files/chunk/{upload_id}/{chunk_index}",
                    files={"file": (file_path.name, data, "application/octet-stream")},
                )
                if resp.status_code == 200:
                    break
                if resp.status_code == 401:
                    raise AuthError("Authentication failed (401)")
                if attempt == max_retries:
                    raise UploadError(
                        f"Chunk {chunk_index} upload failed after {max_retries} retries: {resp.text}"
                    )
            except httpx.TransportError as e:
                if attempt == max_retries:
                    raise UploadError(f"Chunk {chunk_index} transport error: {e}")

    # 3. Complete upload (server verifies SHA256)
    complete_resp = client.post(
        f"/api/v1/analyses/{job_id}/files/complete/{upload_id}",
    )
    if complete_resp.status_code == 400:
        raise FileIntegrityError(file_path.name, expected_sha256)
    if complete_resp.status_code == 401:
        raise AuthError("Authentication failed (401)")
    if complete_resp.status_code != 200:
        raise UploadError(f"Upload complete failed: {complete_resp.text}")


def _read_chunk(path: Path, offset: int, size: int) -> bytes:
    """Read a chunk from a file at the given offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)
