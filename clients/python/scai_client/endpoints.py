"""The single registry for retained server routes used by the Python client."""
from urllib.parse import quote


V1 = "/api/v1"


def _segment(value: str) -> str:
    """Encode one route parameter without allowing it to change the route."""
    encoded = quote(str(value), safe="")
    # RFC 3986 leaves dots unescaped even with ``safe=""``.  Exact dot
    # segments are nevertheless structural to URL resolution, so force their
    # percent-encoded spelling before joining them into a route.
    return encoded.replace(".", "%2E") if encoded in {".", ".."} else encoded


def create_analysis() -> str:
    return f"{V1}/analyses"


def analysis(job_id: str) -> str:
    return f"{V1}/analyses/{_segment(job_id)}"


def start_analysis(job_id: str) -> str:
    return f"{analysis(job_id)}/start"


def analysis_export(job_id: str) -> str:
    return f"{analysis(job_id)}/export"


def analysis_config(job_id: str) -> str:
    return f"{analysis(job_id)}/config/export"


def analysis_artifact(job_id: str, name: str) -> str:
    # The server route is {name:path}; preserve artifact separators while
    # escaping each individual component, including structural dot segments.
    encoded_name = "/".join(_segment(part) for part in str(name).split("/"))
    return f"{analysis(job_id)}/artifacts/{encoded_name}"


def upload_init(job_id: str) -> str:
    return f"{analysis(job_id)}/files/init"


def upload_chunk(job_id: str, upload_id: str, chunk_index: int) -> str:
    return f"{analysis(job_id)}/files/chunk/{_segment(upload_id)}/{int(chunk_index)}"


def upload_complete(job_id: str, upload_id: str) -> str:
    return f"{analysis(job_id)}/files/complete/{_segment(upload_id)}"


def health() -> str:
    return "/api/health"


def login() -> str:
    return "/api/auth/login"


def api_keys() -> str:
    return "/api/auth/api-keys"


def sessions() -> str:
    return "/api/sessions"


def session_import() -> str:
    return "/api/session/import"


def import_status(import_id: str) -> str:
    return f"/api/session/import/{_segment(import_id)}"


def import_cancel(import_id: str) -> str:
    return f"{import_status(import_id)}/cancel"


def session_export(session_id: str) -> str:
    return f"/api/session/{_segment(session_id)}/export"


def stop_session(session_id: str) -> str:
    return f"/api/session/{_segment(session_id)}/stop"


def session_snapshots(session_id: str) -> str:
    return f"/api/session/{_segment(session_id)}/snapshots"


def session_snapshot(session_id: str, snapshot_id: str) -> str:
    return f"{session_snapshots(session_id)}/{_segment(snapshot_id)}"


def restore_snapshot(session_id: str, snapshot_id: str) -> str:
    return f"{session_snapshot(session_id, snapshot_id)}/restore"


OUTSIDE_V1 = {
    "health": "v1 has no version or liveness probe.",
    "login": "v1 has no authentication.",
    "api_keys": "mints the credential used by v1 calls.",
    "sessions": "v1 cannot enumerate sessions.",
    "session_import": "v1 has no portable session import.",
    "import_status": "polls the portable import job.",
    "import_cancel": "cancels the portable import job.",
    "session_export": "v1 export is not the portable session archive.",
    "stop_session": "v1 cannot stop a running analysis.",
    "session_snapshots": "v1 has no decision snapshots.",
    "session_snapshot": "deletes one decision snapshot.",
    "restore_snapshot": "forks a session from one decision snapshot.",
}
