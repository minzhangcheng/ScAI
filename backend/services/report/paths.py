from pathlib import Path
from typing import Optional

from backend.config import get_session_reports
from backend.utils.files import safe_filename


def _sanitize_branch_id(branch_id: str) -> str:
    """Clean a branch identifier for safe filesystem usage."""
    clean = branch_id.replace("/", "_").replace("\\", "_")
    return safe_filename(clean)


def resolve_report_output_dir(session_id: str, branch_id: Optional[str]) -> Path:
    base = get_session_reports(session_id)
    if branch_id:
        return base / "branches" / _sanitize_branch_id(branch_id)
    return base


def resolve_report_target_lineage(branch_id: Optional[str]) -> str:
    return branch_id or "main"
