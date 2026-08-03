import hashlib
import json
import stat
from pathlib import Path
from typing import Optional

from backend.config import get_session_reports
from backend.utils.files import safe_filename
from backend.services.report.output_generation import (
    ReportGenerationError,
    resolve_current_report_dir,
)


def _sanitize_branch_id(branch_id: str) -> str:
    """Map a non-empty lineage ID to one safe, collision-resistant component."""
    if not isinstance(branch_id, str) or not branch_id.strip():
        raise ValueError("branch_id must be a non-empty string")
    readable = safe_filename(branch_id.replace("/", "_").replace("\\", "_"))
    readable = readable.lstrip(".")[:48] or "branch"
    digest = hashlib.sha256(branch_id.encode("utf-8")).hexdigest()
    return f"{readable}--{digest}"


def resolve_report_output_dir(session_id: str, branch_id: Optional[str]) -> Path:
    base = get_session_reports(session_id)
    if branch_id:
        branches = base / "branches"
        canonical = branches / _sanitize_branch_id(branch_id)
        try:
            canonical_metadata = canonical.lstat()
        except FileNotFoundError:
            canonical_metadata = None
        except OSError as exc:
            raise ReportGenerationError("branch report path cannot be inspected") from exc
        if canonical_metadata is not None:
            if stat.S_ISLNK(canonical_metadata.st_mode) or not stat.S_ISDIR(
                canonical_metadata.st_mode
            ):
                raise ReportGenerationError("branch report path is unsafe")
            return canonical

        # Compatibility is allowed only when the old collision-prone directory
        # can prove which lineage owns it from its serialized report metadata.
        legacy_component = safe_filename(
            branch_id.replace("/", "_").replace("\\", "_")
        )
        legacy = (
            branches / legacy_component
            if legacy_component and len(legacy_component.encode("utf-8")) <= 240
            else None
        )
        if legacy is not None and legacy != canonical:
            try:
                legacy_metadata = legacy.lstat()
            except FileNotFoundError:
                legacy_metadata = None
            except OSError as exc:
                raise ReportGenerationError(
                    "legacy branch report path cannot be inspected"
                ) from exc
        else:
            legacy_metadata = None
        if legacy_metadata is not None:
            if stat.S_ISLNK(legacy_metadata.st_mode) or not stat.S_ISDIR(
                legacy_metadata.st_mode
            ):
                raise ReportGenerationError("legacy branch report path is unsafe")
            try:
                published = resolve_current_report_dir(legacy)
                payload = json.loads(
                    (published / "report.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError, ReportGenerationError):
                payload = None
            if isinstance(payload, dict) and payload.get("branch_id") == branch_id:
                return legacy
            if (
                branch_id.startswith("restore-")
                and legacy_component == branch_id
            ):
                return legacy
            raise ReportGenerationError(
                "legacy branch report directory has no unambiguous lineage marker"
            )
        return canonical
    return base


def resolve_report_target_lineage(branch_id: Optional[str]) -> str:
    return branch_id or "main"


def resolve_published_report_dir(
    session_id: str,
    branch_id: Optional[str],
) -> Path:
    """Resolve the immutable published report tree for one lineage."""
    published = resolve_current_report_dir(
        resolve_report_output_dir(session_id, branch_id)
    )
    if branch_id and published.parent.name == ".report-generations":
        try:
            payload = json.loads(
                (published / "report.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportGenerationError(
                "published branch report metadata is invalid"
            ) from exc
        if not isinstance(payload, dict) or payload.get("branch_id") != branch_id:
            raise ReportGenerationError(
                "published report does not belong to the requested lineage"
            )
    return published
