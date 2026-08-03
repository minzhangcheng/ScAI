"""Atomic generation publication for detailed-report output trees."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.utils.files import atomic_write_bytes, fsync_directory, safe_filename


logger = logging.getLogger(__name__)

GENERATIONS_DIR_NAME = ".report-generations"
CURRENT_GENERATION_POINTER = ".report-current.json"
GENERATION_MODE_SENTINEL = ".report-generation-mode"
_STAGING_PREFIX = ".staging-"
_GENERATION_PREFIX = "generation-"
_POINTER_VERSION = 1
REPORT_GENERATION_RETAIN_COUNT = 3
REPORT_GENERATION_RETIRE_GRACE_SECONDS = 60 * 60


class ReportGenerationError(RuntimeError):
    """A report generation could not be safely staged or published."""


def _require_real_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ReportGenerationError(f"report directory is missing: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ReportGenerationError(f"report path is not a real directory: {path}")


def _generation_root(output_root: Path) -> Path:
    output_root = Path(output_root)
    _require_real_directory(output_root, create=True)
    root = output_root / GENERATIONS_DIR_NAME
    _require_real_directory(root, create=True)
    return root


def _decode_pointer(pointer_path: Path) -> str:
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportGenerationError("invalid current report generation pointer") from exc
    if not isinstance(payload, dict) or payload.get("version") != _POINTER_VERSION:
        raise ReportGenerationError("unsupported current report generation pointer")
    generation = payload.get("generation")
    if (
        not isinstance(generation, str)
        or not generation.startswith(_GENERATION_PREFIX)
        or generation != Path(generation).name
        or "/" in generation
        or "\\" in generation
    ):
        raise ReportGenerationError("unsafe current report generation pointer")
    return generation


def resolve_current_report_dir(output_root: str | Path) -> Path:
    """Resolve one immutable published generation, or a legacy flat tree."""
    output_root = Path(output_root)
    pointer_path = output_root / CURRENT_GENERATION_POINTER
    try:
        pointer_metadata = pointer_path.lstat()
    except FileNotFoundError:
        sentinel_path = output_root / GENERATION_MODE_SENTINEL
        try:
            sentinel_metadata = sentinel_path.lstat()
        except FileNotFoundError:
            sentinel_metadata = None
        if sentinel_metadata is not None:
            if stat.S_ISLNK(sentinel_metadata.st_mode) or not stat.S_ISREG(
                sentinel_metadata.st_mode
            ):
                raise ReportGenerationError("report generation marker is unsafe")
            raise ReportGenerationError(
                "report generation store exists without a current pointer"
            )
        generations_root = output_root / GENERATIONS_DIR_NAME
        try:
            generations_metadata = generations_root.lstat()
        except FileNotFoundError:
            generations_metadata = None
        if generations_metadata is not None:
            if stat.S_ISLNK(generations_metadata.st_mode) or not stat.S_ISDIR(
                generations_metadata.st_mode
            ):
                raise ReportGenerationError("report generation store is unsafe")
            for entry in generations_root.iterdir():
                try:
                    metadata = entry.lstat()
                except OSError as exc:
                    raise ReportGenerationError(
                        "report generation store cannot be inspected"
                    ) from exc
                if (
                    entry.name.startswith(_GENERATION_PREFIX)
                    and stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                ):
                    continue
                if (
                    entry.name.startswith(_STAGING_PREFIX)
                    and stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                ):
                    continue
                raise ReportGenerationError(
                    "report generation store contains an unsafe entry"
                )
        return output_root
    if stat.S_ISLNK(pointer_metadata.st_mode) or not stat.S_ISREG(
        pointer_metadata.st_mode
    ):
        raise ReportGenerationError("current report generation pointer is unsafe")

    generation = _decode_pointer(pointer_path)
    generations_root = output_root / GENERATIONS_DIR_NAME
    _require_real_directory(generations_root)
    candidate = generations_root / generation
    _require_real_directory(candidate)
    try:
        resolved_root = generations_root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReportGenerationError("current report generation is missing") from exc
    if (
        resolved_candidate.parent != resolved_root
        or not resolved_candidate.is_dir()
    ):
        raise ReportGenerationError("current report generation is unsafe")
    return resolved_candidate


def report_matches_published_dir(report: Any, published_dir: Path) -> bool:
    """Return whether a cached report belongs to the resolved generation."""
    if published_dir.parent.name != GENERATIONS_DIR_NAME:
        return not (
            isinstance(report, dict) and report.get("artifact_generation_id")
        )
    return bool(
        isinstance(report, dict)
        and report.get("artifact_generation_id") == published_dir.name
    )


def _rebase_generation_paths(value: Any, staging: Path, published: Path) -> Any:
    staging_prefix = str(staging)
    published_prefix = str(published)
    if isinstance(value, str):
        if value == staging_prefix:
            return published_prefix
        prefix = staging_prefix + os.sep
        if value.startswith(prefix):
            return published_prefix + value[len(staging_prefix):]
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _rebase_generation_paths(item, staging, published)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _rebase_generation_paths(item, staging, published)
        return value
    if isinstance(value, tuple):
        return tuple(
            _rebase_generation_paths(item, staging, published) for item in value
        )
    return value


def _validate_asset_path(raw_path: Any, expected_dir: Path, label: str) -> None:
    if raw_path in (None, ""):
        return
    if not isinstance(raw_path, str):
        raise ReportGenerationError(f"staged {label} path is not a string")
    candidate = Path(raw_path)
    try:
        resolved_dir = expected_dir.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReportGenerationError(
            f"staged {label} path is missing: {candidate}"
        ) from exc
    if (
        candidate.is_symlink()
        or not resolved_candidate.is_file()
        or resolved_candidate.parent != resolved_dir
    ):
        raise ReportGenerationError(
            f"staged {label} path escapes its immutable asset directory"
        )


def _validate_report_asset_references(report: dict[str, Any], staging: Path) -> None:
    """Require serialized figure/table paths to reference staged immutable bytes."""
    figures: list[Any] = list(report.get("figures") or [])
    tables: list[Any] = list(report.get("tables") or [])
    for section in report.get("sections") or []:
        if not isinstance(section, dict):
            continue
        figures.extend(section.get("figures") or [])
        tables.extend(section.get("tables") or [])

    for figure in figures:
        if not isinstance(figure, dict):
            continue
        for path_key in ("png_path", "path"):
            _validate_asset_path(
                figure.get(path_key),
                staging / "figures",
                f"figure {path_key}",
            )
        _validate_asset_path(
            figure.get("pdf_path"),
            staging / "figures",
            "figure PDF",
        )
    for table in tables:
        if not isinstance(table, dict):
            continue
        for path_key in ("csv_path", "path"):
            _validate_asset_path(
                table.get(path_key),
                staging / "tables",
                f"table {path_key}",
            )
    _validate_asset_path(
        report.get("graphic_abstract_path"),
        staging / "figures",
        "graphic abstract",
    )


def _rebind_imported_asset_entry(
    entry: dict[str, Any],
    *,
    directory: Path,
    path_keys: tuple[str, ...],
    label: str,
) -> None:
    raw = next(
        (
            entry.get(key)
            for key in path_keys
            if isinstance(entry.get(key), str) and entry.get(key).strip()
        ),
        None,
    )
    filename = Path(str(entry.get("filename") or raw or "")).name
    if not filename:
        return
    target = directory / filename
    if target.is_symlink() or not target.is_file():
        raise ReportGenerationError(
            f"imported {label} metadata references a missing asset: {filename}"
        )
    for key in path_keys:
        if key in entry or key == path_keys[0]:
            entry[key] = str(target)


def _rebind_imported_report_metadata(
    report: dict[str, Any],
    published: Path,
    *,
    generation_id: str | None,
) -> None:
    figures: list[Any] = list(report.get("figures") or [])
    tables: list[Any] = list(report.get("tables") or [])
    for section in report.get("sections") or []:
        if not isinstance(section, dict):
            continue
        figures.extend(section.get("figures") or [])
        tables.extend(section.get("tables") or [])
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        _rebind_imported_asset_entry(
            figure,
            directory=published / "figures",
            path_keys=("png_path", "path"),
            label="figure",
        )
        raw_pdf = figure.get("pdf_path")
        if isinstance(raw_pdf, str) and raw_pdf.strip():
            pdf_target = published / "figures" / Path(raw_pdf).name
            if pdf_target.is_symlink() or not pdf_target.is_file():
                raise ReportGenerationError(
                    "imported figure metadata references a missing PDF"
                )
            figure["pdf_path"] = str(pdf_target)
    for table in tables:
        if not isinstance(table, dict):
            continue
        _rebind_imported_asset_entry(
            table,
            directory=published / "tables",
            path_keys=("csv_path", "path"),
            label="table",
        )
    raw_ga = report.get("graphic_abstract_path")
    if isinstance(raw_ga, str) and raw_ga.strip():
        target = published / "figures" / Path(raw_ga).name
        if target.is_symlink() or not target.is_file():
            raise ReportGenerationError(
                "imported report references a missing graphic abstract"
            )
        report["graphic_abstract_path"] = str(target)
    report["html_path"] = str(published / "report.html")
    if generation_id is not None:
        report["artifact_generation_id"] = generation_id


def validate_and_rebind_imported_reports(reports_root: str | Path) -> None:
    """Validate imported pointers and bind metadata to destination-owned bytes."""
    reports_root = Path(reports_root)
    roots: list[tuple[Path, bool]] = [(reports_root, False)]
    branches = reports_root / "branches"
    try:
        branches_metadata = branches.lstat()
    except FileNotFoundError:
        branches_metadata = None
    if branches_metadata is not None:
        if stat.S_ISLNK(branches_metadata.st_mode) or not stat.S_ISDIR(
            branches_metadata.st_mode
        ):
            raise ReportGenerationError("imported report branches path is unsafe")
        for path in sorted(branches.iterdir()):
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise ReportGenerationError(
                    "imported branch report path cannot be inspected"
                ) from exc
            if (
                path.name.startswith(".")
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                raise ReportGenerationError(
                    "imported branch report entry is unsafe"
                )
            roots.append((path, True))

    for output_root, is_branch in roots:
        pointer = output_root / CURRENT_GENERATION_POINTER
        generations = output_root / GENERATIONS_DIR_NAME
        try:
            pointer.lstat()
            generation_mode = True
        except FileNotFoundError:
            generation_mode = False
        if not generation_mode:
            try:
                generations.lstat()
            except FileNotFoundError:
                legacy_json = output_root / "report.json"
                if not legacy_json.is_file() or legacy_json.is_symlink():
                    if is_branch:
                        raise ReportGenerationError(
                            "imported branch has no valid report publication"
                        )
                    continue
                published = output_root
                generation_id = None
            else:
                published = resolve_current_report_dir(output_root)
                generation_id = (
                    published.name
                    if published.parent.name == GENERATIONS_DIR_NAME
                    else None
                )
        else:
            published = resolve_current_report_dir(output_root)
            generation_id = published.name
        _validate_staged_tree(published)
        try:
            report = json.loads(
                (published / "report.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportGenerationError("imported report.json is invalid") from exc
        if not isinstance(report, dict):
            raise ReportGenerationError("imported report.json is not an object")
        declared_branch = report.get("branch_id")
        if is_branch:
            if isinstance(declared_branch, str) and declared_branch:
                from backend.services.report.paths import _sanitize_branch_id

                legacy_component = safe_filename(
                    declared_branch.replace("/", "_").replace("\\", "_")
                )
                if output_root.name not in {
                    _sanitize_branch_id(declared_branch),
                    legacy_component,
                }:
                    raise ReportGenerationError(
                        "imported branch directory does not match report lineage"
                    )
            elif not (
                "--" not in output_root.name
                and re.fullmatch(
                    r"restore-[A-Za-z0-9][A-Za-z0-9_-]{0,127}",
                    output_root.name,
                )
                and safe_filename(output_root.name) == output_root.name
            ):
                raise ReportGenerationError(
                    "imported branch report has no lineage provenance"
                )
        elif declared_branch not in (None, "", "main"):
            raise ReportGenerationError(
                "imported main report declares a branch lineage"
            )
        declared_generation = report.get("artifact_generation_id")
        if generation_id is not None and declared_generation not in (
            None,
            generation_id,
        ):
            raise ReportGenerationError(
                "imported report metadata belongs to a different generation"
            )
        _rebind_imported_report_metadata(
            report,
            published,
            generation_id=generation_id,
        )
        _validate_report_asset_references(report, published)
        atomic_write_bytes(
            published / "report.json",
            json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            mode=0o600,
        )
        if generation_id is not None:
            atomic_write_bytes(
                output_root / GENERATION_MODE_SENTINEL,
                b'{"version":1}',
                mode=0o600,
            )


def _validate_staged_tree(staging: Path) -> None:
    for required_file in ("report.json", "report.html"):
        path = staging / required_file
        if path.is_symlink() or not path.is_file():
            raise ReportGenerationError(
                f"staged report is missing required file: {required_file}"
            )
    for required_dir in ("figures", "tables"):
        path = staging / required_dir
        if path.is_symlink() or not path.is_dir():
            raise ReportGenerationError(
                f"staged report is missing required directory: {required_dir}"
            )
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ReportGenerationError(f"staged report contains a symlink: {path}")
        if not path.is_dir() and not path.is_file():
            raise ReportGenerationError(
                f"staged report contains a non-regular entry: {path}"
            )


def _fsync_staged_tree(staging: Path) -> None:
    directories = [staging]
    for path in staging.rglob("*"):
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for directory in reversed(directories):
        fsync_directory(directory)


@contextmanager
def report_publication_lock(output_root: str | Path) -> Iterator[None]:
    """Serialize publication, export capture, and retired-generation cleanup."""
    output_root = Path(output_root)
    _require_real_directory(output_root, create=True)
    lock_path = output_root / ".report-publication.lock"
    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def cleanup_report_generations(
    output_root: str | Path,
    *,
    now: float | None = None,
    retain_count: int = REPORT_GENERATION_RETAIN_COUNT,
    grace_seconds: float = REPORT_GENERATION_RETIRE_GRACE_SECONDS,
) -> list[Path]:
    """Remove expired old generations and crashed staging trees.

    The selected generation is resolved again while holding the publication
    lock and is never removed.  A grace interval protects paths already handed
    to response machinery; the newest ``retain_count`` published trees are
    retained even after that interval.
    """
    output_root = Path(output_root)
    if retain_count < 1 or grace_seconds < 0:
        raise ValueError("invalid report generation retention policy")
    removed: list[Path] = []
    current_time = time.time() if now is None else now

    try:
        with report_publication_lock(output_root):
            try:
                current = resolve_current_report_dir(output_root)
            except ReportGenerationError:
                return removed
            generations_root = output_root / GENERATIONS_DIR_NAME
            if not generations_root.is_dir() or generations_root.is_symlink():
                return removed

            published: list[Path] = []
            crashed_staging: list[Path] = []
            for candidate in generations_root.iterdir():
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                if candidate.name.startswith(_GENERATION_PREFIX):
                    published.append(candidate)
                elif candidate.name.startswith(_STAGING_PREFIX):
                    crashed_staging.append(candidate)
            published.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            protected = {path.name for path in published[:retain_count]}
            if current.parent == generations_root:
                protected.add(current.name)

            for candidate in published + crashed_staging:
                if candidate.name in protected:
                    continue
                try:
                    age = current_time - candidate.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age < grace_seconds:
                    continue
                shutil.rmtree(candidate)
                removed.append(candidate)
            if removed:
                fsync_directory(generations_root)
    except FileNotFoundError:
        return removed
    return removed


class ReportGenerationTransaction:
    """Build a complete report privately, then publish one atomic pointer."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)
        self.generations_root = _generation_root(self.output_root)
        token = uuid.uuid4().hex
        self.generation_id = f"{_GENERATION_PREFIX}{token}"
        self.staging_dir = self.generations_root / f"{_STAGING_PREFIX}{token}"
        self.published_dir = self.generations_root / self.generation_id
        self.staging_dir.mkdir(mode=0o700)
        self._committed = False
        self._published_visible = False

    def commit_report(self, report: dict[str, Any]) -> Path:
        """Finalize metadata, validate the tree, and atomically select it."""
        if self._committed:
            raise ReportGenerationError("report generation was already published")

        _validate_report_asset_references(report, self.staging_dir)
        report["artifact_generation_id"] = self.generation_id
        report["html_path"] = str(self.published_dir / "report.html")
        _rebase_generation_paths(report, self.staging_dir, self.published_dir)
        payload = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        atomic_write_bytes(self.staging_dir / "report.json", payload, mode=0o600)
        _validate_staged_tree(self.staging_dir)
        _fsync_staged_tree(self.staging_dir)

        with report_publication_lock(self.output_root):
            os.replace(self.staging_dir, self.published_dir)
            self._published_visible = True
            fsync_directory(self.generations_root)
            pointer_payload = json.dumps(
                {
                    "version": _POINTER_VERSION,
                    "generation": self.generation_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            try:
                atomic_write_bytes(
                    self.output_root / CURRENT_GENERATION_POINTER,
                    pointer_payload,
                    mode=0o600,
                )
            except BaseException:
                try:
                    visible = resolve_current_report_dir(self.output_root)
                except ReportGenerationError:
                    visible = None
                if visible != self.published_dir.resolve():
                    raise
                logger.critical(
                    "Report generation %s is visible but pointer durability "
                    "could not be confirmed",
                    self.generation_id,
                    exc_info=True,
                )
                raise
            try:
                atomic_write_bytes(
                    self.output_root / GENERATION_MODE_SENTINEL,
                    b'{"version":1}',
                    mode=0o600,
                )
            except OSError:
                logger.critical(
                    "Report pointer is durable but generation-mode marker "
                    "could not be persisted",
                    exc_info=True,
                )
            self._committed = True

        try:
            cleanup_report_generations(self.output_root)
        except Exception:
            logger.warning("Failed to sweep retired report generations", exc_info=True)
        return self.published_dir

    def abort(self) -> None:
        """Remove only this transaction's unpublished staging directory."""
        if self._committed:
            return
        if self.staging_dir.parent != self.generations_root:
            raise ReportGenerationError("refusing to clean an unsafe staging path")
        if self.staging_dir.exists() and not self.staging_dir.is_symlink():
            shutil.rmtree(self.staging_dir)
