"""Method-neutral host profiling for uploaded bulk expression tables."""
from __future__ import annotations

from pathlib import Path

from ..models.schemas import BulkInputProfile, DATA_TYPE_BULK_RNASEQ


_PROFILE_CHUNK_ROWS = 10_000
_PROFILE_MAX_ROWS = 250_000


def _profile_file(data_file) -> BulkInputProfile:
    import pandas as pd

    path = Path(data_file.file_path)
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    numeric_counts: dict[str, int] = {}
    integer_counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}
    finite_counts: dict[str, int] = {}
    missing: dict[str, int] = {}
    columns: list[str] = []
    rows_scanned = 0
    complete = True
    reader = pd.read_csv(
        path,
        sep=separator,
        chunksize=_PROFILE_CHUNK_ROWS,
        nrows=_PROFILE_MAX_ROWS + 1,
    )
    for chunk in reader:
        if not columns:
            columns = [str(column) for column in chunk.columns]
            numeric_counts = {column: 0 for column in columns}
            integer_counts = {column: 0 for column in columns}
            negative_counts = {column: 0 for column in columns}
            finite_counts = {column: 0 for column in columns}
            missing = {column: 0 for column in columns}
        remaining = _PROFILE_MAX_ROWS - rows_scanned
        if remaining <= 0:
            complete = False
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]
            complete = False
        rows_scanned += len(chunk)
        for raw_column in chunk.columns:
            column = str(raw_column)
            raw = chunk[raw_column]
            converted = pd.to_numeric(raw, errors="coerce")
            finite = converted[
                converted.notna()
                & (converted != float("inf"))
                & (converted != float("-inf"))
            ]
            numeric_counts[column] += int(converted.notna().sum())
            finite_counts[column] += int(finite.size)
            integer_counts[column] += int(
                ((finite - finite.round()).abs() <= 1e-8).sum()
            )
            negative_counts[column] += int((finite < 0).sum())
            missing[column] += int(raw.isna().sum())
        if not complete:
            break

    fractions = {
        column: round(
            numeric_counts[column] / rows_scanned if rows_scanned else 0.0,
            6,
        )
        for column in columns
    }
    candidates = [
        column for column in columns if fractions.get(column, 0.0) >= 0.95
    ]
    annotation = [column for column in columns if column not in candidates]
    candidate_finite = sum(finite_counts[column] for column in candidates)
    candidate_integer = sum(integer_counts[column] for column in candidates)
    integer_like = (
        candidate_integer / candidate_finite if candidate_finite else 0.0
    )
    return BulkInputProfile(
        source_file_id=str(data_file.id),
        shape=[rows_scanned, len(columns)],
        annotation_columns=annotation,
        candidate_sample_columns=candidates,
        numeric_fraction_by_column=fractions,
        integer_like_fraction=round(integer_like, 6),
        negative_value_count=sum(negative_counts[column] for column in candidates),
        missing_value_summary=missing,
        declared_scale="unknown",
        profile_status="complete" if complete else "sampled",
        complete=complete,
        rows_scanned=rows_scanned,
    )


def ensure_bulk_input_profiles(state) -> list[BulkInputProfile]:
    existing = {item.source_file_id: item for item in state.bulk_input_profiles}
    profiles = []
    for data_file in state.data_files:
        if data_file.data_type != DATA_TYPE_BULK_RNASEQ:
            continue
        key = str(data_file.id)
        profile = existing.get(key)
        if profile is None:
            try:
                profile = _profile_file(data_file)
            except Exception as exc:
                profile = BulkInputProfile(
                    source_file_id=key,
                    profile_status="failed",
                    complete=False,
                    error_code=type(exc).__name__,
                )
        profiles.append(profile)
    state.bulk_input_profiles = profiles
    return profiles
