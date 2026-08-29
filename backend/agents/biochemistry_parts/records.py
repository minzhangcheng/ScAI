"""Normalizers for extracted biochemistry records.

Pure functions: they turn what a data file says about a measurement into the
fields ``BiochemistryIndicator`` stores. Kept out of the agent because the agent
module is at its size limit and these have no dependency on it.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ...models.schemas import AnalysisState


def clean_optional_text(value: Any) -> Optional[str]:
    """Normalize an LLM "null"-ish field to a real None."""
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    return text


def coerce_sample_number(value: Any) -> Optional[int]:
    """Take a replicate index out of a free-form sample identifier."""
    text = clean_optional_text(value)
    if text is None:
        return None
    digits = re.findall(r"\d+", text)
    if not digits:
        return None
    try:
        return int(digits[-1])
    except (ValueError, TypeError):
        return None


def resolve_group_id(value: Any, state: AnalysisState) -> Optional[str]:
    """Map an extracted group LABEL onto a DECLARED TreatmentGroup id, or None.

    ``group_id`` means TreatmentGroup.id everywhere else in the system, so an
    unmatched label must never be returned here — a fabricated id would be joined
    against downstream as though it were real. The label is not lost: the caller
    records it in ``source_group_label`` with a provenance note, which is how the
    PI gets to see "the file says Treated and no such group is declared".
    """
    label = clean_optional_text(value)
    if label is None:
        return None
    for group in (state.treatment_groups or []):
        if str(group.name or "").strip().casefold() == label.casefold():
            return group.id
        if str(group.id or "") == label:
            return group.id
    return None


def reconcile_provenance(
    *,
    file_group_id: Optional[str],
    file_sample_number: Optional[int],
    row_group_label: Any,
    row_sample_value: Any,
    state: AnalysisState,
) -> tuple[Optional[str], Optional[int], Optional[str], list]:
    """Combine upload-declared metadata with what the file body says.

    The upload values were set by a person in the UI and are the trustworthy ones;
    the row values were parsed out of the document by a model. Precedence is
    therefore: declared wins, parsed fills a gap, and a disagreement is RECORDED
    rather than resolved — silently overwriting a declared group with a parsed one
    can change how many independent samples a comparison believes it has.

    Returns (group_id, sample_number, source_group_label, provenance_notes).
    """
    notes: list = []
    label = clean_optional_text(row_group_label)
    row_group_id = resolve_group_id(row_group_label, state)

    group_id = file_group_id
    if file_group_id is None:
        group_id = row_group_id
        if row_group_id is not None:
            notes.append(f"group taken from the file body ({label!r}); not declared at upload")
    elif row_group_id is not None and row_group_id != file_group_id:
        notes.append(
            f"group conflict: upload declared {file_group_id!r}, file body says {label!r}; "
            "kept the declared value"
        )
    if label is not None and row_group_id is None:
        notes.append(
            f"group label {label!r} in the file body matches no declared treatment group"
        )

    row_sample = coerce_sample_number(row_sample_value)
    if row_sample is not None and row_sample < 1:
        notes.append(f"ignored non-positive sample number parsed from {row_sample_value!r}")
        row_sample = None

    sample_number = file_sample_number
    if file_sample_number is None:
        sample_number = row_sample
        if row_sample is not None:
            notes.append(f"sample number {row_sample} read from the file body")
    elif row_sample is not None and row_sample != file_sample_number:
        notes.append(
            f"sample-number conflict: upload declared {file_sample_number}, file body "
            f"suggests {row_sample}; kept the declared value"
        )

    return group_id, sample_number, label, notes


def merge_marker_records(step1_markers: Any, step3_markers: Any):
    """Keep step 1's per-measurement record; let step 3 only ADD what it did not see.

    Identity is the ``measurement_id`` the host stamped on each measurement and
    asked both steps to echo back. Matching on retyped ``(indicator, group,
    sample)`` text was fragile in both directions: a step that omitted the group
    silently became a different measurement, and two groups' readings of one
    indicator collapsed into one row.

    Returns ``(merged_markers, provenance_notes)``. Nothing is rejected and
    nothing blocks: a step-3 entry that cannot be tied to a measurement is
    reported as an unlinked observation rather than being dropped or promoted to
    a located record.
    """
    step1 = [m for m in (step1_markers or []) if isinstance(m, dict)]
    step3 = [m for m in (step3_markers or []) if isinstance(m, dict)]
    notes: list = []

    def mid(marker):
        return str(marker.get("measurement_id") or "").strip()

    def legacy_key(marker):
        return (
            str(marker.get("indicator", "")).strip().casefold(),
            str(marker.get("group_id") or marker.get("group") or "").strip().casefold(),
            str(marker.get("sample_number") or "").strip(),
        )

    merged = []
    by_id = {}
    by_legacy = {}
    for marker in step1:
        entry = dict(marker)
        merged.append(entry)
        key = mid(entry)
        if key:
            by_id.setdefault(key, entry)
        legacy = legacy_key(entry)
        if legacy[0]:
            # A LIST per key, not one entry: setdefault() collapsed duplicates, so
            # the "only match on a unique legacy key" guard below could never see
            # that there had been more than one and matched ambiguously instead.
            by_legacy.setdefault(legacy, []).append(entry)

    matched = set()
    for marker in step3:
        key = mid(marker)
        target = by_id.get(key) if key else None
        if target is None:
            legacy = legacy_key(marker)
            candidates = by_legacy.get(legacy, []) if legacy[0] else []
            target = candidates[0] if len(candidates) == 1 else None
            if len(candidates) > 1:
                notes.append(
                    f"{marker.get('indicator') or 'unnamed'}: the synthesis step gave no "
                    f"measurement_id and {len(candidates)} measurements share that "
                    "indicator/group/sample; left unlinked rather than guessing which"
                )

        if target is None:
            label = str(marker.get("indicator") or "unnamed").strip()
            evidence = str(marker.get("source_evidence") or "").strip()
            if (
                str(marker.get("source_step") or "").strip().casefold() == "step2"
                and evidence
            ):
                entry = dict(marker)
                entry.setdefault("basis", "none")
                entry.setdefault(
                    "significance",
                    "Observed during synthesis from image evidence; no per-measurement "
                    "comparator was available.",
                )
                merged.append(entry)
                notes.append(
                    f"{label}: added by the synthesis step from image evidence ({evidence})"
                )
            elif str(marker.get("source_step") or "").strip().casefold() == "step2":
                # Claiming an image source without naming one is not evidence, and
                # a located marker is what the report cites.
                notes.append(
                    f"{label}: the synthesis step reported a new measurement from image "
                    "evidence but named no source; recorded as an unlinked observation"
                )
            else:
                notes.append(
                    f"{label}: the synthesis step reported a measurement that matches no "
                    "record from the per-indicator analysis and names no image source; "
                    "recorded as an unlinked observation, not as a located measurement"
                )
            continue

        matched.add(id(target))
        for field in ("status", "basis"):
            old = str(target.get(field, "") or "").strip()
            new = str(marker.get(field, "") or "").strip()
            if new and old and new.casefold() != old.casefold():
                target.setdefault("provenance_notes", []).append(
                    f"synthesis proposed {field}={new!r} but the per-indicator analysis "
                    f"recorded {old!r}; the per-indicator record stands"
                )

    if step3:
        for entry in merged[: len(step1)]:
            if id(entry) not in matched:
                entry.setdefault("provenance_notes", []).append(
                    "not carried forward by the synthesis step; retained from the "
                    "per-indicator analysis"
                )
    return merged, notes


def restore_measurement_identity(markers: Any, indicators: Any, state: Any) -> list:
    """Overwrite the model's identity fields with the record they name.

    The prompt asks the model to echo each measurement's ``measurement_id``. That
    makes the LINK reliable; it does not make the model's retyped indicator,
    value, unit, group or sample reliable, and a plausible-looking mismatch is
    invisible downstream. So the host restores those from the stored record and
    leaves the model only what it was asked to decide: status, basis, and the
    explanation.

    A marker naming no known id keeps what the model wrote and is annotated, so
    the PI can see the judgement was not tied to a measurement.
    """
    by_id = {}
    for indicator in (indicators or []):
        ident = str(getattr(indicator, "id", "") or "")
        if ident:
            by_id[ident] = indicator

    group_names = {}
    for group in (getattr(state, "treatment_groups", None) or []):
        group_names[group.id] = group.name

    restored = []
    for marker in (markers or []):
        if not isinstance(marker, dict):
            continue
        entry = dict(marker)
        record = by_id.get(str(entry.get("measurement_id") or "").strip())
        if record is None:
            if entry.get("measurement_id"):
                entry.setdefault("provenance_notes", []).append(
                    "measurement_id does not match any recorded measurement; the "
                    "identity fields are as the model wrote them"
                )
            restored.append(entry)
            continue
        entry["indicator"] = record.indicator
        entry["value"] = record.value
        entry["unit"] = record.unit
        entry["group_id"] = record.group_id
        entry["group"] = group_names.get(record.group_id) or record.source_group_label or ""
        entry["sample_number"] = record.sample_number
        entry["reference_range"] = record.reference_range
        restored.append(entry)
    return restored
