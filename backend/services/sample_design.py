"""Resolve replicate counts without collapsing independent data modalities.

Replicate counts belong to a (modality, treatment group) pair.  A session-wide
minimum is not an experimental design: in a mixed single-cell/bulk session it
lets the single-cell arm understate the replicated bulk arm, while a maximum
would let bulk replication license single-cell inference.

This module deliberately resolves counts only.  It does not infer that a known
sample count makes a condition/batch design full-rank; callers use the narrower
``has_replicated_condition_design`` name for that reason.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import re
from typing import Any, Dict


SCRNA_MODALITY = "scRNA-seq"
BULK_RNASEQ_MODALITY = "bulk RNA-seq"

_EXECUTION_BULK = re.compile(
    r"\b(?:pydeseq2|deseqdataset|deseqstats|edger|limma|merged_counts|"
    r"bulk_(?:counts?|expr\w*|matrix|df|degs?|results?))\b",
    re.I,
)
_EXECUTION_SINGLE_CELL = re.compile(
    r"\b(?:scanpy|anndata|pseudobulk|liana|cellchat|bbknn|harmonypy|scvi)\b|"
    r"\bsc\.(?:pp|tl|pl|external)\.|\badata\w*\b",
    re.I,
)


def canonical_modality(section: str, data_type: str) -> str:
    """Return the analysis modality represented by a stored data type."""
    value = str(data_type or "").strip()
    lowered = value.casefold()
    if str(section or "").casefold() == "sequencing":
        if lowered in {
            "scrna-seq", "scrnaseq", "snrna-seq", "snrnaseq",
            "single-cell rna-seq", "single-nucleus rna-seq", "loom",
        }:
            return SCRNA_MODALITY
        if lowered in {"bulk rna-seq", "bulk rnaseq", "bulk_rna_seq"}:
            return BULK_RNASEQ_MODALITY
    return value or "unknown"


def state_has_modality(state: Any, modality: str) -> bool:
    """Whether uploaded files or profiling records contain a modality."""
    for item in list(getattr(state, "data_files", None) or []) + list(
        getattr(state, "detected_sample_counts", None) or []
    ):
        if getattr(item, "section", None) != "sequencing":
            continue
        if (
            canonical_modality(
                "sequencing", getattr(item, "data_type", ""),
            ) == modality
        ):
            return True
    return False


def infer_execution_modalities(code: str) -> list[str]:
    """Derive evidence modalities from executable Python syntax.

    Comments and string literals are deliberately excluded: neither proves
    that an execution used a modality. Import aliases are expanded so an
    ordinary ``DeseqDataSet as DDS`` call remains classifiable.
    """
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return []

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".")[0]
                aliases[local] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name == "*":
                    continue
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    def resolved_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            prefix = resolved_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    tokens: list[str] = list(aliases.values())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.append(aliases.get(node.id, node.id))
        elif isinstance(node, ast.Attribute):
            tokens.append(resolved_name(node))
        elif isinstance(node, ast.Call):
            tokens.append(resolved_name(node.func))
    value = "\n".join(tokens)
    modalities: list[str] = []
    if _EXECUTION_SINGLE_CELL.search(value):
        modalities.append("single_cell")
    if _EXECUTION_BULK.search(value):
        modalities.append("bulk_rna_seq")
    return modalities


def finding_evidence_errors(
    finding_records: list[dict[str, Any]],
    requested_evidence_ids: set[str],
    evidence_by_id: dict[str, Any],
) -> list[str]:
    """Validate each finding against its own host-classified executions."""
    errors: list[str] = []
    eligible_ids = set(evidence_by_id)
    required_by_modality = {
        "single_cell": {"single_cell"},
        "bulk_rna_seq": {"bulk_rna_seq"},
        "cross_modal": {"single_cell", "bulk_rna_seq"},
    }
    for index, record in enumerate(finding_records, start=1):
        record_ids = {
            str(value).strip()
            for value in record.get("evidence_execution_ids", [])
            if str(value).strip()
        }
        invalid_ids = record_ids - eligible_ids
        unlisted_ids = record_ids - requested_evidence_ids
        if not record_ids:
            errors.append(f"finding {index} cites no execution IDs")
            continue
        if invalid_ids:
            errors.append(
                f"finding {index} cites invalid IDs {sorted(invalid_ids)}"
            )
            continue
        if unlisted_ids:
            errors.append(
                f"finding {index} IDs are absent from the report-level "
                f"evidence list: {sorted(unlisted_ids)}"
            )
            continue
        modality = record.get("modality")
        required = required_by_modality.get(modality)
        if modality is not None and required is None:
            errors.append(f"finding {index} has invalid modality {modality!r}")
            continue
        execution_modalities = [
            set(getattr(evidence_by_id[execution_id], "analysis_modalities", []))
            for execution_id in record_ids
        ]
        supported = {
            value
            for modalities in execution_modalities
            for value in modalities
        }
        has_support = (
            required.issubset(supported)
            if modality == "cross_modal" and required is not None else
            any(required.issubset(modalities) for modalities in execution_modalities)
            if required is not None else True
        )
        if required is not None and not has_support:
            errors.append(
                f"finding {index} declares {modality!r}, but its executed code "
                "does not include dedicated matching evidence; observed "
                f"modalities are {sorted(supported) or ['unresolved']}"
            )
    return errors


def samples_per_group_by_modality(state: Any) -> Dict[str, Dict[str, int]]:
    """Resolve sample counts independently for every sequencing modality.

    Measured group-specific records are authoritative. Per-sample numbered
    uploads provide a count when profiling records are absent. Declarations are
    display metadata only and never determine scientific acceptance. An ungrouped
    matrix keeps the modality present but unresolved:
    its total column count is never divided among groups. Profiling must persist
    a group-specific record after verifying column-to-group membership.
    """
    groups = list(getattr(state, "treatment_groups", None) or [])
    group_by_id = {getattr(group, "id", None): group for group in groups}

    present_modalities: set[str] = set()
    relevant_groups: dict[str, set[Any]] = defaultdict(set)
    numbered: dict[tuple[str, Any], set[Any]] = defaultdict(set)
    measured: dict[tuple[str, Any], int] = {}
    profiled_keys: set[tuple[str, Any]] = set()

    for data_file in (getattr(state, "data_files", None) or []):
        if getattr(data_file, "section", None) != "sequencing":
            continue
        modality = canonical_modality("sequencing", getattr(data_file, "data_type", ""))
        present_modalities.add(modality)
        group_id = getattr(data_file, "group_id", None)
        if group_id in group_by_id:
            relevant_groups[modality].add(group_id)
        sample_number = getattr(data_file, "sample_number", None)
        if group_id in group_by_id and sample_number is not None:
            numbered[(modality, group_id)].add(sample_number)

    for record in (getattr(state, "detected_sample_counts", None) or []):
        if getattr(record, "section", None) != "sequencing":
            continue
        modality = canonical_modality("sequencing", getattr(record, "data_type", ""))
        present_modalities.add(modality)
        group_id = getattr(record, "group_id", None)
        if group_id is None:
            continue
        if group_id not in group_by_id:
            continue
        relevant_groups[modality].add(group_id)
        key = (modality, group_id)
        profiled_keys.add(key)
        count = int(getattr(record, "detected_count", 0) or 0)
        if count < 1:
            continue
        prior = measured.get(key)
        measured[key] = count if prior is None else min(prior, count)

    resolved: Dict[str, Dict[str, int]] = {}
    for modality in sorted(present_modalities):
        # A total for an ungrouped matrix cannot establish how its columns are
        # distributed among treatment groups. Profiling must register one
        # group-specific record per verified allocation; otherwise this
        # modality remains explicitly unresolved.
        group_ids = set(relevant_groups.get(modality, ()))
        counts: Dict[str, int] = {}
        for group_id in group_ids:
            group = group_by_id[group_id]
            key = (modality, group_id)
            if key in measured:
                count = measured[key]
            elif key in profiled_keys:
                # An explicit undetermined measurement is authoritative. File
                # numbering is only a fallback when profiling never produced a
                # record for this modality/group pair.
                count = 0
            else:
                count = len(numbered.get(key, ()))
            if count > 0:
                counts[str(getattr(group, "name", group_id))] = count
        # A partial allocation is not a condition design.  Keeping only the
        # groups that happened to resolve lets (for example) Control=3 with a
        # missing Treatment group pass every ``min(counts.values())`` check.
        # Fail the whole modality closed until every declared comparison group
        # has an assigned independent-unit count.
        expected_names = {
            str(getattr(group, "name", getattr(group, "id", "")))
            for group in groups
        }
        resolved[modality] = (
            counts if expected_names and expected_names.issubset(counts) else {}
        )
    return resolved


def has_replicated_condition_design(state: Any, modality: str) -> bool:
    """Whether every treatment group has at least two verified independent units.

    Declarations remain a compatibility fallback for legacy display consumers,
    but cannot by themselves enable inferential work. This is a necessary
    replication check, not proof that technical batch and condition are
    separable. Planning text must retain that distinction.
    """
    groups = list(getattr(state, "treatment_groups", None) or [])
    if len(groups) < 2:
        return False
    counts = samples_per_group_by_modality(state).get(modality, {})
    expected = {str(getattr(group, "name", "")) for group in groups}
    if not (expected and expected.issubset(counts)):
        return False

    verified: set[str] = set()
    numbered: dict[Any, set[Any]] = defaultdict(set)
    group_by_id = {getattr(group, "id", None): group for group in groups}
    for data_file in (getattr(state, "data_files", None) or []):
        if getattr(data_file, "section", None) != "sequencing":
            continue
        if canonical_modality(
            "sequencing", getattr(data_file, "data_type", ""),
        ) != modality:
            continue
        group_id = getattr(data_file, "group_id", None)
        sample_number = getattr(data_file, "sample_number", None)
        if group_id in group_by_id and sample_number is not None:
            numbered[group_id].add(sample_number)
    for record in (getattr(state, "detected_sample_counts", None) or []):
        if getattr(record, "section", None) != "sequencing":
            continue
        if canonical_modality(
            "sequencing", getattr(record, "data_type", ""),
        ) != modality:
            continue
        detected = int(getattr(record, "detected_count", 0) or 0)
        group_id = getattr(record, "group_id", None)
        if group_id is None:
            continue
        if group_id in group_by_id and detected >= 2:
            verified.add(str(getattr(group_by_id[group_id], "name", "")))
    for group_id, sample_numbers in numbered.items():
        if len(sample_numbers) >= 2:
            verified.add(str(getattr(group_by_id[group_id], "name", "")))

    return verified.issuperset(expected) and all(
        counts[name] >= 2 for name in expected
    )


def single_cell_design_instruction(state: Any) -> str:
    """Render measured single-cell design facts for PI consideration."""
    designs = samples_per_group_by_modality(state)
    counts = designs.get(SCRNA_MODALITY)
    if not counts:
        if not state_has_modality(state, SCRNA_MODALITY):
            return ""
        return (
            "Single-cell independent-unit counts or group membership could not "
            "be resolved. The PI must decide which comparisons are supportable "
            "and state any resulting limitations."
        )
    if len(getattr(state, "treatment_groups", None) or []) < 2:
        return (
            "Only one treatment group is present. There is no between-condition "
            "contrast in the recorded design; any technical batch variable must "
            "be identified separately."
        )
    rendered = ", ".join(f"{group}={count}" for group, count in counts.items())
    if has_replicated_condition_design(state, SCRNA_MODALITY):
        return (
            f"Single-cell independent-unit counts are {rendered}. Sample-level "
            "replication is recorded. Replicate counts alone do not establish "
            "whether condition and technical batch are separable; the PI must "
            "judge any correction or condition-level inference in that context."
        )
    bulk_clause = (
        "Replicated bulk data is recorded separately and is available to the PI "
        "when judging cross-modal support."
        if has_replicated_condition_design(state, BULK_RNASEQ_MODALITY)
        else
        "No separately replicated sequencing modality is recorded."
    )
    return (
        f"Single-cell independent-unit counts are {rendered}. The single-cell arm "
        "does not have replicated conditions, so sample and condition may be "
        "confounded. The PI must decide the analytical approach, inferential "
        f"strength, and required disclosure. {bulk_clause}"
    )


def bulk_design_instruction(state: Any) -> str:
    """Render measured bulk design facts for PI consideration."""
    designs = samples_per_group_by_modality(state)
    counts = designs.get(BULK_RNASEQ_MODALITY)
    if counts is None:
        return ""
    if not counts:
        return (
            "Bulk RNA-seq sample-to-group membership could not be resolved. "
            "The PI must decide which comparisons are supportable and state any "
            "resulting limitations."
        )
    rendered = ", ".join(f"{group}={count}" for group, count in counts.items())
    if has_replicated_condition_design(state, BULK_RNASEQ_MODALITY):
        return (
            f"Bulk RNA-seq independent-unit counts are {rendered}; sample-level "
            "replication is recorded. The PI decides whether and how to use "
            "condition-level inference."
        )
    return (
        f"Bulk RNA-seq independent-unit counts are {rendered}. The bulk arm does "
        "not have replicated conditions; the PI must judge the analytical approach, "
        "inferential strength, and required disclosure."
    )


def sequencing_design_instruction(state: Any) -> str:
    """Return measured sequencing-design facts without making a plan decision."""
    return "\n".join(filter(None, (
        single_cell_design_instruction(state),
        bulk_design_instruction(state),
    )))
