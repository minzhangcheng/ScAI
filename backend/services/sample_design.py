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
    display metadata only and never enter an inferential gate. An ungrouped
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

    Declarations remain a compatibility fallback for display and legacy gates,
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
    """A deterministic planning constraint derived from the single-cell n."""
    designs = samples_per_group_by_modality(state)
    counts = designs.get(SCRNA_MODALITY)
    if not counts:
        if not state_has_modality(state, SCRNA_MODALITY):
            return ""
        return (
            "Single-cell independent-unit counts or group membership could not "
            "be resolved. Do not plan condition-level inference or batch "
            "correction until the sample design is verified."
        )
    if len(getattr(state, "treatment_groups", None) or []) < 2:
        return (
            "Only one treatment group is present. No between-condition inference "
            "is planned; any technical batch correction still requires an "
            "identified, non-biological batch variable."
        )
    rendered = ", ".join(f"{group}={count}" for group, count in counts.items())
    if has_replicated_condition_design(state, SCRNA_MODALITY):
        return (
            f"Single-cell independent-unit counts are {rendered}. Sample-level "
            "condition inference may be planned. Replicate counts alone do not "
            "establish that condition and technical batch are separable, so do "
            "not plan batch correction without a separately verified full-rank "
            "condition-by-batch design."
        )
    bulk_clause = (
        "Replicated bulk data is the condition-level inferential layer and "
        "single-cell data may localize those results to cell types or states."
        if has_replicated_condition_design(state, BULK_RNASEQ_MODALITY)
        else
        "No other modality's sample count can license single-cell condition inference."
    )
    return (
        f"Single-cell independent-unit counts are {rendered}. The single-cell arm "
        "does not have replicated conditions: concatenate without condition-confounded "
        "batch correction; do not plan cell-level or inferential pseudobulk condition "
        f"tests. Single-cell condition contrasts must be descriptive. {bulk_clause}"
    )


def bulk_design_instruction(state: Any) -> str:
    """A deterministic planning constraint derived from bulk replicate records."""
    designs = samples_per_group_by_modality(state)
    counts = designs.get(BULK_RNASEQ_MODALITY)
    if counts is None:
        return ""
    if not counts:
        return (
            "Bulk RNA-seq sample-to-group membership could not be resolved. "
            "Do not plan condition-level DEG, enrichment from DEG, or inferential "
            "cross-modal validation until group-specific replicate records are verified."
        )
    rendered = ", ".join(f"{group}={count}" for group, count in counts.items())
    if has_replicated_condition_design(state, BULK_RNASEQ_MODALITY):
        return (
            f"Bulk RNA-seq independent-unit counts are {rendered}; sample-level "
            "condition inference may be planned."
        )
    return (
        f"Bulk RNA-seq independent-unit counts are {rendered}. The bulk arm does "
        "not have replicated conditions, so condition comparisons must remain descriptive."
    )


def sequencing_design_instruction(state: Any) -> str:
    """Return all sequencing-modality planning constraints."""
    return "\n".join(filter(None, (
        single_cell_design_instruction(state),
        bulk_design_instruction(state),
    )))


_BATCH_CORRECTION_STEP = re.compile(
    r"\b(?:harmony\w*|bbknn\w*|combat\w*|scanorama\w*|batch[-\s]corr\w*|"
    r"scvi\w*[^.;]{0,80}\bbatch(?:_key)?\b)\b",
    re.I,
)
_SCRNA_CONDITION_INFERENCE_STEP = re.compile(
    r"(?:\b(?:sc[-\s]?rna|sn[-\s]?rna|single[-\s]?(?:cell|nucle\w+)|"
    r"pseudobulk|cell[-\s]?type|per[-\s]?cell|cluster)\b[^.;]{0,120}"
    r"\b(?:condition[-\s]?level|differential\s+expression|\bdegs?\b|"
    r"differential\s+abundance|significan\w*|p[-\s]?value|fdr)\b|"
    r"\b(?:condition[-\s]?level|differential\s+expression|\bdegs?\b)\b"
    r"[^.;]{0,120}\b(?:cell[-\s]?type|per[-\s]?cell|cluster|pseudobulk)\b)",
    re.I,
)
_BULK_CONDITION_INFERENCE_STEP = re.compile(
    r"\b(?:bulk(?:[-\s]?rna(?:[-\s]?seq)?)?|pydeseq2?|deseq2?|edger|limma)\b"
    r"[^.;]{0,160}\b(?:differential\s+expression|\bdegs?\b|significan\w*|"
    r"p[-\s]?value|fdr|condition[-\s]?(?:effect|comparison))\b|"
    r"\b(?:differential\s+expression|\bdegs?\b|condition[-\s]?(?:effect|comparison))\b"
    r"[^.;]{0,160}\b(?:bulk(?:[-\s]?rna(?:[-\s]?seq)?)?|pydeseq2?|deseq2?|edger|limma)\b",
    re.I,
)

_CLAUSE_SPLIT = re.compile(r"[.;\n]+")
_EXPLICIT_NONINFERENCE = re.compile(
    r"\b(?:descriptive|exploratory)\s+only\b|"
    r"\b(?:do\s+not|don't|must\s+not|should\s+not|never|without|no)\b"
    r"[^.;\n]{0,60}\b(?:inferen\w*|test\w*|p[-\s]?values?|fdr|"
    r"significan\w*|differential\s+expression|\bdegs?\b)\b",
    re.I,
)
_HARD_INFERENCE_TERM = re.compile(
    r"\b(?:p[-\s]?values?|fdr|q[-\s]?values?|significan\w*|wilcoxon|"
    r"t[-\s]?test|mann[-\s]?whitney|deseq2?|pydeseq2|edger|limma|"
    r"differential\s+expression|\bdegs?\b)\b",
    re.I,
)
_NEGATED_INFERENCE = re.compile(
    r"\b(?:do\s+not|don't|must\s+not|should\s+not|never|avoid|without|"
    r"exclude|skip|no)\b[^.;\n]{0,60}$",
    re.I,
)
_NEGATED_BATCH = re.compile(
    r"\b(?:do\s+not|don't|must\s+not|should\s+not|never|avoid|without|"
    r"exclude|skip|no)\b[^.;\n]{0,50}$",
    re.I,
)
_POSTPOSED_NEGATION = re.compile(
    r"^[^.;\n]{0,60}\b(?:is|are|was|were|will|would|should|must|can|"
    r"cannot|can't|may|might)\s+(?:not|never)\b|"
    r"^[^.;\n]{0,60}\b(?:not|never)\s+(?:be\s+)?(?:used|run|applied|"
    r"performed|included|reported|computed|generated)\b",
    re.I,
)
_BATCH_ASSESSMENT = re.compile(
    r"\b(?:assess|evaluate|inspect|check|determine|decide)\b[^.;\n]{0,80}"
    r"\b(?:whether|need\s+for|necessity\s+of)?\s*batch[-\s]corr\w*\b",
    re.I,
)


def _has_unnegated_batch_correction(text: str) -> bool:
    """Whether plan prose positively proposes a batch-correction method."""
    value = str(text or "")
    for match in _BATCH_CORRECTION_STEP.finditer(value):
        clause_start = max(
            value.rfind(".", 0, match.start()),
            value.rfind(";", 0, match.start()),
            value.rfind("\n", 0, match.start()),
        ) + 1
        clause_end_candidates = [
            pos for pos in (
                value.find(".", match.end()),
                value.find(";", match.end()),
                value.find("\n", match.end()),
            ) if pos >= 0
        ]
        clause_end = min(clause_end_candidates) if clause_end_candidates else len(value)
        clause = value[clause_start:clause_end]
        if _BATCH_ASSESSMENT.search(clause):
            continue
        prefix = value[max(0, match.start() - 60):match.start()]
        suffix = value[match.end():match.end() + 80]
        if (not _NEGATED_BATCH.search(prefix)
                and not _POSTPOSED_NEGATION.search(suffix)):
            return True
    return False


def _has_undisclaimed_inference(text: str, pattern: re.Pattern) -> bool:
    """Find an inferential plan clause that is not explicitly prohibited."""
    for clause in _CLAUSE_SPLIT.split(str(text or "")):
        if not pattern.search(clause):
            continue
        hard_terms = list(_HARD_INFERENCE_TERM.finditer(clause))
        if any(
            not _NEGATED_INFERENCE.search(
                clause[max(0, match.start() - 70):match.start()],
            )
            and not _POSTPOSED_NEGATION.search(
                clause[match.end():match.end() + 80],
            )
            for match in hard_terms
        ):
            return True
        if not hard_terms and not _EXPLICIT_NONINFERENCE.search(clause):
            return True
    return False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _code_calls_batch_correction(code: str) -> bool:
    """Inspect executable calls, ignoring comments, strings, and negated prose."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func).casefold()
        if any(token in name for token in (
            "harmony", "bbknn", "combat", "scanorama",
        )):
            return True
        if "scvi" in name and any(
            str(keyword.arg or "").casefold() in {"batch", "batch_key"}
            for keyword in node.keywords
        ):
            return True
    return False


def plan_sample_design_violations(plan: Any, state: Any) -> list[dict[str, Any]]:
    """Return plan steps that contradict modality-specific replicate designs."""
    if len(getattr(state, "treatment_groups", None) or []) < 2:
        return []
    scrna_replicated = has_replicated_condition_design(state, SCRNA_MODALITY)
    bulk_replicated = has_replicated_condition_design(
        state, BULK_RNASEQ_MODALITY,
    )
    has_scrna = state_has_modality(state, SCRNA_MODALITY)
    has_bulk = state_has_modality(state, BULK_RNASEQ_MODALITY)

    violations: list[dict[str, Any]] = []
    for step in (getattr(plan, "steps", None) or []):
        text = " ".join(str(getattr(step, field, "") or "") for field in (
            "title", "method", "input", "expected_output", "rationale",
        ))
        if has_scrna and _has_unnegated_batch_correction(text):
            violations.append({
                "step_id": getattr(step, "id", None),
                "suggestion": (
                    "Remove single-cell batch correction: replicate counts do "
                    "not verify a full-rank condition-by-batch design. Concatenate "
                    "with explicit labels and assess the uncorrected embedding."
                ),
            })
            continue
        if (has_scrna and not scrna_replicated
                and _has_undisclaimed_inference(
                    text, _SCRNA_CONDITION_INFERENCE_STEP,
                )):
            remedy = (
                "Use replicated bulk samples for condition-level inference."
                if bulk_replicated else
                "No other modality currently establishes condition-level inference."
            )
            violations.append({
                "step_id": getattr(step, "id", None),
                "suggestion": (
                    "Replace inferential single-cell condition testing with "
                    "descriptive effect sizes/ranks and cell-type localization. "
                    + remedy
                ),
            })
            continue
        if (has_bulk and not bulk_replicated
                and _has_undisclaimed_inference(
                    text, _BULK_CONDITION_INFERENCE_STEP,
                )):
            violations.append({
                "step_id": getattr(step, "id", None),
                "suggestion": (
                    "Remove inferential bulk condition testing until independent "
                    "bulk samples are assigned to treatment groups. Keep the bulk "
                    "comparison descriptive or repair the profiling records."
                ),
            })
    return violations


def code_sample_design_violation(code: str, state: Any) -> str:
    """Reject correction calls without a verified full-rank batch design."""
    if len(getattr(state, "treatment_groups", None) or []) < 2:
        return ""
    if not state_has_modality(state, SCRNA_MODALITY):
        return ""
    if not _code_calls_batch_correction(str(code or "")):
        return ""
    if has_replicated_condition_design(state, SCRNA_MODALITY):
        return (
            "Single-cell batch correction is blocked because replicate counts "
            "alone do not verify that condition and technical batch form a "
            "separable, full-rank design. Concatenate with explicit labels and "
            "use the uncorrected representation unless a future structured "
            "batch-design check establishes separability."
        )
    return (
        "Single-cell batch correction is blocked because each condition has "
        "fewer than two independent biological units. Sample/library effects "
        "cannot be separated from condition; concatenate with explicit labels "
        "and use the uncorrected representation for descriptive comparisons."
    )
