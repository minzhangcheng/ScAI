"""Structured completeness checks for analysis-result figures.

The analysis plan says what must be visualized; ``PublicationFigure`` records
what was actually rendered and reviewed.  This module is the small deterministic
bridge between them.  It deliberately does not make integration or narrative
work visual by fiat: a plan step can explicitly declare ``allow_text_only`` and
give a non-empty ``text_only_reason``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional

from backend.services.report.artifacts import (
    archival_render_available,
    is_report_eligible,
)


_TYPE_ALIASES = {
    "box": "violin",
    "boxplot": "violin",
    "box_plot": "violin",
    "bar": "barplot",
    "bar_chart": "barplot",
    "dot": "dotplot",
    "dot_plot": "dotplot",
    "embedding": "umap",
    "pseudotime": "trajectory",
}

_EXPLICIT_FIGURE_TOKENS = (
    ("volcano", ("volcano",)),
    ("heatmap", ("heatmap", "heat map")),
    ("umap", ("umap", "embedding plot")),
    ("violin", ("violin", "boxplot", "box plot")),
    ("dotplot", ("dotplot", "dot plot")),
    ("barplot", ("barplot", "bar plot", "bar chart")),
    ("trajectory", ("trajectory plot", "pseudotime plot")),
    ("network", ("network plot", "network diagram")),
    ("venn", ("venn",)),
)

_ANNOTATION_TOKENS = (
    "cell_type",
    "cell type",
    "cell-type",
    "annotation",
    "annotated",
)

_PLATFORM_BASELINE_TYPES = {
    "annotation": ("umap", _ANNOTATION_TOKENS),
    "batch": ("umap", ("batch", "sample")),
    "cluster": ("umap", ("cluster", "leiden", "louvain")),
    "markers": ("dotplot", ("marker", "markers", "rank_genes_groups")),
}

_TEXT_ONLY_TASK_TOKENS = (
    "narrative", "evidence synthesis", "synthesize existing evidence",
    "literature review", "evidence review", "interpretation summary",
    "discussion", "report writing", "methodology documentation",
    "provenance summary", "limitations summary", "workflow gate",
    "workflow transition", "integration checkpoint", "integrated findings",
    "cross-direction synthesis",
)
_TEXT_ONLY_ESCAPE_RE = re.compile(
    r"\b(?:save|saving)\s+time\b|\btime\s+(?:limit|constraint|budget)\b|"
    r"\b(?:skip|avoid|omit)\s+(?:the\s+)?(?:figure|plot|visual)|"
    r"\b(?:figure|plot|visual)\s+(?:is\s+)?(?:optional|unnecessary|not\s+needed)|"
    r"\bno\s+need\s+(?:for|to\s+(?:draw|make|generate))\b|"
    r"\btool\s+unavailable\b|\btoo\s+(?:hard|difficult|expensive)\b",
    re.I,
)

# Plans persisted before ``figure_requirements`` used prose to make a visual
# explicitly optional.  Keep that compatibility in the shared contract so the
# executor and post-execution audit cannot disagree about the same old step.
_LEGACY_FIGURE_OPTOUT_PATTERNS = (
    re.compile(r"\b(?:text|prose|narrative|table|tabular)[ -]?only\b", re.I),
    re.compile(r"\bonly\s+(?:a\s+)?(?:text|prose|narrative|table|tabular)\b", re.I),
    re.compile(
        r"\b(?:no|without)\s+(?:publication[- ](?:quality|ready)\s+)?"
        r"(?:figure|figures|plot|plots|visuali[sz]ation|visuali[sz]ations|chart|charts)\b",
        re.I,
    ),
    re.compile(
        r"\bdo\s+not\s+(?:generate|create|produce|render|include|make|save|draw)"
        r"\b.{0,60}\b(?:figure|figures|plot|plots|visuali[sz]ation|"
        r"visuali[sz]ations|chart|charts)\b",
        re.I,
    ),
    re.compile(r"\bdo\s+not\s+(?:plot|visuali[sz]e|chart)\b", re.I),
)
_LEGACY_OPTIONAL_FIGURE_PATTERNS = (
    re.compile(
        r"\boptional(?:ly)?\b[^.!?\n]{0,60}\b(?:figure|figures|plot|plots|"
        r"visuali[sz]ation|visuali[sz]ations|chart|charts)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:figure|figures|plot|plots|visuali[sz]ation|visuali[sz]ations|"
        r"chart|charts)\b[^.!?\n]{0,60}\boptional\b",
        re.I,
    ),
    re.compile(
        r"\b(?:figure|figures|plot|plots|visuali[sz]ation|visuali[sz]ations|"
        r"chart|charts)\b[^.!?\n]{0,40}\bif\s+(?:useful|helpful|needed|warranted|"
        r"informative|space permits|time permits)\b",
        re.I,
    ),
)
_LEGACY_OPTIONAL_QUALIFIER_RE = re.compile(
    r"\b(?:optional(?:ly)?|if\s+(?:useful|helpful|needed|warranted|informative|"
    r"space permits|time permits))\b",
    re.I,
)

# Domain-neutral result families for legacy or incomplete plans that predate a
# structured visual contract.  This is intentionally a capability vocabulary,
# not a tissue/disease list.  Stage 1 is excluded at the call site because its
# standard annotation panels are rendered by the platform baseline renderer.
_RESULT_BEARING_PATTERNS = (
    re.compile(r"\b(?:differential(?: gene)? expression|pydeseq2|deseq2?|edger|limma|"
               r"rank_genes_groups|marker (?:gene )?(?:analysis|testing|ranking))\b", re.I),
    re.compile(r"\b(?:gene set|pathway|functional) enrichment\b|\b(?:gsea|enrichr|"
               r"fgsea|ssgsea|over[- ]representation analysis)\b", re.I),
    re.compile(r"\b(?:score_genes|module score|signature score|program score|gene set score|"
               r"pathway activity|transcription factor activity|aucell|decoupler|"
               r"progeny|dorothea)\b", re.I),
    re.compile(r"\b(?:trajectory|pseudotime|diffusion pseudotime|rna velocity|"
               r"scvelo|monocle|slingshot|palantir)\b|\bsc\.tl\.(?:dpt|paga)\b", re.I),
    re.compile(r"\b(?:differential abundance|composition analysis|compositional analysis|"
               r"cell[- ]type proportion|cell[- ]type fraction|milo|sccoda|propeller)\b", re.I),
    re.compile(r"\b(?:cell[- ]cell communication|cell communication|ligand[- ]receptor|"
               r"regulatory network|co[- ]?expression network|interaction network|"
               r"network analysis|regulon|liana|cellchat|nichenet|connectome)\b", re.I),
    re.compile(r"\b(?:correlation analysis|association analysis|deconvolution|survival analysis|"
               r"feature importance)\b", re.I),
)

_RESULT_COMPUTE_ACTION_RE = re.compile(
    r"\b(?:run|perform|apply|compute|calculate|estimate|test|fit|model|infer|derive|"
    r"quantify|score|execute|conduct|analy[sz]e|identify|detect|rank)\b",
    re.I,
)
_GENERIC_QUANTITATIVE_OBJECT_RE = re.compile(
    r"\b(?:quantitative|statistic(?:al)?|metric|effect size|fold change|p[- ]?value|"
    r"coefficient|odds ratio|hazard ratio|correlation|association|estimate|score|"
    r"ranking|cluster|principal component|latent factor|trajectory|pseudotime|proportion|"
    r"fraction|abundance|probability|distribution|distance|similarity|differential|"
    r"enrichment|regression|model|contrast|measurement)\b",
    re.I,
)
_RESULT_REFERENCE_ACTION_RE = re.compile(
    r"\b(?:synthesi[sz]e|review|interpret|summari[sz]e|describe|discuss|"
    r"reconcile|document|explain)\b",
    re.I,
)

_FIGURE_PLAN_LINEAGE_KEY = "figure_plan_lineage_v1"
_MAX_RECORDED_PLAN_LINEAGES = 64
_MAX_INHERITED_PLAN_IDS = 32


def normalize_figure_type(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "other").strip().lower()).strip("_")
    return _TYPE_ALIASES.get(text, text or "other")


@dataclass(frozen=True)
class RequiredFigure:
    key: str
    figure_type: str
    description: str
    minimum_count: int = 1
    plan_step_id: Optional[int] = None
    plan_id: Optional[str] = None
    # A revised plan may carry an already-completed, visually unchanged step.
    # Its reviewed render remains owned by the plan that produced it, so retain
    # those ancestor IDs explicitly instead of rewriting artifact provenance.
    inherited_plan_ids: tuple[str, ...] = ()
    plan_created_at: Optional[datetime] = None
    semantic_tokens: tuple[str, ...] = ()
    require_plan_link: bool = False
    allow_platform_baseline: bool = False
    # Compatibility is deliberately narrow. A figure from before plan-level
    # provenance may satisfy an original legacy plan only when it is not older
    # than that plan. Revisions never enable this fallback.
    allow_legacy_plan_link: bool = False
    # Legacy result-family inference accepts any accurate result-figure type.
    wildcard_result_type: bool = False


@dataclass
class MissingFigure:
    requirement: RequiredFigure
    found_count: int = 0

    def as_dict(self) -> dict:
        req = self.requirement
        return {
            "key": req.key,
            "figure_type": req.figure_type,
            "description": req.description,
            "minimum_count": req.minimum_count,
            "found_count": self.found_count,
            "plan_step_id": req.plan_step_id,
            "plan_id": req.plan_id,
        }


@dataclass
class FigureContractAudit:
    requirements: List[RequiredFigure] = field(default_factory=list)
    missing: List[MissingFigure] = field(default_factory=list)
    eligible_figure_names: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missing

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "n_requirements": len(self.requirements),
            "n_missing": len(self.missing),
            "missing": [item.as_dict() for item in self.missing],
            "eligible_figure_names": list(self.eligible_figure_names),
        }


def _step_text(step: Any) -> str:
    return " ".join(
        str(getattr(step, name, "") or "")
        for name in ("title", "method", "expected_output", "rationale", "input")
    ).lower()


def _step_has_result_family(step: Any) -> bool:
    text = _step_text(step)
    if any(pattern.search(text) for pattern in _RESULT_BEARING_PATTERNS):
        return True
    # Custom methods remain discoverable without classifying every ordinary
    # non-technical step as a result: both a computation action and a
    # quantitative object must be present.
    instruction = " ".join(
        str(getattr(step, name, "") or "") for name in ("title", "method")
    )
    return bool(
        _RESULT_COMPUTE_ACTION_RE.search(instruction)
        and _GENERIC_QUANTITATIVE_OBJECT_RE.search(text)
    )


def _step_computes_result_family(step: Any) -> bool:
    """Distinguish a new quantitative result from a reference to an old one.

    Integration prose commonly names the result families it reconciles. That is
    not equivalent to rerunning them. A strong computation action wins when a
    step mixes both activities; otherwise recognizably explanatory wording may
    cite prior results without manufacturing a new visual obligation.
    """
    if not _step_has_result_family(step):
        return False
    instruction = " ".join(
        str(getattr(step, name, "") or "") for name in ("title", "method")
    )
    if _RESULT_COMPUTE_ACTION_RE.search(instruction):
        return True
    if _RESULT_REFERENCE_ACTION_RE.search(instruction):
        return False
    # A bare method or result-family name (for example ``pydeseq2`` or
    # ``correlation analysis``) is an analytical commitment, not a synthesis.
    return True


def valid_text_only_exception(step: Any, *, plan_phase: str = "") -> bool:
    """Accept only an explained, recognizably non-result visual exception.

    The reason remains visible for audit, but a free-form sentence is not itself
    authority to suppress a result figure.  Positive task markers are kept
    deliberately narrow; technical/setup steps need no exception because they
    should not declare a result figure in the first place.
    """
    if not bool(getattr(step, "allow_text_only", False)):
        return False
    if getattr(step, "figure_requirements", None):
        return False
    reason = str(getattr(step, "text_only_reason", "") or "").strip()
    if len(reason) < 20 or _TEXT_ONLY_ESCAPE_RE.search(reason):
        return False
    # A narrative word must not waive a quantitative result in the same step.
    # For example, "run DE and synthesize integrated findings" still produces a
    # new DE result and therefore needs a visual deliverable.
    if _step_computes_result_family(step):
        return False
    task_text = _step_text(step)
    # Integration is a structured workflow phase, not a wording convention.
    # Pure synthesis/gate work in that phase may be text-only even when its title
    # does not happen to contain one of the legacy narrative phrases above.
    if str(plan_phase or "").strip().lower() == "integration":
        return True
    return any(token in task_text for token in _TEXT_ONLY_TASK_TOKENS)


def _has_explicit_visual_policy_fields(step: Any) -> bool:
    """Whether this step was written after the structured visual contract.

    Pydantic v2 retains the fields present in the input in ``model_fields_set``.
    Old snapshots omit all three visual-policy fields; current planners pass
    them explicitly even when their values are empty/default. This is the only
    reliable way to preserve old prose opt-outs without letting newly generated
    plans manufacture the same escape hatch.
    """
    fields_set = getattr(step, "model_fields_set", None)
    if fields_set is None:
        if isinstance(step, Mapping):
            fields_set = set(step)
        else:
            return False
    return bool({
        "figure_requirements", "allow_text_only", "text_only_reason",
    } & set(fields_set))


def legacy_figure_opt_out(step: Any) -> bool:
    """Recognize figure-only opt-outs in plans from before structured fields.

    This never overrides ``figure_requirements``.  In particular, ``optionally
    plot`` waives only the visual channel; it does not make the analytical
    method itself optional.
    """
    if getattr(step, "figure_requirements", None):
        return False
    if _has_explicit_visual_policy_fields(step):
        return False
    text = _step_text(step)
    if any(pattern.search(text) for pattern in _LEGACY_FIGURE_OPTOUT_PATTERNS):
        return True
    if any(pattern.search(text) for pattern in _LEGACY_OPTIONAL_FIGURE_PATTERNS):
        return True
    # The fallback is clause-scoped. A qualifier in the following sentence
    # ("Plot the UMAP. If useful, run ...") cannot waive the earlier promise.
    for clause in re.split(r"[.!?\n]", text):
        has_named_visual = any(
            token in clause
            for _figure_type, tokens in _EXPLICIT_FIGURE_TOKENS
            for token in tokens
        )
        if has_named_visual and _LEGACY_OPTIONAL_QUALIFIER_RE.search(clause):
            return True
    return False


def _visual_contract_signature(step: Any) -> str:
    """Stable digest of the part of a step that governs its visual output."""
    requirements = []
    for raw in getattr(step, "figure_requirements", None) or []:
        get = (
            raw.get
            if isinstance(raw, Mapping)
            else lambda key, default=None: getattr(raw, key, default)
        )
        requirements.append({
            "figure_type": normalize_figure_type(get("figure_type", "other")),
            "description": str(get("description", "") or "").strip(),
            "minimum_count": int(get("minimum_count", 1) or 1),
            "platform_baseline": str(get("platform_baseline", "") or "").strip().lower(),
        })
    payload = {
        "title": str(getattr(step, "title", "") or "").strip(),
        "method": str(getattr(step, "method", "") or "").strip(),
        "input": str(getattr(step, "input", "") or "").strip(),
        "expected_output": str(getattr(step, "expected_output", "") or "").strip(),
        "rationale": str(getattr(step, "rationale", "") or "").strip(),
        "figure_requirements": requirements,
        "allow_text_only": bool(getattr(step, "allow_text_only", False)),
        "text_only_reason": str(getattr(step, "text_only_reason", "") or "").strip(),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_plan_figure_inheritance(state: Any, plan: Any) -> None:
    """Persist visual-contract ancestry when a revised plan is approved.

    Completed steps are immutable in the incremental planner. When such a step's
    visual signature is unchanged, its parent-plan render remains valid evidence.
    The lineage is stored separately from ``PublicationFigure.plan_id`` so the
    artifact continues to name the plan that actually produced and reviewed it.
    """
    plan_id = str(getattr(plan, "id", "") or "").strip()
    if not plan_id:
        return
    sandbox_state = dict(getattr(state, "sandbox_state", None) or {})
    lineage = dict(sandbox_state.get(_FIGURE_PLAN_LINEAGE_KEY) or {})
    parent_id = str(getattr(plan, "revision_of", "") or "").strip()
    parent_entry = lineage.get(parent_id) if parent_id else None
    parent_steps = (
        dict(parent_entry.get("steps") or {})
        if isinstance(parent_entry, Mapping) else {}
    )
    step_entries: dict[str, dict[str, Any]] = {}
    for step in getattr(plan, "steps", None) or []:
        step_id = str(int(getattr(step, "id", 0) or 0))
        signature = _visual_contract_signature(step)
        inherited: List[str] = []
        if parent_id and str(getattr(step, "status", "") or "") == "completed":
            parent_step = parent_steps.get(step_id)
            # When the parent itself has a lineage record, absence or mismatch
            # is evidence that this is not the same carried step. The permissive
            # completed-step hand-off is only for a parent approved before this
            # metadata existed at all.
            unchanged = (
                isinstance(parent_step, Mapping)
                and str(parent_step.get("signature") or "") == signature
            )
            legacy_parent = not isinstance(parent_entry, Mapping)
            if unchanged or legacy_parent:
                inherited.append(parent_id)
                if isinstance(parent_step, Mapping):
                    inherited.extend(
                        str(value) for value in (parent_step.get("inherited_plan_ids") or [])
                    )
        step_entries[step_id] = {
            "signature": signature,
            "inherited_plan_ids": list(dict.fromkeys(
                value for value in inherited if value
            ))[:_MAX_INHERITED_PLAN_IDS],
        }
    lineage[plan_id] = {"revision_of": parent_id or None, "steps": step_entries}
    if len(lineage) > _MAX_RECORDED_PLAN_LINEAGES:
        for stale_id in list(lineage)[:-_MAX_RECORDED_PLAN_LINEAGES]:
            lineage.pop(stale_id, None)
    sandbox_state[_FIGURE_PLAN_LINEAGE_KEY] = lineage
    state.sandbox_state = sandbox_state


def visual_plan_inheritance_for_plan(state: Any, plan: Any) -> dict[int, tuple[str, ...]]:
    """Return ancestor plan IDs accepted by each unchanged completed step."""
    plan_id = str(getattr(plan, "id", "") or "").strip()
    parent_id = str(getattr(plan, "revision_of", "") or "").strip()
    sandbox_state = getattr(state, "sandbox_state", None) or {}
    lineage = sandbox_state.get(_FIGURE_PLAN_LINEAGE_KEY) or {}
    entry = lineage.get(plan_id) if isinstance(lineage, Mapping) else None
    stored_steps = dict(entry.get("steps") or {}) if isinstance(entry, Mapping) else {}
    result: dict[int, tuple[str, ...]] = {}
    for step in getattr(plan, "steps", None) or []:
        if str(getattr(step, "status", "") or "") != "completed":
            continue
        step_id = int(getattr(step, "id", 0) or 0)
        stored = stored_steps.get(str(step_id))
        inherited = []
        if isinstance(entry, Mapping):
            if isinstance(stored, Mapping) and (
                str(stored.get("signature") or "")
                == _visual_contract_signature(step)
            ):
                inherited.extend(
                    str(value)
                    for value in (stored.get("inherited_plan_ids") or [])
                )
        elif parent_id:
            # Compatibility for the first revision of a session approved before
            # lineage recording existed. The workflow copies completed steps
            # unchanged, so the immediate parent remains a sound hand-off.
            inherited.append(parent_id)
        if inherited:
            result[step_id] = tuple(dict.fromkeys(
                value for value in inherited if value
            ))
    return result


def _plan_provenance(plan: Any) -> tuple[Optional[str], Optional[datetime], bool]:
    plan_id = str(getattr(plan, "id", "") or "").strip() or None
    created_at = getattr(plan, "created_at", None)
    if not isinstance(created_at, datetime):
        created_at = None
    # A revised plan is a new visual contract. It must never inherit an
    # unattributed render from its parent merely because step IDs were reused.
    allow_legacy = not bool(str(getattr(plan, "revision_of", "") or "").strip())
    return plan_id, created_at, allow_legacy


def _structured_requirements(
    step: Any,
    *,
    stage: str = "",
    plan_id: Optional[str] = None,
    plan_created_at: Optional[datetime] = None,
    allow_legacy_plan_link: bool = False,
    inherited_plan_ids: tuple[str, ...] = (),
) -> List[RequiredFigure]:
    result: List[RequiredFigure] = []
    step_id = int(getattr(step, "id", 0) or 0) or None
    for index, raw in enumerate(getattr(step, "figure_requirements", None) or []):
        get = raw.get if isinstance(raw, dict) else lambda key, default=None: getattr(raw, key, default)
        figure_type = normalize_figure_type(get("figure_type", "other"))
        description = str(get("description", "") or "").strip()
        try:
            minimum_count = max(1, min(12, int(get("minimum_count", 1) or 1)))
        except (TypeError, ValueError):
            minimum_count = 1
        baseline_key = str(get("platform_baseline", "") or "").strip().lower()
        baseline_spec = _PLATFORM_BASELINE_TYPES.get(baseline_key)
        baseline_tokens: tuple[str, ...] = ()
        if stage == "stage1" and baseline_spec and baseline_spec[0] == figure_type:
            baseline_tokens = tuple(baseline_spec[1])
        result.append(RequiredFigure(
            key=f"plan.{step_id or 0}.{index + 1}",
            figure_type=figure_type,
            description=description or f"Required {figure_type} result figure",
            minimum_count=minimum_count,
            plan_step_id=step_id,
            plan_id=plan_id,
            inherited_plan_ids=inherited_plan_ids,
            plan_created_at=plan_created_at,
            semantic_tokens=baseline_tokens,
            require_plan_link=True,
            allow_platform_baseline=bool(baseline_tokens),
            # Platform baselines existed before plan provenance. Keep that
            # narrow original-plan compatibility; agent-owned structured
            # deliverables remain strictly linked.
            allow_legacy_plan_link=(
                bool(baseline_tokens) and allow_legacy_plan_link
            ),
        ))
    return result


def _inferred_requirements(
    step: Any,
    *,
    infer_result_family: bool = False,
    plan_id: Optional[str] = None,
    plan_created_at: Optional[datetime] = None,
    allow_legacy_plan_link: bool = False,
    inherited_plan_ids: tuple[str, ...] = (),
) -> List[RequiredFigure]:
    """Recover explicit plot promises from plans written before the contract.

    A named visual form keeps its exact type. For a later-stage recognized
    quantitative/result-producing step, omission of a structured policy falls
    back to one type-agnostic result figure. A narrow action/object rule covers
    custom methods without treating every non-technical step as a result. Stage
    1 baseline rendering remains outside this fallback.
    """
    text = _step_text(step)
    step_id = int(getattr(step, "id", 0) or 0) or None
    found: List[RequiredFigure] = []
    for figure_type, tokens in _EXPLICIT_FIGURE_TOKENS:
        if any(token in text for token in tokens):
            found.append(RequiredFigure(
                key=f"legacy.{step_id or 0}.{figure_type}",
                figure_type=figure_type,
                description=f"Figure explicitly promised by plan step {step_id or '?'}",
                plan_step_id=step_id,
                plan_id=plan_id,
                inherited_plan_ids=inherited_plan_ids,
                plan_created_at=plan_created_at,
                allow_legacy_plan_link=allow_legacy_plan_link,
            ))
    if not found and infer_result_family and _step_has_result_family(step):
        found.append(RequiredFigure(
            key=f"result.{step_id or 0}",
            figure_type="result",
            description=(
                f"Visual summary of the analytical result from plan step "
                f"{step_id or '?'}"
            ),
            plan_step_id=step_id,
            plan_id=plan_id,
            inherited_plan_ids=inherited_plan_ids,
            plan_created_at=plan_created_at,
            require_plan_link=True,
            allow_legacy_plan_link=allow_legacy_plan_link,
            wildcard_result_type=True,
        ))
    return found


def collect_plan_figure_requirements(
    plan: Any,
    *,
    inherited_plan_ids_by_step: Optional[Mapping[int, Iterable[str]]] = None,
) -> List[RequiredFigure]:
    requirements: List[RequiredFigure] = []
    stage = str(getattr(plan, "stage", "") or "")
    phase = str(getattr(plan, "phase", "") or "")
    plan_id, plan_created_at, allow_legacy_plan_link = _plan_provenance(plan)
    inherited_plan_ids_by_step = inherited_plan_ids_by_step or {}
    infer_result_family = stage != "stage1" and (
        stage in ("stage2", "full") or phase == "innovation"
    )
    for step in getattr(plan, "steps", None) or []:
        step_id = int(getattr(step, "id", 0) or 0)
        inherited_plan_ids = tuple(dict.fromkeys(
            str(value) for value in inherited_plan_ids_by_step.get(step_id, ())
            if str(value or "").strip()
        ))
        structured = _structured_requirements(
            step,
            stage=stage,
            plan_id=plan_id,
            plan_created_at=plan_created_at,
            allow_legacy_plan_link=allow_legacy_plan_link,
            inherited_plan_ids=inherited_plan_ids,
        )
        # A text-only exception is auditable only when it explains itself and
        # does not contradict an explicit deliverable. Structured requirements
        # always win over the opt-out flag.
        if (
            not structured
            and valid_text_only_exception(step, plan_phase=phase)
        ):
            continue
        if not structured and legacy_figure_opt_out(step):
            continue
        requirements.extend(structured or _inferred_requirements(
            step,
            infer_result_family=infer_result_family,
            plan_id=plan_id,
            plan_created_at=plan_created_at,
            allow_legacy_plan_link=allow_legacy_plan_link,
            inherited_plan_ids=inherited_plan_ids,
        ))
    return requirements


def stage_baseline_requirements(state: Any, planning_stage: str) -> List[RequiredFigure]:
    """Return platform-level visual invariants for an analysis stage.

    Cell annotation is the only hard Stage-1 baseline.  The deterministic
    renderer also creates cluster, batch and marker views when data permit, but
    missing metadata should not make an otherwise valid run impossible to end.
    """
    if str(planning_stage or "") not in ("stage1", "stage2", "full"):
        return []
    if int(getattr(state, "current_round", 0) or 0) > 1:
        return []
    has_scrna = getattr(state, "has_scrnaseq", None)
    if callable(has_scrna) and not has_scrna():
        return []
    return [RequiredFigure(
        key="stage1.annotation_umap",
        figure_type="umap",
        description="Cell-type annotation UMAP",
        semantic_tokens=_ANNOTATION_TOKENS,
        allow_platform_baseline=True,
    )]


def _current_eligible_figures(
    figures: Iterable[Any], round_number: int, *, lineage_id: str = "main"
) -> List[Any]:
    """Return report-eligible figures attributable to this exact round.

    Restored pre-provenance figures deserialize with ``round_number == 0``.
    Treating zero as a wildcard lets an old heatmap satisfy a newly approved
    plan merely because its type matches.  That is especially dangerous here:
    this audit controls stage advancement.  A legacy figure can still appear in
    archival reporting, but it cannot prove that the current plan rendered its
    required deliverable.
    """
    result: List[Any] = []
    for figure in figures or []:
        if (getattr(figure, "lineage_id", None) or "main") != (lineage_id or "main"):
            continue
        figure_round = int(getattr(figure, "round_number", 0) or 0)
        if figure_round != int(round_number or 0):
            continue
        if is_report_eligible(figure) and archival_render_available(figure):
            result.append(figure)
    return result


def _figure_search_text(figure: Any) -> str:
    return " ".join(
        str(getattr(figure, name, "") or "")
        for name in ("name", "description", "interpretation", "purpose")
    ).lower()


def _legacy_figure_not_older_than_plan(
    requirement: RequiredFigure, figure: Any
) -> bool:
    """Bound the no-plan-id compatibility path to the plan's lifetime."""
    if not requirement.allow_legacy_plan_link:
        return False
    plan_created = requirement.plan_created_at
    figure_created = getattr(figure, "created_at", None)
    if not isinstance(plan_created, datetime) or not isinstance(figure_created, datetime):
        return True
    try:
        return figure_created >= plan_created
    except TypeError:
        # Old snapshots mix naive and timezone-aware ISO timestamps. Compare
        # their wall-clock components rather than failing the audit outright.
        return figure_created.replace(tzinfo=None) >= plan_created.replace(tzinfo=None)


def _plan_matches(requirement: RequiredFigure, figure: Any) -> bool:
    if not requirement.plan_id:
        return True
    figure_plan_id = str(getattr(figure, "plan_id", "") or "").strip()
    if figure_plan_id:
        return figure_plan_id in {
            requirement.plan_id, *requirement.inherited_plan_ids,
        }
    return _legacy_figure_not_older_than_plan(requirement, figure)


def _matches(requirement: RequiredFigure, figure: Any) -> bool:
    purpose = str(getattr(figure, "purpose", "result") or "result").strip().lower()
    is_supporting = bool(getattr(figure, "diagnostic", False)) or purpose in {
        "baseline", "diagnostic",
    }
    if is_supporting and not requirement.allow_platform_baseline:
        return False

    actual_type = normalize_figure_type(getattr(figure, "figure_type", "other"))
    if requirement.wildcard_result_type:
        if actual_type in {"qc", "diagnostic", "baseline"}:
            return False
    elif actual_type != requirement.figure_type:
        return False

    if not _plan_matches(requirement, figure):
        return False

    figure_step = getattr(figure, "plan_step_id", None)
    if requirement.require_plan_link:
        if bool(
            requirement.plan_step_id is not None
            and figure_step is not None
            and int(figure_step) == int(requirement.plan_step_id)
        ):
            return True
        figure_plan_id = str(getattr(figure, "plan_id", "") or "").strip()
        if (
            figure_step is None
            and figure_plan_id
            and figure_plan_id == requirement.plan_id
        ):
            # Transitional records already carried a stable plan identity but
            # predated step-level provenance. Type + exact plan identity are
            # sufficient; inherited plans stay excluded because their missing
            # step cannot be attributed safely to the revised requirement.
            return True
        if (
            requirement.wildcard_result_type
            and figure_step is None
            and not str(getattr(figure, "plan_id", "") or "").strip()
            and requirement.allow_legacy_plan_link
        ):
            # A pre-provenance result may close an original legacy plan. The
            # plan-age check above prevents it from being inherited by a later
            # plan, and revised plans disable this path entirely.
            return True
        if not requirement.allow_platform_baseline:
            return False
        if purpose != "baseline" and not bool(getattr(figure, "diagnostic", False)):
            return False
        text = _figure_search_text(figure)
        return any(token in text for token in requirement.semantic_tokens)
    if requirement.plan_step_id is not None and figure_step is not None:
        return int(figure_step) == int(requirement.plan_step_id)
    if requirement.semantic_tokens:
        text = _figure_search_text(figure)
        return any(token in text for token in requirement.semantic_tokens)
    # Compatibility for old figures and plans: type is the only structured
    # linkage they carry.  New producers set plan_step_id, making this fallback
    # progressively less important without breaking restored sessions.
    return True


def audit_figure_contract(
    state: Any,
    *,
    plan: Any = None,
    planning_stage: str = "full",
    include_stage_baseline: bool = True,
    defer_platform_baseline: bool = False,
    inherited_plan_ids_by_step: Optional[Mapping[int, Iterable[str]]] = None,
) -> FigureContractAudit:
    active_plan = plan or getattr(state, "current_plan", None)
    inherited_plan_ids = (
        visual_plan_inheritance_for_plan(state, active_plan)
        if inherited_plan_ids_by_step is None
        else inherited_plan_ids_by_step
    )
    requirements = collect_plan_figure_requirements(
        active_plan,
        inherited_plan_ids_by_step=inherited_plan_ids,
    )
    if defer_platform_baseline:
        # The deterministic Stage 1 renderer runs immediately after the
        # autonomous executor returns.  Its standard panels cannot exist at the
        # executor's report_findings gate, so defer only those platform-owned
        # promises there. Exceptional/user-requested Stage 1 views remain hard
        # requirements. The workflow-level audit never sets this flag.
        requirements = [
            requirement for requirement in requirements
            if not requirement.allow_platform_baseline
        ]
    if include_stage_baseline:
        # Put plan requirements first.  A platform baseline may also satisfy a
        # plan promise (for example the cell-type UMAP); the invariant should
        # verify that shared deliverable, not demand a duplicate rendering.
        requirements = requirements + stage_baseline_requirements(state, planning_stage)

    round_number = int(getattr(state, "current_round", 0) or 0)
    figures = _current_eligible_figures(
        getattr(state, "publication_figures", None) or [],
        round_number,
        lineage_id=(getattr(state, "restore_branch_id", None) or "main"),
    )
    audit = FigureContractAudit(
        requirements=requirements,
        eligible_figure_names=[str(getattr(fig, "name", "") or "") for fig in figures],
    )
    consumed: set[int] = set()
    for requirement in requirements:
        reusable_baseline = requirement.key.startswith("stage1.")
        matches = [
            index for index, figure in enumerate(figures)
            if (reusable_baseline or index not in consumed)
            and _matches(requirement, figure)
        ]
        selected = matches[: requirement.minimum_count]
        if not reusable_baseline:
            consumed.update(selected)
        if len(selected) < requirement.minimum_count:
            audit.missing.append(MissingFigure(
                requirement=requirement,
                found_count=len(selected),
            ))
    return audit


def format_missing_figure_feedback(audit: FigureContractAudit) -> str:
    items = []
    for missing in audit.missing:
        req = missing.requirement
        step = f" for plan step {req.plan_step_id}" if req.plan_step_id else ""
        items.append(
            f"- {req.figure_type}{step}: {req.description} "
            f"({missing.found_count}/{req.minimum_count} eligible)"
        )
    return "\n".join(items)


__all__ = [
    "FigureContractAudit",
    "MissingFigure",
    "RequiredFigure",
    "audit_figure_contract",
    "collect_plan_figure_requirements",
    "format_missing_figure_feedback",
    "legacy_figure_opt_out",
    "normalize_figure_type",
    "record_plan_figure_inheritance",
    "stage_baseline_requirements",
    "valid_text_only_exception",
    "visual_plan_inheritance_for_plan",
]
