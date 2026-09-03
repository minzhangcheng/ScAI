# Annotations, selection records, plans, snapshots, and research directions.
# Moved verbatim out of backend/models/schemas.py.
"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


from .enums import AnalysisPhaseType, PlanStage, PlanStatus, SnapshotPhase


class ClusterAnnotation(BaseModel):
    """Single cluster annotation for user review during exploration phase"""
    cluster_id: str
    cell_type: str
    subtypes: List[str] = Field(default_factory=list)
    key_markers: List[str] = Field(default_factory=list)
    confidence: str = "low"  # "high" / "medium" / "low"
    evidence: str = ""
    n_cells: int = 0  # Number of cells in this cluster; see n_cells_known.
    n_cells_known: bool = False
    n_cells_source: str = "unknown"
    group_proportions: Dict[str, float] = Field(default_factory=dict)
    alternative_cell_types: List[str] = Field(default_factory=list)  # Other possible cell types
    low_confidence_reason: str = ""  # Why confidence is low/medium
    user_confirmed: bool = False
    user_override: Optional[str] = None  # User-provided cell type override
    # Validation round: signal that current clustering may be insufficient for this cluster
    requires_reclustering: bool = False
    reclustering_reason: str = ""
    suggested_reclustering_scope: Optional[str] = None  # targeted_subcluster | higher_resolution_review | rerun_stage1_clustering

    @model_validator(mode="after")
    def _preserve_known_nonzero_counts(self):
        # Old snapshots predate the explicit known/source fields.  A positive
        # persisted value is unambiguous; zero is not (it was also the old default),
        # so zero remains unknown unless a current producer says otherwise.
        if self.n_cells > 0 and not self.n_cells_known:
            self.n_cells_known = True
            if self.n_cells_source == "unknown":
                self.n_cells_source = "legacy_record"
        return self

class AnnotationConfirmationState(BaseModel):
    """State for cell annotation confirmation pause during exploration phase"""
    cluster_annotations: List[ClusterAnnotation] = Field(default_factory=list)
    tissue: str = ""
    organism: str = ""
    tissue_summary: str = ""
    analysis_summary: str = ""
    pending: bool = False
    confirmed: bool = False
    proportion_group_labels: List[str] = Field(default_factory=list)
    proportion_dimension: str = "group"
    # Annotation validation round (all optional for backward/snapshot compatibility)
    validation_attempt_id: Optional[str] = None
    validation_status: Optional[str] = None  # not_run | completed | failed | partial | skipped
    validation_plan: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    validation_summary: Optional[str] = None
    validation_text_summary: Optional[str] = None
    reclustering_recommendations: Optional[List[Dict[str, Any]]] = None

class SelectionEntry(BaseModel):
    """One direction the user selected at a direction gate."""
    direction: str = ""
    notes: str = ""
    is_custom: bool = False
    candidate_index: Optional[int] = None

class DirectionSelectionRecord(BaseModel):
    """Immutable snapshot of one direction-selection gate event."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    round_number: int = 0
    integration_cycle: Optional[int] = None
    phase: str = "exploration"
    event_type: str = "open"
    timestamp: datetime = Field(default_factory=datetime.now)
    let_pi_decide: bool = False
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    literature_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    selections: List[SelectionEntry] = Field(default_factory=list)
    pi_selected: List[Dict[str, str]] = Field(default_factory=list)
    # What this gate passed over and why. Without it a later gate re-offers a
    # declined direction with no idea it was ever weighed, and the report cannot
    # tell a direction nobody considered from one that was considered and
    # turned down.
    declined: List[Dict[str, Any]] = Field(default_factory=list)
    supplementary_requests: List[str] = Field(default_factory=list)
    # An empty pi_selected has two very different causes. Without these, a tooling
    # failure and a deliberate "nothing here is worth a round" are the same record.
    decision_outcome: Optional[
        Literal["selected", "none_selected", "tooling_failure"]
    ] = None
    selection_rationale: str = ""

class AnnotationClusterEntry(BaseModel):
    """One cluster's final annotation state at confirm time."""
    cluster_id: str = ""
    original_cell_type: str = ""
    final_cell_type: str = ""
    user_override: bool = False
    confidence: str = "low"
    key_markers: List[str] = Field(default_factory=list)
    group_proportions: Dict[str, float] = Field(default_factory=dict)
    alternative_cell_types: List[str] = Field(default_factory=list)
    low_confidence_reason: str = ""
    n_cells: int = 0
    n_cells_known: bool = False
    n_cells_source: str = "unknown"

    @model_validator(mode="after")
    def _preserve_known_nonzero_counts(self):
        if self.n_cells > 0 and not self.n_cells_known:
            self.n_cells_known = True
            if self.n_cells_source == "unknown":
                self.n_cells_source = "legacy_record"
        return self

class AnnotationRecord(BaseModel):
    """Immutable snapshot of the annotation gate at confirm/auto-confirm time."""
    timestamp: datetime = Field(default_factory=datetime.now)
    organism: str = ""
    tissue: str = ""
    tissue_summary: str = ""
    proportion_dimension: str = "group"
    proportion_group_labels: List[str] = Field(default_factory=list)
    clusters: List[AnnotationClusterEntry] = Field(default_factory=list)
    supplementary_requests: List[str] = Field(default_factory=list)
    auto_confirmed: bool = False
    validation_status: Optional[str] = None
    validation_summary: Optional[str] = None
    validation_text_summary: Optional[str] = None

class AgentRoundCache(BaseModel):
    """Per-round agent-result cache for snapshot-restore idempotency.

    Lives in AnalysisState (so snapshot/restore captures it), keyed by
    AgentType value, holding each node's canonical return dict. Cleared at
    the start of a genuinely-new round; preserved on snapshot restore so a
    replayed workflow can short-circuit nodes whose results are already known.
    """
    round_number: int
    results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class DetailedReportSegment(BaseModel):
    """One segment of the incrementally-built detailed report at each HITL checkpoint"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: str  # "pre_annotation" | "exploration_complete" | "supplementary_annotation_N" | "supplementary_direction_N"
    title: str  # Human-readable, e.g. "Pre-Annotation Analysis Summary"
    content_html: str = ""  # Pre-rendered HTML
    timestamp: datetime = Field(default_factory=datetime.now)
    round_number: int = 0

class Snapshot(BaseModel):
    """A saved snapshot of analysis state for debugging rollback.

    Derived compatibility cache for both V1 files and committed Snapshot V2
    manifests. V2 listing projection may populate the optional V2 fields below.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    phase: SnapshotPhase  # Auto-created at key analysis points
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    state_file: str = ""  # Filename within snapshots/ directory
    size_bytes: int = 0
    parent_id: Optional[str] = None  # Parent snapshot ID for tree view
    notebook_file: str = ""  # Filename of ipynb backup within snapshots/ directory
    workspace_manifest: Dict[str, List[str]] = Field(default_factory=dict)
    # e.g. {"figures": ["fig_001.jpg"], "tables": ["markers.csv"], "reproducibility": []}
    branch_label: Optional[str] = None  # "main", "branch-A", "branch-B", etc.
    runtime_state_included: bool = False
    skipped_files: List[Dict[str, str]] = Field(default_factory=list)
    branch_id: Optional[str] = None
    # Snapshot V2 projection fields (optional; absent for pure V1 records)
    format_version: Optional[int] = None
    runtime_status: Optional[Literal["complete", "missing", "state_only", "corrupt"]] = None
    logical_size_bytes: Optional[int] = None
    physical_added_bytes: Optional[int] = None
    source_restore_transaction_id: Optional[str] = None
    continuation_kind: Optional[str] = None

class FigureRequirement(BaseModel):
    """A visual deliverable that must exist before a plan step is complete.

    ``figure_type`` uses the same open vocabulary as ``PublicationFigure``.  It
    intentionally remains a string: analysis methods evolve faster than the API,
    and forcing every new plot family through a schema migration would make the
    contract less useful rather than safer.
    """

    figure_type: str = "other"
    description: str = ""
    minimum_count: int = Field(default=1, ge=1, le=12)
    # Explicit hand-off to one deterministic Stage 1 renderer. Empty means the
    # autonomous executor owns the deliverable. The closed keys prevent a loose
    # word such as "cluster" from misclassifying subclustering/comparison plots.
    platform_baseline: Literal["", "annotation", "batch", "cluster", "markers"] = ""

    @field_validator("platform_baseline", mode="before")
    @classmethod
    def _canonicalize_platform_baseline(cls, value):
        """Accept only the documented renderer aliases before Literal validation."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold()
        return {
            "annotation_overview": "annotation",
            "batch_overview": "batch",
            "cluster_overview": "cluster",
            "marker_overview": "markers",
        }.get(normalized, normalized)


class PlanStep(BaseModel):
    """A single step in an analysis plan."""
    id: int
    title: str
    method: str
    input: str = ""
    expected_output: str = ""
    rationale: str = ""
    figure_requirements: List[FigureRequirement] = Field(default_factory=list)
    # Reserved for genuinely narrative/special tasks.  Ordinary analytical
    # result steps should declare visual deliverables instead of using this as
    # an escape hatch.
    allow_text_only: bool = False
    # Required whenever ``allow_text_only`` is true.  The deterministic figure
    # contract ignores an unexplained exception and never lets it override an
    # explicit ``figure_requirements`` entry.
    text_only_reason: str = ""
    depends_on: List[int] = []
    status: str = "pending"  # pending | in_progress | completed | skipped
    executed_at: Optional[datetime] = None

class AnalysisPlan(BaseModel):
    """Method-level analysis plan generated by bioinformatics agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: AnalysisPhaseType  # "exploration" or "innovation"
    stage: PlanStage = "full"  # "full" or "stage1" (QC+clustering) or "stage2" (downstream)
    overview: str = ""
    steps: List[PlanStep] = Field(default_factory=list)
    key_decision_points: List[str] = Field(default_factory=list)
    expected_runtime: str = ""
    status: PlanStatus = "draft"
    revision_of: Optional[str] = None  # parent plan ID if this is a revision
    revision_reason: Optional[str] = None  # why this plan was revised
    round_number: int = 0
    direction: str = ""
    direction_key: str = ""
    branch_id: Optional[str] = None
    created_from_snapshot_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None

SpeciesTissueQualification = Literal[
    "",
    "matched_species_tissue_available",
    "no_matched_species_tissue_evidence",
    "cross_tissue_only",
    "cross_species_only",
]

class DirectionSearchQuery(BaseModel):
    category: Literal[
        "direct_prior_work",
        "analogous_mechanism",
        "contradictory_expectedness",
        "methodological_precedent",
        "gene_model_relaxed",
        "cell_type_focused",
        "pathway_focused",
        "cross_species_or_tissue",
        "challenge_contradiction",
        "prior_art",
        "tier3_support",
        "tier3_challenge",
        # Session-cached articles, triaged once per candidate rather than once
        # per query. Always degraded, so it never counts toward axis coverage.
        "session_cache",
    ]
    query: str
    result_count: int = 0
    relevant_count: int = 0
    degraded: bool = False
    notes: str = ""

class PubMedArticle(BaseModel):
    """A single PubMed article reference for direction literature evidence."""
    pubmed_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    journal: str
    publication_date: str
    abstract: str = ""
    url: str = ""

class PubMedEvidenceItem(BaseModel):
    pmid: str
    title: str
    year: Optional[str] = None
    url: Optional[str] = None
    relevance: Literal["high", "medium", "low"]
    supports: Literal["direct", "analogous", "contradictory", "methodological"]
    source: Literal["new_search", "session_cache"] = "new_search"
    species_context: str = ""
    tissue_context: str = ""
    why_it_matters: str

class DirectionLiteratureEvidence(BaseModel):
    direction_id: str
    direction: str
    status: Literal["completed", "partial", "failed", "skipped"] = "partial"
    search_tier: Literal["standard", "deep", "failed"] = "standard"
    literature_novelty: Literal["high", "medium", "low", "unknown"] = "unknown"
    confidence: Literal["high", "medium", "low", "failed"] = "low"
    negative_validation_confidence: Literal["high", "medium", "low", "none"] = "none"
    negative_validation_axes: Dict[str, bool] = Field(default_factory=dict)
    direct_evidence_count: int = 0
    analogous_evidence_count: int = 0
    contradictory_evidence_count: int = 0
    methodological_precedent_count: int = 0
    direct_evidence: List[PubMedEvidenceItem] = Field(default_factory=list)
    analogous_evidence: List[PubMedEvidenceItem] = Field(default_factory=list)
    contradictory_evidence: List[PubMedEvidenceItem] = Field(default_factory=list)
    methodological_precedent: List[PubMedEvidenceItem] = Field(default_factory=list)
    queries: List[DirectionSearchQuery] = Field(default_factory=list)
    species_tissue_qualification: SpeciesTissueQualification = ""
    gap_assessment: str = ""
    summary: str = ""
    limitations: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[float] = None
    timed_out: bool = False
    support_level: Literal["strong", "moderate", "weak", "none", "unknown"] = "unknown"
    contradiction_level: Literal["strong", "moderate", "weak", "none", "unknown"] = "unknown"
    prior_art_level: Literal["high", "medium", "low", "unknown"] = "unknown"
    novelty_status: Literal[
        "genuine_novelty", "known_area", "unknown_due_to_failure",
        "unknown_due_to_timeout", "unknown"
    ] = "unknown"
    similar_published_directions: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_gap_summary: str = ""
    fallback_tier: str = ""
    scheduler_limitation: str = ""

class ResearchDirectionCandidate(BaseModel):
    direction_id: str
    direction: str
    rationale: str = ""
    data_support_summary: str = ""
    literature_support_summary: str = ""
    data_reliability: str = ""
    literature_novelty: str = ""
    biology_expectedness: str = ""
    data_novelty: str = ""
    mechanistic_importance: str = ""
    functional_impact: str = ""
    data_sufficiency: str = ""
    methodological_gap: str = ""
    overall_priority: str = ""
    risk_flags: List[str] = Field(default_factory=list)
    prerequisite_dependencies: List[str] = Field(default_factory=list)
    evidence_ref: Optional[str] = None

class DirectionSelectionDecision(BaseModel):
    """Record of user's direction selection decision with selected and rejected directions."""
    selected_directions: List[Dict[str, str]] = Field(default_factory=list)
    rejected_directions: List[Dict[str, str]] = Field(default_factory=list)
    selection_rationale: str = ""

def tolerant_from_llm_dict(cls: type, data: Dict[str, Any]) -> Any:
    """Build a Pydantic model from loosely-typed LLM JSON, dropping any field
    whose value fails type validation so a single malformed field cannot crash
    the whole model.  Dropped fields fall back to their schema defaults.

    This is a generalised version of ExplorationReport.from_llm_dict that
    works for any BaseModel subclass that receives LLM-generated JSON.
    """
    def _normalize_list_field(annotation: Any, value: Any) -> Any:
        """Recover safe dict-shaped LLM output for list-typed fields.

        Models occasionally emit either a single object or an id-keyed object
        map where the schema asks for a list.  Only dict inputs are normalized:
        arbitrary scalar/string inputs remain invalid so prose is not silently
        promoted into structured report fields.
        """
        from typing import get_args, get_origin

        if get_origin(annotation) not in (list, List) or not isinstance(value, dict):
            return value
        args = get_args(annotation)
        if not args:
            return value
        item_annotation = args[0]
        item_adapter = TypeAdapter(item_annotation)

        # An id-keyed map (for example {"dir_1": {...}, "dir_2": {...}})
        # should become its ordered values before considering the whole mapping
        # as one list item.  This ordering also prevents List[Dict[str, Any]]
        # from accepting the outer map as a misleading singleton record.
        mapped_values = list(value.values())
        if mapped_values:
            try:
                for item in mapped_values:
                    item_adapter.validate_python(item)
            except ValidationError:
                pass
            else:
                return mapped_values

        # Otherwise treat the mapping as one structured item when that item
        # itself satisfies the list element schema.
        try:
            item_adapter.validate_python(value)
        except ValidationError:
            return value
        return [value]

    clean: Dict[str, Any] = {}
    for name, value in data.items():
        field = cls.model_fields.get(name)
        if field is None:
            continue
        normalized = _normalize_list_field(field.annotation, value)
        try:
            TypeAdapter(field.annotation).validate_python(normalized)
        except ValidationError:
            logger.warning(
                "tolerant_from_llm_dict(%s) dropping field %r: value of "
                "type %s failed validation against %s",
                cls.__name__, name, type(value).__name__, field.annotation,
            )
            continue
        if normalized is not value:
            logger.warning(
                "tolerant_from_llm_dict(%s) normalized dict field %r to a list",
                cls.__name__, name,
            )
        clean[name] = normalized
    return cls(**clean)
