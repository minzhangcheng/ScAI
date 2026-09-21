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


from .adata_checkpoints import AdataSourceSelection
from .enums import AnalysisPhaseType, PlanStage, PlanStatus


class AnnotationCitation(BaseModel):
    """One article behind a cluster annotation.

    Title and year are copied rather than referenced on purpose. The per-cluster
    evidence that produced them lives only for the duration of the annotation
    call and is never persisted, so a bare PMID stored here would be
    unresolvable the moment that call returns.

    ``used_by_model`` separates the two facts a citation can carry: whether the
    article was put in front of the matcher, and whether the matcher said it
    used it. Recording only the second loses what the annotation actually read;
    recording only the first cannot distinguish support from background.
    """
    pmid: str
    title: str = ""
    year: Optional[str] = None
    url: Optional[str] = None
    source: Literal["focused_search", "background_review"] = "focused_search"
    used_by_model: bool = False
    # Set when a later validation round changed cell_type: the article was
    # evidence for the previous label and may not support the new one.
    stale: bool = False


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
    citations: List[AnnotationCitation] = Field(default_factory=list)
    # An empty `citations` has four different meanings and the reader cannot
    # tell them apart, so the reason is recorded explicitly:
    #   grounded    - the matcher named at least one of the articles it was shown
    #   pool_only   - articles were shown for this cluster, the matcher named none
    #   no_evidence - the focused search ran and returned nothing
    #   unavailable - no per-cluster evidence reached this record at all, which
    #                 is also what every annotation written before this field
    #                 existed reports. Never read it as "no literature support".
    citation_status: Literal[
        "grounded", "pool_only", "no_evidence", "unavailable"
    ] = "unavailable"
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
    result_identity_key: str = ""
    chapter_sha256: str = ""
    supplement_owner_key: str = ""  # Exact future/current round identity for supplementary content.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: str  # "pre_annotation" | "exploration_complete" | "supplementary_annotation_N" | "supplementary_direction_N"
    title: str  # Human-readable, e.g. "Pre-Annotation Analysis Summary"
    content_html: str = ""  # Pre-rendered HTML
    decision_content_html: str = ""  # Complete result inventory for human review.
    timestamp: datetime = Field(default_factory=datetime.now)
    round_number: int = 0

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


class AcceptedFoundationStep(BaseModel):
    """One completed Round 1 step with retained technical provenance."""

    plan_step_id: int = Field(gt=0)
    title: str = Field(default="", max_length=300)
    method: str = Field(default="", max_length=600)
    status: Literal["completed"] = "completed"
    evidence_execution_ids: List[str] = Field(default_factory=list, max_length=64)
    figure_requirement_keys: List[str] = Field(default_factory=list, max_length=64)


class AcceptedFoundationInventory(BaseModel):
    """Bounded, host-observed identity of a prior accepted foundation."""

    inventory_id: str = Field(min_length=1, max_length=240)
    schema_version: Literal[1] = 1
    round_number: int = Field(ge=1)
    plan_id: str = Field(min_length=1, max_length=240)
    planning_stage: Literal["stage1", "stage2", "full"]
    lineage_id: str = Field(default="main", min_length=1, max_length=240)
    inherited_from_lineage_id: Optional[str] = Field(default=None, max_length=240)
    provenance: Literal["captured", "reconstructed", "inherited"] = "captured"
    adata_checkpoint_id: Optional[str] = Field(default=None, max_length=240)
    adata_source_preparation_ids: List[str] = Field(default_factory=list, max_length=32)
    cohort_digest: Optional[str] = Field(default=None, max_length=128)
    steps: List[AcceptedFoundationStep] = Field(default_factory=list, max_length=128)
    observed_adata_structure: Dict[str, List[str]] = Field(default_factory=dict)
    accepted_at: datetime = Field(default_factory=datetime.now)


class FoundationReference(BaseModel):
    """Exact prior foundation identities used by one declared plan scope."""

    inventory_id: str = Field(min_length=1, max_length=240)
    round_number: int = Field(ge=1)
    plan_id: str = Field(min_length=1, max_length=240)
    plan_step_ids: List[int] = Field(default_factory=list, max_length=128)
    execution_ids: List[str] = Field(default_factory=list, max_length=256)
    checkpoint_id: Optional[str] = Field(default=None, max_length=240)


class PlanAnalysisScope(BaseModel):
    """Host-checkable declaration of a step's relationship to prior work."""

    relationship: Literal["new", "reuse", "revise"]
    cohort_scope: Literal["full", "subset"]
    foundation_refs: List[FoundationReference] = Field(default_factory=list, max_length=16)
    reason: str = Field(default="", max_length=2000)
    requested_by: Optional[Literal["user", "pi"]] = None


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
    analysis_scope: Optional[PlanAnalysisScope] = None
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
    scope_contract_version: Optional[int] = None
    foundation_inventory_id: Optional[str] = None
    round_number: int = 0
    direction: str = ""
    direction_key: str = ""
    branch_id: Optional[str] = None
    created_from_snapshot_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None

    # AnnData source selection is a PI-owned plan input. Application identity
    # is separate: adata_source_preparation_id survives same-source plan
    # revisions even when the plan UUID changes.
    adata_source: AdataSourceSelection = Field(default_factory=AdataSourceSelection)
    adata_source_preparation_id: Optional[str] = None
    # Host-owned transcript/resume and settlement scopes are distinct from the
    # source preparation's scope_id.
    bioinformatics_run_scope_id: Optional[str] = None
    bioinformatics_settlement_scope_id: Optional[str] = None

SpeciesTissueQualification = Literal[
    "",
    "matched_species_tissue_available",
    "no_matched_species_tissue_evidence",
    "cross_tissue_only",
    "cross_species_only",
]

# ---------------------------------------------------------------------------
# PI-directed direction-literature search records
#
# These records are deliberately additive.  They describe the technical
# search contract and the evidence handed back to the PI; they do not encode a
# scientific acceptance rule.  The old DirectionSearchQuery and
# DirectionLiteratureEvidence fields below remain in place for historical
# sessions and legacy readers.

DirectionSearchField = Literal[
    "all", "tiab", "title", "author", "mesh", "majr", "pt", "sh", "subh", "nm"
]


class DirectionSearchTerm(BaseModel):
    """One PI-authored literal PubMed term and its field tag."""

    text: str = Field(min_length=1, max_length=240)
    field: DirectionSearchField = "tiab"

    @model_validator(mode="before")
    @classmethod
    def _accept_field_tag_alias(cls, value):
        if isinstance(value, dict) and "field" not in value and "field_tag" in value:
            value = dict(value)
            value["field"] = value.pop("field_tag")
        return value

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("direction search terms must not be empty")
        return value

    @property
    def field_tag(self) -> str:
        return self.field


class DirectionConceptGroup(BaseModel):
    """Ordered OR alternatives; groups are joined by AND by the compiler."""

    group_id: str = Field(min_length=1, max_length=120)
    terms: List[DirectionSearchTerm] = Field(min_length=1, max_length=24)

    @model_validator(mode="before")
    @classmethod
    def _accept_group_alias(cls, value):
        if isinstance(value, dict) and "terms" not in value and "alternatives" in value:
            value = dict(value)
            value["terms"] = value.pop("alternatives")
        return value


class DirectionExclusionGroup(BaseModel):
    """PI-authored exclusions, including the reason shown to the PI later."""

    exclusion_id: str = Field(min_length=1, max_length=120)
    terms: List[DirectionSearchTerm] = Field(min_length=1, max_length=24)
    rationale: str = Field(default="", max_length=1000)


class DirectionSearchScope(BaseModel):
    """References required concept groups without inventing scope anchors."""

    required_group_ids: List[str] = Field(default_factory=list, max_length=24)
    required_constraints: Dict[str, str] = Field(default_factory=dict, max_length=12)
    unrestricted_dimensions: List[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(default="", max_length=2000)


class DirectionSearchTask(BaseModel):
    """A self-contained evidence question approved by the PI."""

    task_id: str = Field(min_length=1, max_length=160)
    direction_id: str = Field(min_length=1, max_length=160)
    evidence_question: str = Field(min_length=1, max_length=2000)
    purpose: str = Field(default="", max_length=1000)
    concept_groups: List[DirectionConceptGroup] = Field(default_factory=list, max_length=12)
    exclusions: List[DirectionExclusionGroup] = Field(default_factory=list, max_length=8)
    scope: Optional[DirectionSearchScope] = None
    mechanism_link: str = Field(default="", max_length=1000)
    can_establish: List[str] = Field(default_factory=list, max_length=12)
    cannot_establish: List[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def _accept_task_aliases(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "concept_groups" not in value and "groups" in value:
                value["concept_groups"] = value.pop("groups")
            if "mechanism_link" not in value and "hypothesis_link" in value:
                value["mechanism_link"] = value.pop("hypothesis_link")
        return value


class DirectionSearchExecution(BaseModel):
    """Technical facts for one compiled task execution."""

    task_id: str
    direction_id: str
    compiled_query: str = ""
    status: Literal[
        "planned", "running", "completed", "zero_result", "failed",
        "timeout", "unexecuted", "partial",
    ] = "planned"
    returned_count: int = Field(default=0, ge=0)
    total_hit_count: Optional[int] = Field(default=None, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    retrieved_pmids: List[str] = Field(default_factory=list, max_length=20)
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = Field(default=None, ge=0)
    budget_limitation: str = ""
    batch: Literal["initial", "followup"] = "initial"


class DirectionArticleInventory(BaseModel):
    """Raw article metadata retained independently of Literature's label."""

    pmid: str = Field(min_length=1, max_length=32)
    title: str = ""
    url: str = ""
    source: Literal["new_search", "session_cache"] = "new_search"
    task_ids: List[str] = Field(default_factory=list, max_length=64)
    query_links: List[str] = Field(default_factory=list, max_length=64)
    abstract: str = ""
    raw_source_cache_ref: Optional[str] = None
    authors: List[str] = Field(default_factory=list, max_length=64)
    journal: str = ""
    year: Optional[str] = None


class DirectionLiteratureExtraction(BaseModel):
    """Literature's bounded, source-linked observation for PI review."""

    pmid: str = Field(min_length=1, max_length=32)
    task_ids: List[str] = Field(default_factory=list, max_length=64)
    study_context: Dict[str, str] = Field(default_factory=dict, max_length=12)
    finding: str = Field(default="", max_length=2000)
    source_passage: str = Field(default="", max_length=1200)
    addressed_link: str = Field(default="", max_length=1000)
    applicability_limits: List[str] = Field(default_factory=list, max_length=12)
    provisional_classification: Literal[
        "direct", "linked", "background", "conflicting", "irrelevant", "uncertain", "unknown"
    ] = "unknown"
    extraction_status: Literal["completed", "failed", "unavailable"] = "completed"
    limitation: Optional[str] = None


class DirectionPIReview(BaseModel):
    """PI feedback and final scientific assessment for one candidate."""

    direction_id: str = Field(min_length=1, max_length=160)
    action: Literal["finish", "search_more"] = "finish"
    requested_detail_pmids: List[str] = Field(default_factory=list, max_length=20)
    follow_up_tasks: List[DirectionSearchTask] = Field(default_factory=list, max_length=4)
    rationale: str = Field(default="", max_length=4000)
    uncertainty: List[str] = Field(default_factory=list, max_length=20)
    cited_article_ids: List[str] = Field(default_factory=list, max_length=40)
    final_scientific_classifications: Dict[str, str] = Field(default_factory=dict, max_length=16)
    assessment_status: Literal["assessed", "uncertain", "unavailable", "unassessed"] = "unassessed"


class DirectionLiteratureInProgress(BaseModel):
    """Crash/retry checkpoint for one stable direction-search gate identity."""

    gate_id: str = Field(min_length=1, max_length=300)
    policy_version: str = "direction_search_v1"
    plan_id: str = ""
    plan_hash: str = ""
    plan_status: Literal["planned", "partial", "completed", "failed", "unavailable", "unknown"] = "unknown"
    planning_errors: Dict[str, str] = Field(default_factory=dict)
    compile_errors: Dict[str, str] = Field(default_factory=dict)
    limitations_by_candidate: Dict[str, List[str]] = Field(default_factory=dict)
    phase: str = "exploration"
    round_number: int = 0
    integration_cycle: Optional[int] = None
    candidate_ids: List[str] = Field(default_factory=list, max_length=256)
    tasks: List[DirectionSearchTask] = Field(default_factory=list, max_length=256)
    task_executions: List[DirectionSearchExecution] = Field(default_factory=list, max_length=256)
    article_inventory: List[DirectionArticleInventory] = Field(default_factory=list, max_length=2048)
    extractions: List[DirectionLiteratureExtraction] = Field(default_factory=list, max_length=2048)
    pi_reviews: List[DirectionPIReview] = Field(default_factory=list, max_length=32)
    unexecuted_task_ids: List[str] = Field(default_factory=list, max_length=256)
    spent_seconds: float = Field(default=0.0, ge=0)
    followup_used_by_candidate: Dict[str, bool] = Field(default_factory=dict)
    task_batches: Dict[str, Literal["initial", "followup", "forbidden"]] = Field(default_factory=dict)
    provided_detail_pmids_by_candidate: Dict[str, List[str]] = Field(default_factory=dict)
    # Wall-clock deadlines survive an in-process retry and remain meaningful
    # after a session is reloaded; monotonic values are derived at execution.
    candidate_deadline_epoch: Dict[str, float] = Field(default_factory=dict)
    status: Literal["in_progress", "partial", "completed", "unavailable"] = "in_progress"

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
        "pi_directed",
        # Session-cached articles, triaged once per candidate rather than once
        # per query. Always degraded, so it never counts toward axis coverage.
        "session_cache",
    ]
    query: str
    result_count: int = 0
    relevant_count: int = 0
    # How many of result_count had already been triaged for this direction under
    # an earlier category and were therefore not sent to the model again. Without
    # it a repeat-heavy query reads as "20 results, 0 relevant", which is a
    # statement about the literature rather than about the deduplication.
    duplicate_count: int = 0
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
    key_evidence: str = ""
    gap_assessment: str = ""
    summary: str = ""
    limitations: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[float] = None
    timed_out: bool = False
    support_level: Literal["strong", "moderate", "weak", "none", "unknown"] = "unknown"
    contradiction_level: Literal["strong", "moderate", "weak", "none", "unknown"] = "unknown"
    prior_art_level: Literal["high", "medium", "low", "unknown"] = "unknown"
    novelty_status: Literal[
        "genuine_novelty", "known_area", "adjacent_work_only",
        "unknown_due_to_failure", "unknown_due_to_timeout", "unknown"
    ] = "unknown"
    similar_published_directions: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_gap_summary: str = ""
    fallback_tier: str = ""
    scheduler_limitation: str = ""
    # Additive PI-directed search contract.  Empty/default values preserve
    # loading of evidence written before this policy existed.
    policy_version: str = ""
    plan_id: str = ""
    plan_hash: str = ""
    plan_status: Literal["planned", "partial", "completed", "failed", "unavailable", "unknown"] = "unknown"
    tasks: List[DirectionSearchTask] = Field(default_factory=list)
    task_executions: List[DirectionSearchExecution] = Field(default_factory=list)
    article_inventory: List[DirectionArticleInventory] = Field(default_factory=list)
    extractions: List[DirectionLiteratureExtraction] = Field(default_factory=list)
    pi_reviews: List[DirectionPIReview] = Field(default_factory=list)
    assessment_status: Literal["assessed", "uncertain", "unavailable", "unassessed"] = "unassessed"
    assessment_uncertainty: List[str] = Field(default_factory=list)
    assessment_limitations: List[str] = Field(default_factory=list)
    unexecuted_task_ids: List[str] = Field(default_factory=list)

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
