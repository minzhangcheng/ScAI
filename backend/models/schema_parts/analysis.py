# Agent messages, code execution, analysis steps, input data, and mechanism programs.
# Moved verbatim out of backend/models/schemas.py.
"""
Data models for the Multi-Agent Single-Cell Analysis System
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


from .enums import AgentType, MessageRole


class AgentError(BaseModel):
    """Structured agent error record for tracking analysis failures"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: AgentType
    error_type: str  # "execution_error", "timeout", "data_missing", "llm_error"
    message: str
    details: Optional[str] = None
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None

class AgentMessage(BaseModel):
    """Message from/to an agent"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: AgentType
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CodeExecution(BaseModel):
    """Record of code execution"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    code: str
    output: Optional[str] = None
    error: Optional[str] = None
    figures: List[str] = Field(default_factory=list)  # Base64 encoded images
    execution_time: Optional[float] = None
    round_number: int = 0
    evidence_eligible: bool = True
    evidence_rejection_reason: Optional[str] = None
    # Derived by the host from the executed code. Findings cite execution IDs
    # and may use only the replicate design of a modality present here.
    analysis_modalities: List[Literal[
        "single_cell", "bulk_rna_seq",
    ]] = Field(default_factory=list)
    # Version 2 records are host-bound execution receipts. None identifies
    # historical executions that predate exact plan/cohort provenance.
    evidence_schema_version: Optional[int] = None
    plan_id: Optional[str] = None
    plan_step_ids: List[int] = Field(default_factory=list)
    execution_purpose: Literal[
        "planned", "supplementary", "repair", "unspecified",
    ] = "unspecified"
    plan_binding_status: Literal[
        "candidate", "confirmed", "supplementary", "unverified",
        "bound", "legacy_unverifiable",
    ] = "legacy_unverifiable"
    plan_binding_reason: Optional[str] = None
    execution_status: Literal["succeeded", "failed", "partial"] = "succeeded"
    lineage_id: str = "main"
    analysis_generation: int = 0
    cohort_generation: int = 0
    cohort_digest: Optional[str] = None
    input_profile_ids: List[str] = Field(default_factory=list)
    declared_persistent_outputs: List[str] = Field(default_factory=list)
    realized_persistent_outputs: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class BulkInputProfile(BaseModel):
    """Host-observed facts about one uploaded bulk expression table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_file_id: str
    shape: List[int] = Field(default_factory=list)
    annotation_columns: List[str] = Field(default_factory=list)
    candidate_sample_columns: List[str] = Field(default_factory=list)
    numeric_fraction_by_column: Dict[str, float] = Field(default_factory=dict)
    integer_like_fraction: float = 0.0
    negative_value_count: int = 0
    missing_value_summary: Dict[str, int] = Field(default_factory=dict)
    declared_scale: Literal[
        "counts", "normalized", "transformed", "unknown",
    ] = "unknown"
    profile_status: Literal["complete", "sampled", "failed"] = "complete"
    complete: bool = True
    rows_scanned: int = 0
    error_code: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class MethodDecision(BaseModel):
    """Agent scientific choice linked to host-observed input and execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modality: Literal["bulk_rna_seq"] = "bulk_rna_seq"
    input_profile_id: str
    claim_level: Literal["inferential", "descriptive", "not_feasible"]
    method: str
    rationale: str
    execution_id: str
    decision_source: Literal[
        "approved_plan_and_bioinformatics",
    ] = "approved_plan_and_bioinformatics"
    round_number: int = 0
    created_at: datetime = Field(default_factory=datetime.now)

class AnalysisStep(BaseModel):
    """A single step in the bioinformatics agent analysis log, covering all tool call types"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    step_type: str  # "execute_code" | "consult_pi" | "search_literature" | "annotate_clusters" | "report_findings" | "generate_publication_figure" | "text_response"
    timestamp: datetime = Field(default_factory=datetime.now)

    # Agent reasoning (LLM response.content, truncated to 2000 chars)
    agent_reasoning: Optional[str] = None

    # execute_code fields
    code: Optional[str] = None
    code_output: Optional[str] = None
    code_error: Optional[str] = None
    code_description: Optional[str] = None
    code_step_type: Optional[str] = None  # "technical" | "biological" (from tool arg step_type)
    plan_id: Optional[str] = None
    plan_step_ids: List[int] = Field(default_factory=list)
    execution_purpose: Optional[str] = None
    execution_id: Optional[str] = None
    saved_figure_filenames: List[str] = Field(default_factory=list)  # disk PNG filenames, no base64

    # consult_pi fields
    pi_question: Optional[str] = None
    pi_context: Optional[str] = None
    pi_options: List[str] = Field(default_factory=list)
    pi_response: Optional[str] = None

    # search_literature fields
    literature_topic: Optional[str] = None
    literature_result: Optional[str] = None

    # annotate_clusters fields
    cluster_markers_summary: Optional[str] = None
    annotation_tissue: Optional[str] = None
    annotation_organism: Optional[str] = None
    annotation_result_summary: Optional[str] = None
    annotation_database_priors_summary: Optional[str] = None

    # report_findings fields
    findings: List[str] = Field(default_factory=list)
    finding_records: List[Dict[str, Any]] = Field(default_factory=list)
    findings_summary: Optional[str] = None
    findings_suggestions: List[str] = Field(default_factory=list)
    evidence_execution_ids: List[str] = Field(default_factory=list)

    # generate_publication_figure fields
    pub_figure_name: Optional[str] = None
    pub_figure_description: Optional[str] = None
    pub_figure_type: Optional[str] = None
    pub_figure_plan_step_id: Optional[int] = None
    pub_figure_code: Optional[str] = None
    pub_figure_result: Optional[str] = None

    # text_response fields
    text_content: Optional[str] = None

    round_number: int = 1

class ScoreDefinition(BaseModel):
    """Definition of a computed score for methodology transparency in the final report"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score_name: str          # e.g., "response_score", "custom_signature"
    method: str              # e.g., "sc.tl.score_genes", "gseapy.ssgsea"
    description: str         # What the score measures biologically
    parameters: Dict[str, Any] = Field(default_factory=dict)
    gene_list: List[str] = Field(default_factory=list)
    gene_list_source: str = ""  # controlled vocabulary, see services/gene_set_provenance
    # Universe an over-representation test was run against. The universe is the
    # null, so a score that does not name it cannot be checked -- see
    # services/statistical_gates.check_ora_background.
    background_source: str = ""
    # Parsed form of gene_list_source. Empty when the score predates provenance
    # enforcement or carries no gene panel at all.
    provenance: Dict[str, Any] = Field(default_factory=dict)
    # Nominal panel size and how many genes actually mapped into the analysed
    # matrix. n_genes_matched is what governs power; without it a panel that
    # looks large can be almost entirely absent from the data.
    n_genes: int = 0
    n_genes_matched: Optional[int] = None
    # Result of services.gene_set_provenance.assess_panel_power.
    power: Dict[str, Any] = Field(default_factory=dict)
    # Plan this score was registered against, and whether the scored panel
    # still matched the frozen one. Empty plan_id means the score predates
    # plan enforcement or carries no gene panel.
    plan_id: str = ""
    panel_hash: str = ""
    plan_drift: Dict[str, Any] = Field(default_factory=dict)
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

class ScoringPlan(BaseModel):
    """A scoring plan frozen before its result is known.

    Curating a gene panel is a legitimate judgement; revising it after seeing
    the enrichment result is selection on the outcome. Freezing the panel makes
    the difference auditable -- see services/analysis_plan for what this does
    and does not prove.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score_name: str                    # links the plan to the score it governs
    method: str                        # intended computation
    hypothesis: str = ""               # what this score is meant to test
    falsifier: str = ""                # what result would argue against it
    # The frozen panel and its digest. panel_hash is what register_score checks.
    panel_genes: List[str] = Field(default_factory=list)
    panel_hash: str = ""
    panel_source: str = ""             # provenance of the panel as scored
    # The unedited database set this panel was derived from, when there is one.
    # Keeping it lets the report show the curated and original side by side
    # instead of only whichever one was finally used.
    baseline_genes: List[str] = Field(default_factory=list)
    baseline_source: str = ""
    edit_operations: List[Dict[str, Any]] = Field(default_factory=list)
    # Every gene set intended to go into the same multiple-testing correction.
    # GSEA q-values depend on which other sets share the call, so a family that
    # is not declared cannot be checked.
    testing_family: List[str] = Field(default_factory=list)
    registered_at_step: int = 0
    round_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class TreatmentGroup(BaseModel):
    """A treatment/experimental group definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str                          # e.g., "Reference", "Condition A", "Treatment A"
    description: str = ""              # Detailed description of what this group represents
    sample_count: int = 1              # Number of samples/replicates in this group
    order: int = 0                     # Display order

class SampleCountRecord(BaseModel):
    """How many biological replicates one (group, data_type) pair actually holds.

    Measured from the data, not declared. ``TreatmentGroup.sample_count`` is a
    single number per group, so it cannot say that a group's sequencing holds
    one specimen while its pathology holds three stains of that same specimen --
    and counting files says the opposite of the truth in both directions. The
    (group, data_type) key is what separates those cases.

    ``detected_count == 0`` means undetermined, not zero samples. Such a record
    preserves an unresolved modality design rather than manufacturing n=0.
    """
    group_id: Optional[str] = None          # None = ungrouped data
    group_name: str = ""
    section: Literal["pathology", "sequencing", "biochemistry"]
    data_type: str
    detected_count: int = 0
    declared_count: Optional[int] = None    # sample_count at detection time
    method: str = "undetermined"            # obs_column:<name> | barcode_suffix
                                            # | matrix_columns | file_records
                                            # | indicator_sample_numbers | undetermined
    evidence: str = ""                      # one line naming what was read;
                                            # recorded on the session, not
                                            # currently shown to the PI or report
    conflicts: bool = False                 # detected_count != declared_count


class DataFile(BaseModel):
    """Universal file record for all uploaded data (pathology, sequencing, biochemistry)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str                     # Absolute path on disk

    # Section: which data section this belongs to
    section: Literal["pathology", "sequencing", "biochemistry"]

    # Data type within the section (see DATA_TYPE_* constants for sequencing types)
    # pathology: "H&E", "Sirius Red", "Perls Blue", "Oil Red O", "PAS", or custom
    # sequencing: DATA_TYPE_SCRNASEQ, DATA_TYPE_BULK_RNASEQ, "loom"
    # biochemistry: "image", "file"
    data_type: str

    # Group and sample assignment
    group_id: Optional[str] = None     # TreatmentGroup.id, or None for "No group distinction"
    sample_number: Optional[int] = None  # 1-based sample number, or None for "No sample distinction"

    # Section-specific metadata
    tiles: List[str] = Field(default_factory=list)       # For pathology: tile paths
    analysis: Optional[str] = None                        # For pathology: analysis result
    description: Optional[str] = None                     # User-provided description
    extracted_data: Optional[Dict[str, Any]] = None       # For biochemistry: extracted values

    timestamp: datetime = Field(default_factory=datetime.now)

class BiochemistryIndicator(BaseModel):
    """Manual biochemistry entry (indicator name + value + unit) with group/sample association"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    indicator: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None

    # Group and sample
    group_id: Optional[str] = None       # TreatmentGroup.id ONLY, never a raw label
    sample_number: Optional[int] = None

    # Provenance of the two fields above. A value read out of a file body is a
    # parse, not a declaration: keeping the raw text and any disagreement with the
    # upload-level metadata lets the PI see what was assumed instead of having a
    # guess presented as fact.
    source_group_label: Optional[str] = None   # group text as written in the file
    provenance_notes: List[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=datetime.now)

class LiteratureMechanismSignal(BaseModel):
    """A biological mechanism extracted from retrieved literature that
    co-occurs with this study's experimental observations (generic, not
    mechanism-hardcoded)."""
    mechanism: str                                  # mechanism name from the literature (not invented)
    supporting_pmids: List[str] = Field(default_factory=list)  # may be empty when PMID unavailable (e.g. Deep Research)
    evidence_type: Literal["experimental", "correlational", "review", "hypothesis", "unknown"] = "unknown"
    relevance_to_current_study: str = ""            # 1-2 sentences
    observed_data_links: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

class DataDrivenMechanismProgram(BaseModel):
    """A recurring, data-derived mechanism pattern captured by the bioinformatics
    agent after downstream analysis. Labels are descriptive only; all downstream
    interpretation relies on the structured evidence fields, never the label."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    round_number: int = 0
    phase: str = ""
    stage: str = ""
    source_agent: Literal["bioinformatics"] = "bioinformatics"

    label: str
    description: str = ""

    cell_types: List[str] = Field(default_factory=list)
    conditions_compared: str = ""

    triggering_observations: List[str] = Field(default_factory=list)
    supporting_genes: List[str] = Field(default_factory=list)
    supporting_pathways: List[str] = Field(default_factory=list)
    supporting_scores: List[str] = Field(default_factory=list)
    supporting_tables: List[str] = Field(default_factory=list)
    evidence_execution_ids: List[str] = Field(default_factory=list)

    bulk_support: Literal[
        "supported", "not_supported", "mixed", "not_available", "unknown",
    ] = "unknown"

    evidence_strength: Literal["strong", "moderate", "weak"] = "weak"
    specificity: Literal["high", "medium", "low"] = "low"

    alternative_explanations: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)

    interpretation: str = ""
    downstream_literature_queries: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_prompt_text(
        self,
        *,
        style: str = "compact",
        max_observations: int = 2,
        max_genes: int = 8,
        max_pathways: int = 8,
        max_alternatives: int = 2,
        max_missing: int = 2,
    ) -> str:
        """Format this program as human-readable text for LLM prompts.

        Styles (each is a deliberate unified format, not a 1:1 copy of any
        single former ad-hoc site):

          "compact"    — single-line with evidence/specificity/bulk, merged
                         genes+pathways, alternatives, missing. Used by
                         supervisor _build_report_prompt.
          "detailed"   — multi-line block with per-field lines. Used by
                         agents.py _build_observed_context_for_literature.
          "direction"  — single-line with [candidate mechanism]/[exploratory signal]
                         tag, strength/bulk, merged genes+pathways. Used by
                         pi_agent generate_raw_direction_candidates and
                         generate_integrated_report.
        """
        obs = "; ".join(self.triggering_observations[:max_observations])
        genes_pathways = ", ".join(
            (self.supporting_genes + self.supporting_pathways)[:max_genes]
        )
        alts = "; ".join(self.alternative_explanations[:max_alternatives])
        miss = "; ".join(self.missing_evidence[:max_missing])

        if style == "compact":
            parts = [f"{self.label} (evidence {self.evidence_strength}, specificity {self.specificity}, bulk {self.bulk_support})"]
            if obs:
                parts.append(f"{obs}")
            if genes_pathways:
                parts.append(f"Genes/pathways: {genes_pathways}")
            if alts:
                parts.append(f"Alternatives: {alts}")
            if miss:
                parts.append(f"Missing evidence: {miss}")
            return ". ".join(parts)

        if style == "direction":
            elevate = (
                self.evidence_strength in ("moderate", "strong")
                and len(self.triggering_observations) >= 2
            )
            tag = "candidate mechanism" if elevate else "exploratory signal"
            cells = ", ".join(self.cell_types) if self.cell_types else "n/a"
            parts = [f"[{tag}] {self.label} (cells: {cells}; strength {self.evidence_strength}, bulk {self.bulk_support})"]
            if obs:
                parts.append(f"{obs}")
            if genes_pathways:
                parts.append(f"Genes: {genes_pathways}")
            if miss:
                parts.append(f"Missing evidence: {miss}")
            return ". ".join(parts)

        # style == "detailed"
        lines = [f"- Program: {self.label}"]
        if self.cell_types:
            lines.append(f"  Cell types: {', '.join(self.cell_types)}")
        if obs:
            lines.append(f"  Observations: {obs}")
        if self.supporting_genes:
            lines.append(f"  Genes: {', '.join(self.supporting_genes[:max_genes])}")
        pathways = ", ".join(self.supporting_pathways[:max_pathways])
        if pathways:
            lines.append(f"  Pathways: {pathways}")
        lines.append(f"  Bulk support: {self.bulk_support}")
        if alts:
            lines.append(f"  Alternatives: {alts}")
        if miss:
            lines.append(f"  Missing evidence: {miss}")
        return "\n".join(lines)

class LiteratureReport(BaseModel):
    """Literature research report"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    report: str
    references: List[Dict[str, str]] = Field(default_factory=list)
    mechanism_signals: Optional[List[LiteratureMechanismSignal]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class CrossModalEvidenceItem(BaseModel):
    source: Literal["biochemistry", "pathology", "bioinformatics", "literature"]
    evidence: str                                   # concrete observation
    strength: Literal["strong", "moderate", "weak"]
    limitation: str = ""

class CrossModalMechanismCandidate(BaseModel):
    # relation_to_current_direction semantics:
    #   "central"       — candidate IS the current direction's main mechanism
    #   "complementary" — supports/extends current direction, not competing
    #   "downstream"    — consequence of current direction's mechanism
    #   "alternative"   — competing hypothesis that would replace current direction
    #   "uncovered"     — not addressed by current direction; distinct evidence base
    theme: str
    mechanism_statement: str
    supporting_evidence: List[CrossModalEvidenceItem] = Field(default_factory=list)
    evidence_consistency: Literal["high", "medium", "low", "conflicting"] = "low"
    confidence: Literal["high", "medium", "low"] = "low"
    novelty_relative_to_current_direction: Literal["high", "medium", "low"] = "low"
    relation_to_current_direction: Literal["central", "complementary", "downstream", "alternative", "uncovered"] = "uncovered"
    missing_evidence: List[str] = Field(default_factory=list)
    report_recommendation: Literal["headline", "secondary", "future_direction", "do_not_emphasize"] = "future_direction"

    def to_summary_dict(self) -> dict:
        """Return a lightweight 5-field dict for WebSocket/API payloads and
        report planning.  Omits supporting_evidence, evidence_consistency,
        novelty_relative_to_current_direction, and missing_evidence to keep
        round-summary payloads small."""
        return {
            "theme": self.theme,
            "mechanism_statement": self.mechanism_statement,
            "confidence": self.confidence,
            "relation_to_current_direction": self.relation_to_current_direction,
            "report_recommendation": self.report_recommendation,
        }

class CrossModalSynthesis(BaseModel):
    matrix_markdown: str = ""                        # Markdown evidence matrix table (server-side only)
    candidates: List[CrossModalMechanismCandidate] = Field(default_factory=list)
    excluded_or_low_support_themes: List[str] = Field(default_factory=list)
