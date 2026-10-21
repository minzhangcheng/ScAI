/**
 * Shared TypeScript types for the ScAI frontend.
 * Types here match backend API response structures (snake_case) and are used
 * across hooks, stores, and components.
 */

import type { AgentType, TaskStatus, Session, TreatmentGroup, DataFile, BiochemistryIndicator, PublicationFigure, PublicationTable, RoundResult, GraphicAbstract, SpeciesInputInfo } from './stores/appStore'
import type { TranslationKey } from './i18n/translations'

// Re-export for convenience
export type { AgentType, TaskStatus, Session, TreatmentGroup, DataFile, BiochemistryIndicator, PublicationFigure, PublicationTable, RoundResult, GraphicAbstract, SpeciesInputInfo, TranslationKey }

// ============================================================
// Backend API response types (snake_case as returned by server)
// ============================================================

/** One cluster annotation as persisted in `annotation_state`. Fields are
 * optional so states written by older releases still parse. */
/** One article recorded behind a cluster annotation. */
export interface RawAnnotationCitation {
  pmid?: string
  title?: string
  year?: string | null
  url?: string | null
  source?: string
  used_by_model?: boolean
  stale?: boolean
}

export interface RawClusterAnnotation {
  cluster_id?: string
  cell_type?: string
  subtypes?: string[]
  key_markers?: string[]
  confidence?: string
  evidence?: string
  alternative_cell_types?: string[]
  low_confidence_reason?: string
  citations?: RawAnnotationCitation[]
  citation_status?: string
  user_confirmed?: boolean
  user_override?: string | null
  group_proportions?: Record<string, number>
  requires_reclustering?: boolean
  reclustering_reason?: string
  suggested_reclustering_scope?: string | null
}

/**
 * Authoritative cell-annotation gate state, as persisted server-side.
 * The `annotation_confirmation` realtime message carries the same payload but
 * is lossy (replay window, stream cap), so the dialog is rebuilt from this on
 * reload.
 */
export interface RawAnnotationState {
  cluster_annotations?: RawClusterAnnotation[]
  tissue?: string
  organism?: string
  tissue_summary?: string
  analysis_summary?: string
  pending?: boolean
  confirmed?: boolean
  proportion_group_labels?: string[]
  proportion_dimension?: string
  validation_status?: string
  validation_summary?: string
  validation_text_summary?: string
  validation_plan?: unknown
  validation_result?: Record<string, unknown>
  reclustering_recommendations?: Array<{
    cluster_id: string
    requires_reclustering: boolean
    reclustering_reason: string
    suggested_reclustering_scope?: string | null
  }>
}

/** Raw session data as returned by GET /session/:id/load */
export interface SessionLoadResponse {
  session_id: string
  name?: string
  status?: string
  current_round?: number
  created_at?: string
  initial_instruction?: string
  analysis_phase?: 'exploration' | 'innovation' | 'integration'
  treatment_groups?: RawTreatmentGroup[]
  data_files?: RawDataFile[]
  biochemistry_indicators?: RawBiochemistryIndicator[]
  round_results?: RawRoundResult[]
  final_summary?: string
  publication_figures?: RawPublicationFigure[]
  publication_tables?: RawPublicationTable[]
  figure_revision_tasks?: RawFigureRevisionTask[]
  bulk_input_profiles?: RawBulkInputProfile[]
  method_decisions?: RawMethodDecision[]
  graphic_abstract?: RawGraphicAbstract
  integrated_report?: string | null
  integrated_report_state?: {
    published?: RawIntegratedReportPublication | null
    pending?: Record<string, unknown> | null
  } | null
  exploration_report_url?: string | null
  integration_report_url?: string
  annotation_report_url?: string
  detailed_report_url?: string
  restore_branch_id?: string | null
  annotation_state?: RawAnnotationState | null
  state_version?: number
  runtime_persistence?: Record<string, unknown>
  round_wait_seconds?: number
  waiting_input_started_at?: string
  agent_errors?: RawAgentError[]
  messages?: RawMessage[]
  species?: string
  species_resolved?: boolean
  species_genes_validated?: boolean
  species_input_info?: Record<string, unknown>
  wait_input_reason?: string | null
  llm_failure_info?: {
    agent?: string
    model?: string
    llm_server?: string
    error?: string
    failure_kind?: string
    node_name?: string
    generation?: number
    wait_seconds?: number
    reason?: 'llm_failure' | 'marker_db_missing'
  } | null
  agent_disabled_info?: {
    agent?: string
    node_name?: string
    generation?: number
  } | null
  literature_config?: {
    mode?: 'gemini_deep_research' | 'pubmed_llm' | 'deep_research' | 'disabled'
    deep_research_model?: string
    deep_research_llm_base_url?: string
    deep_research_api_key?: string
    google_api_key?: string
    pubmed_email?: string
    max_search_rounds?: number
  }
  runtime_restore_status?: 'complete' | 'unavailable' | 'intentionally_kept'
  runtime_restore_warning?: string | null
}

/** Authoritative per-session configuration displayed by the Agents page. */
export interface AgentSettingsResponse {
  prompts: Record<AgentType, string>
  models: Record<AgentType, string>
  temperatures: Record<AgentType, number>
  agent_server_configs: Partial<Record<AgentType, {
    llm_base_url?: string
    api_key?: string
    context_window?: number | null
    effective_context_window?: number
  }>>
  vision_settings: Record<AgentType, boolean>
  literature_config: NonNullable<SessionLoadResponse['literature_config']>
  default_settings_saved?: boolean
}

export interface RawTreatmentGroup {
  id: string
  name: string
  description?: string
  sampleCount?: number
  sample_count?: number
  order?: number
}

export interface RawDataFile {
  id: string
  filename: string
  filePath?: string
  file_path?: string
  section: 'pathology' | 'sequencing' | 'biochemistry'
  dataType?: string
  data_type?: string
  groupId?: string
  group_id?: string
  sampleNumber?: number
  sample_number?: number
  description?: string
  timestamp?: string
}

export interface RawBiochemistryIndicator {
  id: string
  indicator: string
  value: number
  unit: string
  referenceRange?: string
  reference_range?: string
  interpretation?: string
  groupId?: string
  group_id?: string
  sampleNumber?: number
  sample_number?: number
  sourceGroupLabel?: string | null
  source_group_label?: string | null
  provenanceNotes?: string[]
  provenance_notes?: string[]
}

export interface CrossModalCandidate {
  theme: string
  mechanism_statement: string
  confidence: 'high' | 'medium' | 'low'
  relation_to_current_direction: 'central' | 'complementary' | 'downstream' | 'alternative' | 'uncovered'
  report_recommendation: 'headline' | 'secondary' | 'future_direction' | 'do_not_emphasize'
}

export interface PathologyCoverage {
  image_id: string
  image_type: string
  status: 'completed' | 'partial' | 'error' | 'failed' | 'skipped'
  expected_tiles?: number | null
  valid_tiles?: number | null
  coverage_ratio?: number | null
  invalid_batches?: Record<string, unknown>[]
  error?: string
}

export interface RawRoundResult {
  report_segment_id?: string | null
  round?: number
  pi_summary?: string
  summary?: string
  phase?: string
  pathology_findings?: string[]
  pathology_coverage?: PathologyCoverage[]
  bioinformatics_findings?: string[]
  biochemistry_findings?: string[]
  literature_findings?: string[]
  cross_modal_candidates?: CrossModalCandidate[]
  integration_candidates?: { direction: string; score?: number; notes?: string; [key: string]: unknown }[]
  integration_report_summary?: string
  round_status?: 'completed' | 'partial' | 'failed' | 'skipped'
  evidence_status?: 'substantive' | 'empty' | 'failed'
  direction_coverage_eligible?: boolean | null
  bioinformatics_evidence_verified?: boolean | null
  evidence_gate_degraded?: boolean | null
  evidence_execution_ids?: string[]
  evidence_notes?: string[]
  evidence_gate_findings?: Record<string, unknown>[]
  lineage_id?: string
  branch_id?: string | null
  branch_label?: string | null
  restored_from_snapshot_id?: string | null
  direction?: string
  direction_key?: string
  result_identity_key?: string
  wait_policy?: 'manual' | 'countdown' | 'no_wait'
}

export interface RawIntegratedReportFigureRef {
  number: number
  figure_id: string
  name: string
  description?: string
  figure_type?: string
  render_sha256?: string | null
  pdf_sha256?: string | null
  logical_figure_id?: string | null
  review_status?: string
  review_unavailable?: boolean
  revision_count?: number
  round_number?: number | null
  diagnostic?: boolean
  is_publication?: boolean
  pi_report_note?: string | null
}

export interface RawIntegratedReportPublication {
  generation_id: string
  publication_revision: number
  lineage_id?: string
  markdown: string
  figure_refs?: RawIntegratedReportFigureRef[]
  reporting_limitations?: string[]
  catalog_fingerprint: string
  published_at: string
}

export interface RawPublicationFigure {
  id: string
  name: string
  description: string
  figure_type?: string
  figureType?: string
  png_path?: string
  pngPath?: string
  pdf_path?: string
  pdfPath?: string
  png_base64?: string
  pngBase64?: string
  review_status?: string
  reviewStatus?: string
  review_unavailable?: boolean
  reviewUnavailable?: boolean
  pi_feedback?: string
  piFeedback?: string
  revision_count?: number
  revisionCount?: number
  plan_step_id?: number
  planStepId?: number
  plan_id?: string
  planId?: string
  plan_requirement_key?: string
  planRequirementKey?: string
  submitted_plan_step_id?: number
  submittedPlanStepId?: number
  supervisor_binding_decision?: 'accept' | 'rebind' | 'supplementary' | 'reject'
  supervisorBindingDecision?: 'accept' | 'rebind' | 'supplementary' | 'reject'
  supervisor_binding_reason?: string
  supervisorBindingReason?: string
  source_execution_id?: string
  sourceExecutionId?: string
  purpose?: string
  is_publication?: boolean
  isPublication?: boolean
  created_at?: string
  createdAt?: string
}

export interface RawPublicationTable {
  id: string
  name: string
  description: string
  csv_path?: string
  csvPath?: string
  columns?: string[]
  preview_data?: Record<string, unknown>[]
  previewData?: Record<string, unknown>[]
  review_status?: string
  reviewStatus?: string
  pi_feedback?: string
  piFeedback?: string
  created_at?: string
  createdAt?: string
}

export interface RawFigureRevisionTask {
  id: string
  logical_figure_id: string
  plan_id?: string | null
  plan_requirement_key?: string | null
  current_figure_id: string
  source_execution_id?: string | null
  status: 'pending' | 'in_progress' | 'resolved' | 'blocked' | 'abandoned' | 'exhausted_excluded'
  reason: 'visual_revision' | 'reanalysis_required' | 'revision_failed' | 'attempts_exhausted' | 'cohort_changed'
  must_fix: string[]
  suggestions: string[]
  attempt_count: number
  max_attempts: number
  created_at: string
  updated_at: string
}

export interface RawBulkInputProfile {
  id: string
  source_file_id: string
  shape: number[]
  annotation_columns: string[]
  candidate_sample_columns: string[]
  numeric_fraction_by_column: Record<string, number>
  integer_like_fraction: number
  negative_value_count: number
  missing_value_summary: Record<string, number>
  declared_scale: 'counts' | 'normalized' | 'transformed' | 'unknown'
  profile_status: 'complete' | 'sampled' | 'failed'
  complete: boolean
  rows_scanned: number
  error_code?: string | null
  created_at: string
}

export interface RawMethodDecision {
  id: string
  modality?: string
  input_profile_id?: string | null
  plan_step_id?: number | null
  claim_level: 'inferential' | 'descriptive' | 'not_feasible'
  decision_status?: 'selected' | 'not_selected' | 'inconclusive'
  method: string
  rationale: string
  diagnostics?: string[]
  limitations?: string
  execution_id: string
  decision_source: 'approved_plan_and_bioinformatics' | 'bioinformatics_with_pi_consultation'
  pi_consultation_id?: string | null
  round_number: number
  created_at: string
}

export interface RawGraphicAbstract {
  id: string
  prompt?: string
  image_base64?: string
  imageBase64?: string
  text_diagram?: string
  textDiagram?: string
  is_text_mode?: boolean
  isTextMode?: boolean
  created_at?: string
  createdAt?: string
}

export interface RawAgentError {
  id: string
  agent: string
  error_type: string
  message: string
  details?: string
  round_number: number
  timestamp: string
  resolved: boolean
}

export interface RawMessage {
  id?: string
  type?: string
  agent?: string
  content: unknown
  timestamp?: string
}

export interface RealtimeMessagePage {
  messages: RawMessage[]
  count: number
  total?: number
  page_size: number
  has_more: boolean
  next_before?: string | null
}

/** Graphic abstract message content (snake_case/camelCase flexible) */
export interface GraphicAbstractContent {
  id?: string
  prompt?: string
  image_base64?: string
  imageBase64?: string
  text_diagram?: string
  textDiagram?: string
  is_text_mode?: boolean
  isTextMode?: boolean
  created_at?: string
  createdAt?: string
}

// ============================================================
// API response wrappers
// ============================================================

export interface ApiSuccessResponse<T = unknown> {
  success: boolean
  data: T
}

export interface SessionStartResponse {
  success: boolean
  session_id: string
  data: { name: string }
}

export interface SessionInfoUpdateResponse {
  success: boolean
  data: { name: string; instruction: string }
}

export interface GroupAddResponse {
  success: boolean
  data: TreatmentGroup
}

export interface GroupUpdateResponse {
  success: boolean
  data: Partial<TreatmentGroup>
}

export interface IndicatorAddResponse {
  success: boolean
  data: { indicator_id: string }
}

export interface ResumeResponse {
  success: boolean
  data?: { needs_reconnect?: boolean }
}

export interface SessionListItem {
  id: string
  name: string
  status: string
  current_round: number
  created_at?: string
  updated_at?: string
  waiting_input_started_at?: string
  round_wait_seconds?: number
  has_pathology?: boolean
  has_sequencing?: boolean
  has_biochemistry?: boolean
}

export interface SessionListResponse {
  sessions: SessionListItem[]
}

export interface ImportPollResponse {
  status: 'uploading' | 'pending' | 'completed' | 'failed'
  session_id?: string
  error?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  id: string
  delegation_active?: boolean
  pending_figures?: boolean
  chat_review_ids?: string[]
  execution_incomplete?: boolean
  reply_to?: string
  followup_id?: string
  branch_id?: string
  files?: { filename: string; type: string; target_agent?: string }[]
  figures?: {
    name: string
    path?: string
    base64?: string
    image_base64?: string
    imageBase64?: string
    data?: string
    pdfPath?: string
    figure_role?: 'publication' | 'quick_display'
    artifact_id?: string
    review_status?: string
  }[]
}

/** Report data returned by GET /session/:id/report */
export interface ReportResponse {
  final_summary?: string
  round_results?: RawRoundResult[]
}

// ============================================================
// Component prop helper types
// ============================================================

/** Translation function type from useTranslation */
export type TranslationFn = (key: TranslationKey, params?: Record<string, string | number>) => string

/** Lucide icon component type */
export type LucideIconComponent = React.ComponentType<{ className?: string }>

// ============================================================
// Shared component types
// ============================================================

export interface RoundSummary {
  reportSegmentId?: string
  round: number
  maxRounds?: number
  summary: string
  findings: string[]
  dataRequests: {
    pathology: string[]
    biochemistry: string[]
    other: string[]
  }
  status: 'analyzing' | 'waiting_input' | 'completed'
  waitPolicy?: 'manual' | 'countdown' | 'no_wait'
  // Per-agent findings (from round_summary WebSocket message)
  pathologyFindings?: string[]
  bioinformaticsFindings?: string[]
  biochemistryFindings?: string[]
  literatureFindings?: string[]
  crossModalCandidates?: CrossModalCandidate[]
  phase: 'exploration' | 'innovation' | 'integration' | ''
  integrationCandidates: { direction: string; score?: number; notes?: string; [key: string]: unknown }[]
  integrationReportSummary?: string
  lineageId?: string
  directionKey?: string
  resultIdentityKey?: string
}
