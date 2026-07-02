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

/** Raw session data as returned by GET /session/:id/load */
export interface SessionLoadResponse {
  session_id: string
  name?: string
  status?: string
  current_round?: number
  created_at?: string
  initial_instruction?: string
  analysis_phase?: 'exploration' | 'innovation'
  treatment_groups?: RawTreatmentGroup[]
  data_files?: RawDataFile[]
  biochemistry_indicators?: RawBiochemistryIndicator[]
  round_results?: RawRoundResult[]
  final_summary?: string
  publication_figures?: RawPublicationFigure[]
  publication_tables?: RawPublicationTable[]
  graphic_abstract?: RawGraphicAbstract
  integrated_report?: string | null
  exploration_report_url?: string
  state_version?: number
  round_wait_seconds?: number
  waiting_input_started_at?: string
  agent_errors?: RawAgentError[]
  snapshot_count?: number
  messages?: RawMessage[]
  species?: string
  species_resolved?: boolean
  species_genes_validated?: boolean
  species_input_info?: Record<string, unknown>
  literature_config?: {
    mode?: 'gemini_deep_research' | 'pubmed_llm' | 'deep_research' | 'disabled'
    deep_research_model?: string
    deep_research_llm_base_url?: string
    deep_research_api_key?: string
    google_api_key?: string
    pubmed_email?: string
    max_search_rounds?: number
  }
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
}

export interface CrossModalCandidate {
  theme: string
  mechanism_statement: string
  confidence: 'high' | 'medium' | 'low'
  relation_to_current_direction: 'central' | 'complementary' | 'downstream' | 'alternative' | 'uncovered'
  report_recommendation: 'headline' | 'secondary' | 'future_direction' | 'do_not_emphasize'
}

export interface RawRoundResult {
  round?: number
  pi_summary?: string
  summary?: string
  pathology_findings?: string[]
  bioinformatics_findings?: string[]
  biochemistry_findings?: string[]
  literature_findings?: string[]
  cross_modal_candidates?: CrossModalCandidate[]
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
  pi_feedback?: string
  piFeedback?: string
  revision_count?: number
  revisionCount?: number
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
  files?: { filename: string; type: string; target_agent?: string }[]
  figures?: { name: string; path?: string; base64?: string; pdfPath?: string }[]
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
  // Per-agent findings (from round_summary WebSocket message)
  pathologyFindings?: string[]
  bioinformaticsFindings?: string[]
  biochemistryFindings?: string[]
  literatureFindings?: string[]
  crossModalCandidates?: CrossModalCandidate[]
}
