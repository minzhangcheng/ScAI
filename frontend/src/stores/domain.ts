// Domain types shared across the app, plus their defaults.
// Split out of frontend/src/stores/appStore.ts verbatim.



export type AgentType = 'pi' | 'pathology' | 'bioinformatics' | 'biochemistry' | 'literature' | 'painter' | 'supervisor' | 'technique'
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'stopped' | 'waiting_input' | 'waiting_annotation' | 'waiting_plan_review' | 'waiting_species_input' | 'generating_report'
export type FigureReviewStatus = 'pending' | 'approved' | 'accepted_with_reservations' | 'revision_requested'


// Auth types
export interface AuthUser {
  id: string
  username: string
  email?: string
  role?: 'admin' | 'user'
  status?: 'active' | 'pending' | 'readonly'
}


// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ContentBlock = Record<string, any>

export interface Message {
  id: string
  type: 'status' | 'message' | 'code' | 'result' | 'error' | 'figure' | 'step' | 'prompt' | 'response' | 'thinking' | 'tool_call' | 'output' | 'completed' | 'progress' | 'round_summary' | 'data_request' | 'human_input_received' | 'human_input_requested' | 'literature' | 'pi_question' | 'pi_response' | 'findings' | 'figure_review' | 'figure_approved' | 'figure_revision' | 'integrated_report' | 'graphic_abstract' | 'detailed_report' | 'state_sync' | 'parallel_errors' | 'chat_delegation' | 'chat_result' | 'chat_error' | 'supervisor_feedback' | 'supervisor_summary' | 'supervisor_checkpoint' | 'annotation_confirmation' | 'exploration_report' | 'exploration_failed' | 'direction_correlation' | 'plan_review' | 'plan_approved' | 'modality_warning' | 'report_blocked' | 'integration_gate' | 'integration_complete'
  agent?: AgentType
  content: string | ContentBlock
  timestamp: string
}

export interface AgentError {
  id: string
  agent: AgentType
  errorType: string
  message: string
  details?: string
  round: number
  timestamp: string
  resolved: boolean
}

export interface PublicationFigure {
  id: string
  name: string
  description: string
  figureType: string
  pngPath: string
  pdfPath: string
  pngBase64?: string
  reviewStatus: FigureReviewStatus
  piFeedback?: string
  revisionCount: number
  createdAt: string
}

export interface PublicationTable {
  id: string
  name: string
  description: string
  csvPath: string
  columns: string[]
  previewData: Record<string, any>[]
  reviewStatus: FigureReviewStatus
  piFeedback?: string
  createdAt: string
}

export interface GraphicAbstract {
  id: string
  prompt: string
  imageBase64?: string
  textDiagram?: string
  isTextMode?: boolean
  createdAt: string
}

export interface TreatmentGroup {
  id: string
  name: string
  description: string
  sampleCount: number
  order: number
}

export interface DataFile {
  id: string
  filename: string
  filePath: string
  section: 'pathology' | 'sequencing' | 'biochemistry'
  dataType: string
  groupId?: string
  sampleNumber?: number
  description?: string
  timestamp: string
}

export interface BiochemistryIndicator {
  id: string
  indicator: string
  value: number
  unit: string
  referenceRange?: string
  interpretation?: string
  groupId?: string
  sampleNumber?: number
}

export interface RoundResult {
  round: number
  summary: string
  phase: 'exploration' | 'innovation' | 'integration' | ''
  pathologyFindings: string[]
  bioinformaticsFindings: string[]
  biochemistryFindings: string[]
  literatureFindings: string[]
  crossModalCandidates: import('../types').CrossModalCandidate[]
  integrationCandidates: { direction: string; score?: number; notes?: string; [key: string]: unknown }[]
  integrationReportSummary?: string
}

export interface Session {
  id: string
  name: string
  status: TaskStatus
  currentRound: number
  createdAt: string
  instruction: string
  analysisPhase?: 'exploration' | 'innovation' | 'integration'
  treatmentGroups: TreatmentGroup[]
  dataFiles: DataFile[]
  biochemistryIndicators: BiochemistryIndicator[]
  roundResults: RoundResult[]
  finalSummary?: string
  // Publication-quality outputs
  publicationFigures?: PublicationFigure[]
  publicationTables?: PublicationTable[]
  graphicAbstract?: GraphicAbstract
  integratedReport?: string
  explorationReportUrl?: string
  integrationReportUrl?: string
  multiDirectionCorrelation?: string
  stateVersion?: number
  // Species inference state
  species?: string
  speciesResolved?: boolean
  speciesGenesValidated?: boolean
}

export interface AgentPrompts {
  pi: string
  pathology: string
  bioinformatics: string
  biochemistry: string
  literature: string
  painter: string
  supervisor: string
  technique: string
}

export interface AgentModels {
  pi: string
  pathology: string
  bioinformatics: string
  biochemistry: string
  literature: string
  painter: string
  supervisor: string
  technique: string
}

export interface AgentTemperatures {
  pi: number
  pathology: number
  bioinformatics: number
  biochemistry: number
  literature: number
  painter: number
  supervisor: number
  technique: number
}

export interface AgentServerConfig {
  llmBaseUrl: string
  apiKey: string
}

export interface AgentVisionSettings {
  pi: boolean
  pathology: boolean
  bioinformatics: boolean
  biochemistry: boolean
  literature: boolean
  painter: boolean
  supervisor: boolean
  technique: boolean
}

export const DEFAULT_VISION_SETTINGS: AgentVisionSettings = {
  pi: true,
  pathology: true,
  bioinformatics: true,
  biochemistry: true,
  literature: false,
  painter: true,
  supervisor: true,
  technique: false,
}

// Default models
export const DEFAULT_MODELS: AgentModels = {
  pi: 'gpt-5.6-sol',
  pathology: 'gemini-3.1-pro-preview',
  bioinformatics: 'gpt-5.6-sol',
  biochemistry: 'gemini-3.1-pro-preview',
  literature: 'gpt-5.4-mini',
  painter: 'gpt-image-2',
  supervisor: 'gpt-5.4-mini',
  technique: 'gpt-5.4-mini'
}

// Default temperatures
// Low temperature (0.1-0.3): More deterministic, objective analysis
// High temperature (0.5-0.7): More creative, diverse exploration
export const DEFAULT_TEMPERATURES: AgentTemperatures = {
  pi: 0.5,           // PI can be more creative in synthesis and planning
  pathology: 0.2,    // Pathology needs objective, consistent analysis
  bioinformatics: 0.1, // Bioinformatics must be highly deterministic
  biochemistry: 0.2,  // Biochemistry needs objective interpretation
  literature: 0.6,   // Literature search can explore more broadly
  painter: 0.7,       // Painter benefits from creative variation in image generation
  supervisor: 0.1,   // Supervisor needs high-certainty evaluation
  technique: 0.1     // Technique needs precise documentation lookup
}

export interface LlmFailureInfo {
  agent: string
  model: string
  llmServer: string
  error: string
  nodeName: string
  generation: number
  reason?: 'llm_failure' | 'agent_disabled' | 'marker_db_missing'
}

export interface SpeciesInputInfo {
  source: 'pi_inference' | 'gene_validation'
  suggestions: string[]
  context: string
  species?: string
  sampleGenes?: string[]
  piAssessment?: string
  likelySpecies?: string
}

export interface ExplorationFailedInfo {
  round: number
  reason: string
  message?: string
}
