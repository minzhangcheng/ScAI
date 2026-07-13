import { create, type StateCreator } from 'zustand'
import { persist } from 'zustand/middleware'
import { Language, getDefaultPrompts } from '../i18n/translations'
import { getToken, setToken, clearToken } from '../utils/authStorage'
import { PERSIST_STORAGE_KEY, PERSIST_VERSION } from '../constants/storage'
import type { SnapshotRecordV2, SnapshotStoreStatus, SnapshotTransactionRecord, SnapshotFailureInfo } from '../types/snapshotV2'

// Types
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
  pi: 'google/gemini-3.1-pro-preview',
  pathology: 'google/gemini-3.1-pro-preview',
  bioinformatics: 'anthropic/claude-sonnet-4.6',
  biochemistry: 'google/gemini-3.1-pro-preview',
  literature: 'google/gemini-3.1-flash-lite-preview',
  painter: 'google/gemini-3.1-flash-image-preview',  // Image generation model via OpenRouter
  supervisor: 'google/gemini-3.1-flash-lite-preview',
  technique: 'google/gemini-3.1-flash-lite-preview'
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

export interface AppState {
  // Language
  language: Language
  setLanguage: (lang: Language) => void
  
  // API Config
  apiConfig: {
    apiKey: string
    llmBaseUrl: string
    proxyUrl: string
  }
  setApiConfig: (config: Partial<AppState['apiConfig']>) => void
  
  // Current session
  currentSession: Session | null
  setCurrentSession: (session: Session | null) => void
  updateSession: (updates: Partial<Session>) => void
  
  // Messages (real-time)
  messages: Message[]
  addMessage: (message: Message) => void
  setMessages: (messages: Message[]) => void
  clearMessages: () => void
  
  // Prompts
  prompts: AgentPrompts
  setPrompt: (agent: AgentType, prompt: string) => void
  resetPromptsToDefault: () => void
  getDefaultPromptsForLanguage: () => AgentPrompts
  
  // Models
  models: AgentModels
  setModel: (agent: AgentType, model: string) => void
  resetModelsToDefault: () => void
  
  // Temperatures
  temperatures: AgentTemperatures
  setTemperature: (agent: AgentType, temp: number) => void
  resetTemperaturesToDefault: () => void
  
  // Max Rounds
  maxRounds: number
  setMaxRounds: (rounds: number) => void

  // Round Wait Time (-1=manual, 0=no wait, positive=countdown seconds)
  roundWaitSeconds: number
  setRoundWaitSeconds: (seconds: number) => void

  // Per-agent LLM server overrides
  agentServerConfigs: Partial<Record<AgentType, AgentServerConfig>>
  setAgentServerConfig: (agent: AgentType, config: AgentServerConfig) => void

  // Agent vision capability settings
  agentVisionSettings: AgentVisionSettings
  setAgentVisionSetting: (agent: AgentType, enabled: boolean) => void

  // Literature agent config
  literatureMode: 'gemini_deep_research' | 'pubmed_llm' | 'deep_research' | 'disabled'
  setLiteratureMode: (mode: AppState['literatureMode']) => void
  deepResearchModel: string
  setDeepResearchModel: (model: string) => void
  deepResearchLlmBaseUrl: string
  setDeepResearchLlmBaseUrl: (url: string) => void
  deepResearchApiKey: string
  setDeepResearchApiKey: (key: string) => void
  googleApiKey: string
  setGoogleApiKey: (key: string) => void
  pubmedEmail: string
  setPubmedEmail: (email: string) => void
  literatureMaxSearchRounds: number
  setLiteratureMaxSearchRounds: (rounds: number) => void

  // Chat messages (PI Agent free-form chat)
  chatMessages: { id: string; role: 'user' | 'assistant'; content: string; timestamp: string; files?: { filename: string; type: string }[]; figures?: { name: string; path?: string; base64?: string; pdfPath?: string }[] }[]
  addChatMessage: (msg: Omit<AppState['chatMessages'][0], 'id'> & { id?: string }) => void
  setChatMessages: (msgs: AppState['chatMessages']) => void
  chatLoading: boolean
  setChatLoading: (loading: boolean) => void
  chatDelegationActive: boolean
  setChatDelegationActive: (active: boolean) => void
  chatDelegationPlan: string | null
  setChatDelegationPlan: (plan: string | null) => void
  chatAbortController: AbortController | null
  setChatAbortController: (controller: AbortController | null) => void

  // UI state
  activeTab: 'analysis' | 'data' | 'prompts' | 'results' | 'report' | 'chat'
  setActiveTab: (tab: AppState['activeTab']) => void
  
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void

  // Message panel width (percentage)
  messagePanelWidth: number
  setMessagePanelWidth: (width: number) => void

  // Report generation status
  reportGenerated: boolean
  setReportGenerated: (generated: boolean) => void

  // WebSocket
  wsConnected: boolean
  setWsConnected: (connected: boolean) => void

  // State version control for synchronization
  stateVersion: number
  setStateVersion: (version: number) => void

  // Agent errors tracking
  agentErrors: AgentError[]
  addAgentError: (error: AgentError) => void
  clearAgentErrors: () => void
  resolveError: (errorId: string) => void
  getUnresolvedErrors: () => AgentError[]
  getErrorsByAgent: (agent: AgentType) => AgentError[]

  // Config panel trigger callback (set by App.tsx)
  showConfigPanel: (() => void) | null
  setShowConfigPanelCallback: (callback: (() => void) | null) => void

  // WebSocket connect/disconnect callbacks (set by App.tsx, consumed by AnalysisPanel)
  wsConnect: (() => void) | null
  wsDisconnect: (() => void) | null
  setWsCallbacks: (connect: (() => void) | null, disconnect: (() => void) | null) => void

  // Read mode - restricts user from starting new analyses (not persisted)
  readMode: boolean
  setReadMode: (mode: boolean) => void

  // Session loading overlay - true while loadSession is fetching (not persisted)
  sessionLoading: boolean
  setSessionLoading: (loading: boolean) => void

  // Snapshot count and cleanup reminder
  snapshotCount: number
  setSnapshotCount: (count: number) => void
  showSnapshotCleanupReminder: boolean
  setShowSnapshotCleanupReminder: (show: boolean) => void

  // Snapshot V2 state
  snapshotsV2: SnapshotRecordV2[]
  setSnapshotsV2: (snapshots: SnapshotRecordV2[]) => void
  snapshotStoreStatus: SnapshotStoreStatus | null
  setSnapshotStoreStatus: (status: SnapshotStoreStatus | null) => void
  snapshotTransaction: SnapshotTransactionRecord | null
  setSnapshotTransaction: (txn: SnapshotTransactionRecord | null) => void
  snapshotFailure: SnapshotFailureInfo | null
  setSnapshotFailure: (failure: SnapshotFailureInfo | null) => void
  clearSnapshotFailure: () => void
  /** Backend AnalysisState.runtime_restore_status — survives reload via session load. */
  runtimeRestoreStatus: 'complete' | 'unavailable' | 'intentionally_kept'
  setRuntimeRestoreStatus: (status: 'complete' | 'unavailable' | 'intentionally_kept') => void
  runtimeRestoreWarning: string | null
  setRuntimeRestoreWarning: (warning: string | null) => void
  hydrateRuntimeRestoreFromSession: (payload: {
    runtime_restore_status?: string | null
    runtime_restore_warning?: string | null
    snapshot_failure?: SnapshotFailureInfo['failure'] | Record<string, unknown> | null
    snapshot_failure_allowed_actions?: SnapshotFailureInfo['allowed_actions'] | null
  }) => void
  isForwardResumeBlocked: () => boolean
  isRerunEnabled: () => boolean

  // Session browser callback (set by Sidebar, invoked by ConfigPanel's Read Mode)
  openSessionBrowser: (() => void) | null
  setOpenSessionBrowserCallback: (callback: (() => void) | null) => void

  // Server-side API configuration flag (set when backend confirms a key is configured)
  apiConfigured: boolean
  setApiConfigured: (v: boolean) => void

  // Auth state
  authReady: boolean
  authUser: AuthUser | null
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string, email?: string) => Promise<{ id: string; username: string; status: string }>
  updateProfileEmail: (currentPassword: string, newEmail: string) => Promise<void>
  updateProfilePassword: (currentPassword: string, newPassword: string) => Promise<void>
  logout: () => void
  loadAuthFromStorage: () => Promise<void>

  // LLM failure user intervention
  llmFailureInfo: LlmFailureInfo | null
  setLlmFailureInfo: (info: LlmFailureInfo | null) => void
  showLlmTimeoutDialog: boolean
  setShowLlmTimeoutDialog: (show: boolean) => void
  llmModifyPending: boolean
  setLlmModifyPending: (pending: boolean) => void
  llmModifySaved: boolean
  setLlmModifySaved: (saved: boolean) => void
  clearLlmFailureState: () => void

  // Species input user intervention
  speciesInputInfo: SpeciesInputInfo | null
  showSpeciesInputDialog: boolean
  setSpeciesInputInfo: (info: SpeciesInputInfo | null) => void
  setShowSpeciesInputDialog: (show: boolean) => void
  clearSpeciesInputState: () => void

  // Exploration failed user intervention (likely a data problem — prompt user)
  explorationFailedInfo: ExplorationFailedInfo | null
  showExplorationFailedDialog: boolean
  setExplorationFailedInfo: (info: ExplorationFailedInfo | null) => void
  setShowExplorationFailedDialog: (show: boolean) => void
  clearExplorationFailedState: () => void

  // App version (fetched from backend at startup)
  appVersion: string
  setAppVersion: (version: string) => void

  // Internal: true after Zustand persist finishes hydrating from localStorage.
  _hydrated: boolean
}

const createAppState: StateCreator<AppState> = (set, get) => ({
      // Language
      language: 'en' as Language,
      setLanguage: (lang) => {
        // Only update language, don't reset prompts (prompts are always English)
        set({ language: lang })
      },
      
      // API Config
      apiConfig: {
        apiKey: '',
        llmBaseUrl: 'https://openrouter.ai/api/v1',
        proxyUrl: ''
      },
      setApiConfig: (config) => set((state) => ({
        apiConfig: { ...state.apiConfig, ...config }
      })),
      
      // Current session
      currentSession: null,
      setCurrentSession: (session) => set({
        currentSession: session,
        snapshotCount: 0,
        showSnapshotCleanupReminder: false,
        llmFailureInfo: null,
        showLlmTimeoutDialog: false,
        llmModifyPending: false,
        llmModifySaved: false,
        explorationFailedInfo: null,
        showExplorationFailedDialog: false,
      }),
      updateSession: (updates) => set((state) => ({
        currentSession: state.currentSession 
          ? { ...state.currentSession, ...updates }
          : null
      })),
      
      // Messages
      messages: [],
      addMessage: (message) => set((state) => {
        if (message.type === 'progress') {
          // Replace the last progress message from the SAME agent (avoids stacked spinners
          // while preserving independent progress from different agents)
          let lastProgressIdx = -1;
          for (let i = state.messages.length - 1; i >= 0; i--) {
            if (state.messages[i].type === 'progress' && state.messages[i].agent === message.agent) { lastProgressIdx = i; break; }
          }
          if (lastProgressIdx !== -1) {
            const updated = [...state.messages];
            updated[lastProgressIdx] = message;
            return { messages: updated };
          }
        }
        return { messages: [...state.messages, message] };
      }),
      setMessages: (messages) => set({ messages }),
      clearMessages: () => set({ messages: [] }),
      
      // Prompts
      prompts: getDefaultPrompts('en'),
      setPrompt: (agent, prompt) => set((state) => ({
        prompts: { ...state.prompts, [agent]: prompt }
      })),
      resetPromptsToDefault: () => set((state) => ({
        prompts: getDefaultPrompts(state.language)
      })),
      getDefaultPromptsForLanguage: () => {
        const state = get()
        return getDefaultPrompts(state.language)
      },
      
      // Models
      models: { ...DEFAULT_MODELS },
      setModel: (agent, model) => set((state) => ({
        models: { ...state.models, [agent]: model }
      })),
      resetModelsToDefault: () => set({ models: { ...DEFAULT_MODELS } }),
      
      // Temperatures
      temperatures: { ...DEFAULT_TEMPERATURES },
      setTemperature: (agent, temp) => set((state) => ({
        temperatures: { ...state.temperatures, [agent]: temp }
      })),
      resetTemperaturesToDefault: () => set({ temperatures: { ...DEFAULT_TEMPERATURES } }),
      
      // Max Rounds
      maxRounds: 3,
      setMaxRounds: (rounds) => set({ maxRounds: rounds }),

      // Round Wait Time
      roundWaitSeconds: 300,
      setRoundWaitSeconds: (seconds) => set({ roundWaitSeconds: seconds }),

      // Per-agent LLM server overrides
      agentServerConfigs: {},
      setAgentServerConfig: (agent, config) => set((state) => ({
        agentServerConfigs: { ...state.agentServerConfigs, [agent]: config }
      })),

      // Agent vision capability settings
      agentVisionSettings: { ...DEFAULT_VISION_SETTINGS },
      setAgentVisionSetting: (agent, enabled) => set((state) => ({
        agentVisionSettings: { ...state.agentVisionSettings, [agent]: enabled }
      })),

      // Literature agent config
      literatureMode: 'deep_research' as AppState['literatureMode'],
      setLiteratureMode: (mode) => set({ literatureMode: mode }),
      deepResearchModel: 'openai/o4-mini-deep-research',
      setDeepResearchModel: (model) => set({ deepResearchModel: model }),
      deepResearchLlmBaseUrl: '',
      setDeepResearchLlmBaseUrl: (url) => set({ deepResearchLlmBaseUrl: url }),
      deepResearchApiKey: '',
      setDeepResearchApiKey: (key) => set({ deepResearchApiKey: key }),
      googleApiKey: '',
      setGoogleApiKey: (key) => set({ googleApiKey: key }),
      pubmedEmail: '',
      setPubmedEmail: (email) => set({ pubmedEmail: email }),
      literatureMaxSearchRounds: 10,
      setLiteratureMaxSearchRounds: (rounds) => set({ literatureMaxSearchRounds: Math.max(1, Math.min(50, rounds)) }),

      // Chat messages
      chatMessages: [],
      addChatMessage: (msg) => set((state) => {
        const id = msg.id || (typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`)
        return { chatMessages: [...state.chatMessages, { ...msg, id }] }
      }),
      setChatMessages: (msgs) => set({
        chatMessages: msgs.map((m) => ({
          ...m,
          id: m.id || (typeof crypto !== 'undefined' && crypto.randomUUID
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`)
        }))
      }),
      chatLoading: false,
      setChatLoading: (loading) => set({ chatLoading: loading }),
      chatDelegationActive: false,
      setChatDelegationActive: (active) => set({ chatDelegationActive: active }),
      chatDelegationPlan: null,
      setChatDelegationPlan: (plan) => set({ chatDelegationPlan: plan }),
      chatAbortController: null,
      setChatAbortController: (controller) => set({ chatAbortController: controller }),

      // UI state - default to 'data' tab as users typically upload data first
      activeTab: 'data',
      setActiveTab: (tab) => set(s => s.activeTab === tab ? s : { activeTab: tab }),
      
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      // Message panel width (percentage, 20-80)
      messagePanelWidth: 50,
      setMessagePanelWidth: (width) => set({ messagePanelWidth: Math.max(20, Math.min(80, width)) }),

      // Report generation status
      reportGenerated: false,
      setReportGenerated: (generated) => set({ reportGenerated: generated }),

      // WebSocket
      wsConnected: false,
      setWsConnected: (connected) => set({ wsConnected: connected }),

      // State version control
      stateVersion: 0,
      setStateVersion: (version) => set({ stateVersion: version }),

      // Agent errors tracking
      agentErrors: [],
      addAgentError: (error) => set((state) => ({
        agentErrors: [...state.agentErrors, error]
      })),
      clearAgentErrors: () => set({ agentErrors: [] }),
      resolveError: (errorId) => set((state) => ({
        agentErrors: state.agentErrors.map(e =>
          e.id === errorId ? { ...e, resolved: true } : e
        )
      })),
      getUnresolvedErrors: () => {
        return get().agentErrors.filter(e => !e.resolved)
      },
      getErrorsByAgent: (agent) => {
        return get().agentErrors.filter(e => e.agent === agent && !e.resolved)
      },

      // Config panel trigger callback
      showConfigPanel: null,
      setShowConfigPanelCallback: (callback) => set({ showConfigPanel: callback }),

      // WebSocket connect/disconnect callbacks
      wsConnect: null,
      wsDisconnect: null,
      setWsCallbacks: (connect, disconnect) => set({ wsConnect: connect, wsDisconnect: disconnect }),

      // Read mode
      readMode: false,
      setReadMode: (mode) => set({ readMode: mode }),

      appVersion: '',
      setAppVersion: (version) => set({ appVersion: version }),

      sessionLoading: false,
      setSessionLoading: (loading) => set({ sessionLoading: loading }),

      snapshotCount: 0,
      setSnapshotCount: (count) => {
        if (get().snapshotCount !== count) set({ snapshotCount: count })
      },
      showSnapshotCleanupReminder: false,
      setShowSnapshotCleanupReminder: (show) => {
        if (get().showSnapshotCleanupReminder !== show) set({ showSnapshotCleanupReminder: show })
      },

      // Snapshot V2 state
      snapshotsV2: [],
      setSnapshotsV2: (snapshots) => set({ snapshotsV2: snapshots }),
      snapshotStoreStatus: null,
      setSnapshotStoreStatus: (status) => set({ snapshotStoreStatus: status }),
      snapshotTransaction: null,
      setSnapshotTransaction: (txn) => set({
        snapshotTransaction: txn !== null
          ? {
              ...txn,
              recovery_mode: txn.recovery_mode ?? (get().snapshotTransaction?.recovery_mode ?? 'rollback'),
            }
          : null,
      }),
      snapshotFailure: null,
      setSnapshotFailure: (failure) => set({ snapshotFailure: failure }),
      clearSnapshotFailure: () => set({ snapshotFailure: null }),
      runtimeRestoreStatus: 'complete',
      setRuntimeRestoreStatus: (status) => set({ runtimeRestoreStatus: status }),
      runtimeRestoreWarning: null,
      setRuntimeRestoreWarning: (warning) => set({ runtimeRestoreWarning: warning }),
      hydrateRuntimeRestoreFromSession: (payload) => {
        const statusRaw = payload.runtime_restore_status
        const status =
          statusRaw === 'unavailable' || statusRaw === 'intentionally_kept'
            ? statusRaw
            : 'complete'
        const warning =
          typeof payload.runtime_restore_warning === 'string'
            ? payload.runtime_restore_warning
            : null
        const updates: Partial<AppState> = {
          runtimeRestoreStatus: status,
          runtimeRestoreWarning: warning,
        }
        const rawFailure = payload.snapshot_failure
        if (rawFailure === null || rawFailure === undefined) {
          // Explicit clear from session load / state_sync after gate release
          updates.snapshotFailure = null
        } else if (rawFailure && typeof rawFailure === 'object') {
          const f = rawFailure as Record<string, unknown>
          const code = typeof f.code === 'string' ? f.code : ''
          const message = typeof f.message === 'string' ? f.message : ''
          if (code || message) {
            const allowed =
              payload.snapshot_failure_allowed_actions ||
              (Array.isArray(f.allowed_actions)
                ? (f.allowed_actions as SnapshotFailureInfo['allowed_actions'])
                : (['retry', 'manage', 'terminate'] as SnapshotFailureInfo['allowed_actions']))
            updates.snapshotFailure = {
              failure: {
                snapshot_id: typeof f.snapshot_id === 'string' ? f.snapshot_id : '',
                session_id: typeof f.session_id === 'string' ? f.session_id : '',
                failed_at: typeof f.failed_at === 'string' ? f.failed_at : '',
                code,
                message,
                details:
                  f.details && typeof f.details === 'object'
                    ? (f.details as Record<string, unknown>)
                    : undefined,
              },
              allowed_actions: allowed.filter(
                (a): a is 'retry' | 'skip' | 'manage' | 'terminate' =>
                  a === 'retry' || a === 'skip' || a === 'manage' || a === 'terminate',
              ),
            }
          }
        }
        set(updates)
      },
      isForwardResumeBlocked: () => {
        const s = get()
        if (s.snapshotFailure) return true
        return s.runtimeRestoreStatus === 'unavailable'
      },
      isRerunEnabled: () => {
        const s = get()
        return (
          s.runtimeRestoreStatus === 'unavailable' ||
          s.runtimeRestoreStatus === 'intentionally_kept'
        )
      },
      // Session browser callback
      openSessionBrowser: null,
      setOpenSessionBrowserCallback: (callback) => set({ openSessionBrowser: callback }),

      // LLM failure user intervention
      llmFailureInfo: null,
      setLlmFailureInfo: (info) => set({ llmFailureInfo: info }),
      showLlmTimeoutDialog: false,
      setShowLlmTimeoutDialog: (show) => set({ showLlmTimeoutDialog: show }),
      llmModifyPending: false,
      setLlmModifyPending: (pending) => set({ llmModifyPending: pending }),
      llmModifySaved: false,
      setLlmModifySaved: (saved) => set({ llmModifySaved: saved }),
      clearLlmFailureState: () => set({
        llmFailureInfo: null,
        showLlmTimeoutDialog: false,
        llmModifyPending: false,
        llmModifySaved: false,
      }),

      // Species input user intervention
      speciesInputInfo: null,
      showSpeciesInputDialog: false,
      setSpeciesInputInfo: (info) => set({ speciesInputInfo: info }),
      setShowSpeciesInputDialog: (show) => set({ showSpeciesInputDialog: show }),
      clearSpeciesInputState: () => set({
        speciesInputInfo: null,
        showSpeciesInputDialog: false,
      }),

      // Server-side API configuration flag (start as false, set by App.tsx on load)
      apiConfigured: false,
      setApiConfigured: (v) => set({ apiConfigured: v }),

      // Exploration failed user intervention
      explorationFailedInfo: null,
      showExplorationFailedDialog: false,
      setExplorationFailedInfo: (info) => set({ explorationFailedInfo: info }),
      setShowExplorationFailedDialog: (show) => set({ showExplorationFailedDialog: show }),
      clearExplorationFailedState: () => set({
        explorationFailedInfo: null,
        showExplorationFailedDialog: false,
      }),

      // Auth state
      authReady: false,
      _hydrated: false,
      authUser: null,
      login: async (username: string, password: string) => {
        const r = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        })
        if (!r.ok) {
          const body = await r.json().catch(() => ({ detail: 'Invalid credentials' }))
          throw new Error(body.detail || 'Invalid credentials')
        }
        const { access_token, id, username: uname, email, role, status } = await r.json()
        setToken(access_token)
        set({ authUser: id ? { id, username: uname, email, role, status } : null })
      },
      register: async (username: string, password: string, email?: string) => {
        const r = await fetch('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password, email }),
        })
        if (!r.ok) {
          const body = await r.json().catch(() => ({ detail: 'Registration failed' }))
          throw new Error(body.detail || 'Registration failed')
        }
        const data = await r.json()
        return { id: data.id, username: data.username, status: data.status }
      },
      updateProfileEmail: async (currentPassword: string, newEmail: string) => {
        const t = getToken()
        if (!t) throw new Error('Not authenticated')
        const r = await fetch('/api/auth/profile/email', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${t}` },
          body: JSON.stringify({ current_password: currentPassword, new_email: newEmail }),
        })
        if (!r.ok) {
          const body = await r.json().catch(() => ({ detail: 'Update failed' }))
          throw new Error(body.detail || 'Update failed')
        }
        set(state => state.authUser ? { authUser: { ...state.authUser, email: newEmail } } : {})
      },
      updateProfilePassword: async (currentPassword: string, newPassword: string) => {
        const t = getToken()
        if (!t) throw new Error('Not authenticated')
        const r = await fetch('/api/auth/profile/password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${t}` },
          body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
        })
        if (!r.ok) {
          const body = await r.json().catch(() => ({ detail: 'Update failed' }))
          throw new Error(body.detail || 'Update failed')
        }
      },
      logout: () => {
        clearToken()
        set({ authUser: null, currentSession: null, messages: [], chatMessages: [] })
      },
      loadAuthFromStorage: async () => {
        if (get().authReady) return
        const t = getToken()
        if (!t) {
          set({ authReady: true })
          return
        }
        try {
          const r = await fetch('/api/auth/me', { headers: { Authorization: `Bearer ${t}` } })
          if (r.ok) {
            const me = await r.json()
            set({ authUser: me, authReady: true })
          } else {
            clearToken()
            set({ authUser: null, authReady: true })
          }
        } catch {
          clearToken()
          set({ authUser: null, authReady: true })
        }
      },
})

export function partialize(state: AppState) {
  return {
    _persistedAt: Date.now(),
    language: state.language,
    apiConfig: { ...state.apiConfig, apiKey: '' },
    prompts: state.prompts,
    models: state.models,
    temperatures: state.temperatures,
    maxRounds: state.maxRounds,
    roundWaitSeconds: state.roundWaitSeconds,
    agentServerConfigs: state.agentServerConfigs,
    agentVisionSettings: state.agentVisionSettings,
    literatureMode: state.literatureMode,
    deepResearchModel: state.deepResearchModel,
    deepResearchLlmBaseUrl: state.deepResearchLlmBaseUrl,
    pubmedEmail: state.pubmedEmail,
    literatureMaxSearchRounds: state.literatureMaxSearchRounds,
    messagePanelWidth: state.messagePanelWidth,
    // Persist session with base64 stripped
    currentSession: state.currentSession ? {
      ...state.currentSession,
      publicationFigures: state.currentSession.publicationFigures?.map(
        (f: PublicationFigure) => ({ ...f, pngBase64: undefined })
      ),
      graphicAbstract: state.currentSession.graphicAbstract
        ? { ...state.currentSession.graphicAbstract, imageBase64: undefined, image_base64: undefined }
        : undefined,
    } : null,
    // Limit persisted messages and strip ALL base64 content
    chatMessages: state.chatMessages.slice(-100).map(msg => ({
      ...msg,
      figures: msg.figures?.map((f) => f ? { ...f, base64: undefined } : f)
    })),
    messages: state.messages.slice(-200).map(msg => {
      if (msg.type === 'figure') {
        return { ...msg, content: '[image]' }
      }
      if (msg.type === 'graphic_abstract' && msg.content != null && typeof msg.content === 'object') {
        return { ...msg, content: { ...msg.content, image_base64: undefined, imageBase64: undefined } }
      }
      if ((msg.type === 'figure_review' || msg.type === 'figure_approved' || msg.type === 'figure_revision') && msg.content != null && typeof msg.content === 'object') {
        return { ...msg, content: { ...msg.content, png_base64: undefined, pngBase64: undefined } }
      }
      const c = msg.content as unknown as Record<string, unknown> | undefined
      if (c?.figures && Array.isArray(c.figures) && (c.figures as unknown[]).length > 0) {
        return {
          ...msg,
          content: {
            ...c,
            figures: (c.figures as Record<string, unknown>[]).map((f) => f ? { ...f, data: undefined, pngBase64: undefined, png_base64: undefined } : f)
          }
        }
      }
      return msg
    }),
  }
}

/**
 * Migrate persisted state across breaking shape changes.
 *
 * v0 (unversioned) storage may carry legacy, server-derived runtime state
 * (`messages`, `chatMessages`, `currentSession`) whose array-shaped fields were
 * stored differently by older releases and can crash rendering. Drop that
 * volatile runtime state on upgrade; user settings are preserved and
 * sessions/messages are reloaded from the backend on demand. Runs before
 * `merge` (which then layers the migrated state over current defaults).
 */
export function migratePersistedState(persistedState: unknown, version: number): unknown {
  const ps = (persistedState && typeof persistedState === 'object')
    ? { ...(persistedState as Record<string, unknown>) }
    : {}
  if (version < 1) {
    delete ps.messages
    delete ps.chatMessages
    delete ps.currentSession
  }
  return ps
}

export const useAppStore = create<AppState>()(
  persist(
    createAppState,
    {
      name: PERSIST_STORAGE_KEY,
      version: PERSIST_VERSION,
      migrate: migratePersistedState,
      onRehydrateStorage: () => {
        return (_state: unknown, error?: unknown) => {
          if (error) {
            console.error('Zustand hydration failed:', error)
          }
          // setTimeout(0) defers the flag flip out of the rehydration commit,
          // so subscribers see _hydrated as a distinct state change after
          // currentSession is restored.
          setTimeout(() => {
            useAppStore.setState({ _hydrated: true })
          }, 0)
        }
      },
      // Validate critical fields on hydration to prevent crashes from corrupted/outdated localStorage
      merge: (persistedState: unknown, currentState) => {
        const ps = (persistedState && typeof persistedState === 'object') ? persistedState as Record<string, any> : {}
        // Validate key types; fall back to current (default) state for invalid fields
        if (ps.models && (typeof ps.models !== 'object' || Array.isArray(ps.models))) delete ps.models
        if (ps.temperatures && (typeof ps.temperatures !== 'object' || Array.isArray(ps.temperatures))) delete ps.temperatures
        if (ps.prompts && (typeof ps.prompts !== 'object' || Array.isArray(ps.prompts))) delete ps.prompts
        if (ps.agentServerConfigs && (typeof ps.agentServerConfigs !== 'object' || Array.isArray(ps.agentServerConfigs))) delete ps.agentServerConfigs
        if (ps.language && typeof ps.language !== 'string') delete ps.language
        if (ps.maxRounds && typeof ps.maxRounds !== 'number') delete ps.maxRounds
        if (ps.messages && !Array.isArray(ps.messages)) delete ps.messages
        // Expire persisted messages older than 7 days to prevent localStorage bloat.
        // A negative delta means the system clock moved backwards (clock skew or
        // user adjusted the date) — treat that as an invalid timestamp and clear,
        // otherwise the data could effectively live forever from the user's POV.
        const PERSIST_TTL_MS = 7 * 24 * 60 * 60 * 1000
        if (ps._persistedAt) {
          const delta = Date.now() - (ps._persistedAt as number)
          if (delta < 0 || delta > PERSIST_TTL_MS) {
            delete ps.messages
            delete ps.currentSession
            delete ps.chatMessages
          }
        }
        if (ps.apiConfig) {
          ps.apiConfig = { ...ps.apiConfig, apiKey: '' }
        }
        delete ps.deepResearchApiKey
        delete ps.googleApiKey
        const merged = { ...currentState, ...ps }
        return merged
      },
      partialize,
    }
  )
)
