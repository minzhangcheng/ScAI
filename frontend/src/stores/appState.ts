// The AppState shape: every slice of store state and its actions.
// Split out of frontend/src/stores/appStore.ts verbatim.

import { Language } from '../i18n/translations'
import type { ChatMessage } from '../types'
import type { AnnotationData } from '../utils/annotationGate'
import { AgentError, AgentModels, AgentPrompts, AgentServerConfig, AgentTemperatures, AgentType, AgentVisionSettings, AuthUser, ExplorationFailedInfo, LlmFailureInfo, Message, PendingChatFollowup, Session, SpeciesInputInfo } from './domain'

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
  pendingChatFollowup: PendingChatFollowup | null
  setPendingChatFollowup: (draft: PendingChatFollowup | null) => void
  
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
  chatMessages: ChatMessage[]
  addChatMessage: (msg: Omit<ChatMessage, 'id'> & { id?: string }) => void
  setChatMessages: (msgs: ChatMessage[]) => void
  chatLoading: boolean
  setChatLoading: (loading: boolean) => void
  chatDelegationActive: boolean
  setChatDelegationActive: (active: boolean) => void
  chatDelegationPlan: string | null
  setChatDelegationPlan: (plan: string | null) => void
  chatAbortController: AbortController | null
  setChatAbortController: (controller: AbortController | null) => void

  // Initial realtime history paging failure; retry remains owned by the
  // session/reconnect hook and is deliberately not persisted.
  realtimeHistoryError: string | null
  setRealtimeHistoryError: (error: string | null) => void

  // UI state
  activeTab: 'analysis' | 'data' | 'prompts' | 'results' | 'report' | 'chat'
  setActiveTab: (tab: AppState['activeTab']) => void
  
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void

  sidebarWidth: number
  setSidebarWidth: (width: number) => void

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

  /** Backend AnalysisState.runtime_restore_status — survives reload via session load. */
  runtimeRestoreStatus: 'complete' | 'unavailable' | 'intentionally_kept'
  setRuntimeRestoreStatus: (status: 'complete' | 'unavailable' | 'intentionally_kept') => void
  runtimeRestoreWarning: string | null
  setRuntimeRestoreWarning: (warning: string | null) => void
  hydrateRuntimeRestoreFromSession: (payload: {
    runtime_restore_status?: string | null
    runtime_restore_warning?: string | null
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

  // Cell-annotation gate, rebuilt from the server's authoritative
  // annotation_state on every /load. Server-authoritative, so it is not
  // persisted (see partialize) — the annotation_confirmation realtime message
  // is too lossy to be the dialog's only source.
  annotationGateData: AnnotationData | null
  /** Exact session/branch owner of annotationGateData. */
  annotationGateOwnerKey: string | null
  setAnnotationGateData: (data: AnnotationData | null) => void

  // Species input user intervention
  speciesInputInfo: SpeciesInputInfo | null
  /** Exact session/branch owner of speciesInputInfo. */
  speciesInputOwnerKey: string | null
  showSpeciesInputDialog: boolean
  setSpeciesInputInfo: (info: SpeciesInputInfo | null) => void
  setShowSpeciesInputDialog: (show: boolean) => void
  clearSpeciesInputState: () => void

  // Exploration failed user intervention (likely a data problem — prompt user)
  explorationFailedInfo: ExplorationFailedInfo | null
  /** Exact session/branch owner of explorationFailedInfo. */
  explorationFailedOwnerKey: string | null
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
