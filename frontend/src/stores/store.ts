// The Zustand store: initial state, actions, and persist wiring.
// Split out of frontend/src/stores/appStore.ts verbatim.

import { create, type StateCreator } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { Language, getDefaultPrompts } from '../i18n/translations'
import { getToken, setToken, clearToken } from '../utils/authStorage'
import { PERSIST_STORAGE_KEY, PERSIST_VERSION } from '../constants/storage'
import { resolveSnapshotFailureActions } from '../utils/snapshotV2'
import { AppState } from './appState'
import { DEFAULT_MODELS, DEFAULT_TEMPERATURES, DEFAULT_VISION_SETTINGS } from './domain'
import { migratePersistedState, partialize, quotaSafeLocalStorage, scrubAgentServerConfigs } from './persistence'

let fallbackChatIdentity = 0

function createChatIdentity(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes)
  } else {
    // Non-browser test/SSR fallback remains unique within this process without
    // deriving identity from message content or an unseeded random source.
    fallbackChatIdentity += 1
    const value = `${Date.now().toString(16)}${fallbackChatIdentity.toString(16)}`
    for (let index = 0; index < value.length && index < bytes.length * 2; index += 2) {
      bytes[index / 2] = Number.parseInt(value.slice(index, index + 2).padStart(2, '0'), 16)
    }
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

function clearSignedOutSecrets(state: AppState, authReady: boolean) {
  return {
    agentServerConfigs: {},
    apiConfig: { ...state.apiConfig, apiKey: '' },
    deepResearchApiKey: '',
    googleApiKey: '',
    authUser: null,
    authReady,
  }
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
        realtimeHistoryError: null,
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
      literatureMode: 'pubmed_llm' as AppState['literatureMode'],
      setLiteratureMode: (mode) => set({ literatureMode: mode }),
      deepResearchModel: 'o4-mini-deep-research',
      setDeepResearchModel: (model) => set({ deepResearchModel: model }),
      deepResearchLlmBaseUrl: '',
      setDeepResearchLlmBaseUrl: (url) => set({ deepResearchLlmBaseUrl: url }),
      deepResearchApiKey: '',
      setDeepResearchApiKey: (key) => set({ deepResearchApiKey: key }),
      googleApiKey: '',
      setGoogleApiKey: (key) => set({ googleApiKey: key }),
      pubmedEmail: '',
      setPubmedEmail: (email) => set({ pubmedEmail: email }),
      literatureMaxSearchRounds: 20,
      setLiteratureMaxSearchRounds: (rounds) => set({ literatureMaxSearchRounds: Math.max(1, Math.min(50, rounds)) }),

      // Chat messages
      chatMessages: [],
      addChatMessage: (msg) => set((state) => {
        const id = msg.id || createChatIdentity()
        return { chatMessages: [...state.chatMessages, { ...msg, id }] }
      }),
      setChatMessages: (msgs) => set({
        chatMessages: msgs.every(message => Boolean(message.id))
          ? msgs
          : msgs.map(message => message.id ? message : ({ ...message, id: createChatIdentity() })),
      }),
      chatLoading: false,
      setChatLoading: (loading) => set({ chatLoading: loading }),
      chatDelegationActive: false,
      setChatDelegationActive: (active) => set({ chatDelegationActive: active }),
      chatDelegationPlan: null,
      setChatDelegationPlan: (plan) => set({ chatDelegationPlan: plan }),
      chatAbortController: null,
      setChatAbortController: (controller) => set({ chatAbortController: controller }),
      realtimeHistoryError: null,
      setRealtimeHistoryError: (error) => set({ realtimeHistoryError: error }),

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
      annotationGateData: null,
      setAnnotationGateData: (data) => set({ annotationGateData: data }),
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
            const allowed = resolveSnapshotFailureActions(
              f,
              payload.snapshot_failure_allowed_actions,
            )
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
              allowed_actions: allowed,
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
      snapshotV2Enabled: false,
      setSnapshotV2Enabled: (enabled) => set({ snapshotV2Enabled: enabled }),

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
        set((state) => ({
          ...clearSignedOutSecrets(state, state.authReady),
          currentSession: null,
          messages: [],
          chatMessages: [],
        }))
      },
      loadAuthFromStorage: async () => {
        if (get().authReady) return
        const t = getToken()
        if (!t) {
          set((state) => clearSignedOutSecrets(state, true))
          return
        }
        try {
          const r = await fetch('/api/auth/me', { headers: { Authorization: `Bearer ${t}` } })
          if (r.ok) {
            const me = await r.json()
            set({ authUser: me, authReady: true })
          } else {
            clearToken()
            set((state) => clearSignedOutSecrets(state, true))
          }
        } catch {
          clearToken()
          set((state) => clearSignedOutSecrets(state, true))
        }
      },
})
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
        const ps = (persistedState && typeof persistedState === 'object')
          ? { ...(persistedState as Record<string, any>) }
          : {}
        // Validate key types; fall back to current (default) state for invalid fields
        if (ps.models && (typeof ps.models !== 'object' || Array.isArray(ps.models))) delete ps.models
        if (ps.temperatures && (typeof ps.temperatures !== 'object' || Array.isArray(ps.temperatures))) delete ps.temperatures
        if (ps.prompts && (typeof ps.prompts !== 'object' || Array.isArray(ps.prompts))) delete ps.prompts
        ps.agentServerConfigs = scrubAgentServerConfigs(ps.agentServerConfigs)
        if (ps.language && typeof ps.language !== 'string') delete ps.language
        if (ps.maxRounds && typeof ps.maxRounds !== 'number') delete ps.maxRounds
        // Realtime messages are server-authoritative and re-hydrated from /load on
        // every open/refresh/reconnect, so never trust a persisted copy: drop any
        // legacy `messages` left by older releases (prevents stale/crash-prone shapes).
        delete ps.messages
        // Expire persisted session/chat older than 7 days to prevent localStorage bloat.
        // A negative delta means the system clock moved backwards (clock skew or
        // user adjusted the date) — treat that as an invalid timestamp and clear,
        // otherwise the data could effectively live forever from the user's POV.
        const PERSIST_TTL_MS = 7 * 24 * 60 * 60 * 1000
        if (ps._persistedAt) {
          const delta = Date.now() - (ps._persistedAt as number)
          if (delta < 0 || delta > PERSIST_TTL_MS) {
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
      storage: createJSONStorage(() => quotaSafeLocalStorage),
    }
  )
)
