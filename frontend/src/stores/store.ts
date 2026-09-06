import { create, type StateCreator } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

import { clampSidebarWidth } from '../constants/layout'
import { PERSIST_STORAGE_KEY, PERSIST_VERSION } from '../constants/storage'
import type { AppState } from './appState'
import { createAuthActions } from './actions/authActions'
import { createAgentConfigActions, createBasicConfigActions } from './actions/configActions'
import {
  createAgentErrorActions,
  createChatMessageActions,
  createRealtimeMessageActions,
} from './actions/messageActions'
import {
  createCurrentSessionActions,
  createInterventionActions,
  createRuntimeSessionActions,
  createUiSessionActions,
} from './actions/sessionActions'
import { createRuntimeActions } from './actions/runtimeActions'
import {
  migratePersistedState,
  partialize,
  quotaSafeLocalStorage,
  scrubAgentServerConfigs,
} from './persistence'

const createAppState: StateCreator<AppState> = (set, get) => ({
  ...createBasicConfigActions(set),
  ...createCurrentSessionActions(set),
  ...createRealtimeMessageActions(set),
  ...createAgentConfigActions(set, get),
  ...createChatMessageActions(set),
  ...createUiSessionActions(set),
  ...createAgentErrorActions(set, get),
  ...createRuntimeSessionActions(set),
  ...createRuntimeActions(set, get),
  ...createInterventionActions(set),
  ...createAuthActions(set, get),
})

export const useAppStore = create<AppState>()(
  persist(createAppState, {
    name: PERSIST_STORAGE_KEY,
    version: PERSIST_VERSION,
    migrate: migratePersistedState,
    onRehydrateStorage: () => (_state: unknown, error?: unknown) => {
      if (error) console.error('Zustand hydration failed:', error)
      setTimeout(() => {
        useAppStore.setState({ _hydrated: true })
      }, 0)
    },
    merge: (persistedState: unknown, currentState) => {
      const persisted = persistedState && typeof persistedState === 'object'
        ? { ...(persistedState as Record<string, any>) }
        : {}
      if (persisted.models && (typeof persisted.models !== 'object' || Array.isArray(persisted.models))) {
        delete persisted.models
      }
      if (persisted.temperatures
        && (typeof persisted.temperatures !== 'object' || Array.isArray(persisted.temperatures))) {
        delete persisted.temperatures
      }
      if (persisted.prompts && (typeof persisted.prompts !== 'object' || Array.isArray(persisted.prompts))) {
        delete persisted.prompts
      }
      persisted.agentServerConfigs = scrubAgentServerConfigs(persisted.agentServerConfigs)
      if (persisted.language && typeof persisted.language !== 'string') delete persisted.language
      if (persisted.maxRounds && typeof persisted.maxRounds !== 'number') delete persisted.maxRounds
      if (Object.prototype.hasOwnProperty.call(persisted, 'sidebarWidth')) {
        if (typeof persisted.sidebarWidth !== 'number' || !Number.isFinite(persisted.sidebarWidth)) {
          delete persisted.sidebarWidth
        } else {
          persisted.sidebarWidth = clampSidebarWidth(persisted.sidebarWidth)
        }
      }
      if (Object.prototype.hasOwnProperty.call(persisted, 'sidebarOpen')
        && typeof persisted.sidebarOpen !== 'boolean') {
        delete persisted.sidebarOpen
      }
      delete persisted.messages
      const persistTtlMs = 7 * 24 * 60 * 60 * 1000
      if (persisted._persistedAt) {
        const delta = Date.now() - (persisted._persistedAt as number)
        if (delta < 0 || delta > persistTtlMs) {
          delete persisted.currentSession
          delete persisted.chatMessages
        }
      }
      if (persisted.apiConfig) persisted.apiConfig = { ...persisted.apiConfig, apiKey: '' }
      delete persisted.deepResearchApiKey
      delete persisted.googleApiKey
      return { ...currentState, ...persisted }
    },
    partialize,
    storage: createJSONStorage(() => quotaSafeLocalStorage),
  }),
)
