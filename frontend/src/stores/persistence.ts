// localStorage persistence: what gets written, and version migration.
// Split out of frontend/src/stores/appStore.ts verbatim.

import { AppState } from './appState'
import { AgentServerConfig, AgentType, PublicationFigure } from './domain'

const memoryStorage = new Map<string, string>()

function backingStorage(): Storage | null {
  try {
    return typeof window !== 'undefined' ? window.localStorage : null
  } catch {
    return null
  }
}

/** A best-effort local cache: quota/security errors never reach React state. */
export const quotaSafeLocalStorage: Storage = {
  get length() {
    const storage = backingStorage()
    return storage?.length ?? memoryStorage.size
  },
  clear() {
    try {
      backingStorage()?.clear()
    } catch {
      // Ignore storage security failures.
    }
    memoryStorage.clear()
  },
  getItem(key: string) {
    try {
      return backingStorage()?.getItem(key) ?? memoryStorage.get(key) ?? null
    } catch {
      return memoryStorage.get(key) ?? null
    }
  },
  key(index: number) {
    try {
      return backingStorage()?.key(index) ?? Array.from(memoryStorage.keys())[index] ?? null
    } catch {
      return Array.from(memoryStorage.keys())[index] ?? null
    }
  },
  removeItem(key: string) {
    try {
      backingStorage()?.removeItem(key)
    } catch {
      // Ignore storage security failures.
    }
    memoryStorage.delete(key)
  },
  setItem(key: string, value: string) {
    try {
      backingStorage()?.setItem(key, value)
      memoryStorage.set(key, value)
    } catch (error) {
      // A cache write is optional. Keep the last in-memory value and continue.
      console.warn('Local storage quota unavailable; continuing without persistence', error)
    }
  },
}

const AGENT_TYPES: readonly AgentType[] = [
  'pi',
  'pathology',
  'bioinformatics',
  'biochemistry',
  'literature',
  'painter',
  'supervisor',
  'technique',
]

/** Remove credentials while retaining a valid per-agent endpoint hint. */
export function scrubAgentServerConfigs(
  value: unknown,
): Partial<Record<AgentType, AgentServerConfig>> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}

  const source = value as Record<string, unknown>
  const scrubbed: Partial<Record<AgentType, AgentServerConfig>> = {}
  for (const agent of AGENT_TYPES) {
    const candidate = source[agent]
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) continue
    const config = candidate as Record<string, unknown>
    if (typeof config.llmBaseUrl !== 'string') continue
    scrubbed[agent] = { llmBaseUrl: config.llmBaseUrl, apiKey: '' }
  }
  return scrubbed
}

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
    agentServerConfigs: scrubAgentServerConfigs(state.agentServerConfigs),
    agentVisionSettings: state.agentVisionSettings,
    literatureMode: state.literatureMode,
    deepResearchModel: state.deepResearchModel,
    deepResearchLlmBaseUrl: state.deepResearchLlmBaseUrl,
    pubmedEmail: state.pubmedEmail,
    literatureMaxSearchRounds: state.literatureMaxSearchRounds,
    sidebarOpen: state.sidebarOpen,
    sidebarWidth: state.sidebarWidth,
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
      curationReviewContext: undefined,
    } : null,
    // Limit persisted chat messages and strip ALL base64 content
    chatMessages: state.chatMessages.slice(-100).map(msg => ({
      ...msg,
      figures: msg.figures?.map((f) => f ? {
        ...f,
        base64: undefined,
        image_base64: undefined,
        imageBase64: undefined,
        data: undefined,
      } : f)
    })),
    // Review context may contain authoritative/image-heavy transient payloads;
    // it is rebuilt from the session load response.
    // NOTE: realtime `messages` are intentionally NOT persisted here. They are
    // server-authoritative (backend keeps them in AnalysisState.realtime_messages
    // and on disk) and always re-hydrated from /session/{id}/load on open, refresh
    // and WebSocket reconnect. Persisting them to localStorage only risked stale /
    // crash-prone shapes with no upside — see merge() which strips any legacy copy.
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
  if (Object.prototype.hasOwnProperty.call(ps, 'agentServerConfigs')) {
    ps.agentServerConfigs = scrubAgentServerConfigs(ps.agentServerConfigs)
  }
  return ps
}
