import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore, Message, Session, AgentType, RoundResult, type AppState } from '../stores/appStore'
import { transformRawMessage, restoreAgentErrors } from '../utils/messageTransform'
import { transformLoadResponse, transformRoundResults, type StoreUpdates } from '../utils/transformLoadResponse'
import { devLog } from '../utils/logger'
import { buildLiteratureConfigPayload, isMaskedSecret } from '../utils/literatureConfig'
import { downloadBlob } from '../utils/download'

const SNAPSHOT_CLEANUP_THRESHOLD = 20
const SNAPSHOT_REMINDER_INTERVAL = 5
import { getToken, clearToken } from '../utils/authStorage'
import { WS_AUTH_SUBPROTOCOL } from '../constants/auth'
import type {
  SessionLoadResponse,
  SessionInfoUpdateResponse,
  GroupUpdateResponse,
  SessionListResponse,
  ChatMessage,
} from '../types'

const API_BASE = '/api'
export const WS_CONNECT_TIMEOUT_MS = 10_000

export function applyStoreUpdates(updates: StoreUpdates): void {
  const store = useAppStore.getState()
  const setSecretFromLoad = (
    current: string,
    incoming: string,
    setter: (value: string) => void,
  ): void => {
    if (!isMaskedSecret(incoming) || !current || isMaskedSecret(current)) {
      setter(incoming)
    }
  }
  if (updates.stateVersion != null) store.setStateVersion(updates.stateVersion)
  if (updates.snapshotCount != null) store.setSnapshotCount(updates.snapshotCount)
  if (updates.roundWaitSeconds != null) store.setRoundWaitSeconds(updates.roundWaitSeconds)
  if (updates.literatureMode != null) store.setLiteratureMode(updates.literatureMode as AppState['literatureMode'])
  if (updates.deepResearchModel != null) store.setDeepResearchModel(updates.deepResearchModel)
  if (updates.deepResearchLlmBaseUrl != null) store.setDeepResearchLlmBaseUrl(updates.deepResearchLlmBaseUrl)
  if (updates.deepResearchApiKey != null) {
    setSecretFromLoad(store.deepResearchApiKey, updates.deepResearchApiKey, store.setDeepResearchApiKey)
  }
  if (updates.googleApiKey != null) {
    setSecretFromLoad(store.googleApiKey, updates.googleApiKey, store.setGoogleApiKey)
  }
  if (updates.pubmedEmail != null) store.setPubmedEmail(updates.pubmedEmail)
  if (updates.literatureMaxSearchRounds != null) store.setLiteratureMaxSearchRounds(updates.literatureMaxSearchRounds)
}

// Generate UUID using crypto API with fallback for non-HTTPS contexts
export function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback for environments without crypto.randomUUID
  const bytes = new Uint8Array(16)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    crypto.getRandomValues(bytes)
  } else {
    console.warn('crypto.getRandomValues unavailable — falling back to Math.random() for ID generation')
    for (let i = 0; i < 16; i++) bytes[i] = Math.random() * 256 | 0
  }
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`
}

// Get JWT token from localStorage
function getAuthToken(): string | null {
  return getToken()
}

// Build headers with auth token (exported for raw fetch calls that need auth)
export function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const token = getAuthToken()
  const headers: Record<string, string> = { ...extra }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

// Generic fetch wrapper (exported for direct use) — auto-injects Authorization header
export async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const token = getAuthToken()
  const isFormData = options?.body instanceof FormData
  const headers: Record<string, string> = {
    ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
    ...(options?.headers as Record<string, string> || {}),
  }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  // Auto-logout only on auth-related 401 (from middleware, which uses WWW-Authenticate header)
  // Other 401s (e.g., LLM key validation) should NOT clear the user's session
  if (response.status === 401) {
    const isAuthFailure = response.headers.has('WWW-Authenticate')
    if (isAuthFailure) {
      clearToken()
      throw new Error('Authentication failed. Please log in again.')
    }
    // Non-auth 401 — treat as a regular error
    const error = await response.json().catch(() => ({ detail: 'Unauthorized' }))
    throw new Error(error.detail || 'Unauthorized')
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || 'API request failed')
  }

  if (response.status === 204) return undefined as T

  return response.json()
}

// Config hook
export function useConfig() {
  const { apiConfig, setApiConfig } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const saveConfig = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      await apiFetch('/config', {
        method: 'POST',
        body: JSON.stringify({
          api_key: apiConfig.apiKey,
          llm_base_url: apiConfig.llmBaseUrl,
          proxy_url: apiConfig.proxyUrl || null,
        }),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [apiConfig])
  
  const checkStatus = useCallback(async () => {
    try {
      const status = await apiFetch<{
        configured: boolean
        has_api_key: boolean
      }>('/config/status')
      return status
    } catch (e) {
      console.warn('[ScAI] checkStatus failed:', e)
      return { configured: false, has_api_key: false }
    }
  }, [])

  const validateApiKey = useCallback(async () => {
    try {
      const result = await apiFetch<{
        valid: boolean
        message: string
        status_code?: number
        uncertain?: boolean
      }>('/config/validate', {
        method: 'POST',
        body: JSON.stringify({
          api_key: apiConfig.apiKey,
          llm_base_url: apiConfig.llmBaseUrl,
          proxy_url: apiConfig.proxyUrl || null,
        }),
      })
      return result
    } catch {
      return {
        valid: false,
        message: 'Failed to reach validation endpoint',
        uncertain: true,
      }
    }
  }, [apiConfig])

  const validateSavedConfig = useCallback(async () => {
    // Validate the key already saved in the global config (admin has no local key,
    // e.g. after localStorage was cleared). No key is sent in the request body.
    try {
      return await apiFetch<{
        valid: boolean
        message: string
        uncertain?: boolean
      }>('/config/validate-saved', { method: 'POST' })
    } catch {
      return {
        valid: false,
        message: 'Failed to reach validation endpoint',
        uncertain: true,
      }
    }
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return { apiConfig, setApiConfig, saveConfig, checkStatus, validateApiKey, validateSavedConfig, loading, error, clearError }
}

// Session hook
export function useSession() {
  const { currentSession, setCurrentSession, updateSession, setMessages, clearMessages } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const createSession = useCallback(async (instruction: string, name?: string, roundWaitSeconds?: number) => {
    setLoading(true)
    setError(null)

    // Clear messages from previous session
    clearMessages()

    try {
      const response = await apiFetch<{
        success: boolean
        session_id: string
        data: { name: string }
      }>('/session/start', {
        method: 'POST',
        body: JSON.stringify({ instruction, session_name: name, round_wait_seconds: roundWaitSeconds }),
      })
      
      const session: Session = {
        id: response.session_id,
        name: response.data.name,
        status: 'pending',
        currentRound: 0,
        createdAt: new Date().toISOString(),
        instruction,
        treatmentGroups: [],
        dataFiles: [],
        biochemistryIndicators: [],
        roundResults: [],
      }
      
      setCurrentSession(session)
      return session
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create session')
      throw e
    } finally {
      setLoading(false)
    }
  }, [setCurrentSession, clearMessages])
  
  const loadSession = useCallback(async (sessionId: string) => {
    setLoading(true)
    useAppStore.getState().setSessionLoading(true)
    setError(null)

    try {
      const data = await apiFetch<SessionLoadResponse>(`/session/${sessionId}/load`)

      const { sessionUpdates, storeUpdates } = transformLoadResponse(data)

      const session: Session = {
        id: data.session_id,
        name: data.name ?? '',
        status: 'pending',
        currentRound: 0,
        createdAt: data.created_at || new Date().toISOString(),
        instruction: '',
        treatmentGroups: [],
        dataFiles: [],
        biochemistryIndicators: [],
        roundResults: [],
        ...sessionUpdates,
      }

      setCurrentSession(session)

      applyStoreUpdates(storeUpdates)

      if (data.agent_errors && Array.isArray(data.agent_errors)) {
        restoreAgentErrors(data.agent_errors)
      }
      
      if (data.messages && Array.isArray(data.messages)) {
        setMessages(data.messages.map(transformRawMessage))
      } else {
        clearMessages()
      }
      
      return session
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load session')
      throw e
    } finally {
      setLoading(false)
      useAppStore.getState().setSessionLoading(false)
    }
  }, [setCurrentSession, setMessages, clearMessages])
  
  const listSessions = useCallback(async () => {
    const data = await apiFetch<SessionListResponse>('/sessions')
    return data.sessions
  }, [])
  
  const deleteSession = useCallback(async (sessionId: string) => {
    await apiFetch(`/session/${sessionId}`, { method: 'DELETE' })
    
    // If deleting current session, clear it
    if (currentSession?.id === sessionId) {
      setCurrentSession(null)
    }
  }, [currentSession, setCurrentSession])
  
  const saveSession = useCallback(async () => {
    if (!currentSession) return
    await apiFetch(`/session/${currentSession.id}/save`, { method: 'POST' })
  }, [currentSession])

  const updateSessionInfo = useCallback(async (updates: { name?: string; instruction?: string }) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    const body: Record<string, unknown> = {}
    if (updates.name !== undefined) body.name = updates.name
    if (updates.instruction !== undefined) body.instruction = updates.instruction
    const response = await apiFetch<SessionInfoUpdateResponse>(`/session/${session.id}/info`, {
      method: 'PUT',
      body: JSON.stringify(body),
    })
    useAppStore.getState().updateSession({
      name: response.data.name,
      instruction: response.data.instruction,
    })
    return response
  }, [])

  const updateRoundWait = useCallback(async (seconds: number) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    await apiFetch(`/session/${session.id}/round-wait`, {
      method: 'POST',
      body: JSON.stringify({ round_wait_seconds: seconds }),
    })
    useAppStore.getState().setRoundWaitSeconds(seconds)
  }, [])
  
  // Ensure session exists on server (handles server restart)
  const ensureSession = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')

    const data = await apiFetch<any>(`/session/${currentSession.id}/ensure`, {
      method: 'POST',
      body: JSON.stringify({
        instruction: currentSession.instruction,
        session_name: currentSession.name,
      }),
    })
    if (data.data?.created) {
      devLog('Session recreated after server restart')
    }
    return data
  }, [currentSession])

  // --- Group CRUD ---
  const addGroup = useCallback(async (name: string, description: string, sampleCount: number) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    await ensureSession()

    const response = await apiFetch<any>(`/session/${session.id}/groups`, {
      method: 'POST',
      body: JSON.stringify({ name, description, sample_count: sampleCount }),
    })

    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      treatmentGroups: [...(latestSession?.treatmentGroups || []), response.data],
    })
    return response
  }, [ensureSession])

  const updateGroup = useCallback(async (groupId: string, updates: { name?: string; description?: string; sampleCount?: number }) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')

    const body: Record<string, unknown> = {}
    if (updates.name !== undefined) body.name = updates.name
    if (updates.description !== undefined) body.description = updates.description
    if (updates.sampleCount !== undefined) body.sample_count = updates.sampleCount

    const response = await apiFetch<GroupUpdateResponse>(`/session/${session.id}/groups/${groupId}`, {
      method: 'PUT',
      body: JSON.stringify(body),
    })

    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      treatmentGroups: (latestSession?.treatmentGroups || []).map(g =>
        g.id === groupId ? { ...g, ...response.data } : g
      ),
    })
    return response
  }, [])

  const deleteGroup = useCallback(async (groupId: string) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')

    await apiFetch(`/session/${session.id}/groups/${groupId}`, { method: 'DELETE' })

    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      treatmentGroups: (latestSession?.treatmentGroups || []).filter(g => g.id !== groupId),
      dataFiles: (latestSession?.dataFiles || []).filter(f => f.groupId !== groupId),
      biochemistryIndicators: (latestSession?.biochemistryIndicators || []).filter(i => i.groupId !== groupId),
    })
  }, [])

  // --- Unified data upload ---
  const uploadData = useCallback(async (
    file: File,
    section: 'pathology' | 'sequencing' | 'biochemistry',
    dataType: string,
    groupId?: string,
    sampleNumber?: number,
    description?: string,
    signal?: AbortSignal
  ) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    await ensureSession()

    const formData = new FormData()
    formData.append('file', file)
    formData.append('section', section)
    formData.append('data_type', dataType)
    if (groupId) formData.append('group_id', groupId)
    if (sampleNumber != null) formData.append('sample_number', String(sampleNumber))
    if (description) formData.append('description', description)

    const data = await apiFetch<any>(
      `/session/${session.id}/upload/data`,
      { method: 'POST', body: formData, signal }
    )

    // Refresh dataFiles from server to ensure frontend state matches backend
    try {
      const refreshData = await apiFetch<SessionLoadResponse>(`/session/${session.id}/load`)
      const current = useAppStore.getState().currentSession
      if (current && current.id === session.id) {
        const { sessionUpdates } = transformLoadResponse(refreshData)
        useAppStore.getState().updateSession({
          dataFiles: sessionUpdates.dataFiles,
        })
      }
    } catch {
      // Fallback: manually append the new file if refresh fails
      const latestSession = useAppStore.getState().currentSession
      if (latestSession) {
        useAppStore.getState().updateSession({
          dataFiles: [...(latestSession.dataFiles || []), {
            id: data.data.file_id, filename: file.name, filePath: data.data.file_path,
            section, dataType, groupId: data.data.group_id || undefined,
            sampleNumber: data.data.sample_number || undefined, description,
            timestamp: new Date().toISOString(),
          }],
        })
      }
    }
    return data
  }, [ensureSession])

  const deleteDataFile = useCallback(async (fileId: string) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')

    await apiFetch(`/session/${session.id}/data/${fileId}`, { method: 'DELETE' })

    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      dataFiles: (latestSession?.dataFiles || []).filter(f => f.id !== fileId),
    })
  }, [])

  // --- Biochemistry indicators ---
  const addIndicator = useCallback(async (
    indicator: string, value: number, unit: string,
    groupId?: string, sampleNumber?: number
  ) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')

    const response = await apiFetch<any>(`/session/${session.id}/indicators`, {
      method: 'POST',
      body: JSON.stringify({ indicator, value, unit, group_id: groupId, sample_number: sampleNumber }),
    })

    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      biochemistryIndicators: [
        ...(latestSession?.biochemistryIndicators || []),
        { id: response.data.indicator_id, indicator, value, unit, referenceRange: undefined, interpretation: undefined, groupId, sampleNumber },
      ],
    })
    return response
  }, [])

  const deleteIndicator = useCallback(async (indicatorId: string) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')

    await apiFetch(`/session/${session.id}/indicators/${indicatorId}`, { method: 'DELETE' })

    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      biochemistryIndicators: (latestSession?.biochemistryIndicators || []).filter(i => i.id !== indicatorId),
    })
  }, [])
  
  const startAnalysis = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')

    // Get current models, prompts, and temperatures from store
    const { models, prompts, temperatures } = useAppStore.getState()
    const syncFailures: string[] = []
    const agents = ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature', 'painter', 'supervisor', 'technique'] as const

    // Sync all agent settings in parallel (models, prompts, temperatures)
    await Promise.allSettled([
      ...agents.map(agent =>
        apiFetch(`/session/${currentSession.id}/models`, {
          method: 'POST',
          body: JSON.stringify({ session_id: currentSession.id, agent, model: models[agent] }),
        }).catch(e => { syncFailures.push(`model/${agent}`); throw e })
      ),
      ...agents.map(agent =>
        apiFetch(`/session/${currentSession.id}/prompts`, {
          method: 'POST',
          body: JSON.stringify({ session_id: currentSession.id, agent, prompt: prompts[agent] }),
        }).catch(e => { syncFailures.push(`prompt/${agent}`); throw e })
      ),
      ...agents.map(agent =>
        apiFetch(`/session/${currentSession.id}/temperatures`, {
          method: 'POST',
          body: JSON.stringify({ session_id: currentSession.id, agent, temperature: temperatures[agent] }),
        }).catch(e => { syncFailures.push(`temperature/${agent}`); throw e })
      ),
    ])

    // Sync remaining settings in parallel (max_rounds is now dynamically set by direction selection)
    const appState = useAppStore.getState()
    const { roundWaitSeconds } = appState
    const literaturePayload = buildLiteratureConfigPayload(appState)

    await Promise.allSettled([
      apiFetch(`/session/${currentSession.id}/round-wait`, {
        method: 'POST',
        body: JSON.stringify({ round_wait_seconds: roundWaitSeconds }),
      }).catch(e => { syncFailures.push('round_wait'); throw e }),
      apiFetch(`/session/${currentSession.id}/literature-config`, {
        method: 'POST',
        body: JSON.stringify(literaturePayload),
      }).catch(e => { syncFailures.push('literature_config'); throw e }),
    ])

    // P2-9: Abort if any agent config sync failed so analysis does not run with stale config.
    if (syncFailures.length > 0) {
      const summary = syncFailures.join(', ')
      console.error('[startAnalysis] agent config sync failed:', summary)
      const err: Error & { syncFailures?: string[] } = new Error(
        `Agent config sync failed before starting analysis: ${summary}. Analysis aborted.`
      )
      err.syncFailures = syncFailures
      throw err
    }

    await apiFetch(`/session/${currentSession.id}/run`, { method: 'POST' })
    updateSession({ status: 'running' })
    return syncFailures
  }, [currentSession, updateSession])
  
  const pauseAnalysis = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/pause`, { method: 'POST' })
    updateSession({ status: 'paused' })
  }, [currentSession, updateSession])
  
  const resumeAnalysis = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')

    const resp = await apiFetch<any>(`/session/${currentSession.id}/resume`, { method: 'POST' })
    updateSession({ status: 'running' })
    // If workflow doesn't exist (e.g., after backend restart), trigger WebSocket
    // reconnect which will create a new workflow instance on the server
    if (resp?.data?.needs_reconnect) {
      const { wsConnect: connectWs } = useAppStore.getState()
      if (connectWs) setTimeout(() => connectWs(), 100)
    }
  }, [currentSession, updateSession])

  const continueRound = useCallback(async () => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    const resp = await apiFetch<any>(`/session/${session.id}/resume`, { method: 'POST' })
    useAppStore.getState().updateSession({ status: 'running' })
    if (resp?.data?.needs_reconnect) {
      const { wsConnect: connectWs } = useAppStore.getState()
      if (connectWs) setTimeout(() => connectWs(), 100)
    }
  }, [])

  const stopAnalysis = useCallback(async (action: 'immediate' | 'summarize') => {
    if (!currentSession) throw new Error('No active session')
    await apiFetch(`/session/${currentSession.id}/stop`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    })
    if (action === 'immediate') {
      updateSession({ status: 'paused' })
    } else {
      updateSession({ status: 'generating_report' })
    }
  }, [currentSession, updateSession])

  const exportState = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')

    const response = await fetch(`${API_BASE}/session/${currentSession.id}/export`, {
      method: 'GET',
      headers: authHeaders(),
    })

    if (!response.ok) {
      throw new Error('Failed to export state')
    }

    const blob = await response.blob()
    downloadBlob(blob, `${currentSession.id}_export.tar.gz`)
  }, [currentSession])
  
  const getReport = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')
    
    return apiFetch<any>(`/session/${currentSession.id}/report`)
  }, [currentSession])
  
  return {
    currentSession,
    loading,
    error,
    createSession,
    loadSession,
    listSessions,
    deleteSession,
    saveSession,
    updateSessionInfo,
    updateRoundWait,
    addGroup,
    updateGroup,
    deleteGroup,
    uploadData,
    deleteDataFile,
    addIndicator,
    deleteIndicator,
    startAnalysis,
    pauseAnalysis,
    resumeAnalysis,
    continueRound,
    stopAnalysis,
    exportState,
    getReport,
    updateSession,
  }
}

// WebSocket hook
export function useWebSocket() {
  const { currentSession, addMessage, updateSession, setWsConnected, setMessages, addAgentError, setStateVersion } = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const connectingRef = useRef<boolean>(false)  // Prevent multiple simultaneous connect attempts
  const reconnectCountRef = useRef<number>(0)  // Track reconnection attempts for history recovery
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [connected, setConnected] = useState(false)

  const connect = useCallback(() => {
    // Use fresh state to avoid stale closure references
    const session = useAppStore.getState().currentSession
    if (!session) {
      devLog('WebSocket: No current session, skipping connect')
      return
    }

    // Close existing connection before opening a new one
    if (wsRef.current) {
      const state = wsRef.current.readyState
      if (state === WebSocket.CONNECTING || state === WebSocket.OPEN) {
        devLog('WebSocket: Closing existing connection before reconnecting')
        wsRef.current.close()
      }
      wsRef.current = null
    }

    if (connectingRef.current) {
      devLog('WebSocket: Connection already in progress, skipping')
      return
    }

    connectingRef.current = true
    const isReconnect = reconnectCountRef.current > 0

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${session.id}`
    const token = getToken()
    const wsProtocols = token ? [WS_AUTH_SUBPROTOCOL, token] : undefined

    devLog(`WebSocket: Connecting to ${wsUrl}${isReconnect ? ' (reconnect #' + reconnectCountRef.current + ')' : ''}`)

    const ws = new WebSocket(wsUrl, wsProtocols)
    wsRef.current = ws

    // Connection timeout: close and retry if onopen doesn't fire within 10 seconds
    const connectTimeout = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        devLog('WebSocket: Connection timeout (10s), closing')
        connectingRef.current = false
        ws.close()
      }
    }, WS_CONNECT_TIMEOUT_MS)

    ws.onopen = async () => {
      if (wsRef.current !== ws) return
      clearTimeout(connectTimeout)
      connectingRef.current = false
      setConnected(true)
      setWsConnected(true)
      devLog('WebSocket: Connected successfully')

      // Use fresh state to avoid stale closure
      const freshSession = useAppStore.getState().currentSession

      // On reconnect, restore message history from server
      if (isReconnect && freshSession) {
        try {
          devLog('WebSocket: Reconnect detected, loading message history...')
          // Mark that we're loading history to prevent race conditions
          const loadingSessionId = freshSession.id
          const data = await apiFetch<any>(`/session/${freshSession.id}/load`)

          // Verify session hasn't changed during fetch (race condition protection)
          const currentState = useAppStore.getState()
          if (currentState.currentSession?.id !== loadingSessionId) {
            devLog('WebSocket: Session changed during reconnect, skipping history merge')
            reconnectCountRef.current = 0
            return
          }

          if (data) {

            if (data.messages && Array.isArray(data.messages)) {
              // Convert server messages to frontend format
              const serverMessages: Message[] = data.messages.map(transformRawMessage)

              // Transactional merge: read+write inside setState so messages that
              // arrived via WebSocket during the async fetch are preserved.
              let mergedCount = 0
              let localOnlyCount = 0
              useAppStore.setState((state) => {
                const serverMsgIdSet = new Set(serverMessages.map(m => m.id))
                localOnlyCount = state.messages.filter(m => !serverMsgIdSet.has(m.id)).length

                const currentIdSet = new Set(state.messages.map((m: Message) => m.id))
                const newFromServer = serverMessages.filter((m) => !currentIdSet.has(m.id))
                if (newFromServer.length === 0) {
                  return state
                }
                mergedCount = newFromServer.length
                const merged = [...state.messages, ...newFromServer]
                  .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
                return { ...state, messages: merged }
              })
              devLog(`WebSocket: Restored ${serverMessages.length} messages (${mergedCount} new merged), ${localOnlyCount} local-only`)
            }

            // Also restore agent errors if present
            if (data.agent_errors && Array.isArray(data.agent_errors)) {
              restoreAgentErrors(data.agent_errors)
            }

            // Sync full session state from server using shared transform
            const { sessionUpdates, storeUpdates } = transformLoadResponse(data)
            const currentSess = useAppStore.getState().currentSession
            if (currentSess && currentSess.id === loadingSessionId) {
              useAppStore.getState().updateSession(sessionUpdates)
              applyStoreUpdates(storeUpdates)
            }
          }
        } catch (e: any) {
          // Session no longer exists on server - clear stale session to stop reconnect loop
          if (e?.message?.includes('not found') || e?.message?.includes('404')) {
            console.warn('WebSocket: Session not found on server (404), clearing stale session')
            useAppStore.getState().setCurrentSession(null)
            useAppStore.getState().clearMessages()
            ws.close()
            return
          }
          console.error('WebSocket: Failed to restore message history:', e)
        }
      }

      // Reset reconnect count on successful connection
      reconnectCountRef.current = 0
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        // Server heartbeat ping — ignore silently
        if (data.type === 'ping') return

        // Handle error messages that indicate session/config issues
        if (data.error) {
          console.error('WebSocket: Server error:', data.error)
          // If session not found, clear stale session to stop reconnect loop
          // Match specific backend error pattern: "Session {id} not found" or "Session not found"
          if (typeof data.error === 'string' && /session\b.*\bnot found/i.test(data.error)) {
            const currentSessionId = useAppStore.getState().currentSession?.id
            if (currentSessionId) {
              console.warn(`WebSocket: Session ${currentSessionId} not found on server, clearing stale session`)
              useAppStore.getState().setCurrentSession(null)
              useAppStore.getState().clearMessages()
            }
          }
          return
        }

        // Handle state_sync messages - version control and state synchronization
        if (data.type === 'state_sync') {
          const syncContent = data.content
          const currentVersion = useAppStore.getState().stateVersion

          // Check for version conflict - reload full session state from server
          if (syncContent.version < currentVersion) {
            console.warn('WebSocket: Received outdated state sync, current:', currentVersion, 'received:', syncContent.version, '. Reloading session state.')
            const sessionId = useAppStore.getState().currentSession?.id
            if (sessionId) {
              fetch(`${API_BASE}/session/${sessionId}/load`, { headers: authHeaders() })
                .then(resp => resp.ok ? resp.json() : null)
                .then(reloadData => {
                  if (reloadData) {
                    const { sessionUpdates, storeUpdates } = transformLoadResponse(reloadData)
                    useAppStore.getState().updateSession(sessionUpdates)
                    applyStoreUpdates(storeUpdates)
                  }
                })
                .catch(reloadErr => console.error('WebSocket: Failed to reload session state:', reloadErr))
            }
          } else {
            // Update state version and session status
            useAppStore.getState().setStateVersion(syncContent.version)

            // Handle LLM failure state (single source of truth — state_sync only)
            if (syncContent.change_type === 'llm_failure' && syncContent.llm_failure_info) {
              const info = syncContent.llm_failure_info
              const store = useAppStore.getState()
              store.setLlmFailureInfo({
                agent: info.agent,
                model: info.model,
                llmServer: info.llm_server,
                error: info.error,
                nodeName: info.node_name,
                generation: info.generation,
                reason: info.reason,
              })
              store.setShowLlmTimeoutDialog(true)
            } else if (
              syncContent.change_type === 'agent_disabled' &&
              syncContent.agent_disabled_info
            ) {
              const info = syncContent.agent_disabled_info
              const store = useAppStore.getState()
              store.setLlmFailureInfo({
                agent: info.agent,
                model: '',
                llmServer: '',
                error: `Agent "${info.agent}" is disabled but required by step "${info.node_name}".`,
                nodeName: info.node_name,
                generation: info.generation,
                reason: 'agent_disabled',
              })
              store.setShowLlmTimeoutDialog(true)
            } else if (syncContent.change_type === 'llm_retry') {
              useAppStore.getState().clearLlmFailureState()
            } else if (syncContent.change_type === 'agent_resumed') {
              useAppStore.getState().clearLlmFailureState()
            } else if (
              syncContent.change_type === 'species_input' ||
              syncContent.change_type === 'gene_species_mismatch'
            ) {
              if (syncContent.species_input_info) {
                const store = useAppStore.getState()
                const info = syncContent.species_input_info
                store.setSpeciesInputInfo({
                  source: info.source,
                  suggestions: Array.isArray(info.suggestions) ? info.suggestions : [],
                  context: info.context || info.pi_assessment || '',
                  species: info.species,
                  sampleGenes: info.sample_genes,
                  piAssessment: info.pi_assessment,
                  likelySpecies: info.likely_species,
                })
                store.setShowSpeciesInputDialog(true)
              }
            } else if (
              syncContent.change_type === 'species_resolved' ||
              syncContent.change_type === 'species_auto_skipped'
            ) {
              useAppStore.getState().clearSpeciesInputState()
            }

            useAppStore.getState().updateSession({
              status: syncContent.status,
              currentRound: syncContent.current_round,
              ...(syncContent.analysis_phase ? { analysisPhase: syncContent.analysis_phase } : {}),
            })
          }
          return  // Don't add state_sync to message stream
        }

        // Handle message_history - batch restore messages (e.g. reconnect to completed session)
        if (data.type === 'message_history') {
          if (data.content?.messages && Array.isArray(data.content.messages)) {
            const serverMessages = data.content.messages.map(transformRawMessage)
            useAppStore.getState().setMessages(serverMessages)
            devLog(`WebSocket: Restored ${serverMessages.length} messages from history`)
          }
          return  // Don't add message_history itself to the message stream
        }

        // Handle parallel_errors messages - structured error tracking
        if (data.type === 'parallel_errors') {
          const errorContent = data.content
          if (errorContent.errors && Array.isArray(errorContent.errors)) {
            for (const err of errorContent.errors) {
              useAppStore.getState().addAgentError({
                id: err.id,
                agent: err.agent as AgentType,
                errorType: err.error_type || 'execution_error',
                message: err.message,
                details: err.details,
                round: err.round,
                timestamp: data.timestamp || new Date().toISOString(),
                resolved: false
              })
            }
          }
          // Also add to message stream for visibility
        }

        if (data.type === 'snapshot_count') {
          const count = data.content?.count ?? 0
          useAppStore.getState().setSnapshotCount(count)

          if (count >= SNAPSHOT_CLEANUP_THRESHOLD && count % SNAPSHOT_REMINDER_INTERVAL === 0) {
            useAppStore.getState().setShowSnapshotCleanupReminder(true)
          }
          return  // Don't add to MessageStream
        }

        // Handle chat_result - final response from chat delegation
        if (data.type === 'chat_result') {
          // Refresh full chat history from backend to ensure consistency
          const sessionId = useAppStore.getState().currentSession?.id
          if (sessionId) {
            apiFetch<{ messages: ChatMessage[] }>(`/session/${sessionId}/chat`)
              .then((res) => {
                useAppStore.getState().setChatMessages(res.messages || [])
              })
              .catch((e) => { devLog('Chat refresh after delegation failed:', e) })
          }
          useAppStore.getState().setChatLoading(false)
          useAppStore.getState().setChatDelegationActive(false)
          useAppStore.getState().setChatDelegationPlan(null)
          return  // Don't add to MessageStream
        }

        // Handle chat_cancelled - delegation was stopped by user
        if (data.type === 'chat_cancelled') {
          useAppStore.getState().setChatLoading(false)
          useAppStore.getState().setChatDelegationActive(false)
          useAppStore.getState().setChatDelegationPlan(null)
          useAppStore.getState().setChatAbortController(null)
          return  // Don't add to MessageStream
        }

        // Handle chat_error - delegation failed
        if (data.type === 'chat_error') {
          useAppStore.getState().addChatMessage({
            role: 'assistant',
            content: `Error: ${data.content?.error || 'Unknown error during analysis'}`,
            timestamp: new Date().toISOString(),
          })
          useAppStore.getState().setChatLoading(false)
          useAppStore.getState().setChatDelegationActive(false)
          useAppStore.getState().setChatDelegationPlan(null)
          return  // Don't add to MessageStream
        }

        // Handle chat_delegation progress - update delegation status AND show in MessageStream
        if (data.type === 'chat_delegation') {
          const delegContent = data.content
          if (delegContent?.status === 'started') {
            useAppStore.getState().setChatDelegationActive(true)
            useAppStore.getState().setChatDelegationPlan(delegContent.plan || null)
          }
          // Fall through to addMessage() so it shows in MessageStream
        }

        // Handle exploration_failed - exploration (Stage 1) did not complete,
        // most likely a data problem. Surface an actionable dialog so the user
        // can review/re-upload data and retry Stage 1, rather than silently
        // advancing rounds toward max_rounds.
        if (data.type === 'exploration_failed') {
          const c = data.content || {}
          const store = useAppStore.getState()
          store.setExplorationFailedInfo({
            round: c.round,
            reason: c.reason || 'incomplete',
            message: c.message || '',
          })
          store.setShowExplorationFailedDialog(true)
          return  // Don't add to MessageStream — the dialog is the UX
        }

        const message: Message = {
          id: data.id || generateId(),
          type: data.type || 'message',
          agent: data.agent as AgentType | undefined,
          content: data.content,
          timestamp: data.timestamp || new Date().toISOString(),
        }

        useAppStore.getState().addMessage(message)
        
        // Update session status based on message
        if (data.type === 'completed') {
          const updates: Partial<Session> = {
            status: 'completed' as const,
            finalSummary: data.content?.final_summary,
          }
          // Extract round_results if provided (for completed session reconnect)
          const roundResults = data.content?.round_results
          if (roundResults && Array.isArray(roundResults)) {
            updates.roundResults = transformRoundResults(roundResults)
          }
          useAppStore.getState().updateSession(updates)
        } else if (data.type === 'error' && !data.agent) {
          // Only mark as failed if it's a system error, not an agent error
          useAppStore.getState().updateSession({ status: 'failed' })
        }
        
        // Update currentRound from status messages
        if (data.type === 'status' && typeof data.content === 'string') {
          // Only match the analysis round marker ("Starting round X analysis").
          // Must NOT match other status text that mentions "round", e.g. the
          // planning dialogue's "Planning dialogue: round 3/5", which would
          // otherwise overwrite currentRound with the dialogue iteration count.
          const roundMatch = data.content.match(/Starting round\s+(\d+)/i)
          if (roundMatch) {
            const round = parseInt(roundMatch[1], 10)
            if (!isNaN(round)) {
              useAppStore.getState().updateSession({ currentRound: round })
            }
          }
        }
        
        // Update from result messages that contain round info
        if (data.type === 'result' && data.content?.round) {
          useAppStore.getState().updateSession({ currentRound: data.content.round })
        }

        // Handle round_summary messages to update roundResults
        if (data.type === 'round_summary' && data.content) {
          const roundData = data.content
          const latestSession = useAppStore.getState().currentSession
          if (latestSession) {
            const existingRounds = latestSession.roundResults || []
            // Check if this round already exists
            const existingIndex = existingRounds.findIndex((r: RoundResult) => r.round === roundData.round)

            const newRoundResult = {
              round: roundData.round,
              summary: roundData.summary || '',
              pathologyFindings: roundData.pathology_findings || [],
              bioinformaticsFindings: roundData.bioinformatics_findings || [],
              biochemistryFindings: roundData.biochemistry_findings || [],
              literatureFindings: roundData.literature_findings || [],
              crossModalCandidates: roundData.cross_modal_candidates || [],
            }

            let updatedRounds
            if (existingIndex >= 0) {
              // Update existing round
              updatedRounds = [...existingRounds]
              updatedRounds[existingIndex] = newRoundResult
            } else {
              // Add new round
              updatedRounds = [...existingRounds, newRoundResult]
            }

            useAppStore.getState().updateSession({
              roundResults: updatedRounds,
              currentRound: roundData.round,
              status: roundData.status === 'waiting_input' ? 'waiting_input' : latestSession.status,
            })
          }
        }
      } catch (e) {
        console.error('WebSocket: Failed to parse message:', e)
      }
    }
    
    ws.onerror = (error) => {
      clearTimeout(connectTimeout)
      connectingRef.current = false
      console.error('WebSocket: Error:', error)
    }

    ws.onclose = (event) => {
      if (wsRef.current !== ws) return
      clearTimeout(connectTimeout)
      connectingRef.current = false
      setConnected(false)
      setWsConnected(false)
      wsRef.current = null
      devLog('WebSocket: Disconnected, code:', event.code, 'reason:', event.reason)

      // Don't reconnect on auth failure (4401) or if no token
      if (event.code === 4401 || !getToken()) {
        devLog('WebSocket: Auth failure or no token, not reconnecting')
        return
      }

      reconnectCountRef.current += 1  // Track reconnect for history recovery

      // Auto-reconnect if session is still running (not manually disconnected)
      const latestSession = useAppStore.getState().currentSession
      if (latestSession &&
          (latestSession.status === 'running' || latestSession.status === 'waiting_input' || latestSession.status === 'waiting_annotation' || latestSession.status === 'waiting_plan_review') &&
          event.code !== 1000) {  // Not a normal close
        if (reconnectCountRef.current < 5) {  // Max 5 reconnect attempts
          const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current - 1), 10000)  // Exponential backoff, max 10s
          devLog(`WebSocket: Auto-reconnecting in ${delay}ms (attempt ${reconnectCountRef.current})...`)
          reconnectTimerRef.current = setTimeout(() => {
            reconnectTimerRef.current = null
            if (!wsRef.current && !connectingRef.current) {
              connect()
            }
          }, delay)
        } else {
          // Max reconnect attempts reached - notify user
          console.error('WebSocket: Max reconnection attempts reached, giving up')
          addMessage({
            id: `ws-error-${Date.now()}`,
            type: 'error',
            agent: undefined,
            content: 'WebSocket connection lost. Please refresh the page or click reconnect.',
            timestamp: new Date().toISOString()
          })
        }
      }
    }
  }, [currentSession, addMessage, updateSession, setWsConnected, setMessages, addAgentError, setStateVersion])
  
  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (wsRef.current) {
      devLog('WebSocket: Disconnecting manually')
      wsRef.current.close()
      wsRef.current = null
    }
    connectingRef.current = false
  }, [])
  
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])
  
  return { connect, disconnect, connected }
}

// Prompts hook
export function usePrompts() {
  const { currentSession } = useAppStore()
  const { prompts, setPrompt, models, setModel, temperatures, setTemperature } = useAppStore()
  
  const savePrompt = useCallback(async (agent: AgentType, prompt: string) => {
    if (!currentSession) return
    
    await apiFetch(`/session/${currentSession.id}/prompts`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        agent,
        prompt,
      }),
    })
    
    setPrompt(agent, prompt)
  }, [currentSession, setPrompt])
  
  const saveModel = useCallback(async (agent: AgentType, model: string) => {
    if (!currentSession) return
    
    await apiFetch(`/session/${currentSession.id}/models`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        agent,
        model,
      }),
    })
    
    setModel(agent, model)
  }, [currentSession, setModel])
  
  const saveTemperature = useCallback(async (agent: AgentType, temperature: number) => {
    if (!currentSession) return
    
    await apiFetch(`/session/${currentSession.id}/temperatures`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        agent,
        temperature,
      }),
    })
    
    setTemperature(agent, temperature)
  }, [currentSession, setTemperature])
  
  const loadPrompts = useCallback(async () => {
    if (!currentSession) return prompts
    
    try {
      const data = await apiFetch<Record<string, string>>(
        `/session/${currentSession.id}/prompts`
      )
      
      Object.entries(data).forEach(([agent, prompt]) => {
        setPrompt(agent as AgentType, prompt)
      })
      
      return data
    } catch {
      return prompts
    }
  }, [currentSession, prompts, setPrompt])
  
  return { prompts, setPrompt, savePrompt, models, setModel, saveModel, temperatures, setTemperature, saveTemperature, loadPrompts }
}

// User config hook
interface UserConfigStatus {
  configured: boolean
  has_api_key?: boolean
  llm_base_url?: string
  masked_api_key?: string
  has_proxy_url?: boolean
}

interface ValidateResult {
  valid: boolean
  uncertain?: boolean
  message?: string
}

export function useUserConfig() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getUserConfig = useCallback(async (): Promise<UserConfigStatus> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<UserConfigStatus>('/user/config', { method: 'GET' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const saveUserConfig = useCallback(async (
    api_key: string,
    llm_base_url: string,
    proxy_url: string
  ): Promise<void> => {
    setLoading(true)
    setError(null)
    try {
      await apiFetch<void>('/user/config', {
        method: 'POST',
        body: JSON.stringify({ api_key, llm_base_url, proxy_url: proxy_url || null }),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const getUserConfigStatus = useCallback(async (): Promise<UserConfigStatus> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<UserConfigStatus>('/user/config/status', { method: 'GET' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to get user config status')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const validateUserConfig = useCallback(async (
    api_key: string,
    llm_base_url: string,
    proxy_url: string
  ): Promise<ValidateResult> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<ValidateResult>('/user/config/validate', {
        method: 'POST',
        body: JSON.stringify({ api_key, llm_base_url, proxy_url: proxy_url || null }),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to validate user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  const validateSavedConfig = useCallback(async (): Promise<ValidateResult> => {
    setLoading(true)
    setError(null)
    try {
      return await apiFetch<ValidateResult>('/user/config/validate-saved', { method: 'POST' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to validate saved config')
      return { valid: false, message: 'Failed to reach validation endpoint', uncertain: true }
    } finally {
      setLoading(false)
    }
  }, [])

  const deleteUserConfig = useCallback(async (): Promise<void> => {
    setLoading(true)
    setError(null)
    try {
      await apiFetch<void>('/user/config', { method: 'DELETE' })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete user config')
      throw e
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    loading, error,
    getUserConfig, saveUserConfig, getUserConfigStatus,
    validateUserConfig, validateSavedConfig, deleteUserConfig,
  }
}
