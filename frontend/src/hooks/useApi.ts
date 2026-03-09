import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore, Message, Session, AgentType } from '../stores/appStore'

const API_BASE = '/api'

// Generic fetch wrapper (exported for direct use)
export async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || 'API request failed')
  }
  
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
    } catch {
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

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return { apiConfig, setApiConfig, saveConfig, checkStatus, validateApiKey, loading, error, clearError }
}

// Session hook
export function useSession() {
  const { currentSession, setCurrentSession, updateSession, setMessages, clearMessages } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const createSession = useCallback(async (instruction: string, name?: string) => {
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
        body: JSON.stringify({ instruction, session_name: name }),
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
    setError(null)
    
    try {
      const data = await apiFetch<any>(`/session/${sessionId}/load`)

      // Transform round_results from snake_case to camelCase with defensive null checks
      const transformedRoundResults = (data.round_results || []).map((r: any) => {
        if (!r) return null
        return {
          round: r.round ?? 0,
          summary: r.pi_summary || r.summary || '',
          pathologyFindings: Array.isArray(r.pathology_findings) ? r.pathology_findings : [],
          bioinformaticsFindings: Array.isArray(r.bioinformatics_findings) ? r.bioinformatics_findings : [],
          biochemistryFindings: Array.isArray(r.biochemistry_findings) ? r.biochemistry_findings : [],
          literatureFindings: Array.isArray(r.literature_findings) ? r.literature_findings : [],
        }
      }).filter((r: any) => r !== null)

      // Validate status is a known value
      const validStatuses = ['pending', 'running', 'paused', 'waiting_input', 'completed', 'failed', 'generating_report'] as const
      const status = validStatuses.includes(data.status) ? data.status : 'pending'

      const session: Session = {
        id: data.session_id,
        name: data.name,
        status: status as Session['status'],
        currentRound: data.current_round || 0,
        createdAt: data.created_at || new Date().toISOString(),
        instruction: data.initial_instruction || '',
        treatmentGroups: (data.treatment_groups || []).map((g: any) => ({
          id: g.id, name: g.name, description: g.description || '',
          sampleCount: g.sampleCount ?? g.sample_count ?? 1, order: g.order ?? 0,
        })),
        dataFiles: (data.data_files || []).map((f: any) => ({
          id: f.id, filename: f.filename, filePath: f.filePath ?? f.file_path ?? '',
          section: f.section, dataType: f.dataType ?? f.data_type ?? '',
          groupId: f.groupId ?? f.group_id, sampleNumber: f.sampleNumber ?? f.sample_number,
          description: f.description, timestamp: f.timestamp || '',
        })),
        biochemistryIndicators: (data.biochemistry_indicators || []).map((r: any) => ({
          id: r.id, indicator: r.indicator, value: r.value, unit: r.unit,
          referenceRange: r.referenceRange ?? r.reference_range,
          interpretation: r.interpretation,
          groupId: r.groupId ?? r.group_id, sampleNumber: r.sampleNumber ?? r.sample_number,
        })),
        roundResults: transformedRoundResults,
        finalSummary: data.final_summary,
        // Publication outputs (snake_case → camelCase)
        publicationFigures: (data.publication_figures || []).map((f: any) => ({
          id: f.id,
          name: f.name,
          description: f.description,
          figureType: f.figure_type || f.figureType || '',
          pngPath: f.png_path || f.pngPath || '',
          pdfPath: f.pdf_path || f.pdfPath || '',
          pngBase64: f.png_base64 || f.pngBase64,
          reviewStatus: f.review_status || f.reviewStatus || 'pending',
          piFeedback: f.pi_feedback || f.piFeedback,
          revisionCount: f.revision_count ?? f.revisionCount ?? 0,
          createdAt: f.created_at || f.createdAt || '',
        })),
        publicationTables: (data.publication_tables || []).map((t: any) => ({
          id: t.id,
          name: t.name,
          description: t.description,
          csvPath: t.csv_path || t.csvPath || '',
          columns: Array.isArray(t.columns) ? t.columns : [],
          previewData: Array.isArray(t.preview_data) ? t.preview_data : (t.previewData || []),
          reviewStatus: t.review_status || t.reviewStatus || 'pending',
          piFeedback: t.pi_feedback || t.piFeedback,
          createdAt: t.created_at || t.createdAt || '',
        })),
        graphicAbstract: data.graphic_abstract ? {
          id: data.graphic_abstract.id,
          prompt: data.graphic_abstract.prompt || '',
          imageBase64: data.graphic_abstract.image_base64 || data.graphic_abstract.imageBase64 || '',
          createdAt: data.graphic_abstract.created_at || data.graphic_abstract.createdAt || '',
        } : undefined,
        integratedReport: data.integrated_report || undefined,
      }

      setCurrentSession(session)

      // Restore state version
      if (data.state_version !== undefined) {
        useAppStore.getState().setStateVersion(data.state_version)
      }

      // Restore agent errors
      useAppStore.getState().clearAgentErrors()
      if (data.agent_errors && Array.isArray(data.agent_errors)) {
        for (const err of data.agent_errors) {
          if (!err.resolved) {
            useAppStore.getState().addAgentError({
              id: err.id,
              agent: err.agent as AgentType,
              errorType: err.error_type,
              message: err.message,
              details: err.details,
              round: err.round_number,
              timestamp: err.timestamp,
              resolved: false
            })
          }
        }
      }
      
      // Load realtime messages history (WebSocket messages)
      if (data.messages && Array.isArray(data.messages)) {
        const messages: Message[] = data.messages.map((msg: any) => ({
          id: msg.id || crypto.randomUUID(),
          type: msg.type || 'message',
          agent: msg.agent as AgentType | undefined,
          content: msg.content,
          timestamp: msg.timestamp || new Date().toISOString(),
        }))
        setMessages(messages)
      } else {
        clearMessages()
      }
      
      return session
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load session')
      throw e
    } finally {
      setLoading(false)
    }
  }, [setCurrentSession, setMessages, clearMessages])
  
  const listSessions = useCallback(async () => {
    try {
      const data = await apiFetch<{ sessions: any[] }>('/sessions')
      return data.sessions
    } catch (e) {
      console.error('Failed to list sessions:', e)
      return []
    }
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
    const body: any = {}
    if (updates.name !== undefined) body.name = updates.name
    if (updates.instruction !== undefined) body.instruction = updates.instruction
    const response = await apiFetch<any>(`/session/${session.id}/info`, {
      method: 'PUT',
      body: JSON.stringify(body),
    })
    useAppStore.getState().updateSession({
      name: response.data.name,
      instruction: response.data.instruction,
    })
    return response
  }, [])
  
  // Ensure session exists on server (handles server restart)
  const ensureSession = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')
    
    try {
      const response = await fetch(
        `${API_BASE}/session/${currentSession.id}/ensure`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instruction: currentSession.instruction,
            session_name: currentSession.name,
          }),
        }
      )
      
      if (!response.ok) {
        throw new Error('Failed to ensure session')
      }
      
      const data = await response.json()
      if (data.data?.created) {
        console.log('Session recreated after server restart')
      }
      return data
    } catch (e) {
      console.error('Failed to ensure session:', e)
      throw e
    }
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

    const body: any = {}
    if (updates.name !== undefined) body.name = updates.name
    if (updates.description !== undefined) body.description = updates.description
    if (updates.sampleCount !== undefined) body.sample_count = updates.sampleCount

    const response = await apiFetch<any>(`/session/${session.id}/groups/${groupId}`, {
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

    const response = await fetch(
      `${API_BASE}/session/${session.id}/upload/data`,
      { method: 'POST', body: formData, signal }
    )
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(err.detail || 'Upload failed')
    }
    const data = await response.json()

    // Refresh dataFiles from server to ensure frontend state matches backend
    try {
      const refreshData = await apiFetch<any>(`/session/${session.id}/load`)
      const current = useAppStore.getState().currentSession
      if (current && current.id === session.id) {
        useAppStore.getState().updateSession({
          dataFiles: (refreshData.data_files || []).map((f: any) => ({
            id: f.id, filename: f.filename, filePath: f.filePath ?? f.file_path ?? '',
            section: f.section, dataType: f.dataType ?? f.data_type ?? '',
            groupId: f.groupId ?? f.group_id, sampleNumber: f.sampleNumber ?? f.sample_number,
            description: f.description, timestamp: f.timestamp || '',
          })),
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
    
    // Sync all models to backend before starting
    for (const agent of ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature', 'painter'] as const) {
      try {
        await apiFetch(`/session/${currentSession.id}/models`, {
          method: 'POST',
          body: JSON.stringify({
            session_id: currentSession.id,
            agent,
            model: models[agent],
          }),
        })
      } catch (e) {
        console.error(`Failed to sync model for ${agent}:`, e)
      }
    }
    
    // Sync all prompts to backend before starting
    for (const agent of ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature', 'painter'] as const) {
      try {
        await apiFetch(`/session/${currentSession.id}/prompts`, {
          method: 'POST',
          body: JSON.stringify({
            session_id: currentSession.id,
            agent,
            prompt: prompts[agent],
          }),
        })
      } catch (e) {
        console.error(`Failed to sync prompt for ${agent}:`, e)
      }
    }
    
    // Sync all temperatures to backend before starting
    for (const agent of ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature', 'painter'] as const) {
      try {
        await apiFetch(`/session/${currentSession.id}/temperatures`, {
          method: 'POST',
          body: JSON.stringify({
            session_id: currentSession.id,
            agent,
            temperature: temperatures[agent],
          }),
        })
      } catch (e) {
        console.error(`Failed to sync temperature for ${agent}:`, e)
      }
    }
    
    // Sync max rounds to backend before starting
    const maxRounds = useAppStore.getState().maxRounds
    try {
      await apiFetch(`/session/${currentSession.id}/max-rounds`, {
        method: 'POST',
        body: JSON.stringify({
          session_id: currentSession.id,
          max_rounds: maxRounds,
        }),
      })
    } catch (e) {
      console.error('Failed to sync max rounds:', e)
    }

    // Sync round wait seconds to backend before starting
    const roundWaitSeconds = useAppStore.getState().roundWaitSeconds
    try {
      await apiFetch(`/session/${currentSession.id}/round-wait`, {
        method: 'POST',
        body: JSON.stringify({ round_wait_seconds: roundWaitSeconds }),
      })
    } catch (e) {
      console.error('Failed to sync round wait seconds:', e)
    }

    // Sync literature config to backend before starting
    const { literatureMode, googleApiKey, pubmedEmail, literatureMaxSearchRounds, deepResearchModel } = useAppStore.getState()
    try {
      await apiFetch(`/session/${currentSession.id}/literature-config`, {
        method: 'POST',
        body: JSON.stringify({
          mode: literatureMode,
          deep_research_model: deepResearchModel,
          google_api_key: googleApiKey,
          pubmed_email: pubmedEmail,
          max_search_rounds: literatureMaxSearchRounds,
        }),
      })
    } catch (e) {
      console.error('Failed to sync literature config:', e)
    }

    await apiFetch(`/session/${currentSession.id}/run`, { method: 'POST' })
    updateSession({ status: 'running' })
  }, [currentSession, updateSession])
  
  const pauseAnalysis = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/pause`, { method: 'POST' })
    updateSession({ status: 'paused' })
  }, [currentSession, updateSession])
  
  const resumeAnalysis = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')

    await apiFetch(`/session/${currentSession.id}/resume`, { method: 'POST' })
    updateSession({ status: 'running' })
  }, [currentSession, updateSession])

  const continueRound = useCallback(async () => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    await apiFetch(`/session/${session.id}/resume`, { method: 'POST' })
    useAppStore.getState().updateSession({ status: 'running' })
  }, [])
  
  const exportState = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')
    
    // Download as zip file
    const response = await fetch(`${API_BASE}/session/${currentSession.id}/export`, {
      method: 'POST'
    })
    
    if (!response.ok) {
      throw new Error('Failed to export state')
    }
    
    // Create blob and download
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${currentSession.id}_export.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
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
  const [connected, setConnected] = useState(false)

  // Generate UUID compatible with all browsers (including non-HTTPS)
  const generateId = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0
      const v = c === 'x' ? r : (r & 0x3 | 0x8)
      return v.toString(16)
    })
  }

  const connect = useCallback(() => {
    if (!currentSession) {
      console.log('WebSocket: No current session, skipping connect')
      return
    }

    // Check all connection states
    if (wsRef.current) {
      const state = wsRef.current.readyState
      if (state === WebSocket.CONNECTING || state === WebSocket.OPEN) {
        console.log('WebSocket: Already connected or connecting, skipping')
        return
      }
      // If CLOSING or CLOSED, clear the ref
      wsRef.current = null
    }

    if (connectingRef.current) {
      console.log('WebSocket: Connection already in progress, skipping')
      return
    }

    connectingRef.current = true
    const isReconnect = reconnectCountRef.current > 0

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${currentSession.id}`

    console.log(`WebSocket: Connecting to ${wsUrl}${isReconnect ? ' (reconnect #' + reconnectCountRef.current + ')' : ''}`)

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = async () => {
      connectingRef.current = false
      setConnected(true)
      setWsConnected(true)
      console.log('WebSocket: Connected successfully')

      // Use fresh state to avoid stale closure
      const freshSession = useAppStore.getState().currentSession

      // On reconnect, restore message history from server
      if (isReconnect && freshSession) {
        try {
          console.log('WebSocket: Reconnect detected, loading message history...')
          // Mark that we're loading history to prevent race conditions
          const loadingSessionId = freshSession.id
          const response = await fetch(`${API_BASE}/session/${freshSession.id}/load`)

          // Verify session hasn't changed during fetch (race condition protection)
          const currentState = useAppStore.getState()
          if (currentState.currentSession?.id !== loadingSessionId) {
            console.log('WebSocket: Session changed during reconnect, skipping history merge')
            reconnectCountRef.current = 0
            return
          }

          if (response.ok) {
            const data = await response.json()

            if (data.messages && Array.isArray(data.messages)) {
              // Get current local messages
              const localMessages = currentState.messages

              // Convert server messages to frontend format
              const serverMessages: Message[] = data.messages.map((msg: any) => ({
                id: msg.id || generateId(),
                type: msg.type || 'message',
                agent: msg.agent as AgentType | undefined,
                content: msg.content,
                timestamp: msg.timestamp || new Date().toISOString(),
              }))

              // Create composite key for robust deduplication (handles ID regeneration on server restart)
              // Include content fingerprint to avoid collision when same agent/type/timestamp
              const createMsgKey = (m: Message) => {
                const contentStr = typeof m.content === 'string'
                  ? m.content.slice(0, 80)
                  : JSON.stringify(m.content).slice(0, 80)
                return `${m.agent || 'system'}-${m.type}-${m.timestamp}-${contentStr}`
              }
              const serverMsgKeySet = new Set(serverMessages.map(createMsgKey))
              const serverMsgIdSet = new Set(serverMessages.map(m => m.id))

              // Filter local messages that don't exist on server (by composite key OR id)
              const localOnlyMessages = localMessages.filter(m =>
                !serverMsgIdSet.has(m.id) && !serverMsgKeySet.has(createMsgKey(m))
              )

              // Combine and sort by timestamp
              const mergedMessages = [...serverMessages, ...localOnlyMessages]
                .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())

              setMessages(mergedMessages)
              console.log(`WebSocket: Restored ${serverMessages.length} messages, merged ${localOnlyMessages.length} local-only`)
            }

            // Also restore agent errors if present
            if (data.agent_errors && Array.isArray(data.agent_errors)) {
              useAppStore.getState().clearAgentErrors()
              for (const err of data.agent_errors) {
                if (!err.resolved) {
                  useAppStore.getState().addAgentError({
                    id: err.id,
                    agent: err.agent,
                    errorType: err.error_type,
                    message: err.message,
                    details: err.details,
                    round: err.round_number,
                    timestamp: err.timestamp,
                    resolved: false
                  })
                }
              }
            }
          } else if (response.status === 404) {
            // Session no longer exists on server - clear stale session to stop reconnect loop
            console.warn('WebSocket: Session not found on server (404), clearing stale session')
            useAppStore.getState().setCurrentSession(null)
            useAppStore.getState().clearMessages()
            ws.close()
            return
          }
        } catch (e) {
          console.error('WebSocket: Failed to restore message history:', e)
        }
      }

      // Reset reconnect count on successful connection
      reconnectCountRef.current = 0
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

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

          // Check for version conflict
          if (syncContent.version < currentVersion) {
            console.warn('WebSocket: Received outdated state sync, current:', currentVersion, 'received:', syncContent.version)
            // Could trigger full state reload here if needed
          } else {
            // Update state version and session status
            setStateVersion(syncContent.version)
            updateSession({
              status: syncContent.status,
              currentRound: syncContent.current_round
            })
          }
          return  // Don't add state_sync to message stream
        }

        // Handle message_history - batch restore messages (e.g. reconnect to completed session)
        if (data.type === 'message_history') {
          if (data.content?.messages && Array.isArray(data.content.messages)) {
            const serverMessages: Message[] = data.content.messages.map((msg: any) => ({
              id: msg.id || generateId(),
              type: msg.type || 'message',
              agent: msg.agent as AgentType | undefined,
              content: msg.content,
              timestamp: msg.timestamp || new Date().toISOString(),
            }))
            setMessages(serverMessages)
            console.log(`WebSocket: Restored ${serverMessages.length} messages from history`)
          }
          return  // Don't add message_history itself to the message stream
        }

        // Handle parallel_errors messages - structured error tracking
        if (data.type === 'parallel_errors') {
          const errorContent = data.content
          if (errorContent.errors && Array.isArray(errorContent.errors)) {
            for (const err of errorContent.errors) {
              addAgentError({
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

        // Handle chat_result - final response from chat delegation
        if (data.type === 'chat_result') {
          // Refresh full chat history from backend to ensure consistency
          const sessionId = useAppStore.getState().currentSession?.id
          if (sessionId) {
            apiFetch<{ messages: any[] }>(`/session/${sessionId}/chat`)
              .then((res) => {
                useAppStore.getState().setChatMessages(res.messages || [])
              })
              .catch(() => {})
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

        const message: Message = {
          id: data.id || generateId(),
          type: data.type || 'message',
          agent: data.agent as AgentType | undefined,
          content: data.content,
          timestamp: data.timestamp || new Date().toISOString(),
        }

        addMessage(message)
        
        // Update session status based on message
        if (data.type === 'completed') {
          const updates: Partial<Session> = {
            status: 'completed' as const,
            finalSummary: data.content?.final_summary,
          }
          // Extract round_results if provided (for completed session reconnect)
          const roundResults = data.content?.round_results
          if (roundResults && Array.isArray(roundResults)) {
            updates.roundResults = roundResults.map((r: any) => ({
              round: r.round ?? 0,
              summary: r.pi_summary || r.summary || '',
              pathologyFindings: Array.isArray(r.pathology_findings) ? r.pathology_findings : [],
              bioinformaticsFindings: Array.isArray(r.bioinformatics_findings) ? r.bioinformatics_findings : [],
              biochemistryFindings: Array.isArray(r.biochemistry_findings) ? r.biochemistry_findings : [],
              literatureFindings: Array.isArray(r.literature_findings) ? r.literature_findings : [],
            }))
          }
          updateSession(updates)
        } else if (data.type === 'error' && !data.agent) {
          // Only mark as failed if it's a system error, not an agent error
          updateSession({ status: 'failed' })
        }
        
        // Update currentRound from status messages
        if (data.type === 'status' && typeof data.content === 'string') {
          // Parse "Starting round X analysis" or similar messages
          const roundMatch = data.content.match(/[Rr]ound\s+(\d+)/i)
          if (roundMatch) {
            const round = parseInt(roundMatch[1], 10)
            if (!isNaN(round)) {
              updateSession({ currentRound: round })
            }
          }
        }
        
        // Update from result messages that contain round info
        if (data.type === 'result' && data.content?.round) {
          updateSession({ currentRound: data.content.round })
        }

        // Handle round_summary messages to update roundResults
        if (data.type === 'round_summary' && data.content) {
          const roundData = data.content
          const latestSession = useAppStore.getState().currentSession
          if (latestSession) {
            const existingRounds = latestSession.roundResults || []
            // Check if this round already exists
            const existingIndex = existingRounds.findIndex((r: any) => r.round === roundData.round)

            const newRoundResult = {
              round: roundData.round,
              summary: roundData.summary || '',
              pathologyFindings: roundData.pathology_findings || [],
              bioinformaticsFindings: roundData.bioinformatics_findings || [],
              biochemistryFindings: roundData.biochemistry_findings || [],
              literatureFindings: roundData.literature_findings || [],
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

            updateSession({
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
      connectingRef.current = false
      console.error('WebSocket: Error:', error)
    }
    
    ws.onclose = (event) => {
      connectingRef.current = false
      setConnected(false)
      setWsConnected(false)
      wsRef.current = null
      reconnectCountRef.current += 1  // Track reconnect for history recovery
      console.log('WebSocket: Disconnected, code:', event.code, 'reason:', event.reason)

      // Auto-reconnect if session is still running (not manually disconnected)
      const latestSession = useAppStore.getState().currentSession
      if (latestSession &&
          (latestSession.status === 'running' || latestSession.status === 'waiting_input') &&
          event.code !== 1000) {  // Not a normal close
        if (reconnectCountRef.current < 5) {  // Max 5 reconnect attempts
          const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current - 1), 10000)  // Exponential backoff, max 10s
          console.log(`WebSocket: Auto-reconnecting in ${delay}ms (attempt ${reconnectCountRef.current})...`)
          setTimeout(() => {
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
    if (wsRef.current) {
      console.log('WebSocket: Disconnecting manually')
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
