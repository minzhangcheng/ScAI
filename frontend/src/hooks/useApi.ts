import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore, Message, Session, AgentType } from '../stores/appStore'

const API_BASE = '/api'

// Generic fetch wrapper
async function apiFetch<T>(
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
          google_api_key: apiConfig.googleApiKey,
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
        has_google: boolean
      }>('/config/status')
      return status
    } catch {
      return { configured: false, has_api_key: false, has_google: false }
    }
  }, [])
  
  return { apiConfig, setApiConfig, saveConfig, checkStatus, loading, error }
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
        pathologyImages: [],
        biochemistryResults: [],
        biochemistryFiles: [],
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
      
      const session: Session = {
        id: data.session_id,
        name: data.name,
        status: data.status as any,
        currentRound: data.current_round || 0,
        createdAt: data.created_at || new Date().toISOString(),
        instruction: data.initial_instruction || '',
        pathologyImages: data.pathology_images || [],
        biochemistryResults: data.biochemistry_results || [],
        biochemistryFiles: data.biochemistry_files || [],
        controlSeqPath: data.control_seq_path,
        treatmentSeqPath: data.treatment_seq_path,
        controlLoomPath: data.control_loom_path,
        treatmentLoomPath: data.treatment_loom_path,
        roundResults: data.round_results || [],
        finalSummary: data.final_summary,
      }
      
      setCurrentSession(session)
      
      // Load messages history
      if (data.messages && Array.isArray(data.messages)) {
        const messages = data.messages.map((msg: any) => ({
          agent: msg.agent,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          metadata: msg.metadata || {}
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

  const uploadPathology = useCallback(async (file: File, imageType: string) => {
    if (!currentSession) throw new Error('No active session')
    
    // Ensure session exists first
    await ensureSession()
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('image_type', imageType)
    
    const response = await fetch(
      `${API_BASE}/session/${currentSession.id}/upload/pathology`,
      { method: 'POST', body: formData }
    )
    
    if (!response.ok) {
      throw new Error('Failed to upload pathology image')
    }
    
    const data = await response.json()
    
    updateSession({
      pathologyImages: [
        ...currentSession.pathologyImages,
        {
          id: data.data.image_id,
          filename: file.name,
          imageType: imageType,
          preview: URL.createObjectURL(file),
        },
      ],
    })
    
    return data
  }, [currentSession, updateSession, ensureSession])
  
  const uploadSnRNASeq = useCallback(async (file: File, group: 'control' | 'treatment') => {
    if (!currentSession) throw new Error('No active session')
    
    // Ensure session exists first
    await ensureSession()
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('group', group)
    
    const response = await fetch(
      `${API_BASE}/session/${currentSession.id}/upload/snrna_seq`,
      { method: 'POST', body: formData }
    )
    
    if (!response.ok) {
      throw new Error('Failed to upload snRNA-seq data')
    }
    
    const data = await response.json()
    
    // Update the appropriate path based on group
    if (group === 'control') {
      updateSession({ controlSeqPath: data.data.path })
    } else {
      updateSession({ treatmentSeqPath: data.data.path })
    }
    
    return data
  }, [currentSession, updateSession, ensureSession])
  
  const uploadLoom = useCallback(async (file: File, group: 'control' | 'treatment') => {
    if (!currentSession) throw new Error('No active session')
    
    // Ensure session exists first
    await ensureSession()
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('group', group)
    
    const response = await fetch(
      `${API_BASE}/session/${currentSession.id}/upload/loom`,
      { method: 'POST', body: formData }
    )
    
    if (!response.ok) {
      throw new Error('Failed to upload loom file')
    }
    
    const data = await response.json()
    
    // Update the appropriate path based on group
    if (group === 'control') {
      updateSession({ controlLoomPath: data.data.path })
    } else {
      updateSession({ treatmentLoomPath: data.data.path })
    }
    
    return data
  }, [currentSession, updateSession, ensureSession])
  
  const addBiochemistry = useCallback(async (
    indicator: string,
    value: number,
    unit: string
  ) => {
    if (!currentSession) throw new Error('No active session')
    
    const response = await apiFetch<any>(`/session/${currentSession.id}/biochemistry`, {
      method: 'POST',
      body: JSON.stringify({
        session_id: currentSession.id,
        indicator,
        value,
        unit,
      }),
    })
    
    updateSession({
      biochemistryResults: [
        ...currentSession.biochemistryResults,
        {
          id: response.data.result_id,
          indicator,
          value,
          unit,
        },
      ],
    })
    
    return response
  }, [currentSession, updateSession])
  
  const uploadBiochemistryFile = useCallback(async (
    file: File,
    description?: string
  ) => {
    if (!currentSession) throw new Error('No active session')
    
    const formData = new FormData()
    formData.append('file', file)
    if (description) {
      formData.append('description', description)
    }
    
    const response = await fetch(
      `${API_BASE}/session/${currentSession.id}/upload/biochemistry`,
      {
        method: 'POST',
        body: formData,
      }
    )
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
      throw new Error(error.detail || 'Upload failed')
    }
    
    const data = await response.json()
    
    // Get latest session state to avoid closure issues
    const latestSession = useAppStore.getState().currentSession
    const existingFiles = latestSession?.biochemistryFiles || []
    
    const newFile = {
      id: data.data.file_id,
      filename: file.name,
      fileType: data.data.file_type as 'image' | 'file',
    }
    
    // Update session with new file using store's method directly
    useAppStore.getState().updateSession({
      biochemistryFiles: [...existingFiles, newFile],
    })
    
    return data
  }, [currentSession])
  
  const startAnalysis = useCallback(async () => {
    if (!currentSession) throw new Error('No active session')
    
    // Get current models, prompts, and temperatures from store
    const { models, prompts, temperatures } = useAppStore.getState()
    
    // Sync all models to backend before starting
    for (const agent of ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature'] as const) {
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
    for (const agent of ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature'] as const) {
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
    for (const agent of ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature'] as const) {
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
  
  // Delete functions
  const deletePathologyImage = useCallback(async (imageId: string) => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/pathology/${imageId}`, {
      method: 'DELETE',
    })
    
    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      pathologyImages: latestSession?.pathologyImages.filter(img => img.id !== imageId) || [],
    })
  }, [currentSession])
  
  const deleteBiochemistryFile = useCallback(async (fileId: string) => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/biochemistry/file/${fileId}`, {
      method: 'DELETE',
    })
    
    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      biochemistryFiles: latestSession?.biochemistryFiles?.filter(f => f.id !== fileId) || [],
    })
  }, [currentSession])
  
  const deleteBiochemistryResult = useCallback(async (resultId: string) => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/biochemistry/result/${resultId}`, {
      method: 'DELETE',
    })
    
    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      biochemistryResults: latestSession?.biochemistryResults.filter(r => r.id !== resultId) || [],
    })
  }, [currentSession])
  
  const deleteSequencing = useCallback(async (group: 'control' | 'treatment') => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/sequencing/${group}`, {
      method: 'DELETE',
    })
    
    if (group === 'control') {
      useAppStore.getState().updateSession({ controlSeqPath: undefined })
    } else {
      useAppStore.getState().updateSession({ treatmentSeqPath: undefined })
    }
  }, [currentSession])
  
  const deleteLoom = useCallback(async (group: 'control' | 'treatment') => {
    if (!currentSession) throw new Error('No active session')
    
    await apiFetch(`/session/${currentSession.id}/loom/${group}`, {
      method: 'DELETE',
    })
    
    if (group === 'control') {
      useAppStore.getState().updateSession({ controlLoomPath: undefined })
    } else {
      useAppStore.getState().updateSession({ treatmentLoomPath: undefined })
    }
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
    uploadPathology,
    uploadSnRNASeq,
    uploadLoom,
    addBiochemistry,
    uploadBiochemistryFile,
    deletePathologyImage,
    deleteBiochemistryFile,
    deleteBiochemistryResult,
    deleteSequencing,
    deleteLoom,
    startAnalysis,
    pauseAnalysis,
    resumeAnalysis,
    exportState,
    getReport,
    updateSession,
  }
}

// WebSocket hook
export function useWebSocket() {
  const { currentSession, addMessage, updateSession, setWsConnected } = useAppStore()
  const wsRef = useRef<WebSocket | null>(null)
  const connectingRef = useRef<boolean>(false)  // Prevent multiple simultaneous connect attempts
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
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${currentSession.id}`
    
    console.log(`WebSocket: Connecting to ${wsUrl}`)
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    
    ws.onopen = () => {
      connectingRef.current = false
      setConnected(true)
      setWsConnected(true)
      console.log('WebSocket: Connected successfully')
    }
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        // Skip error messages that indicate session/config issues
        if (data.error) {
          console.error('WebSocket: Server error:', data.error)
          return
        }
        
        const message: Message = {
          id: generateId(),
          type: data.type || 'message',
          agent: data.agent as AgentType | undefined,
          content: data.content,
          timestamp: data.timestamp || new Date().toISOString(),
        }
        
        addMessage(message)
        
        // Update session status based on message
        if (data.type === 'completed') {
          updateSession({
            status: 'completed',
            finalSummary: data.content?.final_summary,
          })
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
      console.log('WebSocket: Disconnected, code:', event.code, 'reason:', event.reason)
    }
  }, [currentSession, addMessage, updateSession, setWsConnected])
  
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
