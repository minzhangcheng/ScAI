// Session lifecycle: create/load/list/delete, groups, uploads, analysis control.
// Split out of frontend/src/hooks/useApi.ts verbatim.

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAppStore, Session } from '../../stores/appStore'
import { transformRawMessage, restoreAgentErrors } from '../../utils/messageTransform'
import { transformLoadResponse } from '../../utils/transformLoadResponse'
import { devLog } from '../../utils/logger'
import { buildLiteratureConfigPayload } from '../../utils/literatureConfig'
import { downloadBlob } from '../../utils/download'
import type { SessionLoadResponse, SessionInfoUpdateResponse, GroupUpdateResponse, SessionListResponse } from '../../types'
import { API_BASE, apiFetch, applyStoreUpdates, authHeaders } from './client'

// Session hook
export function useSession() {
  const { currentSession, setCurrentSession, updateSession, setMessages, clearMessages } = useAppStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const loadGenerationRef = useRef(0)
  const activeLoadRef = useRef<{
    generation: number
    controller: AbortController
  } | null>(null)

  useEffect(() => () => {
    const activeLoad = activeLoadRef.current
    if (!activeLoad) return
    activeLoadRef.current = null
    loadGenerationRef.current += 1
    activeLoad.controller.abort()
    useAppStore.getState().setSessionLoading(false)
  }, [])
  
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
    activeLoadRef.current?.controller.abort()
    const generation = loadGenerationRef.current + 1
    loadGenerationRef.current = generation
    const controller = new AbortController()
    activeLoadRef.current = { generation, controller }
    const previousSessionId = useAppStore.getState().currentSession?.id ?? null
    const ownsRequest = () => {
      const activeLoad = activeLoadRef.current
      return (
        activeLoad?.generation === generation
        && activeLoad.controller === controller
        && !controller.signal.aborted
      )
    }
    const ownsResponse = () => {
      const selectedSessionId = useAppStore.getState().currentSession?.id ?? null
      return (
        ownsRequest()
        && (selectedSessionId === previousSessionId || selectedSessionId === sessionId)
      )
    }

    setLoading(true)
    useAppStore.getState().setSessionLoading(true)
    setError(null)

    try {
      const data = await apiFetch<SessionLoadResponse>(
        `/session/${sessionId}/load`,
        { signal: controller.signal },
        ownsResponse,
      )

      if (!ownsResponse()) {
        throw new DOMException('Session load was superseded', 'AbortError')
      }

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
      if (ownsResponse()) {
        setError(e instanceof Error ? e.message : 'Failed to load session')
      }
      throw e
    } finally {
      if (ownsRequest()) {
        activeLoadRef.current = null
        setLoading(false)
        useAppStore.getState().setSessionLoading(false)
      }
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
  
  const continueRound = useCallback(async () => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')

    const resp = await apiFetch<any>(`/session/${session.id}/resume`, { method: 'POST' })
    // A refusal to advance still answers 200: the backend reports it in the
    // body rather than as an error. Marking the session running regardless is
    // what hid the re-run control, which is the only way out of that state.
    // Four refusals reach here — runtime unavailable, recovery blocked,
    // snapshot failure, and a pending plan review — and only the first carries
    // a dedicated flag, so `success === false` has to be honoured as well.
    if (resp?.success === false || resp?.data?.runtime_unavailable) return
    useAppStore.getState().updateSession({ status: 'running' })
    // If workflow doesn't exist (e.g., after backend restart), trigger WebSocket
    // reconnect which will create a new workflow instance on the server
    if (resp?.data?.needs_reconnect) {
      const { wsConnect: connectWs } = useAppStore.getState()
      if (connectWs) setTimeout(() => connectWs(), 100)
    }
  }, [])

  // Resuming a paused session and continuing a finished round are the same
  // POST under the same refusal contract, so they share one implementation.
  const resumeAnalysis = continueRound

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
