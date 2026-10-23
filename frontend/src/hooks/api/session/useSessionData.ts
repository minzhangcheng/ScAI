import { useCallback } from 'react'

import type {
  GroupUpdateResponse,
  SessionInfoUpdateResponse,
  SessionListResponse,
  SessionLoadResponse,
} from '../../../types'
import { useAppStore, type Session } from '../../../stores/appStore'
import { devLog } from '../../../utils/logger'
import {
  loadLastAgentSettings,
  newSessionAgentSettingsPayload,
} from '../../../utils/lastAgentSettings'
import { transformLoadResponse } from '../../../utils/transformLoadResponse'
import { apiFetch } from '../client'

interface SessionDataStore {
  currentSession: Session | null
  setCurrentSession: (session: Session | null) => void
  clearMessages: () => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export function useSessionData(store: SessionDataStore) {
  const createSession = useCallback(async (instruction: string, name?: string, roundWaitSeconds?: number) => {
    store.setLoading(true)
    store.setError(null)
    store.clearMessages()
    try {
      const { authUser, pubmedEmail } = useAppStore.getState()
      const previousDefaults = authUser
        ? loadLastAgentSettings(authUser.id)?.settings
        : undefined
      const response = await apiFetch<{
        success: boolean
        session_id: string
        data: { name: string }
      }>('/session/start', {
        method: 'POST',
        body: JSON.stringify({
          instruction,
          session_name: name,
          round_wait_seconds: roundWaitSeconds,
          ...(previousDefaults
            ? { agent_settings: newSessionAgentSettingsPayload(previousDefaults) }
            : pubmedEmail
              ? { agent_settings: { literature_config: { pubmed_email: pubmedEmail } } }
              : {}),
        }),
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
      store.setCurrentSession(session)
      return session
    } catch (e) {
      store.setError(e instanceof Error ? e.message : 'Failed to create session')
      throw e
    } finally {
      store.setLoading(false)
    }
  }, [store.setCurrentSession, store.clearMessages, store.setLoading, store.setError])

  const listSessions = useCallback(async () => {
    const data = await apiFetch<SessionListResponse>('/sessions')
    return data.sessions
  }, [])

  const deleteSession = useCallback(async (sessionId: string) => {
    await apiFetch(`/session/${sessionId}`, { method: 'DELETE' })
    if (store.currentSession?.id === sessionId) store.setCurrentSession(null)
  }, [store.currentSession, store.setCurrentSession])

  const saveSession = useCallback(async () => {
    if (!store.currentSession) return
    await apiFetch(`/session/${store.currentSession.id}/save`, { method: 'POST' })
  }, [store.currentSession])

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
    useAppStore.getState().updateSession({ name: response.data.name, instruction: response.data.instruction })
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

  const ensureSession = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    const data = await apiFetch<any>(`/session/${store.currentSession.id}/ensure`, {
      method: 'POST',
      body: JSON.stringify({
        instruction: store.currentSession.instruction,
        session_name: store.currentSession.name,
      }),
    })
    if (data.data?.created) devLog('Session recreated after server restart')
    return data
  }, [store.currentSession])

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

  const updateGroup = useCallback(async (
    groupId: string,
    updates: { name?: string; description?: string; sampleCount?: number },
  ) => {
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
      treatmentGroups: (latestSession?.treatmentGroups || []).map(group => (
        group.id === groupId ? { ...group, ...response.data } : group
      )),
    })
    return response
  }, [])

  const deleteGroup = useCallback(async (groupId: string) => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    await apiFetch(`/session/${session.id}/groups/${groupId}`, { method: 'DELETE' })
    const latestSession = useAppStore.getState().currentSession
    useAppStore.getState().updateSession({
      treatmentGroups: (latestSession?.treatmentGroups || []).filter(group => group.id !== groupId),
      dataFiles: (latestSession?.dataFiles || []).filter(file => file.groupId !== groupId),
      biochemistryIndicators: (latestSession?.biochemistryIndicators || []).filter(item => item.groupId !== groupId),
    })
  }, [])

  const uploadData = useCallback(async (
    file: File,
    section: 'pathology' | 'sequencing' | 'biochemistry',
    dataType: string,
    groupId?: string,
    sampleNumber?: number,
    description?: string,
    signal?: AbortSignal,
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
    const data = await apiFetch<any>(`/session/${session.id}/upload/data`, {
      method: 'POST', body: formData, signal,
    })
    try {
      const refreshData = await apiFetch<SessionLoadResponse>(`/session/${session.id}/load`)
      const current = useAppStore.getState().currentSession
      if (current && current.id === session.id) {
        const { sessionUpdates } = transformLoadResponse(refreshData)
        useAppStore.getState().updateSession({ dataFiles: sessionUpdates.dataFiles })
      }
    } catch {
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
      dataFiles: (latestSession?.dataFiles || []).filter(file => file.id !== fileId),
    })
  }, [])

  const addIndicator = useCallback(async (
    indicator: string, value: number, unit: string, groupId?: string, sampleNumber?: number,
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
        { id: response.data.indicator_id, indicator, value, unit, referenceRange: undefined,
          interpretation: undefined, groupId, sampleNumber, origin: 'manual' },
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
      biochemistryIndicators: (latestSession?.biochemistryIndicators || []).filter(item => item.id !== indicatorId),
    })
  }, [])

  return {
    createSession, listSessions, deleteSession, saveSession, updateSessionInfo, updateRoundWait,
    addGroup, updateGroup, deleteGroup, uploadData, deleteDataFile, addIndicator, deleteIndicator,
  }
}
