import { useCallback, useEffect, useRef, useState } from 'react'

import type { AgentSettingsResponse, SessionLoadResponse } from '../../../types'
import { useAppStore, type Session } from '../../../stores/appStore'
import { restoreAgentErrors, transformRawMessage } from '../../../utils/messageTransform'
import { transformAgentSettings, transformLoadResponse } from '../../../utils/transformLoadResponse'
import { apiFetch, applyStoreUpdates } from '../client'
import { loadRealtimeMessageHistory } from '../realtimeHistory'
import type { ActiveSessionHistory, ActiveSessionLoad, SessionLoadingOwnership } from './types'

interface SessionLoadingStore {
  setCurrentSession: (session: Session | null) => void
  setMessages: ReturnType<typeof useAppStore.getState>['setMessages']
  clearMessages: ReturnType<typeof useAppStore.getState>['clearMessages']
}

export function useSessionLoading(store: SessionLoadingStore) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const loadGenerationRef = useRef(0)
  const activeLoadRef = useRef<ActiveSessionLoad | null>(null)
  const activeHistoryRef = useRef<ActiveSessionHistory | null>(null)

  const startHistoryPaging = useCallback((sessionId: string, generation: number) => {
    activeHistoryRef.current?.controller.abort()
    const controller = new AbortController()
    activeHistoryRef.current = { sessionId, generation, controller }
    const ownsHistory = () => {
      const active = activeHistoryRef.current
      return Boolean(
        active
        && active.sessionId === sessionId
        && active.generation === generation
        && active.controller === controller
        && !controller.signal.aborted
        && useAppStore.getState().currentSession?.id === sessionId
      )
    }
    useAppStore.getState().setRealtimeHistoryError(null)
    void loadRealtimeMessageHistory(sessionId, {
      signal: controller.signal,
      ownsResponse: ownsHistory,
      onFailure: error => {
        if (ownsHistory()) {
          useAppStore.getState().setRealtimeHistoryError(
            error instanceof Error ? error.message : 'Failed to load realtime history',
          )
        }
      },
    })
  }, [])

  useEffect(() => () => {
    const activeLoad = activeLoadRef.current
    if (activeLoad) {
      activeLoadRef.current = null
      loadGenerationRef.current += 1
      activeLoad.controller.abort()
    }
    activeHistoryRef.current?.controller.abort()
    activeHistoryRef.current = null
    useAppStore.getState().setSessionLoading(false)
  }, [])

  const loadSession = useCallback(async (sessionId: string) => {
    activeLoadRef.current?.controller.abort()
    activeHistoryRef.current?.controller.abort()
    activeHistoryRef.current = null
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
      return ownsRequest() && (selectedSessionId === previousSessionId || selectedSessionId === sessionId)
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
      if (!ownsResponse()) throw new DOMException('Session load was superseded', 'AbortError')
      const agentSettings = await apiFetch<AgentSettingsResponse>(
        `/session/${sessionId}/agent-settings`,
        { signal: controller.signal },
        ownsResponse,
      )
      if (!ownsResponse()) throw new DOMException('Session load was superseded', 'AbortError')

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

      store.setCurrentSession(session)
      applyStoreUpdates({ ...storeUpdates, ...transformAgentSettings(agentSettings) })
      if (data.agent_errors && Array.isArray(data.agent_errors)) restoreAgentErrors(data.agent_errors)
      if (data.messages && Array.isArray(data.messages)) {
        store.setMessages(data.messages.map(transformRawMessage))
      } else {
        store.clearMessages()
      }
      startHistoryPaging(sessionId, generation)
      return session
    } catch (e) {
      if (ownsResponse()) setError(e instanceof Error ? e.message : 'Failed to load session')
      throw e
    } finally {
      if (ownsRequest()) {
        activeLoadRef.current = null
        setLoading(false)
        useAppStore.getState().setSessionLoading(false)
      }
    }
  }, [store.setCurrentSession, store.setMessages, store.clearMessages, startHistoryPaging])

  const retryRealtimeHistory = useCallback(() => {
    const sessionId = useAppStore.getState().currentSession?.id
    if (!sessionId) return
    startHistoryPaging(sessionId, loadGenerationRef.current)
  }, [startHistoryPaging])

  const ownership: SessionLoadingOwnership = {
    loadGenerationRef,
    activeLoadRef,
    activeHistoryRef,
  }

  return { loading, error, setLoading, setError, loadSession, retryRealtimeHistory, ownership }
}
