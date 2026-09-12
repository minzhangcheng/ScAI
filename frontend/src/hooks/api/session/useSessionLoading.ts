import { useCallback, useEffect, useRef, useState } from 'react'

import type { AgentSettingsResponse, SessionLoadResponse } from '../../../types'
import { useAppStore, type Session } from '../../../stores/appStore'
import { restoreAgentErrors, transformRawMessage } from '../../../utils/messageTransform'
import { transformAgentSettings, transformLoadResponse } from '../../../utils/transformLoadResponse'
import { chatOwnerKey } from '../../../utils/chatOwnership'
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

  const startHistoryPaging = useCallback((
    sessionId: string,
    ownerKey: string | null,
    generation: number,
  ) => {
    activeHistoryRef.current?.controller.abort()
    const controller = new AbortController()
    activeHistoryRef.current = { sessionId, ownerKey, generation, controller }
    const ownsHistory = () => {
      const active = activeHistoryRef.current
      const current = useAppStore.getState().currentSession
      return Boolean(
        active
        && active.sessionId === sessionId
        && active.ownerKey === ownerKey
        && active.generation === generation
        && active.controller === controller
        && !controller.signal.aborted
        && current?.id === sessionId
        && chatOwnerKey(current.id, current.restoreBranchId) === ownerKey
      )
    }
    useAppStore.getState().setRealtimeHistoryError(null)
    void loadRealtimeMessageHistory(sessionId, {
      signal: controller.signal,
      ownerKey,
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
    const previousSession = useAppStore.getState().currentSession
    const previousOwnerKey = chatOwnerKey(
      previousSession?.id, previousSession?.restoreBranchId,
    )
    activeLoadRef.current = {
      generation,
      controller,
      ownerKey: previousOwnerKey,
      requestedSessionId: sessionId,
    }
    const ownsRequest = () => {
      const activeLoad = activeLoadRef.current
      return (
        activeLoad?.generation === generation
        && activeLoad.controller === controller
        && !controller.signal.aborted
      )
    }
    const ownsResponse = () => {
      const selected = useAppStore.getState().currentSession
      const selectedOwnerKey = chatOwnerKey(
        selected?.id, selected?.restoreBranchId,
      )
      const selectedIsPreviousOwner = selectedOwnerKey === previousOwnerKey

      // The response is allowed to install the target only while the owner
      // selected at request start is still selected. If another path has
      // already installed the target (or a different branch of it), this
      // request is stale even when the session id matches; the newer
      // transition owns the store and its messages/settings.
      return ownsRequest() && selectedIsPreviousOwner
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
      // The owner is installed in the same store transaction that clears any
      // previous owner's messages; only then may this load's messages land.
      applyStoreUpdates({ ...storeUpdates, ...transformAgentSettings(agentSettings) })
      if (data.agent_errors && Array.isArray(data.agent_errors)) restoreAgentErrors(data.agent_errors)
      if (data.messages && Array.isArray(data.messages)) {
        store.setMessages(data.messages.map(transformRawMessage))
      } else {
        store.clearMessages()
      }
      const installedOwnerKey = chatOwnerKey(session.id, session.restoreBranchId)
      const activeLoad = activeLoadRef.current
      if (activeLoad && activeLoad.generation === generation) {
        activeLoad.ownerKey = installedOwnerKey
      }
      startHistoryPaging(sessionId, installedOwnerKey, generation)
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
    const current = useAppStore.getState().currentSession
    if (!current?.id) return
    startHistoryPaging(
      current.id,
      chatOwnerKey(current.id, current.restoreBranchId),
      loadGenerationRef.current,
    )
  }, [startHistoryPaging])

  const ownership: SessionLoadingOwnership = {
    loadGenerationRef,
    activeLoadRef,
    activeHistoryRef,
  }

  return { loading, error, setLoading, setError, loadSession, retryRealtimeHistory, ownership }
}
