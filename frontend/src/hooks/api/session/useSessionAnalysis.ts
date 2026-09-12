import { useCallback } from 'react'

import { useAppStore, type Session } from '../../../stores/appStore'
import { buildLiteratureConfigPayload } from '../../../utils/literatureConfig'
import { downloadBlob } from '../../../utils/download'
import { chatOwnerKey } from '../../../utils/chatOwnership'
import { API_BASE, apiFetch, authHeaders } from '../client'

interface SessionAnalysisStore {
  currentSession: Session | null
  updateSession: (updates: Partial<Session>) => void
}

export function useSessionAnalysis(store: SessionAnalysisStore) {
  const startAnalysis = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    const session = store.currentSession
    const ownerKey = chatOwnerKey(session.id, session.restoreBranchId)
    const { models, prompts, temperatures } = useAppStore.getState()
    const syncFailures: string[] = []
    const agents = ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature', 'painter', 'supervisor', 'technique'] as const
    await Promise.allSettled([
      ...agents.map(agent => apiFetch(`/session/${session.id}/models`, {
        method: 'POST',
        body: JSON.stringify({ session_id: session.id, agent, model: models[agent] }),
      }).catch(e => { syncFailures.push(`model/${agent}`); throw e })),
      ...agents.map(agent => apiFetch(`/session/${session.id}/prompts`, {
        method: 'POST',
        body: JSON.stringify({ session_id: session.id, agent, prompt: prompts[agent] }),
      }).catch(e => { syncFailures.push(`prompt/${agent}`); throw e })),
      ...agents.map(agent => apiFetch(`/session/${session.id}/temperatures`, {
        method: 'POST',
        body: JSON.stringify({ session_id: session.id, agent, temperature: temperatures[agent] }),
      }).catch(e => { syncFailures.push(`temperature/${agent}`); throw e })),
    ])

    const appState = useAppStore.getState()
    const literaturePayload = buildLiteratureConfigPayload(appState)
    await Promise.allSettled([
      apiFetch(`/session/${session.id}/round-wait`, {
        method: 'POST',
        body: JSON.stringify({ round_wait_seconds: appState.roundWaitSeconds }),
      }).catch(e => { syncFailures.push('round_wait'); throw e }),
      apiFetch(`/session/${session.id}/literature-config`, {
        method: 'POST',
        body: JSON.stringify(literaturePayload),
      }).catch(e => { syncFailures.push('literature_config'); throw e }),
    ])

    if (syncFailures.length > 0) {
      const summary = syncFailures.join(', ')
      console.error('[startAnalysis] agent config sync failed:', summary)
      const error: Error & { syncFailures?: string[] } = new Error(
        `Agent config sync failed before starting analysis: ${summary}. Analysis aborted.`,
      )
      error.syncFailures = syncFailures
      throw error
    }
    if (chatOwnerKey(useAppStore.getState().currentSession?.id,
      useAppStore.getState().currentSession?.restoreBranchId) !== ownerKey) return syncFailures
    await apiFetch(`/session/${session.id}/run`, { method: 'POST' })
    if (chatOwnerKey(useAppStore.getState().currentSession?.id,
      useAppStore.getState().currentSession?.restoreBranchId) === ownerKey) {
      store.updateSession({ status: 'running' })
    }
    return syncFailures
  }, [store.currentSession, store.updateSession])

  const pauseAnalysis = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    const session = store.currentSession
    const ownerKey = chatOwnerKey(session.id, session.restoreBranchId)
    await apiFetch(`/session/${session.id}/pause`, { method: 'POST' })
    if (chatOwnerKey(useAppStore.getState().currentSession?.id,
      useAppStore.getState().currentSession?.restoreBranchId) === ownerKey) {
      store.updateSession({ status: 'paused' })
    }
  }, [store.currentSession, store.updateSession])

  const continueRound = useCallback(async () => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    const ownerKey = chatOwnerKey(session.id, session.restoreBranchId)
    const response = await apiFetch<any>(`/session/${session.id}/resume`, { method: 'POST' })
    // A delayed response must not project the previous session's runtime
    // refusal onto a session selected while the request was in flight.
    const activeSession = useAppStore.getState().currentSession
    if (
      !activeSession
      || activeSession.id !== session.id
      || chatOwnerKey(activeSession.id, activeSession.restoreBranchId) !== ownerKey
    ) return
    if (response?.data?.runtime_unavailable) {
      // The backend deliberately answers this refusal without reconnecting.
      // Hydrate the durable warning immediately so the existing banner and
      // Re-run action remain visible even when the user does not reload.
      const data = response.data
      useAppStore.getState().hydrateRuntimeRestoreFromSession({
        runtime_restore_status: data.runtime_restore_status ?? 'unavailable',
        runtime_restore_warning: data.runtime_restore_warning
          ?? (response.message
            || 'The saved runtime is unavailable. Re-run the analysis from the input data.'),
      })
      return
    }
    if (response?.success === false) return
    useAppStore.getState().updateSession({ status: 'running' })
    if (response?.data?.needs_reconnect) {
      const { wsConnect } = useAppStore.getState()
      if (wsConnect) setTimeout(() => wsConnect(), 100)
    }
  }, [])
  const resumeAnalysis = continueRound

  const stopAnalysis = useCallback(async (action: 'immediate' | 'summarize') => {
    if (!store.currentSession) throw new Error('No active session')
    const session = store.currentSession
    const ownerKey = chatOwnerKey(session.id, session.restoreBranchId)
    await apiFetch(`/session/${session.id}/stop`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    })
    const activeSession = useAppStore.getState().currentSession
    if (chatOwnerKey(activeSession?.id, activeSession?.restoreBranchId) === ownerKey) {
      store.updateSession({ status: action === 'immediate' ? 'paused' : 'generating_report' })
    }
  }, [store.currentSession, store.updateSession])

  const exportState = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    const response = await fetch(`${API_BASE}/session/${store.currentSession.id}/export`, {
      method: 'GET',
      headers: authHeaders(),
    })
    if (!response.ok) throw new Error('Failed to export state')
    downloadBlob(await response.blob(), `${store.currentSession.id}_export.tar.gz`)
  }, [store.currentSession])

  const getReport = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    return apiFetch<any>(`/session/${store.currentSession.id}/report`)
  }, [store.currentSession])

  return {
    startAnalysis, pauseAnalysis, resumeAnalysis, continueRound, stopAnalysis, exportState, getReport,
  }
}
