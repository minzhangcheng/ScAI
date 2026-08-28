import { useCallback } from 'react'

import { useAppStore, type Session } from '../../../stores/appStore'
import { buildLiteratureConfigPayload } from '../../../utils/literatureConfig'
import { downloadBlob } from '../../../utils/download'
import { API_BASE, apiFetch, authHeaders } from '../client'

interface SessionAnalysisStore {
  currentSession: Session | null
  updateSession: (updates: Partial<Session>) => void
}

export function useSessionAnalysis(store: SessionAnalysisStore) {
  const startAnalysis = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    const { models, prompts, temperatures } = useAppStore.getState()
    const syncFailures: string[] = []
    const agents = ['pi', 'pathology', 'bioinformatics', 'biochemistry', 'literature', 'painter', 'supervisor', 'technique'] as const
    await Promise.allSettled([
      ...agents.map(agent => apiFetch(`/session/${store.currentSession!.id}/models`, {
        method: 'POST',
        body: JSON.stringify({ session_id: store.currentSession!.id, agent, model: models[agent] }),
      }).catch(e => { syncFailures.push(`model/${agent}`); throw e })),
      ...agents.map(agent => apiFetch(`/session/${store.currentSession!.id}/prompts`, {
        method: 'POST',
        body: JSON.stringify({ session_id: store.currentSession!.id, agent, prompt: prompts[agent] }),
      }).catch(e => { syncFailures.push(`prompt/${agent}`); throw e })),
      ...agents.map(agent => apiFetch(`/session/${store.currentSession!.id}/temperatures`, {
        method: 'POST',
        body: JSON.stringify({ session_id: store.currentSession!.id, agent, temperature: temperatures[agent] }),
      }).catch(e => { syncFailures.push(`temperature/${agent}`); throw e })),
    ])

    const appState = useAppStore.getState()
    const literaturePayload = buildLiteratureConfigPayload(appState)
    await Promise.allSettled([
      apiFetch(`/session/${store.currentSession.id}/round-wait`, {
        method: 'POST',
        body: JSON.stringify({ round_wait_seconds: appState.roundWaitSeconds }),
      }).catch(e => { syncFailures.push('round_wait'); throw e }),
      apiFetch(`/session/${store.currentSession.id}/literature-config`, {
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
    await apiFetch(`/session/${store.currentSession.id}/run`, { method: 'POST' })
    store.updateSession({ status: 'running' })
    return syncFailures
  }, [store.currentSession, store.updateSession])

  const pauseAnalysis = useCallback(async () => {
    if (!store.currentSession) throw new Error('No active session')
    await apiFetch(`/session/${store.currentSession.id}/pause`, { method: 'POST' })
    store.updateSession({ status: 'paused' })
  }, [store.currentSession, store.updateSession])

  const continueRound = useCallback(async () => {
    const session = useAppStore.getState().currentSession
    if (!session) throw new Error('No active session')
    const response = await apiFetch<any>(`/session/${session.id}/resume`, { method: 'POST' })
    if (response?.success === false || response?.data?.runtime_unavailable) return
    useAppStore.getState().updateSession({ status: 'running' })
    if (response?.data?.needs_reconnect) {
      const { wsConnect } = useAppStore.getState()
      if (wsConnect) setTimeout(() => wsConnect(), 100)
    }
  }, [])
  const resumeAnalysis = continueRound

  const stopAnalysis = useCallback(async (action: 'immediate' | 'summarize') => {
    if (!store.currentSession) throw new Error('No active session')
    await apiFetch(`/session/${store.currentSession.id}/stop`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    })
    store.updateSession({ status: action === 'immediate' ? 'paused' : 'generating_report' })
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
