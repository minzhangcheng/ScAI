import { useCallback } from 'react'

import { apiFetch, generateId } from '../../../hooks/useApi'
import { useAppStore, type Session } from '../../../stores/appStore'
import type { AnnotationData } from '../../../utils/annotationGate'
import { chatOwnerKey } from '../../../utils/chatOwnership'
import type { ExplorationReportData } from '../types'
import type { RoundSummary } from '../../../types'

interface InterventionActionOptions {
  currentSession: Session | null
  clearExplorationFailedState: () => void
  clearSpeciesInputState: () => void
  setShowSpeciesInputDialog: (show: boolean) => void
  setShowLlmTimeoutDialog: (show: boolean) => void
  setShowStopDialog: (show: boolean) => void
  handleStop: (action: 'immediate' | 'summarize') => Promise<void>
  annotationData: AnnotationData | null
  annotationGateData: AnnotationData | null
  setAnnotationData: (data: AnnotationData | null) => void
  setShowAnnotationDialog: (show: boolean) => void
  integrationGateData: any
  setShowIntegrationGateDialog: (show: boolean) => void
  explorationReportData: ExplorationReportData | null
  setShowExplorationDialog: (show: boolean) => void
  feedbackRoundData: RoundSummary | null
  setShowFeedbackDialog: (show: boolean) => void
  setShowPlanReviewDialog: (show: boolean) => void
  wsConnect: (() => void) | null
  maxExplorationRound: number
  addMessage: ReturnType<typeof useAppStore.getState>['addMessage']
}

function ownerForSession(session: Session | null): string | null {
  return chatOwnerKey(session?.id, session?.restoreBranchId)
}

function currentOwner(): string | null {
  const session = useAppStore.getState().currentSession
  return ownerForSession(session)
}

export function useAnalysisPanelInterventionActions(options: InterventionActionOptions) {
  const reportActionError = useCallback((action: string, error: unknown) => {
    const detail = error instanceof Error ? error.message : String(error)
    options.addMessage({
      id: generateId(),
      type: 'error',
      content: `${action}: ${detail}`,
      timestamp: new Date().toISOString(),
    })
  }, [options.addMessage])

  const handleExplorationFailedGoToData = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    options.clearExplorationFailedState()
    useAppStore.getState().setActiveTab('data')
  }, [options.currentSession, options.clearExplorationFailedState])

  const handleExplorationFailedContinue = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    options.clearExplorationFailedState()
  }, [options.currentSession, options.clearExplorationFailedState])

  const handleExplorationFailedStop = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    options.clearExplorationFailedState()
    options.handleStop('immediate')
  }, [options.currentSession, options.clearExplorationFailedState, options.handleStop])

  const handleLlmSkip = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    const session = options.currentSession
    const owner = ownerForSession(session)
    if (!session || !owner || currentOwner() !== owner || !info) return
    options.setShowLlmTimeoutDialog(false)
    try {
      await apiFetch(`/session/${session.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({ action: 'skip', node_name: info.nodeName, generation: info.generation }),
      })
    } catch (error) {
      console.error('Failed to signal LLM skip:', error)
      if (currentOwner() === owner) {
        options.setShowLlmTimeoutDialog(true)
        reportActionError('llm_skip', error)
      }
    }
  }, [options.currentSession, options.setShowLlmTimeoutDialog, reportActionError])

  const handleLlmRetry = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    const session = options.currentSession
    const owner = ownerForSession(session)
    if (!session || !owner || currentOwner() !== owner || !info) return
    options.setShowLlmTimeoutDialog(false)
    try {
      await apiFetch(`/session/${session.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({ action: 'retry', node_name: info.nodeName, generation: info.generation }),
      })
    } catch (error) {
      console.error('Failed to signal LLM retry:', error)
      if (currentOwner() === owner) {
        options.setShowLlmTimeoutDialog(true)
        reportActionError('llm_retry', error)
      }
    }
  }, [options.currentSession, options.setShowLlmTimeoutDialog, reportActionError])

  const handleSpeciesInput = useCallback(async (input: string) => {
    const session = options.currentSession
    const owner = ownerForSession(session)
    // Capture and validate the exact owner before issuing the request. A
    // callback retained by the source dialog must not post to the source (or
    // clear the child) after a branch-only session switch.
    if (!session || !owner || currentOwner() !== owner) return
    options.setShowSpeciesInputDialog(false)
    try {
      await apiFetch(`/session/${session.id}/species-input`, {
        method: 'POST', body: JSON.stringify({ input }),
      })
      if (currentOwner() === owner) options.clearSpeciesInputState()
    } catch (error) {
      console.error('Failed to submit species input:', error)
      if (currentOwner() === owner) {
        options.setShowSpeciesInputDialog(true)
        reportActionError('species_input', error)
      }
      throw error
    }
  }, [
    options.currentSession, options.clearSpeciesInputState,
    options.setShowSpeciesInputDialog, reportActionError,
  ])

  const handleLlmModifyAgent = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    options.setShowLlmTimeoutDialog(false)
    useAppStore.getState().setLlmModifyPending(true)
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setActiveTab('prompts')
  }, [options.currentSession, options.setShowLlmTimeoutDialog])

  const handleLlmTerminate = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    const session = options.currentSession
    const owner = ownerForSession(session)
    if (!session || !owner || currentOwner() !== owner || !info) return
    options.setShowLlmTimeoutDialog(false)
    try {
      await apiFetch(`/session/${session.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({ action: 'terminate', node_name: info.nodeName, generation: info.generation }),
      })
    } catch (error) {
      console.error('Failed to signal LLM terminate:', error)
      if (currentOwner() === owner) {
        options.setShowLlmTimeoutDialog(true)
        reportActionError('llm_terminate', error)
      }
    }
  }, [options.currentSession, options.setShowLlmTimeoutDialog, reportActionError])

  const handleStopDialogCancel = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    options.setShowStopDialog(false)
    if (useAppStore.getState().llmFailureInfo) options.setShowLlmTimeoutDialog(true)
  }, [options.currentSession, options.setShowStopDialog, options.setShowLlmTimeoutDialog])

  const handleLlmResumeAfterModify = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    useAppStore.getState().setLlmModifyPending(false)
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setActiveTab('analysis')
    handleLlmRetry()
  }, [options.currentSession, handleLlmRetry])

  const handleLlmReturnToSettings = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    useAppStore.getState().setLlmModifySaved(false)
  }, [options.currentSession])

  const handleLlmReturnToTimeout = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setShowLlmTimeoutDialog(true)
  }, [options.currentSession])

  const handleReopenDialog = useCallback(() => {
    const owner = ownerForSession(options.currentSession)
    if (!owner || currentOwner() !== owner) return
    if (options.currentSession?.status === 'waiting_annotation') {
      if (!options.annotationData && options.annotationGateData) {
        options.setAnnotationData(options.annotationGateData)
      }
      options.setShowAnnotationDialog(true)
    } else if (options.currentSession?.status === 'waiting_input') {
      if (options.integrationGateData) {
        options.setShowIntegrationGateDialog(true)
      } else if (options.explorationReportData
        && (options.currentSession.currentRound ?? 0) <= options.maxExplorationRound) {
        options.setShowExplorationDialog(true)
      } else if (options.feedbackRoundData && options.feedbackRoundData.phase !== 'integration') {
        options.setShowFeedbackDialog(true)
      } else {
        options.wsConnect?.()
      }
    } else if (options.currentSession?.status === 'waiting_plan_review') {
      options.setShowPlanReviewDialog(true)
    } else if (options.currentSession?.status === 'waiting_species_input') {
      useAppStore.getState().setShowSpeciesInputDialog(true)
    }
  }, [
    options.currentSession?.status,
    options.currentSession?.currentRound,
    options.explorationReportData,
    options.feedbackRoundData,
    options.integrationGateData,
    options.annotationData,
    options.annotationGateData,
    options.wsConnect,
  ])

  return {
    handleExplorationFailedGoToData,
    handleExplorationFailedContinue,
    handleExplorationFailedStop,
    handleLlmSkip,
    handleLlmRetry,
    handleSpeciesInput,
    handleLlmModifyAgent,
    handleLlmTerminate,
    handleStopDialogCancel,
    handleLlmResumeAfterModify,
    handleLlmReturnToSettings,
    handleLlmReturnToTimeout,
    handleReopenDialog,
  }
}
