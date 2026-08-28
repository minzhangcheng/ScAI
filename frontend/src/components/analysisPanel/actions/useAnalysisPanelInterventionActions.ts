import { useCallback } from 'react'

import { apiFetch } from '../../../hooks/useApi'
import { useAppStore, type Session } from '../../../stores/appStore'
import type { AnnotationData } from '../../../utils/annotationGate'
import type { ExplorationReportData } from '../types'
import type { RoundSummary } from '../../../types'

interface InterventionActionOptions {
  currentSession: Session | null
  clearExplorationFailedState: () => void
  clearSpeciesInputState: () => void
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
}

export function useAnalysisPanelInterventionActions(options: InterventionActionOptions) {
  const handleExplorationFailedGoToData = useCallback(() => {
    options.clearExplorationFailedState()
    useAppStore.getState().setActiveTab('data')
  }, [options.clearExplorationFailedState])

  const handleExplorationFailedContinue = useCallback(() => {
    options.clearExplorationFailedState()
  }, [options.clearExplorationFailedState])

  const handleExplorationFailedStop = useCallback(() => {
    options.clearExplorationFailedState()
    options.handleStop('immediate')
  }, [options.clearExplorationFailedState, options.handleStop])

  const handleLlmSkip = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    if (!options.currentSession || !info) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({ action: 'skip', node_name: info.nodeName, generation: info.generation }),
      })
      options.setShowLlmTimeoutDialog(false)
    } catch (error) {
      console.error('Failed to signal LLM skip:', error)
    }
  }, [options.currentSession])

  const handleLlmRetry = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    if (!options.currentSession || !info) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({ action: 'retry', node_name: info.nodeName, generation: info.generation }),
      })
      options.setShowLlmTimeoutDialog(false)
    } catch (error) {
      console.error('Failed to signal LLM retry:', error)
    }
  }, [options.currentSession])

  const handleSpeciesInput = useCallback(async (input: string) => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/species-input`, {
        method: 'POST', body: JSON.stringify({ input }),
      })
      options.clearSpeciesInputState()
    } catch (error) {
      console.error('Failed to submit species input:', error)
    }
  }, [options.currentSession, options.clearSpeciesInputState])

  const handleLlmModifyAgent = useCallback(() => {
    options.setShowLlmTimeoutDialog(false)
    useAppStore.getState().setLlmModifyPending(true)
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setActiveTab('prompts')
  }, [])

  const handleLlmTerminate = useCallback(async () => {
    const info = useAppStore.getState().llmFailureInfo
    if (!options.currentSession || !info) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/llm-failure-action`, {
        method: 'POST',
        body: JSON.stringify({ action: 'terminate', node_name: info.nodeName, generation: info.generation }),
      })
      options.setShowLlmTimeoutDialog(false)
    } catch (error) {
      console.error('Failed to signal LLM terminate:', error)
    }
  }, [options.currentSession])

  const handleStopDialogCancel = useCallback(() => {
    options.setShowStopDialog(false)
    if (useAppStore.getState().llmFailureInfo) options.setShowLlmTimeoutDialog(true)
  }, [])

  const handleLlmResumeAfterModify = useCallback(() => {
    useAppStore.getState().setLlmModifyPending(false)
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setActiveTab('analysis')
    handleLlmRetry()
  }, [handleLlmRetry])

  const handleLlmReturnToSettings = useCallback(() => {
    useAppStore.getState().setLlmModifySaved(false)
  }, [])

  const handleLlmReturnToTimeout = useCallback(() => {
    useAppStore.getState().setLlmModifySaved(false)
    useAppStore.getState().setShowLlmTimeoutDialog(true)
  }, [])

  const handleReopenDialog = useCallback(() => {
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
