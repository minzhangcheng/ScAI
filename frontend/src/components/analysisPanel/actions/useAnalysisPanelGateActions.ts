import { useCallback, type MutableRefObject } from 'react'

import { apiFetch } from '../../../hooks/useApi'
import { useAppStore, type Session } from '../../../stores/appStore'
import { chatOwnerKey } from '../../../utils/chatOwnership'
import { useGateActionSubmission } from './useGateActionSubmission'

export type { GateContinuationPending } from './useGateActionSubmission'

function buildDirectionsPayload(
  directions: Array<{ direction: string; notes: string; isCustom?: boolean }>,
) {
  return directions.map(direction => ({
    direction: direction.direction,
    notes: direction.notes ?? '',
    is_custom: !!direction.isCustom,
  }))
}

interface GateActionOptions {
  currentSession: Session | null
  stateVersion?: number
  wsConnected: boolean
  wsConnect: (() => void) | null
  setShowAnnotationDialog: (show: boolean) => void
  setShowPlanReviewDialog: (show: boolean) => void
  setShowExplorationDialog: (show: boolean) => void
  setShowIntegrationGateDialog: (show: boolean) => void
  setIntegrationGateData: (data: any) => void
  setIsSupplementaryRunning: (running: boolean) => void
  roundSupplementaryPendingRef: MutableRefObject<boolean>
  onGateActionError?: (error: unknown, action: string) => void
}

export function useAnalysisPanelGateActions(options: GateActionOptions) {
  const { gateContinuationPending, reconnectIfNeeded, submitGateAction } =
    useGateActionSubmission(options)

  const handleConfirmAnnotations = useCallback(async (
    annotations: Array<{ cluster_id: string; user_override?: string; user_confirmed: boolean }>,
  ) => {
    if (!options.currentSession) return
    await submitGateAction(
      'annotation_confirmation',
      token => apiFetch(`/session/${token.sessionId}/confirm-annotations`, {
        method: 'POST', body: JSON.stringify({ annotations }),
      }),
      (ownerKey) => {
        options.setShowAnnotationDialog(false)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }, [options.currentSession, options.setShowAnnotationDialog, reconnectIfNeeded, submitGateAction])

  const handlePlanApprove = useCallback(async () => {
    if (!options.currentSession) return
    await submitGateAction(
      'plan_review',
      token => apiFetch(`/session/${token.sessionId}/plan-review`, {
        method: 'POST', body: JSON.stringify({ action: 'approve' }),
      }),
      (ownerKey) => {
        options.setShowPlanReviewDialog(false)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }, [options.currentSession, options.setShowPlanReviewDialog, reconnectIfNeeded, submitGateAction])

  const handlePlanSkip = useCallback(async () => {
    if (!options.currentSession) return
    await submitGateAction(
      'plan_review',
      token => apiFetch(`/session/${token.sessionId}/plan-review`, {
        method: 'POST', body: JSON.stringify({ action: 'skip' }),
      }),
      (ownerKey) => {
        options.setShowPlanReviewDialog(false)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }, [options.currentSession, options.setShowPlanReviewDialog, reconnectIfNeeded, submitGateAction])

  const handlePlanModify = useCallback(async (userInput: string) => {
    if (!options.currentSession) return
    await submitGateAction(
      'plan_review',
      token => apiFetch(`/session/${token.sessionId}/plan-review`, {
        method: 'POST',
        body: JSON.stringify({ action: 'modify', user_input: userInput }),
      }),
      (ownerKey) => reconnectIfNeeded(ownerKey),
    ).catch(() => undefined)
  }, [options.currentSession, reconnectIfNeeded, submitGateAction])

  const handleChooseDirections = useCallback(async (
    directions: Array<{ direction: string; notes: string; isCustom: boolean }>,
  ) => {
    if (!options.currentSession) return
    await submitGateAction(
      'direction_selection',
      token => apiFetch(`/session/${token.sessionId}/choose-direction`, {
        method: 'POST', body: JSON.stringify({ directions: buildDirectionsPayload(directions) }),
      }),
      (ownerKey) => {
        options.setShowExplorationDialog(false)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }, [options.currentSession, options.setShowExplorationDialog, reconnectIfNeeded, submitGateAction])

  const handleLetPiDecide = useCallback(async () => {
    if (!options.currentSession) return
    await submitGateAction(
      'direction_selection',
      token => apiFetch(`/session/${token.sessionId}/choose-direction`, {
        method: 'POST', body: JSON.stringify({ direction: '', let_pi_decide: true }),
      }),
      (ownerKey) => {
        options.setShowExplorationDialog(false)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }, [options.currentSession, options.setShowExplorationDialog, reconnectIfNeeded, submitGateAction])

  const submitIntegrationDirections = async (selected: any[]) => {
    if (!options.currentSession) return
    await submitGateAction(
      'integration_selection',
      token => apiFetch(`/session/${token.sessionId}/choose-direction`, {
        method: 'POST',
        body: JSON.stringify({
          directions: buildDirectionsPayload(selected.map(item => ({
            direction: item.direction,
            notes: item.notes ?? '',
            isCustom: !!item.isCustom,
          }))),
        }),
      }),
      (ownerKey) => {
        options.setShowIntegrationGateDialog(false)
        options.setIntegrationGateData(null)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }

  const finalizeFromIntegration = async () => {
    if (!options.currentSession) return
    await submitGateAction(
      'finalize_integration',
      token => apiFetch(`/session/${token.sessionId}/finalize-from-integration`, { method: 'POST' }),
      (ownerKey) => {
        options.setShowIntegrationGateDialog(false)
        options.setIntegrationGateData(null)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }

  const handleLetPiDecideIntegration = async () => {
    if (!options.currentSession) return
    await submitGateAction(
      'integration_selection',
      token => apiFetch(`/session/${token.sessionId}/choose-direction`, {
        method: 'POST', body: JSON.stringify({ let_pi_decide: true }),
      }),
      (ownerKey) => {
        options.setShowIntegrationGateDialog(false)
        options.setIntegrationGateData(null)
        reconnectIfNeeded(ownerKey)
      },
    ).catch(() => undefined)
  }

  const handleSupplementaryAnalysis = useCallback(async (
    requestText: string,
    target: 'annotation' | 'direction' | 'round',
  ) => {
    const session = options.currentSession
    const ownerKey = chatOwnerKey(session?.id, session?.restoreBranchId)
    const activeSession = useAppStore.getState().currentSession
    if (
      !session
      || !ownerKey
      || chatOwnerKey(activeSession?.id, activeSession?.restoreBranchId) !== ownerKey
    ) return
    options.setIsSupplementaryRunning(true)
    if (target === 'round') options.roundSupplementaryPendingRef.current = true
    try {
      await apiFetch(`/session/${session.id}/supplementary-analysis`, {
        method: 'POST', body: JSON.stringify({ request_text: requestText, target }),
      })
    } catch (error) {
      console.error('Supplementary analysis error:', error)
      const current = useAppStore.getState().currentSession
      if (chatOwnerKey(current?.id, current?.restoreBranchId) === ownerKey) {
        options.setIsSupplementaryRunning(false)
        options.roundSupplementaryPendingRef.current = false
      }
    }
  }, [options.currentSession, options.setIsSupplementaryRunning])

  return {
    gateContinuationPending,
    handleConfirmAnnotations,
    handlePlanApprove,
    handlePlanSkip,
    handlePlanModify,
    handleChooseDirections,
    handleLetPiDecide,
    submitIntegrationDirections,
    finalizeFromIntegration,
    handleLetPiDecideIntegration,
    handleSupplementaryAnalysis,
  }
}
