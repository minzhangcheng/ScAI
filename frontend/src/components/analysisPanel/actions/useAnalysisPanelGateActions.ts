import { useCallback, type MutableRefObject } from 'react'

import { apiFetch } from '../../../hooks/useApi'
import type { Session } from '../../../stores/appStore'

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
  wsConnected: boolean
  wsConnect: (() => void) | null
  setShowAnnotationDialog: (show: boolean) => void
  setShowPlanReviewDialog: (show: boolean) => void
  setShowExplorationDialog: (show: boolean) => void
  setShowIntegrationGateDialog: (show: boolean) => void
  setIntegrationGateData: (data: any) => void
  setIsSupplementaryRunning: (running: boolean) => void
  roundSupplementaryPendingRef: MutableRefObject<boolean>
}

export function useAnalysisPanelGateActions(options: GateActionOptions) {
  const reconnectIfNeeded = () => {
    if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
  }

  const handleConfirmAnnotations = useCallback(async (
    annotations: Array<{ cluster_id: string; user_override?: string; user_confirmed: boolean }>,
  ) => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/confirm-annotations`, {
        method: 'POST', body: JSON.stringify({ annotations }),
      })
      options.setShowAnnotationDialog(false)
      reconnectIfNeeded()
    } catch (error) {
      console.error('Failed to confirm annotations:', error)
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect])

  const handlePlanApprove = useCallback(async () => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/plan-review`, {
        method: 'POST', body: JSON.stringify({ action: 'approve' }),
      })
      options.setShowPlanReviewDialog(false)
      reconnectIfNeeded()
    } catch (error) {
      console.error('Failed to approve plan:', error)
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect])

  const handlePlanSkip = useCallback(async () => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/plan-review`, {
        method: 'POST', body: JSON.stringify({ action: 'skip' }),
      })
      options.setShowPlanReviewDialog(false)
      reconnectIfNeeded()
    } catch (error) {
      console.error('Failed to skip plan review:', error)
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect])

  const handlePlanModify = useCallback(async (userInput: string) => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/plan-review`, {
        method: 'POST',
        body: JSON.stringify({ action: 'modify', user_input: userInput }),
      })
      reconnectIfNeeded()
    } catch (error) {
      console.error('Failed to submit plan modification:', error)
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect])

  const handleChooseDirections = useCallback(async (
    directions: Array<{ direction: string; notes: string; isCustom: boolean }>,
  ) => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/choose-direction`, {
        method: 'POST', body: JSON.stringify({ directions: buildDirectionsPayload(directions) }),
      })
      options.setShowExplorationDialog(false)
      reconnectIfNeeded()
    } catch (error) {
      console.error('Failed to choose directions:', error)
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect])

  const handleLetPiDecide = useCallback(async () => {
    if (!options.currentSession) return
    try {
      await apiFetch(`/session/${options.currentSession.id}/choose-direction`, {
        method: 'POST', body: JSON.stringify({ direction: '', let_pi_decide: true }),
      })
      options.setShowExplorationDialog(false)
      reconnectIfNeeded()
    } catch (error) {
      console.error('Failed to let PI decide:', error)
    }
  }, [options.currentSession, options.wsConnected, options.wsConnect])

  const submitIntegrationDirections = async (selected: any[]) => {
    if (!options.currentSession) return
    await apiFetch(`/session/${options.currentSession.id}/choose-direction`, {
      method: 'POST',
      body: JSON.stringify({
        directions: buildDirectionsPayload(selected.map(item => ({
          direction: item.direction,
          notes: item.notes ?? '',
          isCustom: !!item.isCustom,
        }))),
      }),
    })
    options.setShowIntegrationGateDialog(false)
    options.setIntegrationGateData(null)
  }

  const finalizeFromIntegration = async () => {
    if (!options.currentSession) return
    await apiFetch(`/session/${options.currentSession.id}/finalize-from-integration`, { method: 'POST' })
    options.setShowIntegrationGateDialog(false)
    options.setIntegrationGateData(null)
  }

  const handleLetPiDecideIntegration = async () => {
    if (!options.currentSession) return
    await apiFetch(`/session/${options.currentSession.id}/choose-direction`, {
      method: 'POST', body: JSON.stringify({ let_pi_decide: true }),
    })
    options.setShowIntegrationGateDialog(false)
    options.setIntegrationGateData(null)
  }

  const handleSupplementaryAnalysis = useCallback(async (
    requestText: string,
    target: 'annotation' | 'direction' | 'round',
  ) => {
    if (!options.currentSession) return
    options.setIsSupplementaryRunning(true)
    if (target === 'round') options.roundSupplementaryPendingRef.current = true
    try {
      await apiFetch(`/session/${options.currentSession.id}/supplementary-analysis`, {
        method: 'POST', body: JSON.stringify({ request_text: requestText, target }),
      })
    } catch (error) {
      console.error('Supplementary analysis error:', error)
      options.setIsSupplementaryRunning(false)
      options.roundSupplementaryPendingRef.current = false
    }
  }, [options.currentSession])

  return {
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
