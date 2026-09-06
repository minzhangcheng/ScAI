import { useCallback, type MutableRefObject } from 'react'

import { apiFetch, generateId } from '../../../hooks/useApi'
import { useAppStore, type Session, type TaskStatus } from '../../../stores/appStore'
import type {
  SnapshotFailureAction,
  SnapshotFailureInfo,
  SnapshotTransactionRecord,
} from '../../../types/snapshotV2'

/** Narrow a rejected apiFetch to its error envelope without importing the
 * class: this module is loaded by tests that replace the whole `useApi` module,
 * where an `instanceof` against a mocked-away export throws on its own. Only
 * `code` and `message` are read here. */
function asApiErrorEnvelope(
  error: unknown,
): { code: string; message?: string } | null {
  if (!error || typeof error !== 'object') return null
  const candidate = error as { code?: unknown; message?: unknown }
  if (typeof candidate.code !== 'string') return null
  return {
    code: candidate.code,
    message: typeof candidate.message === 'string' ? candidate.message : undefined,
  }
}

interface SnapshotFailureActionOptions {
  currentSession: Session | null
  snapshotFailure: SnapshotFailureInfo | null
  clearSnapshotFailure: () => void
  monitorSnapshotRetry: (sessionId: string, seed: SnapshotTransactionRecord) => void
  snapshotAutoOpenedForRef: MutableRefObject<string | null>
  setShowSnapshotPanel: (show: boolean) => void
  wsConnected: boolean
  wsConnect: (() => void) | null
  addMessage: ReturnType<typeof useAppStore.getState>['addMessage']
  language: string
}

export function useAnalysisPanelSnapshotFailureActions(options: SnapshotFailureActionOptions) {
  // Every runtime-gate action can 409. Swallowing that into console.error left
  // a modal with three buttons that could only 409 again and no close control,
  // so a terminated session showed a dialog the user could not act on or
  // dismiss. INACTIVE specifically means the gate is gone -- the session ended,
  // or another client answered it -- and the dialog is then stale, not live.
  const runSnapshotFailureCapture = useCallback(async (action: 'retry' | 'proceed') => {
    if (!options.currentSession) return
    try {
      const response = await apiFetch<{
        success?: boolean
        data?: {
          status?: TaskStatus
          still_blocked?: boolean
          snapshot_failure?: Record<string, unknown> | null
          snapshot_failure_allowed_actions?: SnapshotFailureAction[]
          transaction_id?: string
          snapshot_transaction?: SnapshotTransactionRecord
        }
      }>(`/session/${options.currentSession.id}/snapshot-failure/${action}`, {
        method: 'POST',
        body: JSON.stringify({ regenerate_reports: false }),
      })
      if (response?.success !== true) {
        const rawFailure = response?.data?.snapshot_failure
        if (rawFailure && typeof rawFailure === 'object') {
          useAppStore.getState().hydrateRuntimeRestoreFromSession({
            snapshot_failure: rawFailure,
            snapshot_failure_allowed_actions: response?.data?.snapshot_failure_allowed_actions,
          })
        }
        return
      }
      const transaction = response?.data?.snapshot_transaction
      if (transaction) {
        options.snapshotAutoOpenedForRef.current = transaction.transaction_id
        options.setShowSnapshotPanel(true)
        options.monitorSnapshotRetry(options.currentSession.id, transaction)
      }
      if (response?.data?.status) useAppStore.getState().updateSession({ status: response.data.status })
      options.clearSnapshotFailure()
      if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
    } catch (error) {
      console.error(`Snapshot failure ${action} failed:`, error)
    }
  }, [
    options.currentSession,
    options.clearSnapshotFailure,
    options.monitorSnapshotRetry,
    options.wsConnected,
    options.wsConnect,
  ])

  // A failed runtime save no longer opens a gate, so there is no
  // runtime-persistence retry to offer. This handler now serves only Snapshot
  // V2's capture failure, which keeps its own gate until A2-5.
  const handleSnapshotFailureRetry = useCallback(async () => {
    if (!options.currentSession) return
    await runSnapshotFailureCapture('retry')
  }, [
    options.currentSession,
    runSnapshotFailureCapture,
  ])

  const handleSnapshotFailureProceed = useCallback(
    () => runSnapshotFailureCapture('proceed'),
    [runSnapshotFailureCapture],
  )

  const handleSnapshotFailureSkip = useCallback(async () => {
    if (!options.currentSession) return
    try {
      const response = await apiFetch<{ success?: boolean; data?: { status?: TaskStatus } }>(
        `/session/${options.currentSession.id}/snapshot-failure/skip`,
        { method: 'POST' },
      )
      if (response?.success !== true) return
      if (response?.data?.status) useAppStore.getState().updateSession({ status: response.data.status })
      options.clearSnapshotFailure()
      if (!options.wsConnected) setTimeout(() => options.wsConnect?.(), 100)
    } catch (error) {
      console.error('Snapshot failure skip failed:', error)
    }
  }, [options.currentSession, options.clearSnapshotFailure, options.wsConnected, options.wsConnect])

  const handleSnapshotFailureManage = useCallback(async () => {
    if (!options.currentSession) return
    options.setShowSnapshotPanel(true)
  }, [
    options.currentSession,
  ])

  const handleSnapshotFailureTerminate = useCallback(async () => {
    if (!options.currentSession) return
    try {
      const endpoint = `/session/${options.currentSession.id}/snapshot-failure/terminate`
      const response = await apiFetch<{
        success?: boolean
        message?: string
        data?: {
          status?: TaskStatus
          still_blocked?: boolean
          action_pending?: boolean
          cleanup_incomplete?: boolean
          termination_committed?: boolean
        }
      }>(endpoint, { method: 'POST' })
      if (response?.data?.status) useAppStore.getState().updateSession({ status: response.data.status })
      if (response?.data?.termination_committed
        || (response?.success && response?.data?.still_blocked === false)) {
        options.clearSnapshotFailure()
      }
      if (!response?.success) {
        options.addMessage({
          id: generateId(),
          type: 'error',
          content: response?.message || (options.language === 'zh'
            ? '终止已提交，但后台资源清理尚未完成；系统将继续重试。'
            : 'Termination was committed, but background resource cleanup is incomplete and will be retried.'),
          timestamp: new Date().toISOString(),
        })
      }
    } catch (error) {
      console.error('Snapshot failure terminate failed:', error)
    }
  }, [
    options.currentSession,
    options.clearSnapshotFailure,
    options.addMessage,
    options.language,
  ])

  return {
    handleSnapshotFailureRetry,
    handleSnapshotFailureProceed,
    handleSnapshotFailureSkip,
    handleSnapshotFailureManage,
    handleSnapshotFailureTerminate,
  }
}
