import { useCallback, useEffect, useRef, useState } from 'react'

import {
  cancelSnapshotTransaction, deleteSnapshotV2, getSnapshotTransaction,
  listSnapshotsV2, restoreSnapshotV2,
} from '../../hooks/useApi'
import type { Language } from '../../i18n/translations'
import { useAppStore } from '../../stores/appStore'
import type {
  RestoreSnapshotV2Request, SnapshotRecordV2, SnapshotStoreStatus,
  SnapshotTransactionRecord,
} from '../../types/snapshotV2'
import { SNAPSHOT_TERMINAL_STATUSES } from '../../utils/snapshotV2'
import type { RuntimeChoice } from '../snapshots/RuntimeChoiceDialog'

const POLL_INTERVAL_MS = 2000

// The transaction journal spells the code upper-case, the HTTP error payload
// lower-case; both mean the same refusal.
const isPreRestoreUnavailable = (code?: string | null): boolean =>
  (code || '').toUpperCase() === 'PRE_RESTORE_UNAVAILABLE'

export interface SnapshotPanelProps {
  sessionId: string
  language: Language
  onClose: () => void
  onRestore: () => void
  /** Optional: resume polling an in-flight restore after reload */
  activeTransactionId?: string | null
}

function asApiError(err: unknown): { status: number; code: string; runtimeStatus: string | null; message: string; details: Record<string, unknown> | null } | null {
  if (err && typeof err === 'object' && 'code' in err) {
    const e = err as { code: string; status?: number; runtimeStatus?: string | null; message?: string; details?: Record<string, unknown> | null }
    return { status: e.status ?? 500, code: e.code, runtimeStatus: e.runtimeStatus ?? null, message: e.message ?? '', details: e.details ?? null }
  }
  return null
}

export function useSnapshotPanelController({
  sessionId,
  language,
  onClose,
  onRestore,
  activeTransactionId = null,
}: SnapshotPanelProps) {
  const isZh = language === 'zh'
  const [snapshots, setSnapshots] = useState<SnapshotRecordV2[]>([])
  const [storeStatus, setStoreStatus] = useState<SnapshotStoreStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [restoring, setRestoring] = useState(false)
  const [restoreError, setRestoreError] = useState<string | null>(null)
  const [confirmRestoreId, setConfirmRestoreId] = useState<string | null>(null)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [deleteBranch, setDeleteBranch] = useState(false)
  const [showDiscard, setShowDiscard] = useState(false)
  const [pendingRestoreId, setPendingRestoreId] = useState<string | null>(null)
  const [showRuntimeChoice, setShowRuntimeChoice] = useState(false)
  const [runtimeStatusHint, setRuntimeStatusHint] = useState('missing')
  const [pendingRuntimeRestoreId, setPendingRuntimeRestoreId] = useState<string | null>(null)
  const [localTxn, setLocalTxn] = useState<SnapshotTransactionRecord | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const lastStatusRef = useRef<string | null>(null)
  const [pollingTxnId, setPollingTxnId] = useState<string | null>(activeTransactionId)
  const storeTxn = useAppStore((s) => s.snapshotTransaction)
  const setSnapshotTransaction = useAppStore((s) => s.setSnapshotTransaction)
  const activeTxn: SnapshotTransactionRecord | null =
    localTxn ||
    (storeTxn && storeTxn.transaction_id
      ? (storeTxn as SnapshotTransactionRecord)
      : null)
  const fetchSnapshots = useCallback(async () => {
    try {
      const data = await listSnapshotsV2(sessionId)
      setSnapshots(data.snapshots)
      setStoreStatus(data.store_status)
      useAppStore.getState().setSnapshotsV2(data.snapshots)
      useAppStore.getState().setSnapshotStoreStatus(data.store_status)
      useAppStore.getState().setSnapshotCount(data.snapshots.length)
    } catch (e) {
      console.error('Failed to fetch V2 snapshots:', e)
    } finally {
      setLoading(false)
    }
  }, [sessionId])
  useEffect(() => {
    fetchSnapshots()
  }, [fetchSnapshots])
  useEffect(() => {
    if (snapshots.length > 0 && !selectedId) {
      setSelectedId(snapshots[0].snapshot_id)
    }
  }, [snapshots, selectedId])
  const offerDiscardRestore = useCallback((snapshotId: string) => {
    setPendingRestoreId(snapshotId)
    setShowDiscard(true)
  }, [])
  const applyTxnUpdate = useCallback(
    (txn: SnapshotTransactionRecord) => {
      const key = `${txn.transaction_id}:${txn.status}`
      if (lastStatusRef.current === key) return
      lastStatusRef.current = key
      setLocalTxn(txn)
      setSnapshotTransaction(txn)
      if (txn.status === 'committed') {
        onRestore()
        fetchSnapshots()
      }
      if (
        txn.status === 'failed'
        && isPreRestoreUnavailable(txn.failure?.code as string | undefined)
        && txn.snapshot_id
      ) {
        offerDiscardRestore(txn.snapshot_id)
      }
    },
    [setSnapshotTransaction, onRestore, fetchSnapshots, offerDiscardRestore],
  )
  useEffect(() => {
    if (!storeTxn?.transaction_id) return
    applyTxnUpdate(storeTxn as SnapshotTransactionRecord)
  }, [storeTxn, applyTxnUpdate])
  const applyTxnUpdateRef = useRef(applyTxnUpdate)
  applyTxnUpdateRef.current = applyTxnUpdate
  useEffect(() => {
    if (activeTransactionId) setPollingTxnId(activeTransactionId)
  }, [activeTransactionId])
  useEffect(() => {
    if (!pollingTxnId) return
    let cancelled = false
    const poll = async () => {
      try {
        const txn = await getSnapshotTransaction(sessionId, pollingTxnId)
        if (cancelled) return
        applyTxnUpdateRef.current(txn)
        if (SNAPSHOT_TERMINAL_STATUSES.has(txn.status)) {
          setPollingTxnId(null)
        }
      } catch (e) {
        console.error('Failed to poll snapshot transaction:', e)
      }
    }
    void poll()
    const timer = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [sessionId, pollingTxnId])
  const selectedRecord = snapshots.find((s) => s.snapshot_id === selectedId) || null
  const beginRestore = async (
    snapshotId: string,
    request: RestoreSnapshotV2Request,
  ) => {
    setRestoring(true)
    setRestoreError(null)
    setShowDiscard(false)
    try {
      const resp = await restoreSnapshotV2(sessionId, snapshotId, request)
      const seed: SnapshotTransactionRecord = {
        transaction_id: resp.transaction_id,
        session_id: sessionId,
        snapshot_id: snapshotId,
        status: (resp.status as SnapshotTransactionRecord['status']) || 'queued',
        recovery_mode: 'rollback',
        failure: null,
      }
      applyTxnUpdate(seed)
      setPollingTxnId(resp.transaction_id)
      setConfirmRestoreId(null)
      setPendingRestoreId(null)
    } catch (e) {
      const apiErr = asApiError(e)
      if (apiErr) {
        const upper = apiErr.code.toUpperCase()
        if (isPreRestoreUnavailable(apiErr.code)) {
          offerDiscardRestore(snapshotId)
        } else if (
          upper === 'RUNTIME_CHOICE_REQUIRED' ||
          upper === 'RUNTIME_UNAVAILABLE' ||
          (apiErr.status === 409 && /runtime/i.test(apiErr.message) && /choice|required|unavailable/i.test(apiErr.message))
        ) {
          setPendingRuntimeRestoreId(snapshotId)
          setRuntimeStatusHint(apiErr.runtimeStatus || 'missing')
          setShowRuntimeChoice(true)
        } else {
          setRestoreError(apiErr.message || (isZh ? '回退失败' : 'Restore failed'))
        }
      } else {
        setRestoreError(e instanceof Error ? e.message : (isZh ? '回退失败' : 'Restore failed'))
      }
    } finally {
      setRestoring(false)
    }
  }
  const handleConfirmRestore = async (snapshotId: string) => {
    await beginRestore(snapshotId, {
      snapshot_id: snapshotId,
      current_state_choice: 'abort',
      discard_confirmation: false,
    })
  }
  const handleConfirmDiscard = async () => {
    if (!pendingRestoreId) return
    await beginRestore(pendingRestoreId, {
      snapshot_id: pendingRestoreId,
      current_state_choice: 'discard_current',
      discard_confirmation: true,
      runtime_choice: 'reset',
    })
  }
  const handleRuntimeChoice = async (choice: RuntimeChoice) => {
    if (!pendingRuntimeRestoreId) return
    setShowRuntimeChoice(false)
    const snapshotId = pendingRuntimeRestoreId
    setPendingRuntimeRestoreId(null)
    await beginRestore(snapshotId, {
      snapshot_id: snapshotId,
      current_state_choice: 'abort',
      discard_confirmation: false,
      runtime_choice: choice,
    })
  }
  const handleDelete = async (snapshotId: string, branch: boolean) => {
    setDeleteError(null)
    try {
      await deleteSnapshotV2(sessionId, snapshotId, branch)
    } catch (e) {
      const err = asApiError(e)
      const reason = err?.details?.reason as string | undefined
      setDeleteError(
        [
          err?.message
            || (e instanceof Error ? e.message : '')
            || (isZh ? '删除失败' : 'Delete failed'),
          reason ? `(${reason})` : '',
        ]
          .filter(Boolean)
          .join(' '),
      )
      return
    } finally {
      await fetchSnapshots()
    }
    setConfirmDeleteId(null)
    setDeleteBranch(false)
  }
  const handleCancelTxn = async () => {
    const txnId = activeTxn?.transaction_id
    if (!txnId) return
    setCancelling(true)
    setCancelError(null)
    try {
      await cancelSnapshotTransaction(sessionId, txnId)
      const reloaded = await getSnapshotTransaction(sessionId, txnId)
      applyTxnUpdate(reloaded)
      if (SNAPSHOT_TERMINAL_STATUSES.has(reloaded.status)) {
        setPollingTxnId(null)
      }
    } catch (e) {
      const err = asApiError(e)
      setCancelError(
        (err?.message || (e instanceof Error ? e.message : ''))
        || (isZh ? '取消恢复失败' : 'Cancel restore failed'),
      )
    } finally {
      setCancelling(false)
    }
  }
  const isProtected = (record: SnapshotRecordV2) => {
    return record.manifest.capsule.kind === 'terminal'
  }
  const isRestorable = (record: SnapshotRecordV2) => record.failure === null
  const showProgress = Boolean(activeTxn)
  return {
    activeTxn, cancelError, cancelling, confirmDeleteId, confirmRestoreId, deleteBranch,
    deleteError, handleCancelTxn, handleConfirmDiscard, handleConfirmRestore, handleDelete,
    handleRuntimeChoice, isProtected, isRestorable, isZh, lastStatusRef, loading, onClose,
    pendingRestoreId, restoring, restoreError, runtimeStatusHint, selectedId, selectedRecord,
    setCancelError, setConfirmDeleteId, setConfirmRestoreId, setDeleteBranch, setDeleteError,
    setLocalTxn, setPendingRestoreId, setPendingRuntimeRestoreId, setRestoreError,
    setSelectedId, setShowDiscard, setShowRuntimeChoice, setSnapshotTransaction,
    showDiscard, showProgress, showRuntimeChoice, snapshots, storeStatus,
  }
}

export type SnapshotPanelController = ReturnType<typeof useSnapshotPanelController>
