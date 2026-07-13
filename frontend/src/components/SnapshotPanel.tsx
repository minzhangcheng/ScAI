import { useState, useEffect, useCallback, useRef } from 'react'
import { Camera, RotateCcw, Trash2, X } from 'lucide-react'
import { Language } from '../i18n/translations'
import {
  listSnapshotsV2,
  deleteSnapshotV2,
  restoreSnapshotV2,
  cancelSnapshotTransaction,
  getSnapshotTransaction,
} from '../hooks/useApi'
import { useAppStore } from '../stores/appStore'
import { StorageSummary } from './snapshots/StorageSummary'
import { SnapshotTree } from './snapshots/SnapshotTree'
import { SnapshotPreview } from './snapshots/SnapshotPreview'
import { DiscardCurrentDialog } from './snapshots/DiscardCurrentDialog'
import { RestoreProgressDialog } from './snapshots/RestoreProgressDialog'
import { RuntimeChoiceDialog, type RuntimeChoice } from './snapshots/RuntimeChoiceDialog'
import { SNAPSHOT_TERMINAL_STATUSES } from '../utils/snapshotV2'
import type {
  SnapshotRecordV2,
  SnapshotStoreStatus,
  SnapshotTransactionRecord,
  RestoreSnapshotV2Request,
} from '../types/snapshotV2'

const POLL_INTERVAL_MS = 2000

interface SnapshotPanelProps {
  sessionId: string
  language: Language
  onClose: () => void
  onRestore: () => void
  /** Optional: resume polling an in-flight restore after reload */
  activeTransactionId?: string | null
}

function asApiError(err: unknown): { status: number; code: string; runtimeStatus: string | null; message: string } | null {
  if (err && typeof err === 'object' && 'code' in err) {
    const e = err as { code: string; status?: number; runtimeStatus?: string | null; message?: string }
    return { status: e.status ?? 500, code: e.code, runtimeStatus: e.runtimeStatus ?? null, message: e.message ?? '' }
  }
  return null
}

export function SnapshotPanel({
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
  const [deleteBranch, setDeleteBranch] = useState(false)
  const [showDiscard, setShowDiscard] = useState(false)
  const [pendingRestoreId, setPendingRestoreId] = useState<string | null>(null)
  const [showRuntimeChoice, setShowRuntimeChoice] = useState(false)
  const [runtimeStatusHint, setRuntimeStatusHint] = useState('missing')
  const [pendingRuntimeRestoreId, setPendingRuntimeRestoreId] = useState<string | null>(null)
  const [localTxn, setLocalTxn] = useState<SnapshotTransactionRecord | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const lastStatusRef = useRef<string | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

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
    },
    [setSnapshotTransaction, onRestore, fetchSnapshots],
  )

  // Sync WS store updates into local progress (dedupe by status)
  useEffect(() => {
    if (!storeTxn?.transaction_id) return
    applyTxnUpdate(storeTxn as SnapshotTransactionRecord)
  }, [storeTxn, applyTxnUpdate])

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const startPolling = useCallback(
    (transactionId: string) => {
      stopPolling()
      const poll = async () => {
        try {
          const txn = await getSnapshotTransaction(sessionId, transactionId)
          applyTxnUpdate(txn)
          if (SNAPSHOT_TERMINAL_STATUSES.has(txn.status)) {
            stopPolling()
          }
        } catch (e) {
          console.error('Failed to poll snapshot transaction:', e)
        }
      }
      void poll()
      pollTimerRef.current = setInterval(poll, POLL_INTERVAL_MS)
    },
    [sessionId, applyTxnUpdate, stopPolling],
  )

  useEffect(() => {
    if (activeTransactionId) {
      startPolling(activeTransactionId)
    }
    return () => stopPolling()
  }, [activeTransactionId, startPolling, stopPolling])

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
      startPolling(resp.transaction_id)
      setConfirmRestoreId(null)
      setPendingRestoreId(null)
    } catch (e) {
      const apiErr = asApiError(e)
      if (apiErr) {
        const upper = apiErr.code.toUpperCase()
        if (upper === 'PRE_RESTORE_UNAVAILABLE' || apiErr.code === 'pre_restore_unavailable') {
          setPendingRestoreId(snapshotId)
          setShowDiscard(true)
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
    await deleteSnapshotV2(sessionId, snapshotId, branch)
    setConfirmDeleteId(null)
    setDeleteBranch(false)
    await fetchSnapshots()
  }

  const handleCancelTxn = async () => {
    const txnId = activeTxn?.transaction_id
    if (!txnId) return
    setCancelling(true)
    try {
      await cancelSnapshotTransaction(sessionId, txnId)
      const reloaded = await getSnapshotTransaction(sessionId, txnId)
      applyTxnUpdate(reloaded)
      if (SNAPSHOT_TERMINAL_STATUSES.has(reloaded.status)) {
        stopPolling()
      }
    } catch (e) {
      console.error('Cancel restore failed:', e)
    } finally {
      setCancelling(false)
    }
  }

  const isProtected = (record: SnapshotRecordV2) => {
    return record.manifest.capsule.kind === 'terminal'
  }

  const showProgress = Boolean(activeTxn)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl mx-4 max-h-[80vh] flex flex-col" role="dialog" aria-label="Analysis Snapshots">
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-accent-600" />
            <h2 className="text-lg font-semibold text-slate-900">
              {isZh ? '分析快照' : 'Analysis Snapshots'}
            </h2>
            {snapshots.length > 0 && (
              <span className="text-xs text-slate-400 ml-1">({snapshots.length})</span>
            )}
          </div>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg" aria-label="Close">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4">
          {loading ? (
            <div className="text-center text-slate-400 py-8">{isZh ? '加载中...' : 'Loading...'}</div>
          ) : snapshots.length === 0 ? (
            <div className="text-center text-slate-400 py-8">
              {isZh ? '暂无快照（快照会在分析关键节点自动创建）' : 'No snapshots (snapshots are created automatically at key analysis points)'}
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {storeStatus && <StorageSummary status={storeStatus} />}

              <div className="flex gap-4">
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
                    {isZh ? '快照列表' : 'Snapshot Lineage'}
                  </div>
                  <SnapshotTree
                    snapshots={snapshots}
                    selectedId={selectedId}
                    onSelect={setSelectedId}
                  />
                </div>

                {selectedRecord && (
                  <div className="w-72 shrink-0">
                    <SnapshotPreview record={selectedRecord} />
                    <div className="flex items-center gap-2 mt-3 flex-wrap">
                      {confirmRestoreId === selectedRecord.snapshot_id ? (
                        <div className="flex flex-col gap-2 w-full">
                          {restoreError && (
                            <div className="p-2 bg-red-50 rounded text-xs text-red-700">{restoreError}</div>
                          )}
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-amber-600">{isZh ? '确认恢复？' : 'Confirm restore?'}</span>
                            <button
                              type="button"
                              onClick={() => handleConfirmRestore(selectedRecord.snapshot_id)}
                              disabled={restoring}
                              className="px-2 py-1 text-xs bg-amber-500 text-white rounded-lg hover:bg-amber-600 disabled:opacity-50"
                            >
                              {restoring ? '...' : (isZh ? '开始恢复' : 'Start Restore')}
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setConfirmRestoreId(null)
                                setRestoreError(null)
                              }}
                              className="px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 rounded-lg"
                            >
                              {isZh ? '取消' : 'No'}
                            </button>
                          </div>
                        </div>
                      ) : confirmDeleteId === selectedRecord.snapshot_id ? (
                        <div className="flex flex-col gap-2 w-full">
                          <label className="flex items-center gap-1 text-xs text-slate-600">
                            <input
                              type="checkbox"
                              checked={deleteBranch}
                              onChange={(e) => setDeleteBranch(e.target.checked)}
                            />
                            {isZh ? '删除整支分支' : 'Delete entire branch'}
                          </label>
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => handleDelete(selectedRecord.snapshot_id, deleteBranch)}
                              className="px-2 py-1 text-xs bg-red-600 text-white rounded-lg hover:bg-red-700"
                            >
                              {isZh ? '确认删除' : 'Confirm Delete'}
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setConfirmDeleteId(null)
                                setDeleteBranch(false)
                              }}
                              className="px-2 py-1 text-xs text-slate-500 hover:bg-slate-100 rounded-lg"
                            >
                              {isZh ? '取消' : 'No'}
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <button
                            type="button"
                            onClick={() => setConfirmRestoreId(selectedRecord.snapshot_id)}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs text-accent-600 hover:bg-accent-50 rounded-lg border border-accent-200"
                            title={isZh ? '恢复' : 'Restore'}
                          >
                            <RotateCcw className="w-3.5 h-3.5" />
                            {isZh ? '恢复' : 'Restore'}
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteId(selectedRecord.snapshot_id)}
                            disabled={isProtected(selectedRecord)}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs text-red-500 hover:bg-red-50 rounded-lg border border-red-200 disabled:opacity-40 disabled:cursor-not-allowed"
                            title={isZh ? '删除' : 'Delete'}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            {isZh ? '删除' : 'Delete'}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {showDiscard && (
        <DiscardCurrentDialog
          isZh={isZh}
          onAbort={() => {
            setShowDiscard(false)
            setPendingRestoreId(null)
          }}
          onConfirm={handleConfirmDiscard}
        />
      )}

      <RuntimeChoiceDialog
        open={showRuntimeChoice}
        runtimeStatus={runtimeStatusHint}
        isZh={isZh}
        onChoose={(choice) => {
          void handleRuntimeChoice(choice)
        }}
        onAbort={() => {
          setShowRuntimeChoice(false)
          setPendingRuntimeRestoreId(null)
        }}
      />

      {showProgress && activeTxn && (
        <RestoreProgressDialog
          transaction={activeTxn}
          isZh={isZh}
          onCancel={
            SNAPSHOT_TERMINAL_STATUSES.has(activeTxn.status) ? undefined : handleCancelTxn
          }
          onDismiss={
            SNAPSHOT_TERMINAL_STATUSES.has(activeTxn.status)
              ? () => {
                  setLocalTxn(null)
                  lastStatusRef.current = null
                  setSnapshotTransaction(null)
                }
              : undefined
          }
          cancelling={cancelling}
        />
      )}
    </div>
  )
}
