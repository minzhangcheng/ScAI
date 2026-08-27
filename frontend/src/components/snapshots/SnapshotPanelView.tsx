import { Camera, RotateCcw, Trash2, X } from 'lucide-react'

import { SNAPSHOT_TERMINAL_STATUSES, formatBytes } from '../../utils/snapshotV2'
import { DiscardCurrentDialog } from '../snapshots/DiscardCurrentDialog'
import { RestoreProgressDialog } from '../snapshots/RestoreProgressDialog'
import { RuntimeChoiceDialog } from '../snapshots/RuntimeChoiceDialog'
import { SnapshotPreview } from '../snapshots/SnapshotPreview'
import { SnapshotTree } from '../snapshots/SnapshotTree'
import { StorageSummary } from '../snapshots/StorageSummary'
import type { SnapshotPanelController } from './useSnapshotPanelController'

interface SnapshotPanelViewProps {
  controller: SnapshotPanelController
}

export function SnapshotPanelView({ controller }: SnapshotPanelViewProps) {
  const {
    activeTxn, cancelError, cancelling, confirmDeleteId, confirmRestoreId, deleteBranch,
    deleteError, handleCancelTxn, handleConfirmDiscard, handleConfirmRestore, handleDelete,
    handleRuntimeChoice, isProtected, isRestorable, isZh, lastStatusRef, loading, onClose,
    restoring, restoreError, runtimeStatusHint, selectedId, selectedRecord, setCancelError,
    setConfirmDeleteId, setConfirmRestoreId, setDeleteBranch, setDeleteError, setLocalTxn,
    setPendingRestoreId, setPendingRuntimeRestoreId, setRestoreError, setSelectedId,
    setShowDiscard, setShowRuntimeChoice, setSnapshotTransaction, showDiscard, showProgress,
    showRuntimeChoice, snapshots, storeStatus,
  } = controller
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-surface-900 rounded-2xl shadow-2xl w-full max-w-3xl mx-4 max-h-[80vh] flex flex-col" role="dialog" aria-label="Analysis Snapshots">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2">
            <Camera className="w-5 h-5 text-accent-600" />
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {isZh ? '分析快照' : 'Analysis Snapshots'}
            </h2>
            {/*
              The feature ships disabled and is reached only by an operator who
              set SCAI_ENABLE_SNAPSHOT_V2, so this panel rendering at all means
              it is on. The marker is unconditional rather than tied to a flag:
              anyone looking at this screen is looking at the experimental
              feature, and it is deliberately undocumented for end users.
            */}
            <span
              className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-amber-100 text-amber-700 border border-amber-200"
              title={isZh
                ? '实验性功能：仍在开发中，行为与存储格式可能变化，请勿作为唯一的成果保护手段。'
                : 'Experimental: in development. Behaviour and storage format may change; do not rely on it as your only safeguard.'}
            >
              {isZh ? '实验性' : 'Experimental'}
            </span>
            {snapshots.length > 0 && (
            <span className="text-xs text-slate-400 dark:text-slate-500 ml-1">({snapshots.length})</span>
            )}
          </div>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg" aria-label="Close">
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <div className="px-6 py-2 bg-amber-50/60 dark:bg-amber-950/30 border-b border-amber-100 dark:border-amber-800 text-[11px] text-amber-800 dark:text-amber-200">
          {isZh
            ? '此功能仍在开发中，尚不稳定：快照格式可能在后续版本中变更，旧快照不保证能被恢复。请勿将其作为唯一的成果保护手段。'
            : 'This feature is in development and not yet stable: the snapshot format may change in a later build, and older snapshots are not guaranteed to restore. Do not rely on it as your only safeguard.'}
        </div>

        <div className="flex-1 overflow-auto p-4">
          {loading ? (
            <div className="text-center text-slate-400 dark:text-slate-500 py-8">{isZh ? '加载中...' : 'Loading...'}</div>
          ) : snapshots.length === 0 ? (
            <div className="text-center text-slate-400 dark:text-slate-500 py-8">
              {isZh ? '暂无快照（快照会在分析关键节点自动创建）' : 'No snapshots (snapshots are created automatically at key analysis points)'}
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {storeStatus?.quota_warning && (
                <div
                  role="alert"
                  className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-800"
                >
                  {isZh
                    ? `快照已占用会话配额的 ${Math.round((storeStatus.quota_usage_ratio ?? 0) * 100)}%（${formatBytes(storeStatus.physical_bytes)} / ${formatBytes(storeStatus.quota_bytes ?? 0)}）。请删除不再需要的快照以释放空间：配额用尽后，后续快照会失败并暂停分析。`
                    : `Snapshots use ${Math.round((storeStatus.quota_usage_ratio ?? 0) * 100)}% of the session quota (${formatBytes(storeStatus.physical_bytes)} / ${formatBytes(storeStatus.quota_bytes ?? 0)}). Delete snapshots you no longer need to free space — once the quota is reached, further captures fail and pause the analysis.`}
                </div>
              )}

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
                    isZh={isZh}
                  />
                </div>

                {selectedRecord && (
                  <div className="w-72 shrink-0">
                    <SnapshotPreview record={selectedRecord} isZh={isZh} />
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
                          {deleteError && (
                            <div className="p-2 bg-red-50 rounded text-xs text-red-700">{deleteError}</div>
                          )}
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
                                setDeleteError(null)
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
                            disabled={!isRestorable(selectedRecord)}
                            className="flex items-center gap-1 px-3 py-1.5 text-xs text-accent-600 hover:bg-accent-50 rounded-lg border border-accent-200 disabled:opacity-40 disabled:cursor-not-allowed"
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
                  setCancelError(null)
                  setSnapshotTransaction(null)
                }
              : undefined
          }
          cancelling={cancelling}
        />
      )}

      {/* Sits above the progress dialog's own z-[60] overlay: the refusal has to
          be readable next to the Cancel button that produced it. */}
      {showProgress && cancelError && (
        <div className="fixed inset-x-0 top-4 z-[70] flex justify-center px-4 pointer-events-none">
          <div className="pointer-events-auto max-w-md w-full p-2 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700">
            {isZh ? '取消恢复失败：' : 'Cancel restore failed: '}
            {cancelError}
          </div>
        </div>
      )}
    </div>
  )

}
