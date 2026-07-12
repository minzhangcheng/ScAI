import type { SnapshotTransactionRecord } from '../../types/snapshotV2'

const TERMINAL = new Set(['committed', 'failed', 'cancelled', 'timed_out'])

interface RestoreProgressDialogProps {
  transaction: SnapshotTransactionRecord
  isZh?: boolean
  onCancel?: () => void
  onDismiss?: () => void
  cancelling?: boolean
}

export function RestoreProgressDialog({
  transaction,
  isZh = false,
  onCancel,
  onDismiss,
  cancelling = false,
}: RestoreProgressDialogProps) {
  const status = transaction.status
  const isTerminal = TERMINAL.has(status)
  const failureMessage =
    transaction.failure && typeof transaction.failure === 'object'
      ? String((transaction.failure as { message?: string }).message || (transaction.failure as { code?: string }).code || '')
      : ''

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40" role="dialog" aria-label="Restore progress">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4 p-5 space-y-3">
        <h3 className="text-base font-semibold text-slate-900">
          {isZh ? '恢复进度' : 'Restore Progress'}
        </h3>
        <div className="text-sm text-slate-700">
          <span className="text-slate-500">{isZh ? '状态：' : 'Status: '}</span>
          <span className="font-medium capitalize">{status.replace(/_/g, ' ')}</span>
        </div>
        {transaction.recovery_mode && (
          <div className="text-xs text-slate-500">
            {isZh ? '恢复模式：' : 'Recovery mode: '}
            {transaction.recovery_mode}
          </div>
        )}
        {status === 'timed_out' && (
          <div className="p-2 bg-amber-50 text-amber-800 text-xs rounded-lg">
            {isZh ? '等待安全点超时' : 'Timed out waiting for a safe point'}
            {failureMessage ? `: ${failureMessage}` : ''}
          </div>
        )}
        {status === 'failed' && failureMessage && (
          <div className="p-2 bg-red-50 text-red-700 text-xs rounded-lg">{failureMessage}</div>
        )}
        <div className="flex justify-end gap-2 pt-1">
          {!isTerminal && onCancel && (
            <button
              type="button"
              onClick={onCancel}
              disabled={cancelling}
              className="px-3 py-1.5 text-xs text-red-600 hover:bg-red-50 rounded-lg border border-red-200 disabled:opacity-50"
            >
              {cancelling ? '...' : (isZh ? '取消' : 'Cancel')}
            </button>
          )}
          {isTerminal && onDismiss && (
            <button
              type="button"
              onClick={onDismiss}
              className="px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100 rounded-lg border border-slate-200"
            >
              {isZh ? '关闭' : 'Dismiss'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
