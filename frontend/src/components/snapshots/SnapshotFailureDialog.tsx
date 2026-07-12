import type { SnapshotFailureInfo } from '../../types/snapshotV2'

export interface SnapshotFailureRetryOptions {
  /** Retry re-enters capture only; never regenerates completed reports. */
  regenerateReports: false
}

interface SnapshotFailureDialogProps {
  failure: SnapshotFailureInfo
  isZh?: boolean
  onRetry: (options: SnapshotFailureRetryOptions) => void
  onManage: () => void
  onTerminate: () => void
  onClose?: () => void
}

function extractNamedVariables(details?: Record<string, unknown>): string[] {
  if (!details) return []
  const names = new Set<string>()
  const raw = details.required_variables
  if (Array.isArray(raw)) {
    for (const item of raw) {
      if (typeof item === 'string' && item.trim()) names.add(item.trim())
    }
  }
  if (typeof details.name === 'string' && details.name.trim()) {
    names.add(details.name.trim())
  }
  return Array.from(names)
}

function phaseOf(details?: Record<string, unknown>): string | null {
  if (!details) return null
  if (typeof details.phase === 'string') return details.phase
  return null
}

export function SnapshotFailureDialog({
  failure,
  isZh = false,
  onRetry,
  onManage,
  onTerminate,
  onClose,
}: SnapshotFailureDialogProps) {
  const record = failure.failure
  const actions = new Set(failure.allowed_actions || [])
  const variables = extractNamedVariables(record.details)
  const phase = phaseOf(record.details)
  const isPreRestore = phase === 'pre_restore' || /pre_restore/i.test(record.code || '')
  const isPostReport =
    phase === 'post_report' || /post_report/i.test(record.code || '')

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40"
      role="dialog"
      aria-label="Snapshot failure"
      aria-modal="true"
    >
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4 p-5 space-y-4">
        <div className="flex items-start justify-between gap-3">
          <h3 className="text-base font-semibold text-red-700">
            {isZh ? '快照失败' : 'Snapshot Failure'}
          </h3>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="text-xs text-slate-400 hover:text-slate-600"
            >
              {isZh ? '关闭' : 'Close'}
            </button>
          )}
        </div>

        <div className="space-y-2 text-sm text-slate-700">
          <div>
            <span className="font-medium text-slate-500">{isZh ? '错误码' : 'Code'}: </span>
            <code className="text-red-700 font-mono text-xs">{record.code}</code>
          </div>
          <p className="text-slate-600">{record.message}</p>
        </div>

        {variables.length > 0 && (
          <div className="bg-amber-50 border border-amber-100 rounded-lg p-3">
            <div className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">
              {isZh ? '必需变量诊断' : 'Required variable diagnostics'}
            </div>
            <ul className="list-disc list-inside text-sm text-amber-900 space-y-0.5">
              {variables.map((name) => (
                <li key={name}>
                  <code className="font-mono text-xs">{name}</code>
                </li>
              ))}
            </ul>
          </div>
        )}

        {isPreRestore && (
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-600">
            <div className="font-semibold text-slate-700 mb-1">pre_restore</div>
            {isZh
              ? '预恢复快照捕获失败。当前工作流保持可恢复状态，尚未对线上状态做任何变更。'
              : 'Pre-restore capture failed. The current workflow remains resumable; no live mutation occurred.'}
          </div>
        )}

        {isPostReport && (
          <p className="text-xs text-slate-500">
            {isZh
              ? '重试仅重新进入捕获流程，不会重新生成已完成的报告。'
              : 'Retry re-enters capture only and does not regenerate completed reports.'}
          </p>
        )}

        {!isPostReport && (
          <p className="text-xs text-slate-500">
            {isZh
              ? '重试仅重新进入捕获流程。'
              : 'Retry re-enters capture only.'}
          </p>
        )}

        <div className="flex flex-col gap-2 pt-1">
          {actions.has('retry') && (
            <button
              type="button"
              onClick={() => onRetry({ regenerateReports: false })}
              className="w-full px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
            >
              {isZh ? '重试' : 'Retry'}
            </button>
          )}
          {actions.has('manage') && (
            <button
              type="button"
              onClick={onManage}
              className="w-full px-3 py-2 text-sm bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg font-medium"
            >
              {isZh ? '管理快照' : 'Manage Snapshots'}
            </button>
          )}
          {actions.has('terminate') && (
            <button
              type="button"
              onClick={onTerminate}
              className="w-full px-3 py-2 text-sm bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 rounded-lg font-medium"
            >
              {isZh ? '终止' : 'Terminate'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
