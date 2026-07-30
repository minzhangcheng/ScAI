import type { SnapshotFailureInfo } from '../../types/snapshotV2'

export interface SnapshotFailureRetryOptions {
  /** Retry re-enters capture only; never regenerates completed reports. */
  regenerateReports: false
}

interface SnapshotFailureDialogProps {
  failure: SnapshotFailureInfo
  isZh?: boolean
  onRetry: (options: SnapshotFailureRetryOptions) => void
  onProceed: () => void
  onSkip: () => void
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
  onProceed,
  onSkip,
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
  // The memory preflight refused before allocating anything. There is no Retry
  // on this gate: the check is deterministic, so it would refuse again.
  const isMemoryRefusal = record.code === 'snapshot_memory_insufficient'

  // The backend message can be a short sentence or a multi-thousand-character
  // kernel traceback. Split off the first line as a human summary and keep the
  // full text in a bounded, scrollable box so the dialog never outgrows the
  // viewport (which would push the action buttons off-screen).
  const fullMessage = (record.message || '').trim()
  const firstLine = fullMessage.split('\n', 1)[0]
  const hasMoreDetail = fullMessage.length > firstLine.length

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-label="Snapshot failure"
      aria-modal="true"
    >
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md max-h-[85vh] flex flex-col">
        <div className="flex items-start justify-between gap-3 p-5 pb-3 shrink-0 border-b border-slate-100">
          <h3 className="text-base font-semibold text-red-700">
            {isZh ? '快照失败' : 'Snapshot Failure'}
          </h3>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="text-xs text-slate-400 hover:text-slate-600 shrink-0"
            >
              {isZh ? '关闭' : 'Close'}
            </button>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-5 py-4 space-y-4">
          <div className="space-y-2 text-sm text-slate-700">
            <div>
              <span className="font-medium text-slate-500">{isZh ? '错误码' : 'Code'}: </span>
              <code className="text-red-700 font-mono text-xs break-all">{record.code}</code>
            </div>
            {firstLine && (
              <p className="text-slate-600 break-words">{firstLine}</p>
            )}
            {hasMoreDetail && (
              <details className="group">
                <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-700 select-none">
                  {isZh ? '技术细节' : 'Technical details'}
                </summary>
                <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words bg-slate-50 border border-slate-200 rounded-lg p-2 font-mono text-[11px] leading-snug text-slate-600">
                  {fullMessage}
                </pre>
              </details>
            )}
          </div>

          {variables.length > 0 && (
            <div className="bg-amber-50 border border-amber-100 rounded-lg p-3">
              <div className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">
                {isZh ? '必需变量诊断' : 'Required variable diagnostics'}
              </div>
              <ul className="list-disc list-inside text-sm text-amber-900 space-y-0.5">
                {variables.map((name) => (
                  <li key={name}>
                    <code className="font-mono text-xs break-all">{name}</code>
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

          {!isPostReport && actions.has('retry') && (
            <p className="text-xs text-slate-500">
              {isZh
                ? '重试仅重新进入捕获流程。'
                : 'Retry re-enters capture only.'}
            </p>
          )}
          {isMemoryRefusal && (
            <p className="text-xs text-slate-500">
              {isZh
                ? '快照会先复制内存中的数据集再写盘，当前可用内存不足以完成这次复制。分析本身未被中断，只有快照被拦下。重试没有意义 —— 同样的检查会得到同样的结果。'
                : 'A snapshot copies the in-memory dataset before writing it, and the available memory is not enough for that copy. The analysis itself was not interrupted — only the snapshot was refused. Retrying would not help: the same check would reach the same result.'}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-2 p-5 pt-3 shrink-0 border-t border-slate-100">
          {actions.has('retry') && (
            <button
              type="button"
              onClick={() => onRetry({ regenerateReports: false })}
              className="w-full px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
            >
              {isZh ? '重试' : 'Retry'}
            </button>
          )}
          {actions.has('proceed') && (
            <button
              type="button"
              onClick={onProceed}
              className="w-full px-3 py-2 text-sm bg-orange-50 hover:bg-orange-100 text-orange-800 border border-orange-300 rounded-lg font-medium"
              title={
                isZh
                  ? '忽略内存预检，仍然创建快照。内存不足时内核可能被终止，正在进行的分析状态将丢失。'
                  : 'Ignore the memory preflight and capture anyway. If memory runs out the kernel may be killed and the live analysis state lost.'
              }
            >
              {isZh ? '仍然创建快照（有风险）' : 'Capture anyway (risky)'}
            </button>
          )}
          {actions.has('skip') && !isPreRestore && (
            <button
              type="button"
              onClick={onSkip}
              className="w-full px-3 py-2 text-sm bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-200 rounded-lg font-medium"
              title={
                isZh
                  ? '跳过本次快照创建，继续分析（该检查点将无法用于恢复）'
                  : 'Skip creating this snapshot and continue the analysis (this checkpoint will not be restorable)'
              }
            >
              {isZh ? '跳过快照并继续' : 'Skip snapshot & continue'}
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
