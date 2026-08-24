export type RuntimeChoice = 'reset' | 'keep'

interface RuntimeChoiceDialogProps {
  open: boolean
  runtimeStatus: string
  isZh?: boolean
  onChoose: (choice: RuntimeChoice) => void
  onAbort: () => void
}

export function RuntimeChoiceDialog({
  open,
  runtimeStatus,
  isZh = false,
  onChoose,
  onAbort,
}: RuntimeChoiceDialogProps) {
  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40"
      role="dialog"
      aria-label="Runtime choice"
      aria-modal="true"
    >
      <div className="bg-white dark:bg-surface-900 rounded-xl shadow-xl w-full max-w-md mx-4 p-5 space-y-4">
        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          {isZh ? '运行时不可用' : 'Runtime Unavailable'}
        </h3>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          {isZh
            ? `快照运行时状态为「${runtimeStatus}」。必须显式选择重置或保留当前内核，不会静默默认。`
            : `Snapshot runtime status is “${runtimeStatus}”. Choose reset or keep explicitly — there is no silent default.`}
        </p>

        <div className="space-y-2">
          <button
            type="button"
            onClick={() => onChoose('reset')}
            className="w-full text-left px-3 py-2.5 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800"
          >
            <div className="text-sm font-medium text-slate-800">
              {isZh ? '重置为空内核' : 'Reset to empty kernel'}
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              {isZh
                ? '恢复业务状态与工作区，隔离现有运行时，并阻止继续分析，直到从原始数据重新运行。'
                : 'Restore business state and workspace, quarantine live runtime, and block forward resume until a full re-run from raw inputs.'}
            </div>
          </button>

          <button
            type="button"
            onClick={() => onChoose('keep')}
            className="w-full text-left px-3 py-2.5 rounded-lg border border-amber-200 bg-amber-50 hover:bg-amber-100"
          >
            <div className="text-sm font-medium text-amber-900">
              {isZh ? '保留当前内核（有风险）' : 'Keep current kernel (risk)'}
            </div>
            <div className="text-xs text-amber-800 mt-0.5">
              {isZh
                ? '这不是完整一致的恢复：内核可能与恢复后的业务状态不一致。风险将在重新加载后持续提示。'
                : 'Not a complete restore: the live kernel may be inconsistent with restored business state. This risk warning persists after reload.'}
            </div>
          </button>
        </div>

        <div className="flex justify-end">
          <button
            type="button"
            onClick={onAbort}
            className="px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-100 rounded-lg border border-slate-200"
          >
            {isZh ? '中止' : 'Abort'}
          </button>
        </div>
      </div>
    </div>
  )
}
