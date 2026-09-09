export type RuntimeRestoreStatus = 'complete' | 'unavailable' | 'intentionally_kept'

interface RuntimeWarningBannerProps {
  runtimeRestoreStatus: RuntimeRestoreStatus | string | null | undefined
  warning?: string | null
  isZh?: boolean
}

export function RuntimeWarningBanner({
  runtimeRestoreStatus,
  warning,
  isZh = false,
}: RuntimeWarningBannerProps) {
  if (!runtimeRestoreStatus || runtimeRestoreStatus === 'complete') {
    return null
  }

  if (runtimeRestoreStatus === 'intentionally_kept') {
    return (
      <div
        role="status"
        className="rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 px-3 py-2 text-sm text-amber-900 dark:text-amber-200"
      >
        <div className="font-medium">
          {isZh ? '已有意保留运行时（风险）' : 'Runtime intentionally kept (risk)'}
        </div>
        <p className="text-xs mt-0.5 text-amber-800 dark:text-amber-200">
          {warning ||
            (isZh
              ? '恢复后保留了当前内核，可能与业务状态不一致。该警告在重新加载后仍会显示。'
              : 'Live kernel was retained after restore and may be inconsistent with restored state. This risk warning persists after reload.')}
        </p>
      </div>
    )
  }

  // unavailable (reset restore)
  return (
    <div
      role="status"
      className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 px-3 py-2 text-sm text-slate-700 dark:text-slate-200"
    >
      <div className="font-medium">
        {isZh ? '无法继续 — 可只读查看或恢复' : 'Cannot continue — view read-only or recover'}
      </div>
      <p className="text-xs mt-0.5 text-slate-600 dark:text-slate-300">
        {warning ||
          (isZh
            ? '当前内核状态与本会话记录的工作对不上，继续分析已阻止。「恢复并继续」会回到最近一次保存的内核状态并从那里继续，只有在没有可用状态时才从原始数据重跑；也可以只读查看结果。'
            : 'The kernel state does not match the work this session records, so forward analysis is blocked. Recover returns to the last saved runtime and continues from there, and re-runs from the raw inputs only if none can be used. Results stay viewable read-only.')}
      </p>
    </div>
  )
}
