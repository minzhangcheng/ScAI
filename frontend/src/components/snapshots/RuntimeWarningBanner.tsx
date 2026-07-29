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
        className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900"
      >
        <div className="font-medium">
          {isZh ? '已有意保留运行时（风险）' : 'Runtime intentionally kept (risk)'}
        </div>
        <p className="text-xs mt-0.5 text-amber-800">
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
      className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
    >
      <div className="font-medium">
        {isZh ? '运行时不可用 — 只读 / 需重新运行' : 'Runtime unavailable — read-only / re-run required'}
      </div>
      <p className="text-xs mt-0.5 text-slate-600">
        {warning ||
          (isZh
            ? '会话在无一致内核运行时的情况下恢复。继续分析已阻止；可只读查看，或点击“重新运行”从原始数据重新计算。'
            : 'Session was restored without a consistent kernel runtime. Forward resume is blocked; view read-only, or use the Re-run button to recompute from the raw inputs.')}
      </p>
    </div>
  )
}
