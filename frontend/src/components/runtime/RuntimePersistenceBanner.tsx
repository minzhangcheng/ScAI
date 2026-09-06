export type RuntimeCoherence = 'current' | 'stale' | 'inconsistent' | 'unavailable'

export interface RuntimePersistenceRecord {
  coherence?: string | null
  /** ISO time of the last **successful** commit, never the last attempt. */
  exposure_since?: string | null
  error_summary?: string | null
}

interface RuntimePersistenceBannerProps {
  record: RuntimePersistenceRecord | null | undefined
  isZh?: boolean
}

function formatSince(raw: string | null | undefined, isZh: boolean): string | null {
  if (!raw) return null
  const at = new Date(raw)
  if (Number.isNaN(at.getTime())) return null
  return at.toLocaleString(isZh ? 'zh-CN' : undefined)
}

/**
 * What the last committed runtime generation actually promises. Spec 5.1.
 *
 * Persistent and non-blocking by design: the capture gate that used to stop the
 * analysis here is gone, and there is no action a dialog could offer — the
 * previous generation is intact and the run continues either way. What the user
 * needs is to know that "continue where I left off" is no longer guaranteed,
 * and from when.
 *
 * The time shown is the last **successful** commit. Measuring from the last
 * attempt would keep reading "saved a moment ago" through an unbounded run of
 * failures, which is precisely the lie spec 5.1 forbids.
 */
export function RuntimePersistenceBanner({
  record,
  isZh = false,
}: RuntimePersistenceBannerProps) {
  const coherence = record?.coherence
  if (!coherence || coherence === 'current') return null

  const since = formatSince(record?.exposure_since, isZh)

  if (coherence === 'unavailable') {
    return (
      <div
        role="status"
        className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-sm text-red-900 dark:text-red-200"
      >
        <div className="font-medium">
          {isZh ? '没有已提交的运行时' : 'No committed runtime'}
        </div>
        <p className="text-xs mt-0.5">
          {isZh
            ? '本会话没有可恢复的内核状态。结果仍可查看和导出，但要继续分析需要从数据重新运行。'
            : 'This session has no kernel state to restore. Results stay viewable and exportable, but continuing the analysis means re-running from the data.'}
        </p>
      </div>
    )
  }

  return (
    <div
      role="status"
      className="rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 px-3 py-2 text-sm text-amber-900 dark:text-amber-200"
    >
      <div className="font-medium">
        {isZh ? '运行时状态落后于本次分析' : 'The saved runtime is behind this analysis'}
      </div>
      <p className="text-xs mt-0.5">
        {isZh
          ? '结果仍可查看和导出。但最近一次成功保存之后的工作没有落盘，因此不保证能从中断处精确继续。'
          : 'Results stay viewable and exportable. Work done since the last successful save is not on disk, so continuing exactly where the analysis stopped is not guaranteed.'}
      </p>
      {since && (
        <p className="text-xs mt-0.5 opacity-80">
          {isZh ? `上次成功保存：${since}` : `Last successful save: ${since}`}
        </p>
      )}
      {record?.error_summary && (
        <p className="text-xs mt-0.5 opacity-80">{record.error_summary}</p>
      )}
    </div>
  )
}
