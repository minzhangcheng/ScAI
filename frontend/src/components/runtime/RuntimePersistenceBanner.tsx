export type RuntimeCoherence = 'current' | 'stale' | 'inconsistent' | 'unavailable'

export interface RuntimePersistenceRecord {
  coherence?: string | null
  /** ISO time of the last **successful** commit, never the last attempt. */
  exposure_since?: string | null
  error_summary?: string | null
  /** Outcome of the last save *attempt*, which is what separates a kernel that
      is merely ahead of its last generation from one whose save is failing. */
  last_attempt_status?: string | null
  /** The record's own lifecycle state (``live_dirty``, ``current``, ...). Not
      read here -- ``coherence`` answers the question this banner asks -- but
      declared because it travels with the record. */
  status?: string | null
  /** Written by a pair restore (spec B2-3): what continuing from an earlier
      generation cost. Present only until the next committed generation. */
  pair_restore?: {
    at?: string | null
    restored_revision?: number | null
    discarded_revisions?: number | null
    discarded_since?: string | null
  } | null
}

interface RuntimePersistenceBannerProps {
  record: RuntimePersistenceRecord | null | undefined
  /** The session's workflow status, or undefined when it is not known. */
  sessionStatus?: string | null
  isZh?: boolean
}

/**
 * Statuses in which the workflow still holds the session and its kernel.
 *
 * `paused` is deliberately absent, and so is `pending`: a pause can outlive the
 * process, and after a restart the namespace really is gone -- which is exactly
 * the case the warning exists for.
 */
const LIVE_STATUSES: ReadonlySet<string> = new Set([
  'running',
  'generating_report',
  'waiting_input',
  'waiting_annotation',
  'waiting_plan_review',
  'waiting_species_input',
])

function formatSince(raw: string | null | undefined, isZh: boolean): string | null {
  if (!raw) return null
  const at = new Date(raw)
  if (Number.isNaN(at.getTime())) return null
  return at.toLocaleString(isZh ? 'zh-CN' : undefined)
}

/**
 * What the last committed runtime generation actually promises. Spec 5.1.
 *
 * Non-blocking by design: the capture gate that used to stop the analysis here
 * is gone, and there is no action a dialog could offer — the previous generation
 * is intact and the run continues either way. A live kernel normally advances
 * between successful save points, so that ordinary gap stays silent. A failed
 * save, an inconsistent generation, or a non-live stale session still needs the
 * warning because exact continuation is then at risk beyond normal live work.
 *
 * The time shown is the last **successful** commit. Measuring from the last
 * attempt would keep reading "saved a moment ago" through an unbounded run of
 * failures, which is precisely the lie spec 5.1 forbids.
 */
export function RuntimePersistenceBanner({
  record,
  sessionStatus,
  isZh = false,
}: RuntimePersistenceBannerProps) {
  const coherence = record?.coherence

  // Before the `current` early-return, and deliberately: a pair restore leaves
  // a record whose current_revision equals its committed_revision, so
  // coherence reads `current` and every branch below would render nothing --
  // silence at the one moment there is something real to say. The generation
  // is coherent; the work after it is gone.
  const pairRestore = record?.pair_restore
  if (pairRestore) {
    const resumedFrom = formatSince(pairRestore.discarded_since, isZh)
    const discarded = pairRestore.discarded_revisions ?? 0
    return (
      <div
        role="status"
        className="rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 px-3 py-2 text-sm text-amber-900 dark:text-amber-200"
      >
        <div className="font-medium">
          {isZh
            ? '已从上次成功保存处继续'
            : 'Resumed from the last successful save'}
        </div>
        <p className="text-xs mt-0.5">
          {isZh
            ? '分析在那次保存之后中断了。运行时已从该保存点恢复，其后的工作没有落盘，无法找回。'
            : 'The analysis was interrupted after that save. The runtime was restored from it, and the work done afterwards never reached disk and cannot be recovered.'}
        </p>
        {resumedFrom && (
          <p className="text-xs mt-0.5 opacity-80">
            {isZh ? `继续自：${resumedFrom}` : `Resumed from: ${resumedFrom}`}
          </p>
        )}
        {discarded > 0 && (
          <p className="text-xs mt-0.5 opacity-80">
            {isZh
              ? `丢弃了 ${discarded} 次单元格执行`
              : `${discarded} cell execution${discarded === 1 ? '' : 's'} discarded`}
          </p>
        )}
      </div>
    )
  }

  if (!coherence || coherence === 'current') return null

  const since = formatSince(record?.exposure_since, isZh)

  // A live kernel is ahead of disk by construction: before the first save it
  // is `unavailable`, and after that `current_revision` can advance past the
  // committed generation and make it `stale`. Both are expected between
  // successful save points, so the ordinary live state stays silent. Save
  // failures and inconsistent generations still reach the actionable warning
  // below.
  const liveUnsavedWork =
    (coherence === 'stale' || coherence === 'unavailable') &&
    record?.last_attempt_status !== 'failed' &&
    LIVE_STATUSES.has(sessionStatus ?? '')

  if (liveUnsavedWork) {
    return null
  }

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
