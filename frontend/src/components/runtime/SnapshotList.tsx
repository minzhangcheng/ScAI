import { useCallback, useEffect, useState } from 'react'
import { History, Loader2, RotateCcw, Trash2 } from 'lucide-react'

import {
  deleteSnapshot,
  getRestoreStatus,
  listSnapshots,
  restoreSnapshot,
  type SnapshotListing,
  type SnapshotRow,
} from '../../hooks/api/snapshots'

/**
 * The decision snapshots of one session, and what can be done with them.
 *
 * Part B spec 4.9. No tree: within one session the decisions are linear and
 * stay linear, because restoring creates a *new* session -- branching is
 * expressed by sessions, which is what made V2's branch model unnecessary.
 *
 * Nothing here creates a snapshot. Capture is automatic before each decision
 * the analysis asks about itself; a button that made one on request would be a
 * second producer with different semantics.
 */

interface SnapshotListProps {
  sessionId: string
  isZh?: boolean
  /** Open the session a restore produced. */
  onOpenSession?: (sessionId: string) => void
}

const DECISION_LABEL: Record<string, { en: string; zh: string }> = {
  annotation_confirmation: {
    en: 'before confirming the cell annotation',
    zh: '确认细胞注释之前',
  },
  direction_selection: {
    en: 'before choosing the research direction',
    zh: '选择研究方向之前',
  },
  integration_selection: {
    en: 'before choosing the direction after integration',
    zh: '整合分析后选择方向之前',
  },
  round_wait: {
    en: 'before accepting the round',
    zh: '验收本轮结果之前',
  },
}

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function formatWhen(raw: string, isZh: boolean): string {
  const at = new Date(raw)
  return Number.isNaN(at.getTime()) ? raw : at.toLocaleString(isZh ? 'zh-CN' : undefined)
}

export function SnapshotList({ sessionId, isZh = false, onOpenSession }: SnapshotListProps) {
  const [listing, setListing] = useState<SnapshotListing | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setListing(await listSnapshots(sessionId))
    } catch {
      // A listing that cannot be read is not worth a banner of its own: the
      // panel simply does not appear, exactly as it does not on an
      // installation that takes no snapshots.
      setListing(null)
    }
  }, [sessionId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const onRestore = useCallback(async (row: SnapshotRow) => {
    setBusy(row.snapshot_id)
    setMessage(isZh ? '正在创建新会话…' : 'Creating the new session…')
    try {
      const { job_id: jobId } = await restoreSnapshot(sessionId, row.snapshot_id)
      // Polled rather than awaited: the copy is bounded by the size of the
      // session, and a request left open for minutes is a request a proxy
      // closes.
      for (;;) {
        const status = await getRestoreStatus(jobId)
        if (status.status === 'completed' && status.new_session_id) {
          setMessage(null)
          await refresh()
          onOpenSession?.(status.new_session_id)
          return
        }
        if (status.status === 'failed') {
          setMessage(
            (isZh ? '恢复失败：' : 'Restore failed: ') + (status.error || 'unknown'),
          )
          return
        }
        await new Promise(resolve => setTimeout(resolve, 2000))
      }
    } catch (error) {
      setMessage((isZh ? '恢复失败：' : 'Restore failed: ') + String(error))
    } finally {
      setBusy(null)
    }
  }, [isZh, onOpenSession, refresh, sessionId])

  const onDelete = useCallback(async (row: SnapshotRow) => {
    setBusy(row.snapshot_id)
    try {
      await deleteSnapshot(sessionId, row.snapshot_id)
      await refresh()
    } catch (error) {
      setMessage((isZh ? '删除失败：' : 'Delete failed: ') + String(error))
    } finally {
      setBusy(null)
    }
  }, [isZh, refresh, sessionId])

  // Not enabled on this installation, or nothing captured yet. Rendering an
  // empty panel would read as "this analysis produced none".
  if (!listing?.enabled) return null
  if (!listing.snapshots.length && !listing.failed.length) return null

  return (
    <section
      className="mb-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-surface-900"
      aria-label={isZh ? '决策快照' : 'Decision snapshots'}
    >
      <header className="flex items-center gap-2 px-3 py-2 border-b border-slate-200 dark:border-slate-700">
        <History className="w-4 h-4 text-slate-500" />
        <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
          {isZh ? '决策快照' : 'Decision snapshots'}
        </span>
        <span className="text-xs text-slate-500">
          {isZh
            ? '恢复会新建一个会话，当前会话保持不变'
            : 'Restoring opens a new session and leaves this one untouched'}
        </span>
      </header>

      <ul className="divide-y divide-slate-100 dark:divide-slate-800">
        {listing.snapshots.map(row => {
          const label = DECISION_LABEL[row.gate]
          const forks = listing.restored_sessions?.[row.snapshot_id] ?? []
          return (
            <li key={row.snapshot_id} className="px-3 py-2 text-sm">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-medium text-slate-700 dark:text-slate-200">
                  {isZh ? `第 ${row.round} 轮` : `Round ${row.round}`}
                </span>
                <span className="text-slate-500">
                  {label ? (isZh ? label.zh : label.en) : row.gate}
                </span>
                {row.superseded && (
                  <span className="text-xs rounded px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 text-slate-500">
                    {isZh ? '已被更新的快照取代' : 'superseded'}
                  </span>
                )}
              </div>
              <div className="mt-0.5 text-xs text-slate-500 flex items-center gap-3 flex-wrap">
                <span>{formatWhen(row.created_at, isZh)}</span>
                <span>{formatBytes(row.bytes_total)}</span>
                {row.skipped_count > 0 && (
                  <span>
                    {isZh
                      ? `${row.skipped_count} 个值未能保存`
                      : `${row.skipped_count} value${row.skipped_count === 1 ? '' : 's'} not saved`}
                  </span>
                )}
                <span>
                  {isZh
                    ? `空间不足时第 ${row.deletion_order} 个删除`
                    : `deleted ${row.deletion_order}${row.deletion_order === 1 ? 'st' : row.deletion_order === 2 ? 'nd' : row.deletion_order === 3 ? 'rd' : 'th'} if space runs short`}
                </span>
              </div>
              {forks.length > 0 && (
                <div className="mt-0.5 text-xs text-slate-500">
                  {isZh ? '已恢复为：' : 'Restored into: '}
                  {forks.map(id => (
                    <button
                      key={id}
                      type="button"
                      className="underline mr-2"
                      onClick={() => onOpenSession?.(id)}
                    >
                      {id.slice(0, 8)}
                    </button>
                  ))}
                </div>
              )}
              <div className="mt-1 flex items-center gap-2">
                <button
                  type="button"
                  disabled={busy !== null}
                  onClick={() => void onRestore(row)}
                  className="inline-flex items-center gap-1 text-xs rounded px-2 py-1 border border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50"
                >
                  {busy === row.snapshot_id
                    ? <Loader2 className="w-3 h-3 animate-spin" />
                    : <RotateCcw className="w-3 h-3" />}
                  {isZh ? '恢复到新会话' : 'Restore into a new session'}
                </button>
                <button
                  type="button"
                  disabled={busy !== null}
                  onClick={() => void onDelete(row)}
                  className="inline-flex items-center gap-1 text-xs rounded px-2 py-1 text-slate-500 hover:text-red-600 disabled:opacity-50"
                >
                  <Trash2 className="w-3 h-3" />
                  {isZh ? '删除' : 'Delete'}
                </button>
              </div>
            </li>
          )
        })}

        {listing.failed.map(row => (
          <li key={row.snapshot_id} className="px-3 py-2 text-xs text-slate-500">
            {isZh
              ? `第 ${row.round} 轮的这个决策没有快照（${row.reason}）`
              : `Round ${row.round}: this decision has no snapshot (${row.reason})`}
          </li>
        ))}
      </ul>

      {message && (
        <p className="px-3 py-2 text-xs text-slate-600 dark:text-slate-300 border-t border-slate-200 dark:border-slate-700">
          {message}
        </p>
      )}
      <p className="px-3 py-2 text-xs text-slate-500 border-t border-slate-200 dark:border-slate-700">
        {isZh
          ? '恢复出的会话不包含报告，也可能缺少原会话无法序列化的值。'
          : 'A restored session carries no reports, and may be missing values the original could not serialise.'}
      </p>
    </section>
  )
}
