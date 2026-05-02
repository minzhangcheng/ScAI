import type { SessionListItem } from '../types'

const RUNNING_STATUSES = new Set([
  'running', 'waiting_input', 'waiting_annotation',
  'generating_report', 'paused',
])

export function isRunningStatus(status: string): boolean {
  return RUNNING_STATUSES.has(status)
}

export function getTargetTab(status: string): 'data' | 'analysis' | 'report' {
  if (status === 'pending') return 'data'
  if (isRunningStatus(status)) return 'analysis'
  return 'report'
}

const toTs = (v?: string | null) => (v ? new Date(v).getTime() : 0)

function sortByTsDesc(key: 'created_at' | 'updated_at') {
  return (a: SessionListItem, b: SessionListItem) =>
    toTs(b[key]) - toTs(a[key])
}

export function selectPrioritySession(
  sessions: SessionListItem[]
): SessionListItem | null {
  const waitingInput: SessionListItem[] = []
  const normal: SessionListItem[] = []
  const paused: SessionListItem[] = []

  for (const s of sessions) {
    switch (s.status) {
      case 'waiting_input':
        waitingInput.push(s); break
      case 'waiting_annotation':
      case 'running':
      case 'generating_report':
        normal.push(s); break
      case 'paused':
        paused.push(s); break
    }
  }

  const byCreatedDesc = sortByTsDesc('created_at')

  waitingInput.sort((a, b) => {
    const aInf = a.round_wait_seconds === -1 || !a.waiting_input_started_at
    const bInf = b.round_wait_seconds === -1 || !b.waiting_input_started_at
    if (aInf !== bInf) return aInf ? 1 : -1
    if (!aInf) {
      const aDeadline = new Date(a.waiting_input_started_at!).getTime() + (a.round_wait_seconds ?? 0) * 1000
      const bDeadline = new Date(b.waiting_input_started_at!).getTime() + (b.round_wait_seconds ?? 0) * 1000
      if (aDeadline !== bDeadline) return aDeadline - bDeadline
    }
    return byCreatedDesc(a, b)
  })
  normal.sort(byCreatedDesc)
  paused.sort(byCreatedDesc)

  return waitingInput[0] ?? normal[0] ?? paused[0] ?? null
}

function selectLatestByStatus(
  sessions: SessionListItem[],
  statuses: ReadonlySet<string>,
  key: 'created_at' | 'updated_at'
): SessionListItem | null {
  let best: SessionListItem | null = null
  let bestTs = 0
  for (const s of sessions) {
    if (statuses.has(s.status)) {
      const ts = toTs(s[key])
      if (ts > bestTs) { bestTs = ts; best = s }
    }
  }
  return best
}

const COMPLETED_STATUSES = new Set(['completed', 'failed'])
const PENDING_STATUSES = new Set(['pending'])

export const selectCompletedSession = (sessions: SessionListItem[]) =>
  selectLatestByStatus(sessions, COMPLETED_STATUSES, 'updated_at')

export const selectPendingSession = (sessions: SessionListItem[]) =>
  selectLatestByStatus(sessions, PENDING_STATUSES, 'created_at')

