import { apiFetch } from './client'

/**
 * Decision snapshots. Part B spec 4.8.
 *
 * A snapshot is taken automatically before each decision an analysis asks
 * about itself, and there is deliberately no way to make one from here: a
 * create call would be a second producer with different semantics.
 */

export interface SnapshotRow {
  snapshot_id: string
  round: number
  /** One of the four decision gates the platform parks on. */
  gate: string
  runtime_revision: number
  created_at: string
  bytes_total: number
  capture_seconds: number
  /** Importance rank: 1 is highest and 4 is lowest. Set at capture, never recomputed. */
  retention_rank: number
  /** 1 goes first. Computed by the server so every client agrees. */
  deletion_order: number
  /** A later capture of the same decision replaced this one; both restorable. */
  superseded: boolean
  saved_names: number
  skipped_count: number
  executor_checkpoint_present: boolean
  scai_version: string
}

export interface FailedCaptureRow {
  snapshot_id: string
  round: number
  gate: string
  reason: string
  at?: string | null
}

export interface SnapshotListing {
  /** False when the installation takes no snapshots at all, which is not the
      same as a session that has produced none. */
  enabled: boolean
  snapshots: SnapshotRow[]
  failed: FailedCaptureRow[]
  /** What this session's snapshots occupy, counting a dump two of them share
      once. Not the sum of the rows: sharing is a hardlink, so the filesystem
      is the only thing that knows. */
  bytes_total: number
  /** snapshot id -> the sessions it was forked into. */
  restored_sessions?: Record<string, string[]>
}

type SnapshotListingChangedListener = (sessionId: string) => void

const snapshotListingChangedListeners = new Set<SnapshotListingChangedListener>()

/** Publish the transport-only invalidation sent after a capture finishes. */
export function notifySnapshotListingChanged(sessionId: string): void {
  for (const listener of snapshotListingChangedListeners) listener(sessionId)
}

/** Subscribe without putting an internal invalidation into the message stream. */
export function subscribeSnapshotListingChanged(
  listener: SnapshotListingChangedListener,
): () => void {
  snapshotListingChangedListeners.add(listener)
  return () => snapshotListingChangedListeners.delete(listener)
}

export async function listSnapshots(sessionId: string): Promise<SnapshotListing> {
  return apiFetch<SnapshotListing>(`/session/${sessionId}/snapshots`)
}

/**
 * Fork this session as it was before one decision. The source is untouched.
 *
 * Reported through the import-job registry, because a restore is an import
 * from a local bundle; the terminal record carries `new_session_id`.
 */
export async function restoreSnapshot(
  sessionId: string,
  snapshotId: string,
): Promise<{ job_id: string }> {
  return apiFetch<{ job_id: string }>(
    `/session/${sessionId}/snapshots/${snapshotId}/restore`,
    { method: 'POST' },
  )
}

export async function deleteSnapshot(
  sessionId: string,
  snapshotId: string,
): Promise<void> {
  await apiFetch(`/session/${sessionId}/snapshots/${snapshotId}`, {
    method: 'DELETE',
  })
}

export interface RestoreJobStatus {
  status: string
  error?: string
  new_session_id?: string
}

export async function getRestoreStatus(jobId: string): Promise<RestoreJobStatus> {
  return apiFetch<RestoreJobStatus>(`/session/import/${jobId}`)
}
