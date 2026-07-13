/**
 * Shared Snapshot V2 parsers and formatters.
 *
 * Consolidates duplicated formatting, status checks, and failure/import
 * parsing across the frontend into a single source of truth.
 */

/** Statuses that indicate a snapshot transaction has finished. */
export const SNAPSHOT_TERMINAL_STATUSES = new Set([
  'committed',
  'failed',
  'cancelled',
  'timed_out',
])

/**
 * Format a byte count into a human-readable string.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

/**
 * Parse a snapshot failure payload from the backend into a typed structure.
 */
export interface ParsedSnapshotFailure {
  failure: {
    snapshot_id: string
    session_id: string
    failed_at: string
    code: string
    message: string
    details?: Record<string, unknown>
  }
  allowed_actions: Array<'retry' | 'manage' | 'terminate'>
}

export function parseSnapshotFailure(
  raw: unknown,
): ParsedSnapshotFailure | null {
  if (!raw || typeof raw !== 'object') return null

  const f = raw as Record<string, unknown>
  const code = typeof f.code === 'string' ? f.code : ''
  const message = typeof f.message === 'string' ? f.message : ''
  if (!code && !message) return null

  const rawActions = Array.isArray(f.allowed_actions)
    ? f.allowed_actions
    : ['retry', 'manage', 'terminate']

  return {
    failure: {
      snapshot_id: typeof f.snapshot_id === 'string' ? f.snapshot_id : '',
      session_id: typeof f.session_id === 'string' ? f.session_id : '',
      failed_at: typeof f.failed_at === 'string' ? f.failed_at : '',
      code,
      message,
      details:
        f.details && typeof f.details === 'object'
          ? (f.details as Record<string, unknown>)
          : undefined,
    },
    allowed_actions: rawActions.filter(
      (a): a is 'retry' | 'manage' | 'terminate' =>
        a === 'retry' || a === 'manage' || a === 'terminate',
    ),
  }
}

/**
 * Parse an import result from the backend into a typed structure.
 */
export interface ParsedImportResult {
  success: boolean
  imported_snapshot_ids: string[]
  committed_count: number
  skipped_runtime_snapshot_ids: string[]
  errors: string[]
  warnings: string[]
}

export function parseImportResult(raw: unknown): ParsedImportResult | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  if (typeof r.success !== 'boolean') return null
  return {
    success: r.success,
    imported_snapshot_ids: Array.isArray(r.imported_snapshot_ids)
      ? r.imported_snapshot_ids.map(String)
      : [],
    committed_count: typeof r.committed_count === 'number' ? r.committed_count : 0,
    skipped_runtime_snapshot_ids: Array.isArray(r.skipped_runtime_snapshot_ids)
      ? r.skipped_runtime_snapshot_ids.map(String)
      : [],
    errors: Array.isArray(r.errors) ? r.errors.map(String) : [],
    warnings: Array.isArray(r.warnings) ? r.warnings.map(String) : [],
  }
}
