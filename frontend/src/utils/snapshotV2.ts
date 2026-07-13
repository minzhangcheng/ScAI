/**
 * Shared Snapshot V2 parsers and formatters.
 *
 * Consolidates duplicated formatting, status checks, and failure/import
 * parsing across the frontend into a single source of truth.
 */

import type {
  SnapshotRecordV2,
  SnapshotManifestV2,
  SnapshotFailureRecord,
  ContinuationCapsule,
} from '../types/snapshotV2'

/**
 * Normalize one raw snapshot entry from the backend list endpoint into the
 * wrapped {@link SnapshotRecordV2} shape the panel/tree/preview consume.
 *
 * The backend serves each committed snapshot as a *flat* manifest dump using
 * the authoritative field names (`created_at`, `branch_id`, `continuation`) with
 * no `manifest` envelope, and corrupt entries as `{ snapshot_id, status:
 * 'corrupt', manifest: null, error_code, error_message }`. The UI reads
 * `record.manifest.capsule.kind` / `.captured_at` / `.branch` unconditionally,
 * so this adapter bridges the contract drift and stays defensive against
 * missing fields (corrupt entries) to avoid crashing the render.
 */
export function normalizeSnapshotRecordV2(raw: unknown): SnapshotRecordV2 {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const snapshotId = typeof r.snapshot_id === 'string' ? r.snapshot_id : ''

  // Tolerate a future wrapped shape (`r.manifest`) and today's flat dump (`r`).
  const src = (r.manifest && typeof r.manifest === 'object' ? r.manifest : r) as Record<string, unknown>
  const isCorrupt = r.status === 'corrupt' || r.manifest === null

  const capsule = (src.capsule ?? src.continuation ?? { kind: 'idle', graph_version: '' }) as ContinuationCapsule

  const manifest = {
    ...src,
    capsule,
    captured_at: (src.captured_at ?? src.created_at ?? '') as string,
    branch: (src.branch ?? src.branch_label ?? src.branch_id ?? 'main') as string,
    generation: (src.generation ?? src.round_number ?? 0) as number,
    workspace: Array.isArray(src.workspace) ? src.workspace : [],
    objects: (src.objects && typeof src.objects === 'object' ? src.objects : {}),
    snapshot_id: snapshotId,
  } as unknown as SnapshotManifestV2

  let failure: SnapshotFailureRecord | null = null
  if (isCorrupt) {
    failure = {
      snapshot_id: snapshotId,
      session_id: typeof src.session_id === 'string' ? src.session_id : '',
      failed_at: '',
      code: typeof r.error_code === 'string' ? r.error_code : 'corrupt',
      message: typeof r.error_message === 'string' ? r.error_message : '',
    }
  } else if (r.failure && typeof r.failure === 'object') {
    failure = r.failure as SnapshotFailureRecord
  }

  return { snapshot_id: snapshotId, manifest, failure }
}

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
