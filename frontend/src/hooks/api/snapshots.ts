// Snapshot V2 REST calls.
// Split out of frontend/src/hooks/useApi.ts verbatim.

import type { SnapshotV2ListResult, SnapshotV2PreviewResult, SnapshotStoreStatus, SnapshotTransactionResponse, SnapshotTransactionRecord, RestoreSnapshotV2Request, DeleteSnapshotV2Result, ExportV2Result, ImportV2Result } from '../../types/snapshotV2'
import { normalizeSnapshotRecordV2 } from '../../utils/snapshotV2'
import { apiFetch } from './client'

export async function listSnapshotsV2(sessionId: string): Promise<SnapshotV2ListResult> {
  const response = await apiFetch<{ success: boolean; data: { snapshots: unknown[]; store_status: SnapshotStoreStatus } }>(
    `/session/${sessionId}/snapshots-v2`,
  )
  const raw = response.data
  return {
    snapshots: (Array.isArray(raw.snapshots) ? raw.snapshots : []).map(normalizeSnapshotRecordV2),
    store_status: raw.store_status,
  }
}

export async function previewSnapshotV2(sessionId: string, snapshotId: string): Promise<SnapshotV2PreviewResult> {
  const response = await apiFetch<{ success: boolean; data: SnapshotV2PreviewResult }>(
    `/session/${sessionId}/snapshots-v2/${snapshotId}`,
  )
  return response.data
}

export async function restoreSnapshotV2(
  sessionId: string,
  snapshotId: string,
  request: RestoreSnapshotV2Request,
): Promise<SnapshotTransactionResponse> {
  return apiFetch<SnapshotTransactionResponse>(
    `/session/${sessionId}/snapshots/${snapshotId}/restore-v2`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    },
  )
}

export async function cancelSnapshotTransaction(sessionId: string, transactionId: string): Promise<void> {
  await apiFetch<void>(
    `/session/${sessionId}/snapshot-transactions/${transactionId}/cancel`,
    { method: 'POST' },
  )
}

export async function getSnapshotTransaction(
  sessionId: string,
  transactionId: string,
): Promise<SnapshotTransactionRecord> {
  const response = await apiFetch<{ success: boolean; data: SnapshotTransactionRecord }>(
    `/session/${sessionId}/snapshot-transactions/${transactionId}`,
  )
  return response.data
}

export async function deleteSnapshotV2(
  sessionId: string,
  snapshotId: string,
  branch?: boolean,
): Promise<DeleteSnapshotV2Result> {
  const query = branch ? '?branch=true' : ''
  const response = await apiFetch<{ success: boolean; data: DeleteSnapshotV2Result }>(
    `/session/${sessionId}/snapshots-v2/${snapshotId}${query}`,
    { method: 'DELETE' },
  )
  return response.data
}

export async function exportSnapshotsV2(
  sessionId: string,
  options?: { include_runtime?: boolean; snapshot_ids?: string[] },
): Promise<ExportV2Result> {
  const params = new URLSearchParams()
  if (options?.include_runtime !== undefined) {
    params.set('include_runtime', String(options.include_runtime))
  }
  if (options?.snapshot_ids && options.snapshot_ids.length > 0) {
    params.set('snapshot_ids', options.snapshot_ids.join(','))
  }
  const qs = params.toString()
  return apiFetch<ExportV2Result>(
    `/session/${sessionId}/export-v2${qs ? `?${qs}` : ''}`,
  )
}

export async function importSnapshotsV2(
  sessionId: string,
  file: File,
  trust?: boolean,
): Promise<ImportV2Result> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/gzip',
    'X-Filename': file.name,
    'X-Trust': trust ? 'true' : 'false',
  }
  const options: RequestInit = {
    method: 'POST',
    headers,
    body: file,
  }
  return apiFetch<ImportV2Result>(`/session/${sessionId}/import-v2`, options)
}
