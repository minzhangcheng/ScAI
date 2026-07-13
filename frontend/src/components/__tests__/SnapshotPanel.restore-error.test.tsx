import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { SnapshotPanel } from '../SnapshotPanel'

const mockListSnapshotsV2 = vi.fn()
const mockRestoreSnapshotV2 = vi.fn()
const mockDeleteSnapshotV2 = vi.fn()
const mockCancelSnapshotTransaction = vi.fn()
const mockGetSnapshotTransaction = vi.fn()

vi.mock('../../hooks/useApi', () => {
  class MockApiError extends Error {
    status: number
    code: string
    runtimeStatus: string | null
    details: Record<string, unknown> | null
    constructor(status: number, code: string, message: string, runtimeStatus?: string | null, details?: Record<string, unknown> | null) {
      super(message)
      this.name = 'ApiError'
      this.status = status
      this.code = code
      this.runtimeStatus = runtimeStatus ?? null
      this.details = details ?? null
    }
  }
  return {
    ApiError: MockApiError,
    apiFetch: vi.fn(),
    listSnapshotsV2: (...args: unknown[]) => mockListSnapshotsV2(...args),
    deleteSnapshotV2: (...args: unknown[]) => mockDeleteSnapshotV2(...args),
    restoreSnapshotV2: (...args: unknown[]) => mockRestoreSnapshotV2(...args),
    cancelSnapshotTransaction: (...args: unknown[]) => mockCancelSnapshotTransaction(...args),
    getSnapshotTransaction: (...args: unknown[]) => mockGetSnapshotTransaction(...args),
  }
})

vi.mock('../../stores/appStore', () => {
  const useAppStore = Object.assign(
    (selector?: (s: Record<string, unknown>) => unknown) => {
      const state = {
        snapshotTransaction: null,
        setSnapshotTransaction: vi.fn(),
        setSnapshotsV2: vi.fn(),
        setSnapshotStoreStatus: vi.fn(),
        setSnapshotCount: vi.fn(),
      }
      return selector ? selector(state) : state
    },
    {
      getState: () => ({
        snapshotTransaction: null,
        setSnapshotTransaction: vi.fn(),
        setSnapshotsV2: vi.fn(),
        setSnapshotStoreStatus: vi.fn(),
        setSnapshotCount: vi.fn(),
      }),
    },
  )
  return { useAppStore }
})

const oneSnapshotV2 = {
  snapshots: [
    {
      snapshot_id: 's1',
      manifest: {
        format_version: 2,
        session_id: 'sess',
        snapshot_id: 's1',
        captured_at: new Date().toISOString(),
        branch: 'main',
        generation: 1,
        parent_id: null,
        root_hash: 'abc',
        capsule: { kind: 'idle', graph_version: '1.0' },
        workspace: [],
        objects: {},
      },
      failure: null,
    },
  ],
  store_status: {
    object_count: 0,
    total_bytes: 0,
    snapshot_count: 1,
    gc_eligible_bytes: 0,
  },
}

beforeEach(() => {
  mockListSnapshotsV2.mockReset()
  mockRestoreSnapshotV2.mockReset()
  mockDeleteSnapshotV2.mockReset()
  mockCancelSnapshotTransaction.mockReset()
  mockGetSnapshotTransaction.mockReset()
  mockListSnapshotsV2.mockResolvedValue(oneSnapshotV2)
  mockRestoreSnapshotV2.mockRejectedValue(
    new Error('Rollback failed: current_round type mismatch'),
  )
})

describe('SnapshotPanel restore error', () => {
  it('shows an inline error when restore fails', async () => {
    render(
      <SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={vi.fn()} />,
    )
    await screen.findByTitle('Restore')
    fireEvent.click(screen.getByTitle('Restore'))
    fireEvent.click(await screen.findByText('Start Restore'))
    await waitFor(() => {
      expect(screen.getByText(/Rollback failed/)).toBeInTheDocument()
    })
  })
})
