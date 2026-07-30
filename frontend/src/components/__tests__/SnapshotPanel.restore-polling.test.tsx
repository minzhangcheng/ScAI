import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { SnapshotPanel } from '../SnapshotPanel'

const mockListSnapshotsV2 = vi.fn()
const mockRestoreSnapshotV2 = vi.fn()
const mockDeleteSnapshotV2 = vi.fn()
const mockCancelSnapshotTransaction = vi.fn()
const mockGetSnapshotTransaction = vi.fn()

vi.mock('../../hooks/useApi', () => ({
  ApiError: class extends Error {},
  apiFetch: vi.fn(),
  listSnapshotsV2: (...a: unknown[]) => mockListSnapshotsV2(...a),
  deleteSnapshotV2: (...a: unknown[]) => mockDeleteSnapshotV2(...a),
  restoreSnapshotV2: (...a: unknown[]) => mockRestoreSnapshotV2(...a),
  cancelSnapshotTransaction: (...a: unknown[]) => mockCancelSnapshotTransaction(...a),
  getSnapshotTransaction: (...a: unknown[]) => mockGetSnapshotTransaction(...a),
}))

vi.mock('../../stores/appStore', () => {
  const state = {
    snapshotTransaction: null,
    setSnapshotTransaction: vi.fn(),
    setSnapshotsV2: vi.fn(),
    setSnapshotStoreStatus: vi.fn(),
    setSnapshotCount: vi.fn(),
  }
  const useAppStore = Object.assign(
    (selector?: (s: Record<string, unknown>) => unknown) =>
      (selector ? selector(state) : state),
    { getState: () => state },
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

const queuedTxn = {
  transaction_id: 'restore-job-1',
  session_id: 'sess',
  snapshot_id: 's1',
  status: 'waiting_safe_point' as const,
  recovery_mode: 'rollback' as const,
  failure: null,
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  for (const m of [
    mockListSnapshotsV2, mockRestoreSnapshotV2, mockDeleteSnapshotV2,
    mockCancelSnapshotTransaction, mockGetSnapshotTransaction,
  ]) m.mockReset()
  mockListSnapshotsV2.mockResolvedValue(oneSnapshotV2)
  mockRestoreSnapshotV2.mockResolvedValue({
    transaction_id: 'restore-job-1',
    status: 'queued',
  })
  mockGetSnapshotTransaction.mockResolvedValue(queuedTxn)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('SnapshotPanel restore polling', () => {
  /**
   * The polling effect used to depend on `startPolling`, whose identity chains
   * through `applyTxnUpdate` to the `onRestore` prop — an inline arrow in
   * AnalysisPanel, i.e. a new value on every parent render. The effect
   * therefore tore down and re-created on every render, and its cleanup killed
   * the interval `beginRestore` had just started; the panel's own poller never
   * ran once. The dialog only kept moving because the parent's retry monitor
   * polled the same transaction, and that loop hard-stops at exactly the
   * backend's 300s safe-point timeout.
   */
  it('keeps polling across parent re-renders that change onRestore identity', async () => {
    const { rerender } = render(
      <SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={() => {}} />,
    )
    await screen.findByTitle('Restore')
    fireEvent.click(screen.getByTitle('Restore'))
    fireEvent.click(await screen.findByText('Start Restore'))

    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalled())
    const afterStart = mockGetSnapshotTransaction.mock.calls.length

    // A parent render with a brand-new onRestore closure — exactly what
    // AnalysisPanel does on every state change.
    rerender(
      <SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={() => {}} />,
    )

    await vi.advanceTimersByTimeAsync(5000)
    expect(mockGetSnapshotTransaction.mock.calls.length).toBeGreaterThan(afterStart)
  })

  it('stops polling once the transaction is terminal', async () => {
    mockGetSnapshotTransaction.mockResolvedValue({ ...queuedTxn, status: 'failed' })

    render(
      <SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={() => {}} />,
    )
    await screen.findByTitle('Restore')
    fireEvent.click(screen.getByTitle('Restore'))
    fireEvent.click(await screen.findByText('Start Restore'))

    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalled())
    const afterTerminal = mockGetSnapshotTransaction.mock.calls.length
    await vi.advanceTimersByTimeAsync(6000)
    expect(mockGetSnapshotTransaction.mock.calls.length).toBe(afterTerminal)
  })

  it('offers the discard escape hatch when the pre-restore capture fails', async () => {
    // The pre-restore safety capture fails inside the background job, so this
    // is where the client learns about it — not from the initial 202. Nothing
    // has been mutated yet, which is exactly when discard_current is valid.
    mockGetSnapshotTransaction.mockResolvedValue({
      ...queuedTxn,
      status: 'failed',
      failure: { code: 'PRE_RESTORE_UNAVAILABLE', message: 'kernel not capturable' },
    })

    render(
      <SnapshotPanel sessionId="sess" language="en" onClose={vi.fn()} onRestore={() => {}} />,
    )
    await screen.findByTitle('Restore')
    fireEvent.click(screen.getByTitle('Restore'))
    fireEvent.click(await screen.findByText('Start Restore'))

    // The failure arrives on the first poll tick, one microtask after the
    // effect re-runs with the new pollingTxnId.
    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalled())
    await vi.advanceTimersByTimeAsync(50)

    expect(
      await screen.findByRole('dialog', { name: 'Discard current' }),
    ).toBeInTheDocument()
  })
})
