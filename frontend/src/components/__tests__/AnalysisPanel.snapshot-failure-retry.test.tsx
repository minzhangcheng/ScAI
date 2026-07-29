import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { useAppStore, type Session } from '../../stores/appStore'
import { apiFetch, getSnapshotTransaction, listSnapshotsV2 } from '../../hooks/useApi'
import type { SnapshotTransactionRecord } from '../../types/snapshotV2'

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  getSnapshotTransaction: vi.fn(),
  listSnapshotsV2: vi.fn(async () => ({
    snapshots: [],
    store_status: {
      logical_bytes: 0,
      physical_bytes: 0,
      dedup_savings: 0,
      object_count: 0,
      manifest_count: 0,
    },
  })),
  deleteSnapshotV2: vi.fn(),
  restoreSnapshotV2: vi.fn(),
  cancelSnapshotTransaction: vi.fn(),
  generateId: () => 'test-id',
  useConfig: () => ({
    checkStatus: vi.fn(),
    validateApiKey: vi.fn(),
    validateSavedConfig: vi.fn(),
    saveConfig: vi.fn(),
  }),
  useSession: () => ({
    startAnalysis: vi.fn(),
    resumeAnalysis: vi.fn(),
    continueRound: vi.fn(),
    pauseAnalysis: vi.fn(),
    stopAnalysis: vi.fn(),
  }),
  useUserConfig: () => ({
    getUserConfigStatus: vi.fn(),
    validateSavedConfig: vi.fn(),
  }),
}))

const mockApiFetch = apiFetch as unknown as ReturnType<typeof vi.fn>
const mockGetSnapshotTransaction = getSnapshotTransaction as unknown as ReturnType<typeof vi.fn>
const mockListSnapshotsV2 = listSnapshotsV2 as unknown as ReturnType<typeof vi.fn>

function makeSession(): Session {
  return {
    id: 'sess-snap-retry',
    name: 'Snapshot retry gate',
    status: 'waiting_input',
    currentRound: 1,
    analysisPhase: 'exploration',
    createdAt: '2026-07-12T00:00:00Z',
    instruction: 'test',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }
}

const blockedFailure = {
  failure: {
    snapshot_id: 'snap-1',
    session_id: 'sess-snap-retry',
    failed_at: '2026-07-12T00:00:00Z',
    code: 'QUOTA',
    message: 'disk full',
    details: { phase: 'post_report' },
  },
  allowed_actions: ['retry', 'manage', 'terminate'] as Array<'retry' | 'manage' | 'terminate'>,
}

describe('AnalysisPanel snapshot-failure retry gate', () => {
  beforeEach(() => {
    mockApiFetch.mockReset()
    mockGetSnapshotTransaction.mockReset()
    useAppStore.setState({
      currentSession: makeSession(),
      messages: [],
      language: 'en',
      wsConnected: true,
      snapshotFailure: blockedFailure,
      snapshotV2Enabled: true,
      snapshotTransaction: null,
      runtimeRestoreStatus: 'complete',
      runtimeRestoreWarning: null,
    })
  })

  it('hides Snapshot V2 UI and starts no Snapshot requests when disabled', async () => {
    useAppStore.setState({
      snapshotV2Enabled: false,
      snapshotFailure: blockedFailure,
      snapshotTransaction: {
        transaction_id: 'disabled-restore',
        session_id: 'sess-snap-retry',
        snapshot_id: 'snapshot-disabled',
        status: 'queued',
        recovery_mode: 'rollback',
        failure: null,
      },
      showSnapshotCleanupReminder: true,
      runtimeRestoreStatus: 'unavailable',
      runtimeRestoreWarning: 'snapshot runtime unavailable',
    })

    render(<AnalysisPanel />)
    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.queryByRole('dialog', { name: /snapshot failure/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('dialog', { name: /analysis snapshots/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/snapshot count reminder/i)).not.toBeInTheDocument()
    // The runtime-restore banner is NOT Snapshot V2 UI. An ordinary session
    // *import* sets runtime_restore_status='unavailable' with Snapshot V2 off,
    // and gating the banner on the feature flag left the Resume button disabled
    // with no explanation of why.
    expect(screen.getByText(/snapshot runtime unavailable/i)).toBeInTheDocument()
    expect(mockGetSnapshotTransaction).not.toHaveBeenCalled()
    expect(mockListSnapshotsV2).not.toHaveBeenCalled()
  })

  it('explains a runtime-unavailable session even with Snapshot V2 disabled', async () => {
    useAppStore.setState({
      snapshotV2Enabled: false,
      snapshotFailure: null,
      snapshotTransaction: null,
      showSnapshotCleanupReminder: false,
      runtimeRestoreStatus: 'unavailable',
      runtimeRestoreWarning: 'imported without kernel runtime state',
    })

    render(<AnalysisPanel />)
    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByText(/imported without kernel runtime state/i)).toBeInTheDocument()
  })

  it('does not clear gate when retry returns success:false / still_blocked', async () => {
    mockApiFetch.mockResolvedValueOnce({
      success: false,
      message: 'Snapshot retry failed; still blocked',
      data: {
        status: 'waiting_input',
        wait_input_reason: 'snapshot_failure',
        still_blocked: true,
        snapshot_failure: {
          code: 'QUOTA',
          message: 'still full',
          snapshot_id: 'snap-1',
          session_id: 'sess-snap-retry',
          failed_at: '2026-07-12T00:01:00Z',
          allowed_actions: ['retry', 'manage', 'terminate'],
        },
      },
    })

    render(<AnalysisPanel />)
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))

    await waitFor(() => {
      expect(mockApiFetch).toHaveBeenCalledWith(
        '/session/sess-snap-retry/snapshot-failure/retry',
        expect.objectContaining({ method: 'POST' }),
      )
    })

    await waitFor(() => {
      const failure = useAppStore.getState().snapshotFailure
      expect(failure).not.toBeNull()
      expect(failure?.failure.code).toBe('QUOTA')
      expect(failure?.failure.message).toBe('still full')
    })
    expect(screen.getByRole('dialog', { name: /snapshot failure/i })).toBeInTheDocument()
  })

  it('lets the snapshot manager own the modal layer until it closes', async () => {
    render(<AnalysisPanel />)

    fireEvent.click(screen.getByRole('button', { name: /manage snapshots/i }))

    expect(
      await screen.findByRole('dialog', { name: /analysis snapshots/i }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('dialog', { name: /snapshot failure/i }),
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /^close$/i }))

    expect(
      screen.queryByRole('dialog', { name: /analysis snapshots/i }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole('dialog', { name: /snapshot failure/i }),
    ).toBeInTheDocument()
    expect(useAppStore.getState().snapshotFailure).not.toBeNull()
  })

  it('clears gate only when retry returns success:true', async () => {
    mockApiFetch.mockResolvedValueOnce({
      success: true,
      message: 'Snapshot recaptured; report finalized',
      data: {
        status: 'completed',
        wait_input_reason: null,
        snapshot_failure: null,
      },
    })

    render(<AnalysisPanel />)
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))

    await waitFor(() => {
      expect(useAppStore.getState().snapshotFailure).toBeNull()
    })
  })

  it('hydrates and polls the queued pre-restore retry transaction', async () => {
    const queued = {
      transaction_id: 'restore-job-retry',
      session_id: 'sess-snap-retry',
      snapshot_id: 'target-snapshot',
      status: 'queued' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
      retry_of_transaction_id: 'restore-job-original',
    }
    const committed = { ...queued, status: 'committed' as const }
    mockApiFetch.mockResolvedValueOnce({
      success: true,
      message: 'Restore retry queued',
      data: {
        status: 'paused',
        wait_input_reason: null,
        snapshot_failure: null,
        transaction_id: queued.transaction_id,
        snapshot_transaction: queued,
      },
    })
    mockGetSnapshotTransaction.mockResolvedValueOnce(committed)

    render(<AnalysisPanel />)
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))

    await waitFor(() => {
      expect(mockGetSnapshotTransaction).toHaveBeenCalledWith(
        'sess-snap-retry',
        'restore-job-retry',
      )
      expect(useAppStore.getState().snapshotTransaction).toEqual(committed)
      expect(useAppStore.getState().snapshotFailure).toBeNull()
    })
  })

  it('cancels retry polling when the active session changes', async () => {
    const queued = {
      transaction_id: 'restore-job-retry',
      session_id: 'sess-snap-retry',
      snapshot_id: 'target-snapshot',
      status: 'queued' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
    }
    let resolvePoll!: (value: SnapshotTransactionRecord) => void
    mockApiFetch.mockResolvedValueOnce({
      success: true,
      data: {
        status: 'paused',
        snapshot_failure: null,
        snapshot_transaction: queued,
      },
    })
    mockGetSnapshotTransaction.mockReturnValueOnce(new Promise<SnapshotTransactionRecord>(resolve => {
      resolvePoll = resolve
    }))

    render(<AnalysisPanel />)
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))
    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalledOnce())

    await act(async () => {
      useAppStore.setState({
        currentSession: { ...makeSession(), id: 'another-session' },
      })
    })
    await act(async () => {
      resolvePoll({ ...queued, status: 'committed' })
      await Promise.resolve()
    })

    expect(useAppStore.getState().snapshotTransaction?.status).toBe('queued')
  })

  it('resumes polling a hydrated non-terminal retry transaction', async () => {
    const queued = {
      transaction_id: 'restore-job-reload',
      session_id: 'sess-snap-retry',
      snapshot_id: 'target-snapshot',
      status: 'waiting_safe_point' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
      retry_of_transaction_id: 'restore-job-original',
    }
    useAppStore.setState({ snapshotTransaction: queued })
    mockGetSnapshotTransaction.mockResolvedValueOnce({
      ...queued, status: 'committed' as const,
    })
    render(<AnalysisPanel />)
    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalledWith(
      'sess-snap-retry', 'restore-job-reload'
    ))
  })

  it('does not let an old poll overwrite a newer transaction', async () => {
    const oldTransaction = {
      transaction_id: 'restore-job-old',
      session_id: 'sess-snap-retry',
      snapshot_id: 'snapshot-old',
      status: 'waiting_safe_point' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
    }
    const newTransaction = {
      ...oldTransaction,
      transaction_id: 'restore-job-new',
      snapshot_id: 'snapshot-new',
    }
    let resolveOld!: (value: SnapshotTransactionRecord) => void
    useAppStore.setState({ snapshotTransaction: oldTransaction })
    mockGetSnapshotTransaction.mockReturnValueOnce(
      new Promise<SnapshotTransactionRecord>((resolve) => { resolveOld = resolve })
    )
    render(<AnalysisPanel />)
    await waitFor(() => expect(mockGetSnapshotTransaction).toHaveBeenCalledOnce())
    act(() => useAppStore.getState().setSnapshotTransaction(newTransaction))
    await act(async () => {
      resolveOld({ ...oldTransaction, status: 'failed' })
      await Promise.resolve()
    })
    expect(useAppStore.getState().snapshotTransaction).toEqual(newTransaction)
  })

  it('opens progress and keeps a polled failure visible', async () => {
    const queued = {
      transaction_id: 'restore-job-retry',
      session_id: 'sess-snap-retry',
      snapshot_id: 'target-snapshot',
      status: 'queued' as const,
      recovery_mode: 'rollback' as const,
      failure: null,
      retry_of_transaction_id: 'restore-job-original',
    }
    const failed = {
      ...queued,
      status: 'failed' as const,
      failure: {
        code: 'RESTORE_APPLY_FAILED',
        message: 'Restore apply failed safely',
      },
    }
    mockApiFetch.mockResolvedValueOnce({
      success: true,
      data: {
        status: 'paused',
        snapshot_failure: null,
        snapshot_transaction: queued,
      },
    })
    mockApiFetch.mockResolvedValue({ snapshots: [], store_status: null })
    mockGetSnapshotTransaction.mockResolvedValueOnce(failed)
    render(<AnalysisPanel />)
    fireEvent.click(screen.getByRole('button', { name: /retry/i }))

    expect(await screen.findByRole(
      'dialog', { name: /restore progress/i }
    )).toBeInTheDocument()
    expect(await screen.findByText('Restore apply failed safely')).toBeInTheDocument()
    expect(useAppStore.getState().snapshotTransaction).toEqual(failed)
    expect(useAppStore.getState().snapshotFailure).toBeNull()
  })
})
