import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { useAppStore, type Session } from '../../stores/appStore'
import { apiFetch, getSnapshotTransaction } from '../../hooks/useApi'
import type { SnapshotTransactionRecord } from '../../types/snapshotV2'

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
  getSnapshotTransaction: vi.fn(),
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
      runtimeRestoreStatus: 'complete',
      runtimeRestoreWarning: null,
    })
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
})
