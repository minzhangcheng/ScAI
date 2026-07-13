import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { AnalysisPanel } from '../AnalysisPanel'
import { useAppStore, type Session } from '../../stores/appStore'
import { apiFetch } from '../../hooks/useApi'

vi.mock('../../hooks/useApi', () => ({
  apiFetch: vi.fn(),
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
})
