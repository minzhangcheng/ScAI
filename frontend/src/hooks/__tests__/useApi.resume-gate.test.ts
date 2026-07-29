import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useSession } from '../api/useSession'
import { useAppStore } from '../../stores/appStore'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

vi.mock('../../utils/authStorage', () => ({
  getToken: vi.fn(() => 'test-token'),
  clearToken: vi.fn(),
}))

function resumeResponse(data: Record<string, unknown>) {
  return {
    ok: true,
    status: 200,
    json: async () => ({ success: true, message: '', data }),
  }
}

describe('resume refused by the backend', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAppStore.setState({
      currentSession: {
        id: 'test-session',
        name: 'Test Session',
        status: 'paused',
        currentRound: 3,
        createdAt: new Date().toISOString(),
        instruction: '',
        treatmentGroups: [],
        dataFiles: [],
        biochemistryIndicators: [],
        roundResults: [],
      },
    })
  })

  // The refusal arrives as a 200 with a flag in the body, so an unconditional
  // `updateSession({status:'running'})` marked a session running that the
  // backend had just declined to advance. The re-run control is gated on
  // `!isRunning`, so that optimistic write hid the only way forward and a page
  // reload was the only way back.
  it('does not mark the session running when runtime is unavailable', async () => {
    mockFetch.mockResolvedValueOnce(
      resumeResponse({ needs_reconnect: false, runtime_unavailable: true }),
    )

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.resumeAnalysis()
    })

    expect(useAppStore.getState().currentSession?.status).toBe('paused')
  })

  it('does not mark the session running when continuing a round is refused', async () => {
    mockFetch.mockResolvedValueOnce(
      resumeResponse({ needs_reconnect: false, runtime_unavailable: true }),
    )

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.continueRound()
    })

    expect(useAppStore.getState().currentSession?.status).toBe('paused')
  })

  it('still marks the session running on an ordinary resume', async () => {
    mockFetch.mockResolvedValueOnce(resumeResponse({ needs_reconnect: false }))

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.resumeAnalysis()
    })

    expect(useAppStore.getState().currentSession?.status).toBe('running')
  })
})
