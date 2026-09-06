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

// Only the runtime-unavailable refusal carries a dedicated flag. Recovery
// blocked and a pending plan review answer 200 with `success: false` and
// nothing else to key off, so that half of the guard needs a response shape of
// its own to be exercised at all.
function refusedResponse(data: Record<string, unknown> = {}) {
  return {
    ok: true,
    status: 200,
    json: async () => ({ success: false, message: 'refused', data }),
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

  it('does not mark the session running when the refusal carries no flag', async () => {
    mockFetch.mockResolvedValueOnce(refusedResponse({ needs_reconnect: false }))

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.resumeAnalysis()
    })

    expect(useAppStore.getState().currentSession?.status).toBe('paused')
  })

  it('does not mark a round continuable when the refusal carries no flag', async () => {
    mockFetch.mockResolvedValueOnce(refusedResponse({ needs_reconnect: false }))

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.continueRound()
    })

    expect(useAppStore.getState().currentSession?.status).toBe('paused')
  })

  // A refusal that still reconnects is worse than one that only mislabels the
  // status: the handler on the other end dispatches the workflow, so the run
  // the backend just declined would start anyway.
  it('does not reconnect the WebSocket when the backend refuses', async () => {
    const wsConnect = vi.fn()
    useAppStore.setState({ wsConnect })
    mockFetch.mockResolvedValueOnce(refusedResponse({ needs_reconnect: true }))

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.resumeAnalysis()
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 150))
    })

    expect(wsConnect).not.toHaveBeenCalled()
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
