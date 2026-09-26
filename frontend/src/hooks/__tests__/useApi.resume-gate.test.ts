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
      runtimeRestoreStatus: 'complete',
      runtimeRestoreWarning: null,
    })
  })

  it('hydrates the runtime warning and leaves rerun enabled on refusal', async () => {
    mockFetch.mockResolvedValueOnce(
      resumeResponse({
        needs_reconnect: false,
        runtime_unavailable: true,
        runtime_restore_status: 'unavailable',
        runtime_restore_warning: 'The committed runtime is stale; re-run to recompute it.',
      }),
    )

    const { result } = renderHook(() => useSession())
    await act(async () => {
      await result.current.continueRound()
    })

    const state = useAppStore.getState()
    expect(state.currentSession?.status).toBe('paused')
    expect(state.runtimeRestoreStatus).toBe('unavailable')
    expect(state.runtimeRestoreWarning).toBe(
      'The committed runtime is stale; re-run to recompute it.',
    )
    expect(state.isRerunEnabled()).toBe(true)
  })

  it('does not apply a late refusal to a newly selected session', async () => {
    let resolveFetch: (value: any) => void = () => undefined
    mockFetch.mockImplementationOnce(
      () => new Promise(resolve => { resolveFetch = resolve }),
    )

    const { result } = renderHook(() => useSession())
    let pending!: Promise<void>
    await act(async () => {
      pending = result.current.continueRound()
      useAppStore.setState({
        currentSession: {
          id: 'new-session',
          name: 'New Session',
          status: 'paused',
          currentRound: 1,
          createdAt: new Date().toISOString(),
          instruction: '',
          treatmentGroups: [],
          dataFiles: [],
          biochemistryIndicators: [],
          roundResults: [],
        },
      })
      resolveFetch(resumeResponse({
        needs_reconnect: false,
        runtime_unavailable: true,
        runtime_restore_warning: 'Old session warning',
      }))
      await pending
    })

    const state = useAppStore.getState()
    expect(state.currentSession?.id).toBe('new-session')
    expect(state.currentSession?.status).toBe('paused')
    expect(state.runtimeRestoreStatus).toBe('complete')
    expect(state.runtimeRestoreWarning).toBeNull()
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
