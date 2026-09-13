import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useWebSocket } from '../useApi'
import { useAppStore } from '../../stores/appStore'

/**
 * A `state_sync` producer that omits a field must not erase it.
 *
 * `updateSession` spreads its argument over the session, so a key present with
 * value `undefined` overwrites the real value. The producer that first exposed
 * this was the snapshot restore job, whose transaction sync carried only
 * `change_type` — no `version`, no `status`, no `current_round` — and blanked
 * `status`/`currentRound` on every progress event. That producer is gone; the
 * rule is not. `AnalysisPanel` derives every header control from
 * `currentSession.status`, so with it undefined none of the branches match and
 * the control bar renders empty until a manual page reload.
 */
describe('state_sync with a partial payload', () => {
  const wsInstances: Array<{ onmessage: ((event: MessageEvent) => void) | null }> = []

  class MockWebSocket {
    public readyState: number = WebSocket.OPEN
    public close = vi.fn()
    public url: string
    private _onopen: ((event: Event) => void) | null = null
    public onmessage: ((event: MessageEvent) => void) | null = null
    public onclose: ((event: CloseEvent) => void) | null = null
    public onerror: ((event: Event) => void) | null = null

    constructor(url: string) {
      this.url = url
      wsInstances.push(this as never)
    }

    set onopen(handler: ((event: Event) => void) | null) {
      this._onopen = handler
      if (handler) handler(new Event('open'))
    }

    get onopen(): ((event: Event) => void) | null {
      return this._onopen
    }
  }

  const session = {
    id: 'session-restore',
    name: 'Restoring',
    status: 'running' as const,
    currentRound: 3,
    createdAt: new Date().toISOString(),
    instruction: '',
    treatmentGroups: [],
    dataFiles: [],
    biochemistryIndicators: [],
    roundResults: [],
  }

  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket as never)
    wsInstances.length = 0
    useAppStore.setState({
      currentSession: session,
      messages: [],
      wsConnected: false,
      stateVersion: 7,
      llmFailureInfo: null,
      showLlmTimeoutDialog: false,
    })
  })

  it('a sync carrying only a change_type leaves status, round and version intact', async () => {
    const { result } = renderHook(() => useWebSocket())

    await act(async () => {
      result.current.connect()
    })

    const ws = wsInstances[wsInstances.length - 1]
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'state_sync',
          change_type: 'runtime_persistence_committed',
        }),
      } as MessageEvent)
    })

    const state = useAppStore.getState()
    expect(state.currentSession?.status).toBe('running')
    expect(state.currentSession?.currentRound).toBe(3)
    expect(state.stateVersion).toBe(7)
  })

  it('a full sync still applies status, round and version', async () => {
    const { result } = renderHook(() => useWebSocket())

    await act(async () => {
      result.current.connect()
    })

    const ws = wsInstances[wsInstances.length - 1]
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'state_sync',
          change_type: 'round_complete',
          version: 8,
          status: 'waiting_input',
          current_round: 4,
        }),
      } as MessageEvent)
    })

    const state = useAppStore.getState()
    expect(state.currentSession?.status).toBe('waiting_input')
    expect(state.currentSession?.currentRound).toBe(4)
    expect(state.stateVersion).toBe(8)
  })

  it('opens the LLM gate with its effective wait and clears it on skip', async () => {
    const { result } = renderHook(() => useWebSocket())

    await act(async () => {
      result.current.connect()
    })

    const ws = wsInstances[wsInstances.length - 1]
    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'state_sync',
          change_type: 'llm_failure',
          version: 8,
          status: 'waiting_input',
          llm_failure_info: {
            agent: 'report_generator',
            model: 'test-model',
            llm_server: 'https://llm.example/v1',
            error: 'connection failed',
            node_name: 'generate_detailed_report',
            generation: 1,
            wait_seconds: -1,
          },
        }),
      } as MessageEvent)
    })

    expect(useAppStore.getState().llmFailureInfo).toMatchObject({
      nodeName: 'generate_detailed_report',
      waitSeconds: -1,
    })
    expect(useAppStore.getState().showLlmTimeoutDialog).toBe(true)

    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({
          type: 'state_sync',
          change_type: 'llm_skip',
          version: 9,
          status: 'running',
        }),
      } as MessageEvent)
    })

    expect(useAppStore.getState().llmFailureInfo).toBeNull()
    expect(useAppStore.getState().showLlmTimeoutDialog).toBe(false)
  })
})
