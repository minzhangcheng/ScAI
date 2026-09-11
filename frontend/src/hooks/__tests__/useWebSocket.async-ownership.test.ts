import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWebSocket } from '../api/useWebSocket'
import { subscribeSnapshotListingChanged } from '../api/snapshots'
import { reloadSessionState } from '../api/websocket/sessionReload'
import { useAppStore } from '../../stores/appStore'
import { setToken } from '../../utils/authStorage'


const session = (id: string) => ({
  id,
  name: id,
  status: 'running' as const,
  currentRound: 0,
  createdAt: new Date().toISOString(),
  instruction: '',
  treatmentGroups: [],
  dataFiles: [],
  biochemistryIndicators: [],
  roundResults: [],
})


class SocketMock {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 3
  static instances: SocketMock[] = []
  readyState = SocketMock.OPEN
  onopen: ((event: Event) => unknown) | null = null
  onmessage: ((event: MessageEvent) => unknown) | null = null
  onerror: ((event: Event) => unknown) | null = null
  onclose: ((event: CloseEvent) => unknown) | null = null

  constructor(public url: string | URL) {
    SocketMock.instances.push(this)
  }

  close() {
    this.readyState = SocketMock.CLOSED
  }
}


const jsonResponse = (body: unknown, status = 200) => new Response(
  JSON.stringify(body),
  { status, headers: { 'Content-Type': 'application/json' } },
)


describe('WebSocket async response ownership', () => {
  beforeEach(() => {
    SocketMock.instances = []
    vi.restoreAllMocks()
    vi.stubGlobal('WebSocket', SocketMock)
    useAppStore.setState({
      currentSession: session('old'),
      messages: [],
      chatMessages: [],
      wsConnected: false,
      stateVersion: 0,
    })
  })

  it('drops a chat refresh started by a retired socket', async () => {
    let resolveOld!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveOld = resolve }),
    ))
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const old = SocketMock.instances[0]

    act(() => old.onmessage?.({
      data: JSON.stringify({ type: 'chat_result' }),
    } as MessageEvent))
    act(() => useAppStore.getState().setCurrentSession(session('new')))
    act(() => result.current.connect())

    await act(async () => {
      resolveOld(jsonResponse({
        messages: [{ role: 'assistant', content: 'stale chat' }],
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.id).toBe('new')
    expect(useAppStore.getState().chatMessages).toEqual([])
  })

  it.each(['none', 'session', 'manual', 'newer'])(
    'refreshes the effective budget with %s concurrent change', async change => {
      const pending: Array<(response: Response) => void> = []
      vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => pending.push(resolve))))
      useAppStore.setState({ agentServerConfigs: {
        pi: { llmBaseUrl: '', apiKey: '', contextWindow: 200000, effectiveContextWindow: 200000 },
      } })
      const { result } = renderHook(() => useWebSocket())
      act(() => result.current.connect())
      const socket = SocketMock.instances[0]
      const notify = () => socket.onmessage?.({ data: JSON.stringify({
        type: 'agent_context_updated', session_id: 'old', agent: 'pi',
      }) } as MessageEvent)
      act(() => { notify() })
      if (change === 'session') {
        act(() => useAppStore.getState().setCurrentSession(session('new')))
      } else if (change === 'manual') {
        act(() => useAppStore.getState().setAgentServerConfig('pi', {
          llmBaseUrl: '', apiKey: '', effectiveContextWindow: 200000,
        }))
      } else if (change === 'newer') {
        act(() => { notify() })
        await act(async () => {
          pending[1](jsonResponse({ pi: { effective_context_window: 128000 } }))
        })
      }
      await act(async () => {
        pending[0](jsonResponse({ pi: { effective_context_window: 160000 } }))
      })
      const config = useAppStore.getState().agentServerConfigs.pi
      expect(config?.effectiveContextWindow).toBe(
        change === 'none' ? 160000 : change === 'newer' ? 128000 : 200000,
      )
      expect(config?.contextWindow).toBe(change === 'manual' ? undefined : 200000)
    },
  )

  it('drops old-session chat terminal events before the new socket connects', () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const controller = new AbortController()
    useAppStore.setState({
      chatLoading: true,
      chatDelegationActive: true,
      chatDelegationPlan: 'old-session work',
      chatAbortController: controller,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const old = SocketMock.instances[0]

    act(() => useAppStore.getState().setCurrentSession(session('new')))
    // Deliberately do not call connect(): this is the ownership gap that the
    // previous test skipped by retiring the old socket immediately.
    act(() => {
      old.onmessage?.({ data: JSON.stringify({ type: 'chat_result' }) } as MessageEvent)
      old.onmessage?.({ data: JSON.stringify({ type: 'chat_cancelled' }) } as MessageEvent)
      old.onmessage?.({
        data: JSON.stringify({ type: 'chat_error', content: { error: 'from old' } }),
      } as MessageEvent)
    })

    const store = useAppStore.getState()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(store.chatLoading).toBe(true)
    expect(store.chatDelegationActive).toBe(true)
    expect(store.chatDelegationPlan).toBe('old-session work')
    expect(store.chatAbortController).toBe(controller)
    expect(store.chatMessages).toEqual([])
  })

  it('preserves identity on an id-bearing chat error with empty history', () => {
    vi.stubGlobal('fetch', vi.fn())
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]

    act(() => socket.onmessage?.({
      data: JSON.stringify({
        type: 'chat_error',
        content: {
          id: 'chat-error-terminal',
          error: 'delegation failed',
          followup_id: 'fu-error',
          branch_id: 'branch-error',
          reply_to: 'chat-user',
        },
      }),
    } as MessageEvent))

    expect(useAppStore.getState().chatMessages).toEqual([expect.objectContaining({
      id: 'chat-error-terminal',
      followup_id: 'fu-error',
      branch_id: 'branch-error',
      reply_to: 'chat-user',
    })])
  })

  it('drops every old-session websocket event before reconnect', () => {
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const old = SocketMock.instances[0]
    act(() => useAppStore.getState().setCurrentSession(session('new')))

    act(() => old.onmessage?.({
      data: JSON.stringify({
        type: 'completed',
        content: { final_summary: 'old summary' },
      }),
    } as MessageEvent))

    const store = useAppStore.getState()
    expect(store.currentSession?.id).toBe('new')
    expect(store.currentSession?.status).toBe('running')
    expect(store.messages).toEqual([])
  })

  it('drops a same-session conflict reload from an older generation', async () => {
    let resolveOld!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveOld = resolve }),
    ))
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 10,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const old = SocketMock.instances[0]

    act(() => old.onmessage?.({
      data: JSON.stringify({ type: 'state_sync', version: 9 }),
    } as MessageEvent))
    act(() => result.current.connect())

    await act(async () => {
      resolveOld(jsonResponse({
        id: 'same',
        status: 'completed',
        current_round: 99,
        state_version: 9,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.currentRound).toBe(4)
    expect(useAppStore.getState().stateVersion).toBe(10)
  })

  it('accepts an authoritative lower reload and then applies the next sync', async () => {
    let resolveReload!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveReload = resolve }),
    ))
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 50,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]

    act(() => socket.onmessage?.({
      data: JSON.stringify({ type: 'state_sync', version: 49 }),
    } as MessageEvent))
    await act(async () => {
      resolveReload(jsonResponse({
        session_id: 'same',
        status: 'running',
        current_round: 3,
        state_version: 47,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().stateVersion).toBe(47)
    expect(useAppStore.getState().currentSession?.currentRound).toBe(3)

    act(() => socket.onmessage?.({
      data: JSON.stringify({
        type: 'state_sync',
        content: { version: 48, current_round: 4 },
      }),
    } as MessageEvent))

    expect(useAppStore.getState().stateVersion).toBe(48)
    expect(useAppStore.getState().currentSession?.currentRound).toBe(4)
  })

  it('drops a pending reload after a newer sync in the same generation', async () => {
    let resolveReload!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveReload = resolve }),
    ))
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 10,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]

    act(() => socket.onmessage?.({
      data: JSON.stringify({ type: 'state_sync', version: 9 }),
    } as MessageEvent))
    act(() => socket.onmessage?.({
      data: JSON.stringify({
        type: 'state_sync', content: { version: 11, current_round: 5 },
      }),
    } as MessageEvent))

    await act(async () => {
      resolveReload(jsonResponse({
        session_id: 'same', current_round: 99, state_version: 10,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.currentRound).toBe(5)
    expect(useAppStore.getState().stateVersion).toBe(11)
  })

  it('drops a pending reload after an equal-version sync in the same generation', async () => {
    let resolveReload!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveReload = resolve }),
    ))
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 10,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]

    act(() => socket.onmessage?.({
      data: JSON.stringify({ type: 'state_sync', version: 9 }),
    } as MessageEvent))
    act(() => socket.onmessage?.({
      data: JSON.stringify({
        type: 'state_sync', content: { version: 10, current_round: 5 },
      }),
    } as MessageEvent))

    await act(async () => {
      resolveReload(jsonResponse({
        session_id: 'same', current_round: 99, state_version: 10,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.currentRound).toBe(5)
    expect(useAppStore.getState().stateVersion).toBe(10)
  })

  it('drops a pending reload when the same session ID gets a new object', async () => {
    let resolveReload!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveReload = resolve }),
    ))
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 10,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]

    act(() => socket.onmessage?.({
      data: JSON.stringify({ type: 'state_sync', version: 9 }),
    } as MessageEvent))
    act(() => useAppStore.getState().setCurrentSession({
      ...session('same'), currentRound: 7,
    }))

    await act(async () => {
      resolveReload(jsonResponse({
        session_id: 'same', current_round: 99, state_version: 9,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.currentRound).toBe(7)
  })

  it('does not let an old lifecycle error retire the new generation', async () => {
    let resolveOld!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(
      new Promise<Response>(resolve => { resolveOld = resolve }),
    ))
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 10,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const old = SocketMock.instances[0]
    act(() => old.onmessage?.({
      data: JSON.stringify({ type: 'state_sync', version: 9 }),
    } as MessageEvent))
    act(() => result.current.connect())

    await act(async () => {
      resolveOld(jsonResponse({
        detail: { code: 'SESSION_GONE', message: 'old generation is gone' },
      }, 410))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(useAppStore.getState().currentSession?.id).toBe('same')
    expect(useAppStore.getState().currentSession?.currentRound).toBe(4)
  })

  it('propagates an owned lifecycle error after retiring the session', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: { code: 'SESSION_GONE', message: 'session is gone' },
    }, 410)))
    useAppStore.setState({ currentSession: session('gone') })

    await expect(reloadSessionState('gone', () => true)).rejects.toMatchObject({
      code: 'SESSION_GONE',
    })
    expect(useAppStore.getState().currentSession).toBeNull()
  })

  it('reloads authoritative state before overlap-bounded reconnect paging', async () => {
    vi.useFakeTimers()
    setToken('test-token')
    const invalidated = vi.fn()
    const unsubscribe = subscribeSnapshotListingChanged(invalidated)
    const resolvers: Array<(response: Response) => void> = []
    const fetchMock = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => {
      resolvers.push(resolve)
    }))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({ currentSession: session('history') })
    const { result } = renderHook(() => useWebSocket())

    act(() => result.current.connect())
    const first = SocketMock.instances[0]
    await act(async () => {
      await first.onopen?.(new Event('open'))
    })
    act(() => first.onclose?.({ code: 1006, reason: '' } as CloseEvent))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    const second = SocketMock.instances[1]
    let reconnectOpen!: Promise<unknown>
    await act(async () => {
      reconnectOpen = Promise.resolve(second.onopen?.(new Event('open')))
      await Promise.resolve()
    })

    expect(invalidated).toHaveBeenCalledTimes(1)
    expect(invalidated).toHaveBeenCalledWith('history')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(String(fetchMock.mock.calls[0][0])).toBe('/api/session/history/load')
    await act(async () => {
      resolvers[0](jsonResponse({ session_id: 'history', status: 'running', messages: [] }))
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    })

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(String(fetchMock.mock.calls[1][0])).toContain('/api/session/history/messages')
    await act(async () => {
      resolvers[1](jsonResponse({ session_id: 'history', status: 'running', messages: [], has_more: false }))
      await reconnectOpen
    })
    unsubscribe()
    vi.useRealTimers()
  })

  it('continues reconnect history paging when a sync overtakes the state reload', async () => {
    vi.useFakeTimers()
    setToken('test-token')
    const resolvers: Array<(response: Response) => void> = []
    const fetchMock = vi.fn<typeof fetch>(() => new Promise<Response>(resolve => {
      resolvers.push(resolve)
    }))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({
      currentSession: { ...session('history'), currentRound: 1 },
      stateVersion: 1,
    })
    const { result } = renderHook(() => useWebSocket())

    act(() => result.current.connect())
    const first = SocketMock.instances[0]
    await act(async () => {
      await first.onopen?.(new Event('open'))
    })
    act(() => first.onclose?.({ code: 1006, reason: '' } as CloseEvent))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })
    const second = SocketMock.instances[1]
    let reconnectOpen!: Promise<unknown>
    await act(async () => {
      reconnectOpen = Promise.resolve(second.onopen?.(new Event('open')))
      await Promise.resolve()
    })

    act(() => second.onmessage?.({
      data: JSON.stringify({
        type: 'state_sync', content: { version: 2, current_round: 2 },
      }),
    } as MessageEvent))
    await act(async () => {
      resolvers[0](jsonResponse({
        session_id: 'history', current_round: 99, state_version: 1,
      }))
      await vi.advanceTimersByTimeAsync(0)
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    })

    expect(String(fetchMock.mock.calls[1][0])).toContain('/messages?limit=20')
    await act(async () => {
      resolvers[1](jsonResponse({ messages: [], has_more: false }))
      await reconnectOpen
    })
    expect(useAppStore.getState().currentSession?.currentRound).toBe(2)
    expect(useAppStore.getState().stateVersion).toBe(2)
    vi.useRealTimers()
  })

  it('surfaces reconnect history failure as retryable state', async () => {
    vi.useFakeTimers()
    setToken('test-token')
    const resolvers: Array<(response: Response) => void> = []
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => {
      resolvers.push(resolve)
    })))
    useAppStore.setState({
      currentSession: session('history-failure'),
      realtimeHistoryError: null,
    })
    const { result } = renderHook(() => useWebSocket())

    act(() => result.current.connect())
    const first = SocketMock.instances[0]
    await act(async () => {
      await first.onopen?.(new Event('open'))
    })
    act(() => first.onclose?.({ code: 1006, reason: '' } as CloseEvent))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    const second = SocketMock.instances[1]
    let reconnectOpen!: Promise<unknown>
    await act(async () => {
      reconnectOpen = Promise.resolve(second.onopen?.(new Event('open')))
      await Promise.resolve()
    })
    await act(async () => {
      resolvers[0](jsonResponse({
        session_id: 'history-failure',
        status: 'running',
      }))
      await vi.waitFor(() => expect(resolvers).toHaveLength(2))
      resolvers[1](jsonResponse({ detail: 'history unavailable' }, 503))
      await reconnectOpen
    })

    expect(useAppStore.getState().realtimeHistoryError).toContain('history unavailable')
    vi.useRealTimers()
  })

  it('stops reconnect recovery and surfaces an authoritative load failure', async () => {
    vi.useFakeTimers()
    setToken('test-token')
    const resolvers: Array<(response: Response) => void> = []
    const fetchMock = vi.fn(() => new Promise<Response>(resolve => {
      resolvers.push(resolve)
    }))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({
      currentSession: session('load-failure'),
      realtimeHistoryError: null,
    })
    const { result } = renderHook(() => useWebSocket())

    act(() => result.current.connect())
    const first = SocketMock.instances[0]
    await act(async () => {
      await first.onopen?.(new Event('open'))
    })
    act(() => first.onclose?.({ code: 1006, reason: '' } as CloseEvent))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    const second = SocketMock.instances[1]
    let reconnectOpen!: Promise<unknown>
    await act(async () => {
      reconnectOpen = Promise.resolve(second.onopen?.(new Event('open')))
      await Promise.resolve()
    })
    await act(async () => {
      resolvers[0](jsonResponse({ detail: 'load unavailable' }, 503))
      await reconnectOpen
    })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(useAppStore.getState().realtimeHistoryError).toContain('load unavailable')
    vi.useRealTimers()
  })
})
