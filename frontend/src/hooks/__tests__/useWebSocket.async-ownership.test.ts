import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWebSocket } from '../api/useWebSocket'
import { useAppStore } from '../../stores/appStore'


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

  it('scopes restore-commit dedupe to session plus socket generation', async () => {
    const resolvers: Array<(response: Response) => void> = []
    const fetchMock = vi.fn().mockImplementation(() => (
      new Promise<Response>(resolve => { resolvers.push(resolve) })
    ))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({
      currentSession: { ...session('same'), currentRound: 4 },
      stateVersion: 10,
    })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const old = SocketMock.instances[0]
    const commit = {
      type: 'state_sync',
      content: {
        change_type: 'snapshot_transaction',
        snapshot_transaction: {
          transaction_id: 'tx-shared',
          session_id: 'same',
          status: 'committed',
        },
      },
    }

    act(() => old.onmessage?.({ data: JSON.stringify(commit) } as MessageEvent))
    act(() => result.current.connect())
    const current = SocketMock.instances[1]
    act(() => current.onmessage?.({ data: JSON.stringify(commit) } as MessageEvent))
    expect(fetchMock).toHaveBeenCalledTimes(2)

    await act(async () => {
      resolvers[0](jsonResponse({
        id: 'same', current_round: 88, state_version: 8,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useAppStore.getState().currentSession?.currentRound).toBe(4)

    await act(async () => {
      resolvers[1](jsonResponse({
        id: 'same', current_round: 7, state_version: 11,
      }))
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useAppStore.getState().currentSession?.currentRound).toBe(7)
    expect(useAppStore.getState().stateVersion).toBe(11)
  })

  it('deduplicates repeated restore commits within one generation', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      id: 'same', current_round: 3, state_version: 2,
    }))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({ currentSession: session('same') })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]
    const commit = JSON.stringify({
      type: 'state_sync',
      content: {
        change_type: 'snapshot_transaction',
        snapshot_transaction: {
          transaction_id: 'tx-one',
          session_id: 'same',
          status: 'committed',
        },
      },
    })

    act(() => {
      socket.onmessage?.({ data: commit } as MessageEvent)
      socket.onmessage?.({ data: commit } as MessageEvent)
    })
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('deduplicates a replayed transaction after another commit in the same generation', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({
      id: 'same', current_round: 3, state_version: 2,
    }))
    vi.stubGlobal('fetch', fetchMock)
    useAppStore.setState({ currentSession: session('same') })
    const { result } = renderHook(() => useWebSocket())
    act(() => result.current.connect())
    const socket = SocketMock.instances[0]
    const commit = (transactionId: string) => JSON.stringify({
      type: 'state_sync',
      content: {
        change_type: 'snapshot_transaction',
        snapshot_transaction: {
          transaction_id: transactionId,
          session_id: 'same',
          status: 'committed',
        },
      },
    })

    act(() => {
      socket.onmessage?.({ data: commit('tx-one') } as MessageEvent)
      socket.onmessage?.({ data: commit('tx-two') } as MessageEvent)
      socket.onmessage?.({ data: commit('tx-one') } as MessageEvent)
    })
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})
