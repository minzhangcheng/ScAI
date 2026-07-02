import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useWebSocket } from '../useApi'
import { useAppStore } from '../../stores/appStore'

describe('useWebSocket', () => {
  const closeMock = vi.fn()
  const wsInstances: Array<{
    url: string
    close: ReturnType<typeof vi.fn>
    onopen: ((event: Event) => void) | null
    onmessage: ((event: MessageEvent) => void) | null
    onclose: ((event: CloseEvent) => void) | null
    onerror: ((event: Event) => void) | null
  }> = []

  class MockWebSocket {
    public readyState: number = WebSocket.OPEN
    public close = closeMock
    public url: string
    private _onopen: ((event: Event) => void) | null = null
    public onmessage: ((event: MessageEvent) => void) | null = null
    public onclose: ((event: CloseEvent) => void) | null = null
    public onerror: ((event: Event) => void) | null = null

    constructor(url: string) {
      this.url = url
      wsInstances.push(this)
    }

    set onopen(handler: ((event: Event) => void) | null) {
      this._onopen = handler
      // Fire synchronously so connectingRef.current gets reset before connect() returns
      if (handler) {
        handler(new Event('open'))
      }
    }

    get onopen(): ((event: Event) => void) | null {
      return this._onopen
    }
  }

  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket as any)
    closeMock.mockClear()
    wsInstances.length = 0
    useAppStore.setState({
      currentSession: null,
      messages: [],
      wsConnected: false,
      stateVersion: 0,
    })
  })

  it('old socket onclose does not disturb the new socket after session switch', async () => {
    const { result } = renderHook(() => useWebSocket())

    const sessionA = {
      id: 'session-A',
      name: 'Session A',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    const sessionB = {
      id: 'session-B',
      name: 'Session B',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    act(() => {
      useAppStore.getState().setCurrentSession(sessionA)
    })

    await act(async () => {
      result.current.connect()
    })

    expect(wsInstances.length).toBe(1)
    expect(result.current.connected).toBe(true)

    act(() => {
      useAppStore.getState().setCurrentSession(sessionB)
    })

    await act(async () => {
      result.current.connect()
    })

    expect(wsInstances.length).toBe(2)
    expect(closeMock).toHaveBeenCalledTimes(1)

    const A = wsInstances[0]
    act(() => {
      A.onclose?.({ code: 1005 } as CloseEvent)
    })

    expect(result.current.connected).toBe(true)
    expect(wsInstances.length).toBe(2)
  })

  it('closes old socket and opens new on session switch', async () => {
    const { result } = renderHook(() => useWebSocket())

    expect(result.current.connected).toBe(false)

    const sessionA = {
      id: 'session-A',
      name: 'Session A',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    const sessionB = {
      id: 'session-B',
      name: 'Session B',
      status: 'completed' as const,
      currentRound: 0,
      createdAt: new Date().toISOString(),
      instruction: '',
      treatmentGroups: [],
      dataFiles: [],
      biochemistryIndicators: [],
      roundResults: [],
    }

    // Connect to session A
    act(() => {
      useAppStore.getState().setCurrentSession(sessionA)
    })

    await act(async () => {
      result.current.connect()
    })

    expect(wsInstances.length).toBe(1)
    expect(wsInstances[0].url).toContain('session-A')
    expect(closeMock).not.toHaveBeenCalled()

    // Switch to session B and connect
    act(() => {
      useAppStore.getState().setCurrentSession(sessionB)
    })

    await act(async () => {
      result.current.connect()
    })

    // Old socket should have been closed before opening new one
    expect(closeMock).toHaveBeenCalledTimes(1)
    // A new WebSocket should have been created for session B
    expect(wsInstances.length).toBe(2)
    expect(wsInstances[1].url).toContain('session-B')
  })
})
